//! The face-embedding worker thread: owns the model and runs every embedding request in turn.
//! The session sends jobs and receives outcomes over channels, so the UI thread never touches
//! the model.  Every job runs under `catch_unwind`: a panic inside candle becomes an error
//! outcome, drops the (possibly inconsistent) model and leaves the worker ready for the next
//! job.  The model is unloaded after [`IDLE_UNLOAD`] without requests.
//!
//! Modelled on the face-detection worker (`crate::face::worker`).

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use lightcraft_catalog::PhotoId;

/// Unload the model after this long without a request.
pub const IDLE_UNLOAD: Duration = Duration::from_secs(10 * 60);

/// One request: compute the embedding for a single face crop.
pub struct Job {
    /// The model folder (containing `adaface_ir18.safetensors`).
    pub dir: PathBuf,
    /// Which photo the face belongs to.
    pub photo: PhotoId,
    /// Index of the face region within the photo's regions (so the caller can match it back).
    pub region_index: usize,
    /// 8-bit RGB pixels of the face crop, row-major, exactly 112×112×3 bytes.
    pub rgb: Vec<u8>,
    /// Where to send the result.
    pub reply: mpsc::Sender<Outcome>,
}

/// A finished embedding request.
pub struct Outcome {
    pub photo: PhotoId,
    /// Which face region this embedding belongs to.
    pub region_index: usize,
    pub result: Result<Vec<f32>, String>,
}

/// Counters the session reads without locking.
#[derive(Default)]
pub struct Shared {
    /// Requests queued or running.
    pub pending: AtomicUsize,
    /// The model is in memory.
    pub loaded: AtomicBool,
    /// Loading the model right now (the slow part).
    pub loading: AtomicBool,
}

/// Decrements a counter when dropped (also when unwinding).
struct Done<'a>(&'a AtomicUsize);

impl Drop for Done<'_> {
    fn drop(&mut self) {
        let _ = self.0.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| Some(n.saturating_sub(1)));
    }
}

/// Sets a flag while alive.
struct Flag<'a>(&'a AtomicBool);

impl<'a> Flag<'a> {
    fn raise(f: &'a AtomicBool) -> Self {
        f.store(true, Ordering::SeqCst);
        Flag(f)
    }
}

impl Drop for Flag<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// The session's handle on the worker thread (started on the first request, restarted if it
/// ever ends).
#[derive(Default)]
pub struct Worker {
    tx: Option<mpsc::Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    pub shared: Arc<Shared>,
}

impl Worker {
    /// Whether the thread has ended (it never should); its counters are then meaningless.
    fn dead(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| t.is_finished())
    }

    pub fn pending(&self) -> usize {
        if self.dead() { 0 } else { self.shared.pending.load(Ordering::SeqCst) }
    }

    pub fn loading(&self) -> bool {
        !self.dead() && self.shared.loading.load(Ordering::SeqCst)
    }

    pub fn loaded(&self) -> bool {
        !self.dead() && self.shared.loaded.load(Ordering::SeqCst)
    }

    /// Queue `job`; returns at once.
    pub fn submit(&mut self, job: Job) -> Result<(), String> {
        if self.dead() {
            self.tx = None;
            self.thread = None;
            self.shared = Arc::new(Shared::default());
        }
        self.shared.pending.fetch_add(1, Ordering::SeqCst);
        let job = match &self.tx {
            Some(tx) => match tx.send(job) {
                Ok(()) => return Ok(()),
                Err(mpsc::SendError(job)) => job,
            },
            None => job,
        };
        // (re)start the thread
        let (tx, rx) = mpsc::channel();
        let shared = self.shared.clone();
        match std::thread::Builder::new().name("face-embed".into()).spawn(move || run(&rx, &shared, IDLE_UNLOAD)) {
            Ok(t) => {
                self.thread = Some(t);
                self.tx = Some(tx.clone());
                tx.send(job).map_err(|_| "the face embedding worker stopped".to_string())
            }
            Err(e) => {
                let _ = self.shared.pending.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| Some(n.saturating_sub(1)));
                Err(format!("could not start the face embedding worker: {e}"))
            }
        }
    }
}

/// The loaded model.
#[derive(Default)]
struct State {
    model: Option<(PathBuf, lightcraft_face_embed::FaceEmbedder)>,
}

fn run(rx: &mpsc::Receiver<Job>, shared: &Shared, idle: Duration) {
    let mut state = State::default();
    loop {
        let job = match rx.recv_timeout(idle) {
            Ok(j) => j,
            Err(RecvTimeoutError::Timeout) => {
                if state.model.is_some() {
                    state = State::default();
                    shared.loaded.store(false, Ordering::SeqCst);
                    log::info!("face embedding model unloaded after {} minutes without use", idle.as_secs() / 60);
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => return,
        };
        let _done = Done(&shared.pending);
        let result = match std::panic::catch_unwind(AssertUnwindSafe(|| run_job(&mut state, shared, &job))) {
            Ok(r) => r,
            Err(p) => {
                let why = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_default();
                log::error!("face embedding model panicked: {why}");
                state = State::default();
                Err("the face embedding model failed unexpectedly (it will be reloaded on the next try)".to_string())
            }
        };
        shared.loaded.store(state.model.is_some(), Ordering::SeqCst);
        let _ = job.reply.send(Outcome { photo: job.photo, region_index: job.region_index, result });
    }
}

/// Load the model from `dir` if not already loaded (or if the dir changed).
fn model<'a>(state: &'a mut State, shared: &Shared, dir: &std::path::Path) -> Result<&'a mut lightcraft_face_embed::FaceEmbedder, String> {
    if state.model.as_ref().is_none_or(|(d, _)| d != dir) {
        state.model = None;
        let _busy = Flag::raise(&shared.loading);
        let t = web_time::Instant::now();
        let m = lightcraft_face_embed::FaceEmbedder::load(dir).map_err(|e| e.to_string())?;
        log::info!("face embedding model loaded in {:?}", t.elapsed());
        state.model = Some((dir.to_path_buf(), m));
        shared.loaded.store(true, Ordering::SeqCst);
    }
    state.model.as_mut().map(|(_, m)| m).ok_or_else(|| "the model did not load".to_string())
}

fn run_job(state: &mut State, shared: &Shared, job: &Job) -> Result<Vec<f32>, String> {
    let m = model(state, shared, &job.dir)?;
    let t = web_time::Instant::now();
    let embedding = m.embed(&job.rgb).map_err(|e| e.to_string())?;
    log::info!("face embedding computed in {:?}", t.elapsed());
    Ok(embedding.vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(reply: &mpsc::Sender<Outcome>) -> Job {
        // An empty folder: loading fails with an error (no model files).
        let dir = std::env::temp_dir().join(format!("bruce-embed-worker-none-{}", std::process::id()));
        Job { dir, photo: PhotoId(1), region_index: 0, rgb: vec![0u8; 112 * 112 * 3], reply: reply.clone() }
    }

    #[test]
    fn errors_come_back_and_the_counters_return_to_zero() {
        let mut w = Worker::default();
        let (tx, rx) = mpsc::channel();
        for _ in 0..3 {
            w.submit(job(&tx)).ok();
        }
        for _ in 0..3 {
            let o = rx.recv_timeout(Duration::from_secs(30));
            if let Ok(o) = o {
                assert!(o.result.is_err());
            }
        }
        let t = web_time::Instant::now();
        while w.pending() > 0 && t.elapsed() < Duration::from_secs(5) {
            std::thread::yield_now();
        }
        assert_eq!(w.pending(), 0);
        assert!(!w.loading() && !w.loaded());
    }
}
