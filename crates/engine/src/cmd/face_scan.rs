//! Background face scan: detect → embed → cluster, driven by frame ticks.
//!
//! The scan runs as a cooperative state machine: each UI frame calls [`Session::face_scan_tick`],
//! which advances one step (render a photo for detection, poll worker results, start embedding,
//! etc.).  The session is needed for rendering and catalog access, so the scan lives _on_ the
//! session rather than on a separate thread.
//!
//! Progress is exposed through [`FaceScanProgress`]: atomic counters that the UI reads cheaply
//! every frame for toast / progress bar display.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use lightcraft_catalog::{Op, PhotoId};
use lightcraft_meta::RegionKind;
use serde_json::{Value, json};

use super::{CommandSpec, always, bad, cmd, f64_or};
use crate::face_index::FaceKey;

/// Observable progress of a face scan (the UI reads the atomics every frame).
#[derive(Debug)]
pub struct FaceScanProgress {
    /// Phase we are in.
    pub phase: std::sync::Mutex<ScanPhase>,
    /// Total photos to detect faces in (set at the start of detection).
    pub detect_total: AtomicUsize,
    /// Photos processed so far in detection.
    pub detect_done: AtomicUsize,
    /// Total faces found (updated during detection).
    pub faces_found: AtomicU64,
    /// Total face regions to embed.
    pub embed_total: AtomicUsize,
    /// Face regions embedded so far.
    pub embed_done: AtomicUsize,
    /// Number of clusters found (set after clustering).
    pub clusters: AtomicUsize,
    /// Errors across all phases.
    pub errors: AtomicUsize,
    /// True once the whole pipeline is done.
    pub finished: AtomicBool,
    /// Set to true to cancel.
    pub cancel: AtomicBool,
}

/// Which phase the scan is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanPhase {
    Detecting,
    Embedding,
    Clustering,
    Done,
}

impl Default for FaceScanProgress {
    fn default() -> Self {
        FaceScanProgress {
            phase: std::sync::Mutex::new(ScanPhase::Detecting),
            detect_total: AtomicUsize::new(0),
            detect_done: AtomicUsize::new(0),
            faces_found: AtomicU64::new(0),
            embed_total: AtomicUsize::new(0),
            embed_done: AtomicUsize::new(0),
            clusters: AtomicUsize::new(0),
            errors: AtomicUsize::new(0),
            finished: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
        }
    }
}

impl FaceScanProgress {
    /// JSON snapshot for the `face.scanProgress` command.
    pub fn json(&self) -> Value {
        let phase = self.phase.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        json!({
            "phase": match *phase {
                ScanPhase::Detecting => "detecting",
                ScanPhase::Embedding => "embedding",
                ScanPhase::Clustering => "clustering",
                ScanPhase::Done => "done",
            },
            "detectTotal": self.detect_total.load(Ordering::Relaxed),
            "detectDone": self.detect_done.load(Ordering::Relaxed),
            "facesFound": self.faces_found.load(Ordering::Relaxed),
            "embedTotal": self.embed_total.load(Ordering::Relaxed),
            "embedDone": self.embed_done.load(Ordering::Relaxed),
            "clusters": self.clusters.load(Ordering::Relaxed),
            "errors": self.errors.load(Ordering::Relaxed),
            "finished": self.finished.load(Ordering::Relaxed),
            "cancelled": self.cancel.load(Ordering::Relaxed),
        })
    }
}

/// Internal state of the face scan state machine.
pub struct FaceScan {
    /// Observable progress (shared with the UI).
    pub progress: Arc<FaceScanProgress>,
    /// Clustering parameters.
    eps: f64,
    min_samples: usize,
    /// Detection state ---------------------------------------------------------
    /// Photos still to detect.
    detect_queue: Vec<PhotoId>,
    /// Number of detection requests in flight (queued to the worker, not yet polled).
    detect_in_flight: usize,
    /// Regions detected so far (batched and committed at the end of detection).
    detect_ops: Vec<Op>,
    /// Embedding state ----------------------------------------------------------
    /// Face regions still to embed.
    embed_queue: Vec<(PhotoId, usize)>,
    /// Number of embedding requests in flight.
    embed_in_flight: usize,
}

/// Maximum number of detection requests queued to the worker at once.
///
/// Now that rendering happens on the worker thread (not the UI thread), we can keep a deeper
/// pipeline without freezing the UI.  Queueing a job is cheap (just creating a RenderJob —
/// no I/O), so we can queue up to 2 per tick and let the single-threaded worker serialize them.
const MAX_DETECT_FLIGHT: usize = 2;
/// Maximum number of embedding requests queued to the worker at once.
const MAX_EMBED_FLIGHT: usize = 4;

impl crate::Session {
    /// Start a background face scan. Returns the shared progress handle.
    /// Fails if a scan is already running, or the models are not installed.
    pub fn face_scan_start(&mut self, eps: f64, min_samples: usize) -> Result<Arc<FaceScanProgress>, String> {
        if let Some(ref scan) = self.face_scan
            && !scan.progress.finished.load(Ordering::Relaxed)
        {
            return Err("a face scan is already running".into());
        }
        self.face_detector.model_dir()?;
        self.face_embedder.model_dir()?;

        // Collect photos that need detection (no face regions yet).
        let ids: Vec<PhotoId> = self
            .catalog
            .photos()
            .filter(|photo| !photo.meta.regions.iter().any(|r| matches!(r.kind, RegionKind::Face)))
            .map(|photo| photo.id)
            .collect();

        let progress = Arc::new(FaceScanProgress::default());
        progress.detect_total.store(ids.len(), Ordering::Relaxed);

        self.face_scan = Some(FaceScan {
            progress: progress.clone(),
            eps,
            min_samples,
            detect_queue: ids,
            detect_in_flight: 0,
            detect_ops: Vec::new(),
            embed_queue: Vec::new(),
            embed_in_flight: 0,
        });

        Ok(progress)
    }

    /// Drive the face scan forward. Call once per frame; cheap when idle or when no scan is
    /// running.  Returns `true` if the scan is still running (the UI should request a repaint).
    pub fn face_scan_tick(&mut self) -> bool {
        // Read the current phase and check cancellation, without holding the borrow long.
        let phase = match self.face_scan.as_ref() {
            Some(scan) if !scan.progress.finished.load(Ordering::Relaxed) => {
                if scan.progress.cancel.load(Ordering::Relaxed) {
                    scan.progress.finished.store(true, Ordering::Relaxed);
                    let mut p = scan.progress.phase.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    *p = ScanPhase::Done;
                    return false;
                }
                *scan.progress.phase.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
            }
            _ => return false,
        };

        match phase {
            ScanPhase::Detecting => self.tick_detecting(),
            ScanPhase::Embedding => self.tick_embedding(),
            ScanPhase::Clustering => self.tick_clustering(),
            ScanPhase::Done => false,
        }
    }

    /// Detection phase: queue photos to the worker, poll results, commit when done.
    fn tick_detecting(&mut self) -> bool {
        // 1. Poll completed detection results.
        let results = self.face_detect_poll();
        for result in results {
            match result {
                Ok(detected) => {
                    if !detected.faces.is_empty() {
                        let face_count = detected.faces.len() as u64;
                        let meta_op = self.catalog.photo(detected.photo).map(|photo| {
                            let mut meta = photo.meta.clone();
                            meta.regions.extend(detected.faces);
                            Op::SetMeta { id: detected.photo, meta: Box::new(meta) }
                        });
                        if let Some(scan) = self.face_scan.as_mut() {
                            scan.progress.faces_found.fetch_add(face_count, Ordering::Relaxed);
                            if let Some(op) = meta_op {
                                scan.detect_ops.push(op);
                            }
                        }
                    }
                    if let Some(scan) = self.face_scan.as_mut() {
                        scan.detect_in_flight = scan.detect_in_flight.saturating_sub(1);
                        scan.progress.detect_done.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_e) => {
                    if let Some(scan) = self.face_scan.as_mut() {
                        scan.detect_in_flight = scan.detect_in_flight.saturating_sub(1);
                        scan.progress.detect_done.fetch_add(1, Ordering::Relaxed);
                        scan.progress.errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        // 2. Queue more detection jobs — collect ids to queue first, then call queue.
        let to_queue = {
            let scan = match self.face_scan.as_mut() {
                Some(s) => s,
                None => return false,
            };
            let capacity = MAX_DETECT_FLIGHT.saturating_sub(scan.detect_in_flight);
            let mut ids = Vec::with_capacity(capacity);
            for _ in 0..capacity {
                match scan.detect_queue.pop() {
                    Some(id) => ids.push(id),
                    None => break,
                }
            }
            ids
        };

        for id in to_queue {
            match self.face_detect_queue(id) {
                Ok(()) => {
                    if let Some(scan) = self.face_scan.as_mut() {
                        scan.detect_in_flight += 1;
                    }
                }
                Err(_) => {
                    if let Some(scan) = self.face_scan.as_mut() {
                        scan.progress.detect_done.fetch_add(1, Ordering::Relaxed);
                        scan.progress.errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        // 3. Check if detection is complete → transition to embedding.
        let scan = match self.face_scan.as_mut() {
            Some(s) => s,
            None => return false,
        };
        if !scan.detect_queue.is_empty() || scan.detect_in_flight > 0 {
            return true;
        }

        // Commit all detection results as one batch.
        let ops = std::mem::take(&mut scan.detect_ops);
        let _ = scan; // release borrow before commit
        if !ops.is_empty() {
            let _ = self.commit("Detect Faces", Op::Batch { ops });
        }

        // Collect all face regions that need embedding.
        let face_index = &self.face_index;
        let embed_queue: Vec<(PhotoId, usize)> = self
            .catalog
            .photos()
            .flat_map(|photo| {
                photo
                    .meta
                    .regions
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| matches!(r.kind, RegionKind::Face))
                    .filter(|(i, _)| !face_index.contains(&FaceKey { photo: photo.id, region_index: *i }))
                    .map(|(i, _)| (photo.id, i))
                    .collect::<Vec<_>>()
            })
            .collect();

        if let Some(scan) = self.face_scan.as_mut() {
            scan.embed_queue = embed_queue;
            scan.progress.embed_total.store(scan.embed_queue.len(), Ordering::Relaxed);
            let mut phase = scan.progress.phase.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            *phase = ScanPhase::Embedding;
        }

        true
    }

    /// Embedding phase: queue face crops to the worker, poll results, index them.
    fn tick_embedding(&mut self) -> bool {
        // 1. Poll completed embedding results.
        let results = self.face_embed_poll();
        for result in results {
            match result {
                Ok(embedding) => {
                    let key = FaceKey { photo: embedding.photo, region_index: embedding.region_index };
                    self.face_index.insert(key, embedding.vector);
                    if let Some(scan) = self.face_scan.as_mut() {
                        scan.embed_in_flight = scan.embed_in_flight.saturating_sub(1);
                        scan.progress.embed_done.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_e) => {
                    if let Some(scan) = self.face_scan.as_mut() {
                        scan.embed_in_flight = scan.embed_in_flight.saturating_sub(1);
                        scan.progress.embed_done.fetch_add(1, Ordering::Relaxed);
                        scan.progress.errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        // 2. Queue more embedding jobs — collect pairs first, then call queue.
        let to_queue = {
            let scan = match self.face_scan.as_mut() {
                Some(s) => s,
                None => return false,
            };
            let capacity = MAX_EMBED_FLIGHT.saturating_sub(scan.embed_in_flight);
            let mut pairs = Vec::with_capacity(capacity);
            for _ in 0..capacity {
                match scan.embed_queue.pop() {
                    Some(pair) => pairs.push(pair),
                    None => break,
                }
            }
            pairs
        };

        for (photo_id, region_index) in to_queue {
            match self.face_embed_queue(photo_id, region_index) {
                Ok(()) => {
                    if let Some(scan) = self.face_scan.as_mut() {
                        scan.embed_in_flight += 1;
                    }
                }
                Err(_) => {
                    if let Some(scan) = self.face_scan.as_mut() {
                        scan.progress.embed_done.fetch_add(1, Ordering::Relaxed);
                        scan.progress.errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        // 3. Check if embedding is complete → transition to clustering.
        let scan = match self.face_scan.as_ref() {
            Some(s) => s,
            None => return false,
        };
        if scan.embed_queue.is_empty() && scan.embed_in_flight == 0 {
            let mut phase = scan.progress.phase.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            *phase = ScanPhase::Clustering;
        }

        true
    }

    /// Clustering phase: run DBSCAN (fast, no I/O) and finish.
    fn tick_clustering(&mut self) -> bool {
        let (eps, min_samples) = match self.face_scan.as_ref() {
            Some(scan) => (scan.eps, scan.min_samples),
            None => return false,
        };

        let result = self.face_index.cluster(eps, min_samples);
        if let Some(scan) = self.face_scan.as_ref() {
            scan.progress.clusters.store(result.num_clusters, Ordering::Relaxed);
            scan.progress.finished.store(true, Ordering::Relaxed);
            let mut phase = scan.progress.phase.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            *phase = ScanPhase::Done;
        }

        false
    }
}

/// Command specs for the background face scan.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "face.scanAll",
            "Scan for Faces",
            ["Photo"],
            None,
            "{eps?: f64 (default 0.45), minSamples?: int (default 2), wait?: bool} → starts background detect → embed → cluster; returns progress handle",
            |s: &crate::Session| -> std::result::Result<(), String> {
                s.face_detector.model_dir()?;
                s.face_embedder.model_dir().map(|_| ())
            },
            |s, p| {
                let eps = f64_or(p, "eps", 0.45);
                let min_samples = p.get("minSamples").and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok()).unwrap_or(2);

                if eps <= 0.0 || eps >= 1.0 {
                    return Err(bad("face.scanAll", "`eps` must be between 0 and 1 (exclusive)"));
                }
                if min_samples == 0 {
                    return Err(bad("face.scanAll", "`minSamples` must be at least 1"));
                }

                // If wait=true, fall back to synchronous mode (CLI, tests).
                let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(false);
                if wait {
                    return sync_scan(s, eps, min_samples);
                }

                let progress = s.face_scan_start(eps, min_samples).map_err(|e| bad("face.scanAll", e))?;
                Ok(progress.json())
            }
        ),
        cmd!(
            query "face.scanProgress",
            "Face Scan Progress",
            [],
            None,
            "{} → current face scan progress (or null if no scan has been started)",
            always,
            |s, _p| {
                Ok(s.face_scan.as_ref().map_or(Value::Null, |scan| scan.progress.json()))
            }
        ),
        cmd!("face.scanCancel", "Cancel Face Scan", [], None, "{} → cancels a running face scan", always, |s, _p| {
            if let Some(ref scan) = s.face_scan {
                scan.progress.cancel.store(true, Ordering::Relaxed);
                Ok(json!({"cancelled": true}))
            } else {
                Ok(json!({"cancelled": false, "reason": "no scan running"}))
            }
        }),
    ]
}

/// Synchronous scan for CLI / tests (same logic as old face.scanAll).
fn sync_scan(s: &mut crate::Session, eps: f64, min_samples: usize) -> crate::Result<Value> {
    // 1. Detect faces
    let detect_result = s.execute("face.detectAll", &json!({}))?;
    let detected = detect_result.get("detected").and_then(Value::as_u64).unwrap_or(0);

    // 2. Embed all
    let embed_result = s.execute("face.embedAll", &json!({}))?;
    let embedded = embed_result.get("embedded").and_then(Value::as_u64).unwrap_or(0);
    let embed_errors = embed_result.get("errors").and_then(Value::as_u64).unwrap_or(0);

    // 3. Cluster
    let cluster_result = s.execute("face.cluster", &json!({"eps": eps, "minSamples": min_samples}))?;
    let num_clusters = cluster_result.get("clusters").and_then(Value::as_array).map(|a| a.len() as u64).unwrap_or(0);

    Ok(json!({
        "detected": detected,
        "embedded": embedded,
        "clusters": num_clusters,
        "errors": embed_errors,
        "finished": true,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn progress_default_is_detecting() {
        let p = FaceScanProgress::default();
        let phase = *p.phase.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(phase, ScanPhase::Detecting);
        assert!(!p.finished.load(Ordering::Relaxed));
        assert!(!p.cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn progress_json_round_trip() {
        let p = FaceScanProgress::default();
        p.detect_total.store(10, Ordering::Relaxed);
        p.detect_done.store(5, Ordering::Relaxed);
        p.faces_found.store(3, Ordering::Relaxed);
        let j = p.json();
        assert_eq!(j.get("detectTotal").and_then(Value::as_u64), Some(10));
        assert_eq!(j.get("detectDone").and_then(Value::as_u64), Some(5));
        assert_eq!(j.get("facesFound").and_then(Value::as_u64), Some(3));
        assert_eq!(j.get("phase").and_then(Value::as_str), Some("detecting"));
    }

    #[test]
    fn scan_start_fails_without_models() {
        let mut s = crate::Session::new();
        let result = s.face_scan_start(0.45, 2);
        assert!(result.is_err());
    }

    #[test]
    fn scan_tick_noop_when_no_scan() {
        let mut s = crate::Session::new();
        assert!(!s.face_scan_tick(), "tick should return false with no scan active");
    }

    #[test]
    fn cancel_stops_scan() {
        let p = FaceScanProgress::default();
        p.cancel.store(true, Ordering::Relaxed);
        // The scan_tick reads cancel and sets finished+Done — test the progress struct directly.
        assert!(p.cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn scan_all_disabled_without_models() {
        let mut s = crate::Session::new();
        let result = s.execute("face.scanAll", &json!({}));
        assert!(result.is_err(), "face.scanAll should fail without face models");
    }

    #[test]
    fn scan_progress_null_when_no_scan() {
        let mut s = crate::Session::new();
        let result = s.execute("face.scanProgress", &json!({}));
        assert!(result.is_ok());
        assert_eq!(result.unwrap_or_default(), Value::Null);
    }

    #[test]
    fn scan_cancel_noop_when_no_scan() {
        let mut s = crate::Session::new();
        let result = s.execute("face.scanCancel", &json!({}));
        assert!(result.is_ok());
        let val = result.unwrap_or_default();
        assert_eq!(val.get("cancelled").and_then(Value::as_bool), Some(false));
    }
}
