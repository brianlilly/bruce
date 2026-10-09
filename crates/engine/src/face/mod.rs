//! Face detection: SCRFD-500M (`lightcraft-face`, cargo feature `face`; the desktop app
//! enables it).
//!
//! **Nothing requires the model.** The user downloads the weights themselves (≈ 2.4 MB
//! safetensors from the insightface model zoo, converted with `tools/convert_scrfd.py`) and
//! puts them in the face model folder. Without the weights, face detection requests fail with
//! a clear "not installed" error.
//!
//! Everything that touches the model (loading it, running detection on a rendered photo) runs
//! on one worker thread ([`worker`]); the session talks to it over channels and never waits on
//! a lock. With [`FaceDetector::background`] (the desktop app) commands return at once and
//! results are polled; without it (CLI, tests) commands wait for the worker.

#[cfg(feature = "face")]
pub(crate) mod worker;

use std::path::PathBuf;

use lightcraft_catalog::PhotoId;
use lightcraft_meta::Region;

use crate::Session;

/// How errors about a missing model start.
pub const NOT_INSTALLED: &str = "The SCRFD face detection model is not installed";

/// Longest a waiting command (CLI) waits for the model.
#[cfg_attr(not(feature = "face"), allow(dead_code))]
const WAIT: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Maximum edge (longest side) of the image fed to the detector. The model accepts 640×640;
/// the photo is scaled down to fit this (keeping aspect, no letterbox needed — the face crate
/// handles letterboxing internally).
#[cfg(feature = "face")]
const DETECT_EDGE: usize = 1024;

/// A face detection result ready for the session: the photo and its detected regions.
pub struct DetectedFaces {
    pub photo: PhotoId,
    pub faces: Vec<Region>,
}

/// The face-detection worker and its configuration.
pub struct FaceDetector {
    /// Where the `scrfd_500m.safetensors` file lives (set by the app; `None`: no face detection).
    pub dir: Option<PathBuf>,
    /// Run requests in the background and poll results (the desktop app); otherwise commands
    /// wait for their result (CLI, tests).
    pub background: bool,
    #[cfg(feature = "face")]
    worker: worker::Worker,
    #[cfg(feature = "face")]
    results: (std::sync::mpsc::Sender<worker::Outcome>, std::sync::mpsc::Receiver<worker::Outcome>),
}

#[allow(clippy::derivable_impls)]
impl Default for FaceDetector {
    fn default() -> Self {
        FaceDetector {
            dir: None,
            background: false,
            #[cfg(feature = "face")]
            worker: worker::Worker::default(),
            #[cfg(feature = "face")]
            results: std::sync::mpsc::channel(),
        }
    }
}

impl FaceDetector {
    /// Whether this build can detect faces at all.
    pub const AVAILABLE: bool = cfg!(feature = "face");

    /// Whether the model's weights file is in place.
    pub fn installed(&self) -> bool {
        #[cfg(feature = "face")]
        if let Some(dir) = &self.dir {
            return lightcraft_face::is_model_dir(dir);
        }
        false
    }

    /// Whether requests are queued or running.
    pub fn busy(&self) -> bool {
        #[cfg(feature = "face")]
        return self.worker.pending() > 0;
        #[cfg(not(feature = "face"))]
        false
    }

    /// Whether the model is being loaded right now.
    pub fn loading(&self) -> bool {
        #[cfg(feature = "face")]
        return self.worker.loading();
        #[cfg(not(feature = "face"))]
        false
    }

    /// Whether the model is in memory.
    pub fn loaded(&self) -> bool {
        #[cfg(feature = "face")]
        return self.worker.loaded();
        #[cfg(not(feature = "face"))]
        false
    }

    /// The model folder, if the model is there; otherwise an error that says what to do.
    pub fn model_dir(&self) -> Result<PathBuf, String> {
        if !Self::AVAILABLE {
            return Err("Face detection is not available in this build".into());
        }
        let dir = self.dir.clone().ok_or("no folder is set for the SCRFD face detection model")?;
        if self.installed() {
            return Ok(dir);
        }
        Err(format!("{NOT_INSTALLED}. Put scrfd_500m.safetensors (converted with tools/convert_scrfd.py) in {}.", dir.display()))
    }
}

/// Convert pixel-coordinate `lightcraft_face::Face` boxes into normalized `Region` values
/// suitable for the catalog. `img_w` and `img_h` are the dimensions of the image that was
/// actually fed to the detector (after any downscale).
#[cfg(feature = "face")]
fn faces_to_regions(faces: &[lightcraft_face::Face], img_w: usize, img_h: usize) -> Vec<Region> {
    if img_w == 0 || img_h == 0 {
        return Vec::new();
    }
    let iw = img_w as f64;
    let ih = img_h as f64;
    faces
        .iter()
        .map(|f| {
            let x = (f64::from(f.x) / iw).clamp(0.0, 1.0);
            let y = (f64::from(f.y) / ih).clamp(0.0, 1.0);
            let w = (f64::from(f.w) / iw).clamp(0.0, 1.0 - x);
            let h = (f64::from(f.h) / ih).clamp(0.0, 1.0 - y);
            Region { rect: lightcraft_meta::Rect::from_xywh(x, y, w, h), kind: lightcraft_meta::RegionKind::Face, name: None, description: None }
        })
        .collect()
}

impl Session {
    /// Render a photo's RGB pixels for face detection: the photo as-imported (no develop
    /// settings, no crop), scaled to fit within `DETECT_EDGE × DETECT_EDGE`.
    #[cfg(feature = "face")]
    fn face_detect_render(&mut self, id: PhotoId) -> Result<(Vec<u8>, usize, usize), String> {
        let _exists = self.catalog.photo(id).ok_or("no such photo")?;
        let settings = lightcraft_develop::DevelopSettings::default();
        let job = self.preview_job(id, DETECT_EDGE, DETECT_EDGE, false, &settings).ok_or("could not render the photo for face detection")?;
        let rendered = job.run();
        let img = rendered.rendered.map_err(|e| format!("could not render the photo for face detection: {e}"))?;
        let (w, h) = (img.image.width, img.image.height);
        let rgb: Vec<u8> = img.image.data.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        Ok((rgb, w, h))
    }

    /// Detect faces in photo `id`, waiting for the worker (CLI, tests, or non-background mode).
    pub fn face_detect_sync(&mut self, id: PhotoId) -> Result<DetectedFaces, String> {
        self.face_detector.model_dir()?;
        #[cfg(feature = "face")]
        {
            let (rgb, w, h) = self.face_detect_render(id)?;
            let dir = self.face_detector.model_dir()?;
            let (tx, rx) = std::sync::mpsc::channel();
            let job = worker::Job { dir, photo: id, rgb, width: w, height: h, reply: tx };
            self.face_detector.worker.submit(job)?;
            match rx.recv_timeout(WAIT) {
                Ok(o) => {
                    let faces = o.result?;
                    let regions = faces_to_regions(&faces, w, h);
                    Ok(DetectedFaces { photo: id, faces: regions })
                }
                Err(_) => Err("the face detection model took too long; try again".into()),
            }
        }
        #[cfg(not(feature = "face"))]
        {
            let _ = id;
            Err("Face detection is not available in this build".into())
        }
    }

    /// Background mode: queue face detection for photo `id`; the result is collected by
    /// [`Session::face_detect_poll`].
    pub fn face_detect_queue(&mut self, id: PhotoId) -> Result<(), String> {
        self.face_detector.model_dir()?;
        #[cfg(feature = "face")]
        {
            let (rgb, w, h) = self.face_detect_render(id)?;
            let dir = self.face_detector.model_dir()?;
            let reply = self.face_detector.results.0.clone();
            let job = worker::Job { dir, photo: id, rgb, width: w, height: h, reply };
            self.face_detector.worker.submit(job)
        }
        #[cfg(not(feature = "face"))]
        {
            let _ = id;
            Err("Face detection is not available in this build".into())
        }
    }

    /// Collect finished background face-detection results. Cheap when nothing is pending:
    /// call it every frame.
    pub fn face_detect_poll(&mut self) -> Vec<Result<DetectedFaces, String>> {
        #[cfg(feature = "face")]
        {
            let mut results = Vec::new();
            for o in self.face_detector.results.1.try_iter() {
                match o.result {
                    Ok(faces) => {
                        let regions = faces_to_regions(&faces, o.width, o.height);
                        results.push(Ok(DetectedFaces { photo: o.photo, faces: regions }));
                    }
                    Err(e) => {
                        results.push(Err(e));
                    }
                }
            }
            results
        }
        #[cfg(not(feature = "face"))]
        Vec::new()
    }
}

#[cfg(all(test, feature = "face"))]
mod tests {
    use super::*;

    #[test]
    fn faces_to_regions_normalizes_correctly() {
        let faces = vec![lightcraft_face::Face { x: 100.0, y: 200.0, w: 50.0, h: 60.0, score: 0.9 }];
        let regions = faces_to_regions(&faces, 1000, 800);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert!((r.rect.x0 - 0.1).abs() < 1e-5);
        assert!((r.rect.y0 - 0.25).abs() < 1e-5);
        let w = r.rect.x1 - r.rect.x0;
        let h = r.rect.y1 - r.rect.y0;
        assert!((w - 0.05).abs() < 1e-5);
        assert!((h - 0.075).abs() < 1e-5);
        assert!(matches!(r.kind, lightcraft_meta::RegionKind::Face));
    }

    #[test]
    fn faces_to_regions_clamps_to_unit() {
        // A face that extends beyond image bounds
        let faces = vec![lightcraft_face::Face { x: 900.0, y: 750.0, w: 200.0, h: 100.0, score: 0.8 }];
        let regions = faces_to_regions(&faces, 1000, 800);
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert!(r.rect.x1 <= 1.0 + 1e-5, "x1={}", r.rect.x1);
        assert!(r.rect.y1 <= 1.0 + 1e-5, "y1={}", r.rect.y1);
    }

    #[test]
    fn faces_to_regions_empty_for_zero_image() {
        let faces = vec![lightcraft_face::Face { x: 10.0, y: 10.0, w: 5.0, h: 5.0, score: 0.9 }];
        assert!(faces_to_regions(&faces, 0, 0).is_empty());
    }
}
