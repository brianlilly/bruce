//! Face embedding: ArcFace/AdaFace IResNet-18 (`lightcraft-face-embed`, cargo feature
//! `face-embed`; the desktop app enables it).
//!
//! **Nothing requires the model.** The user downloads the weights themselves (≈ 93 MB
//! safetensors, converted with `tools/convert_adaface.py`) and puts them in the face model
//! folder. Without the weights, embedding requests fail with a clear "not installed" error.
//!
//! Everything that touches the model runs on one worker thread ([`worker`]); the session talks
//! to it over channels and never waits on a lock.

#[cfg(feature = "face-embed")]
pub(crate) mod worker;

use std::path::PathBuf;

use lightcraft_catalog::PhotoId;

use crate::Session;

/// How errors about a missing model start.
pub const NOT_INSTALLED: &str = "The AdaFace face embedding model is not installed";

/// Longest a waiting command (CLI) waits for the model.
#[cfg_attr(not(feature = "face-embed"), allow(dead_code))]
const WAIT: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Edge size for the face crop fed to the embedding model (112×112).
#[cfg(feature = "face-embed")]
const EMBED_EDGE: usize = lightcraft_face_embed::INPUT_SIZE;

/// A computed face embedding ready for the session.
pub struct FaceEmbedding {
    pub photo: PhotoId,
    /// Index of the face region within the photo's regions.
    pub region_index: usize,
    /// The 512-d embedding vector, L2-normalized.
    pub vector: Vec<f32>,
}

/// The face-embedding worker and its configuration.
pub struct FaceEmbedModule {
    /// Where the `adaface_ir18.safetensors` file lives (set by the app; `None`: no embedding).
    pub dir: Option<PathBuf>,
    /// Run requests in the background and poll results (the desktop app); otherwise commands
    /// wait for their result (CLI, tests).
    pub background: bool,
    #[cfg(feature = "face-embed")]
    worker: worker::Worker,
    #[cfg(feature = "face-embed")]
    results: (std::sync::mpsc::Sender<worker::Outcome>, std::sync::mpsc::Receiver<worker::Outcome>),
}

#[allow(clippy::derivable_impls)]
impl Default for FaceEmbedModule {
    fn default() -> Self {
        FaceEmbedModule {
            dir: None,
            background: false,
            #[cfg(feature = "face-embed")]
            worker: worker::Worker::default(),
            #[cfg(feature = "face-embed")]
            results: std::sync::mpsc::channel(),
        }
    }
}

impl FaceEmbedModule {
    /// Whether this build can compute face embeddings at all.
    pub const AVAILABLE: bool = cfg!(feature = "face-embed");

    /// Whether the model's weights file is in place.
    pub fn installed(&self) -> bool {
        #[cfg(feature = "face-embed")]
        if let Some(dir) = &self.dir {
            return lightcraft_face_embed::is_model_dir(dir);
        }
        false
    }

    /// Whether requests are queued or running.
    pub fn busy(&self) -> bool {
        #[cfg(feature = "face-embed")]
        return self.worker.pending() > 0;
        #[cfg(not(feature = "face-embed"))]
        false
    }

    /// Whether the model is being loaded right now.
    pub fn loading(&self) -> bool {
        #[cfg(feature = "face-embed")]
        return self.worker.loading();
        #[cfg(not(feature = "face-embed"))]
        false
    }

    /// Whether the model is in memory.
    pub fn loaded(&self) -> bool {
        #[cfg(feature = "face-embed")]
        return self.worker.loaded();
        #[cfg(not(feature = "face-embed"))]
        false
    }

    /// The model folder, if the model is there; otherwise an error that says what to do.
    pub fn model_dir(&self) -> Result<PathBuf, String> {
        if !Self::AVAILABLE {
            return Err("Face embedding is not available in this build".into());
        }
        let dir = self.dir.clone().ok_or("no folder is set for the AdaFace face embedding model")?;
        if self.installed() {
            return Ok(dir);
        }
        Err(format!("{NOT_INSTALLED}. Put adaface_ir18.safetensors (converted with tools/convert_adaface.py) in {}.", dir.display()))
    }
}

impl Session {
    /// Render a 112×112 RGB face crop for embedding: the face region expanded to a square
    /// (with some margin for hair and chin), scaled to 112×112.
    #[cfg(feature = "face-embed")]
    fn face_embed_render(&mut self, id: PhotoId, region: &lightcraft_meta::Region) -> Result<Vec<u8>, String> {
        let p = self.catalog.photo(id).ok_or("no such photo")?.clone();

        // The region rect is in normalized coordinates on the upright photo.
        // Build a crop that's a square centered on the face, 1.3× the face size for margin.
        let face = region.rect;
        let orient = p.develop.orientation;
        let face = orient.map_norm_rect(face);

        let (w, h) = (f64::from(p.width.max(1)), f64::from(p.height.max(1)));
        let (w, h) = if orient.swaps_axes() { (h, w) } else { (w, h) };

        // Square side: 1.3× the face's larger dimension (hair/chin margin, less than the 1.7×
        // used for People cards, since the embedding model expects a tighter crop)
        let side_px = ((face.x1 - face.x0) * w).max((face.y1 - face.y0) * h) * 1.3;
        let side_px = if side_px.is_finite() { side_px.clamp(1.0, w.min(h)) } else { w.min(h) };

        let (cx, cy) = ((face.x0 + face.x1) / 2.0 * w, (face.y0 + face.y1) / 2.0 * h);
        let (cx, cy) = (if cx.is_finite() { cx } else { w / 2.0 }, if cy.is_finite() { cy } else { h / 2.0 });
        let (x0, y0) = (
            cx.clamp(side_px / 2.0, w - side_px / 2.0) - side_px / 2.0,
            cy.clamp(side_px / 2.0, h - side_px / 2.0) - side_px / 2.0,
        );
        let rect = lightcraft_geom::Rect {
            x0: x0 / w,
            y0: y0 / h,
            x1: (x0 + side_px) / w,
            y1: (y0 + side_px) / h,
        };

        let settings = lightcraft_develop::DevelopSettings {
            crop: lightcraft_develop::Crop {
                geometry: lightcraft_geom::CropGeometry { rect, angle: 0.0 },
                ..Default::default()
            },
            ..Default::default()
        };

        // Render at a size that keeps the face sharp, then let the pipeline scale
        let needed = (EMBED_EDGE as f64 * w.max(h) / side_px).ceil().min(2560.0);
        let job = self.preview_job(id, needed as usize, needed as usize, true, &settings)
            .ok_or("could not render the face crop for embedding")?;
        let rendered = job.run();
        let img = rendered.rendered.map_err(|e| format!("could not render the face crop for embedding: {e}"))?;

        // Scale the rendered crop to exactly 112×112
        let src = &img.image;
        let (sw, sh) = (src.width, src.height);
        let mut rgb = vec![0u8; EMBED_EDGE * EMBED_EDGE * 3];
        for dy in 0..EMBED_EDGE {
            for dx in 0..EMBED_EDGE {
                // Nearest-neighbour (fine for ≤ 2× downscale of a face crop)
                let sx = (dx * sw) / EMBED_EDGE;
                let sy = (dy * sh) / EMBED_EDGE;
                let px = src.data.get(sy * sw + sx).copied().unwrap_or_default();
                let off = (dy * EMBED_EDGE + dx) * 3;
                rgb[off] = px[0];
                rgb[off + 1] = px[1];
                rgb[off + 2] = px[2];
            }
        }

        Ok(rgb)
    }

    /// Compute the embedding for face region `region_index` of photo `id`, waiting for the
    /// worker (CLI, tests, or non-background mode).
    pub fn face_embed_sync(&mut self, id: PhotoId, region_index: usize) -> Result<FaceEmbedding, String> {
        self.face_embedder.model_dir()?;
        #[cfg(feature = "face-embed")]
        {
            let regions = self.catalog.photo(id).ok_or("no such photo")?.meta.regions.clone();
            let region = regions.get(region_index).ok_or("no such face region")?;
            if !matches!(region.kind, lightcraft_meta::RegionKind::Face) {
                return Err("region is not a face".into());
            }
            let rgb = self.face_embed_render(id, region)?;
            let dir = self.face_embedder.model_dir()?;
            let (tx, rx) = std::sync::mpsc::channel();
            let job = worker::Job { dir, photo: id, region_index, rgb, reply: tx };
            self.face_embedder.worker.submit(job)?;
            match rx.recv_timeout(WAIT) {
                Ok(o) => {
                    let vector = o.result?;
                    Ok(FaceEmbedding { photo: id, region_index, vector })
                }
                Err(_) => Err("the face embedding model took too long; try again".into()),
            }
        }
        #[cfg(not(feature = "face-embed"))]
        {
            let _ = (id, region_index);
            Err("Face embedding is not available in this build".into())
        }
    }

    /// Background mode: queue embedding for face region `region_index` of photo `id`;
    /// the result is collected by [`Session::face_embed_poll`].
    pub fn face_embed_queue(&mut self, id: PhotoId, region_index: usize) -> Result<(), String> {
        self.face_embedder.model_dir()?;
        #[cfg(feature = "face-embed")]
        {
            let regions = self.catalog.photo(id).ok_or("no such photo")?.meta.regions.clone();
            let region = regions.get(region_index).ok_or("no such face region")?;
            if !matches!(region.kind, lightcraft_meta::RegionKind::Face) {
                return Err("region is not a face".into());
            }
            let rgb = self.face_embed_render(id, region)?;
            let dir = self.face_embedder.model_dir()?;
            let reply = self.face_embedder.results.0.clone();
            let job = worker::Job { dir, photo: id, region_index, rgb, reply };
            self.face_embedder.worker.submit(job)
        }
        #[cfg(not(feature = "face-embed"))]
        {
            let _ = (id, region_index);
            Err("Face embedding is not available in this build".into())
        }
    }

    /// Collect finished background face-embedding results. Cheap when nothing is pending:
    /// call it every frame.
    pub fn face_embed_poll(&mut self) -> Vec<Result<FaceEmbedding, String>> {
        #[cfg(feature = "face-embed")]
        {
            let mut results = Vec::new();
            for o in self.face_embedder.results.1.try_iter() {
                match o.result {
                    Ok(vector) => {
                        results.push(Ok(FaceEmbedding { photo: o.photo, region_index: o.region_index, vector }));
                    }
                    Err(e) => {
                        results.push(Err(e));
                    }
                }
            }
            results
        }
        #[cfg(not(feature = "face-embed"))]
        Vec::new()
    }
}
