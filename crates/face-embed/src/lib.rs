//! ArcFace/AdaFace face embedding in pure Rust on candle.
//!
//! Produces a 512-dimensional embedding vector from a 112x112 aligned face crop
//! using an IResNet-18 backbone. Embeddings can be compared via cosine similarity
//! for face recognition and clustering.
//!
//! The weights are not part of Bruce: they live in a user-supplied directory
//! containing `adaface_ir18.safetensors` (converted from a PyTorch checkpoint).
//!
//! No UI dependencies (L3).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(not(target_arch = "wasm32"))]
mod iresnet;
#[cfg(not(target_arch = "wasm32"))]
mod weights;

#[cfg(not(target_arch = "wasm32"))]
use candle_core::{DType, Device, Tensor};
#[cfg(not(target_arch = "wasm32"))]
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};

/// The weights filename we look for.
pub const WEIGHTS_FILE: &str = "adaface_ir18.safetensors";

/// Model input size (square): 112x112.
pub const INPUT_SIZE: usize = 112;

/// Embedding vector dimension.
pub const EMBED_DIM: usize = 512;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("face embedding model: {0}")]
    Model(String),
    #[error("face embedding model not found in {0} (expected {WEIGHTS_FILE})")]
    Missing(PathBuf),
    #[cfg(not(target_arch = "wasm32"))]
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A face embedding: a 512-d vector suitable for cosine similarity.
#[derive(Clone, Debug)]
pub struct Embedding {
    /// The 512-dimensional embedding vector, L2-normalized.
    pub vector: Vec<f32>,
}

impl Embedding {
    /// Cosine similarity to another embedding (-1 to 1; higher = more similar).
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }
        let dot: f32 = self.vector.iter().zip(&other.vector).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

/// Whether `dir` holds a usable face embedding checkpoint.
pub fn is_model_dir(dir: &Path) -> bool {
    dir.join(WEIGHTS_FILE).is_file()
}

#[cfg(not(target_arch = "wasm32"))]
/// Best device: Metal on macOS (when available), else CPU.
pub fn best_device() -> Device {
    #[cfg(target_os = "macos")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

/// ArcFace/AdaFace face embedding model (IResNet-18).
#[cfg(not(target_arch = "wasm32"))]
pub struct FaceEmbedder {
    device: Device,
    net: iresnet::IResNet,
}

#[cfg(not(target_arch = "wasm32"))]
impl FaceEmbedder {
    /// Load the model from `dir` on the best available device.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_on(dir, best_device())
    }

    /// Load the model from `dir` on a specific device.
    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        let path = dir.join(WEIGHTS_FILE);
        if !path.is_file() {
            return Err(Error::Missing(dir.to_path_buf()));
        }
        let w = weights::Weights::open(&path)?;
        let vb = VarBuilder::from_backend(Box::new(weights::Backend(w)), DType::F32, device.clone());
        let net = iresnet::IResNet::load(&vb).map_err(|e| Error::Model(format!("load: {e}")))?;
        log::info!("face embedding model loaded from {}", path.display());
        Ok(Self { device, net })
    }

    /// Compute the embedding for a 112x112 RGB face crop.
    ///
    /// `rgb` must be exactly `INPUT_SIZE * INPUT_SIZE * 3` bytes (row-major, R G B).
    /// The crop should be aligned (eyes roughly horizontal) and tightly framed.
    pub fn embed(&self, rgb: &[u8]) -> Result<Embedding> {
        let expected = INPUT_SIZE * INPUT_SIZE * 3;
        if rgb.len() != expected {
            return Err(Error::Model(format!(
                "expected {expected} bytes ({}x{}x3), got {}",
                INPUT_SIZE, INPUT_SIZE, rgb.len()
            )));
        }
        let input = self.preprocess(rgb)?;
        let output = self.net.forward(&input).map_err(|e| Error::Model(format!("forward: {e}")))?;
        self.postprocess(&output)
    }

    /// Compute embeddings for a batch of 112x112 RGB face crops.
    ///
    /// Each element of `crops` is `INPUT_SIZE * INPUT_SIZE * 3` bytes.
    /// Returns one Embedding per crop.
    pub fn embed_batch(&self, crops: &[&[u8]]) -> Result<Vec<Embedding>> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let expected = INPUT_SIZE * INPUT_SIZE * 3;
        // Build batch tensor
        let mut batch_data = Vec::with_capacity(crops.len() * 3 * INPUT_SIZE * INPUT_SIZE);
        for (i, rgb) in crops.iter().enumerate() {
            if rgb.len() != expected {
                return Err(Error::Model(format!(
                    "crop {i}: expected {expected} bytes, got {}",
                    rgb.len()
                )));
            }
            // Normalize and convert to CHW
            for c in 0..3 {
                for y in 0..INPUT_SIZE {
                    for x in 0..INPUT_SIZE {
                        let idx = (y * INPUT_SIZE + x) * 3 + c;
                        let v = rgb.get(idx).copied().unwrap_or(0) as f32;
                        batch_data.push((v - 127.5) / 128.0);
                    }
                }
            }
        }
        let input = Tensor::from_vec(batch_data, (crops.len(), 3, INPUT_SIZE, INPUT_SIZE), &self.device)?;
        let output = self.net.forward(&input).map_err(|e| Error::Model(format!("forward: {e}")))?;
        // output: [batch, 512]
        let mut embeddings = Vec::with_capacity(crops.len());
        for i in 0..crops.len() {
            let row = output.get(i)?;
            let e = self.postprocess(&row.unsqueeze(0)?)?;
            embeddings.push(e);
        }
        Ok(embeddings)
    }

    /// Preprocess a single RGB crop to [1, 3, 112, 112] tensor.
    fn preprocess(&self, rgb: &[u8]) -> Result<Tensor> {
        // Convert HWC RGB bytes to CHW float, normalized: (pixel - 127.5) / 128.0
        let mut chw = Vec::with_capacity(3 * INPUT_SIZE * INPUT_SIZE);
        for c in 0..3 {
            for y in 0..INPUT_SIZE {
                for x in 0..INPUT_SIZE {
                    let idx = (y * INPUT_SIZE + x) * 3 + c;
                    let v = rgb.get(idx).copied().unwrap_or(0) as f32;
                    chw.push((v - 127.5) / 128.0);
                }
            }
        }
        Ok(Tensor::from_vec(chw, (1, 3, INPUT_SIZE, INPUT_SIZE), &self.device)?)
    }

    /// Postprocess: extract f32 vector and L2-normalize.
    fn postprocess(&self, output: &Tensor) -> Result<Embedding> {
        let v: Vec<f32> = output.squeeze(0)?.to_vec1().map_err(|e| Error::Model(format!("postprocess: {e}")))?;
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let vector = if norm > 1e-10 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v
        };
        Ok(Embedding { vector })
    }
}

/// Cosine similarity between two raw f32 embedding vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_empty_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_length_mismatch_returns_zero() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn embedding_cosine_similarity() {
        let a = Embedding { vector: vec![1.0, 0.0, 0.0] };
        let b = Embedding { vector: vec![0.7071, 0.7071, 0.0] };
        let sim = a.cosine_similarity(&b);
        // cos(45) ~= 0.7071
        assert!((sim - 0.7071).abs() < 0.01);
    }
}
