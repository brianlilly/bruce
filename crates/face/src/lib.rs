//! SCRFD-500M face detection in pure Rust on candle.
//!
//! Produces face bounding boxes from an RGB image using the SCRFD-500M model
//! (0.57M params, 2.4MB safetensors). Architecture: MobileNetV1 backbone,
//! PAFPN neck, SCRFDHead with depthwise-separable convs.
//!
//! The weights are not part of Bruce: they live in a user-supplied directory
//! containing `scrfd_500m.safetensors` (converted from the insightface checkpoint).
//!
//! No UI dependencies (L3).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(not(target_arch = "wasm32"))]
mod backbone;
#[cfg(not(target_arch = "wasm32"))]
mod head;
#[cfg(not(target_arch = "wasm32"))]
mod neck;
#[cfg(not(target_arch = "wasm32"))]
mod weights;

#[cfg(not(target_arch = "wasm32"))]
use candle_core::{DType, Device, Tensor};
#[cfg(not(target_arch = "wasm32"))]
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};

/// The weights filename we look for.
pub const WEIGHTS_FILE: &str = "scrfd_500m.safetensors";

/// Model input size (square).
const INPUT_SIZE: usize = 640;

/// Number of detection strides.
const NUM_STRIDES: usize = 3;
/// Stride values.
const STRIDES: [usize; NUM_STRIDES] = [8, 16, 32];
/// Number of anchors per cell.
const NUM_ANCHORS: usize = 2;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("SCRFD model: {0}")]
    Model(String),
    #[error("SCRFD model not found in {0} (expected {WEIGHTS_FILE})")]
    Missing(PathBuf),
    #[cfg(not(target_arch = "wasm32"))]
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A detected face bounding box, in pixel coordinates of the original image.
#[derive(Clone, Debug)]
pub struct Face {
    /// Left edge (x) in pixels.
    pub x: f32,
    /// Top edge (y) in pixels.
    pub y: f32,
    /// Width in pixels.
    pub w: f32,
    /// Height in pixels.
    pub h: f32,
    /// Detection confidence, 0..1.
    pub score: f32,
}

/// Whether `dir` holds a usable SCRFD checkpoint.
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

#[cfg(not(target_arch = "wasm32"))]
/// A loaded SCRFD-500M model.
pub struct Scrfd {
    #[allow(dead_code)]
    device: Device,
    backbone: backbone::Backbone,
    neck: neck::Neck,
    head: head::Head,
}

#[cfg(not(target_arch = "wasm32"))]
impl Scrfd {
    /// Load from a directory holding `scrfd_500m.safetensors`.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_on(dir, best_device())
    }

    /// Load onto a specific device.
    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        if !is_model_dir(dir) {
            return Err(Error::Missing(dir.to_path_buf()));
        }
        let w = weights::Weights::open(&dir.join(WEIGHTS_FILE))?;
        let vb = VarBuilder::from_backend(Box::new(weights::Backend(w)), DType::F32, device.clone());

        let backbone = backbone::Backbone::load(&vb.pp("backbone"))?;
        let neck = neck::Neck::load(&vb.pp("neck"))?;
        let head = head::Head::load(&vb.pp("bbox_head"))?;

        Ok(Self { device, backbone, neck, head })
    }

    /// Detect faces in an RGB image (8-bit, row-major, `w × h` pixels).
    ///
    /// `score_threshold` filters weak detections (recommended: 0.5).
    /// `nms_threshold` controls IoU-based non-maximum suppression (recommended: 0.45).
    ///
    /// Returns face bounding boxes in pixel coordinates of the original image.
    pub fn detect(&self, rgb: &[u8], img_w: usize, img_h: usize, score_threshold: f32, nms_threshold: f32) -> Result<Vec<Face>> {
        let expected = img_w.checked_mul(img_h).and_then(|v| v.checked_mul(3)).ok_or_else(|| Error::Model("image dimensions overflow".into()))?;
        if rgb.len() != expected {
            return Err(Error::Model(format!("expected {expected} bytes for {img_w}×{img_h} RGB, got {}", rgb.len())));
        }

        // Aspect-preserving resize to INPUT_SIZE × INPUT_SIZE with letterboxing
        let (resized, scale, pad_x, pad_y) = preprocess(rgb, img_w, img_h)?;

        // Run backbone → neck → head
        let features = self.backbone.forward(&resized)?;
        let neck_out = self.neck.forward(&features)?;
        let (cls_scores, bbox_preds) = self.head.forward(&neck_out)?;

        // Generate anchor centers
        let anchors = generate_anchors(INPUT_SIZE, INPUT_SIZE)?;

        // Post-process: decode boxes, apply threshold and NMS
        let faces = postprocess(&cls_scores, &bbox_preds, &anchors, scale, pad_x, pad_y, img_w, img_h, score_threshold, nms_threshold)?;

        Ok(faces)
    }
}

/// Preprocess an RGB image into a normalized, letterboxed tensor.
///
/// Returns (tensor [1,3,INPUT_SIZE,INPUT_SIZE], scale, pad_x, pad_y).
#[cfg(not(target_arch = "wasm32"))]
fn preprocess(rgb: &[u8], w: usize, h: usize) -> Result<(Tensor, f32, f32, f32)> {
    let scale_w = INPUT_SIZE as f32 / w as f32;
    let scale_h = INPUT_SIZE as f32 / h as f32;
    let scale = scale_w.min(scale_h);

    let new_w = (w as f32 * scale).round() as usize;
    let new_h = (h as f32 * scale).round() as usize;
    let pad_x = (INPUT_SIZE.saturating_sub(new_w)) as f32 / 2.0;
    let pad_y = (INPUT_SIZE.saturating_sub(new_h)) as f32 / 2.0;
    let pad_left = pad_x.floor() as usize;
    let pad_top = pad_y.floor() as usize;

    // Bilinear resize
    let mut resized_rgb = vec![0u8; new_w * new_h * 3];
    for dst_y in 0..new_h {
        for dst_x in 0..new_w {
            let src_xf = (dst_x as f32 + 0.5) / scale - 0.5;
            let src_yf = (dst_y as f32 + 0.5) / scale - 0.5;

            let x0 = src_xf.floor().max(0.0) as usize;
            let y0 = src_yf.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(w.saturating_sub(1));
            let y1 = (y0 + 1).min(h.saturating_sub(1));

            let fx = src_xf - x0 as f32;
            let fy = src_yf - y0 as f32;

            for c in 0..3 {
                let p00 = rgb.get(y0 * w * 3 + x0 * 3 + c).copied().unwrap_or(0) as f32;
                let p01 = rgb.get(y0 * w * 3 + x1 * 3 + c).copied().unwrap_or(0) as f32;
                let p10 = rgb.get(y1 * w * 3 + x0 * 3 + c).copied().unwrap_or(0) as f32;
                let p11 = rgb.get(y1 * w * 3 + x1 * 3 + c).copied().unwrap_or(0) as f32;
                let val = p00 * (1.0 - fx) * (1.0 - fy) + p01 * fx * (1.0 - fy) + p10 * (1.0 - fx) * fy + p11 * fx * fy;
                if let Some(dst) = resized_rgb.get_mut(dst_y * new_w * 3 + dst_x * 3 + c) {
                    *dst = val.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    // Create normalized float tensor with letterbox padding
    // Normalization: (pixel - 127.5) / 128.0
    let pad_val: f32 = (0.0 - 127.5) / 128.0; // black padding normalized
    let mut pixels = vec![pad_val; 3 * INPUT_SIZE * INPUT_SIZE];

    for y in 0..new_h {
        for x in 0..new_w {
            let dst_y = y + pad_top;
            let dst_x = x + pad_left;
            if dst_y < INPUT_SIZE && dst_x < INPUT_SIZE {
                for c in 0..3 {
                    let val = resized_rgb.get(y * new_w * 3 + x * 3 + c).copied().unwrap_or(0) as f32;
                    let norm = (val - 127.5) / 128.0;
                    if let Some(dst) = pixels.get_mut(c * INPUT_SIZE * INPUT_SIZE + dst_y * INPUT_SIZE + dst_x) {
                        *dst = norm;
                    }
                }
            }
        }
    }

    let tensor = Tensor::from_vec(pixels, (1, 3, INPUT_SIZE, INPUT_SIZE), &Device::Cpu)?;
    Ok((tensor, scale, pad_x, pad_y))
}

/// Anchor center for a detection. Coordinates are in the INPUT_SIZE pixel space.
#[cfg(not(target_arch = "wasm32"))]
struct Anchor {
    cx: f32,
    cy: f32,
    stride: usize,
}

/// Generate anchor centers for all strides.
///
/// For each stride and each (y, x) cell, there are NUM_ANCHORS anchors at the cell center.
#[cfg(not(target_arch = "wasm32"))]
fn generate_anchors(input_h: usize, input_w: usize) -> Result<Vec<Anchor>> {
    let mut anchors = Vec::new();
    for &stride in &STRIDES {
        let fh = input_h / stride;
        let fw = input_w / stride;
        for y in 0..fh {
            for x in 0..fw {
                let cx = (x as f32 + 0.5) * stride as f32;
                let cy = (y as f32 + 0.5) * stride as f32;
                for _ in 0..NUM_ANCHORS {
                    anchors.push(Anchor { cx, cy, stride });
                }
            }
        }
    }
    Ok(anchors)
}

/// Post-process: decode boxes from distance predictions, threshold, NMS, and
/// map back to original image coordinates.
#[cfg(not(target_arch = "wasm32"))]
fn postprocess(
    cls_scores: &Tensor,
    bbox_preds: &Tensor,
    anchors: &[Anchor],
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    img_w: usize,
    img_h: usize,
    score_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<Face>> {
    // Squeeze batch dimension and move to CPU
    let scores = cls_scores.squeeze(0)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let preds = bbox_preds.squeeze(0)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;

    let num_anchors = scores.dim(0)?;
    let scores_data = scores.flatten_all()?.to_vec1::<f32>()?;
    let preds_data = preds.flatten_all()?.to_vec1::<f32>()?;

    if anchors.len() != num_anchors {
        return Err(Error::Model(format!("anchor count mismatch: expected {num_anchors}, generated {}", anchors.len())));
    }

    // Decode boxes and apply sigmoid + threshold
    let mut candidates: Vec<Face> = Vec::new();
    for (i, anchor) in anchors.iter().enumerate() {
        let raw_score = scores_data.get(i).copied().unwrap_or(f32::NEG_INFINITY);
        // Sigmoid
        let conf = 1.0 / (1.0 + (-raw_score).exp());
        if conf < score_threshold {
            continue;
        }

        // Decode distance predictions: l, t, r, b (distances from anchor center)
        let base = i * 4;
        let l = preds_data.get(base).copied().unwrap_or(0.0) * anchor.stride as f32;
        let t = preds_data.get(base + 1).copied().unwrap_or(0.0) * anchor.stride as f32;
        let r = preds_data.get(base + 2).copied().unwrap_or(0.0) * anchor.stride as f32;
        let b = preds_data.get(base + 3).copied().unwrap_or(0.0) * anchor.stride as f32;

        // Convert distances to box in INPUT_SIZE space
        let x1 = anchor.cx - l;
        let y1 = anchor.cy - t;
        let x2 = anchor.cx + r;
        let y2 = anchor.cy + b;

        // Map back to original image coordinates
        let orig_x1 = (x1 - pad_x) / scale;
        let orig_y1 = (y1 - pad_y) / scale;
        let orig_x2 = (x2 - pad_x) / scale;
        let orig_y2 = (y2 - pad_y) / scale;

        // Clamp to image bounds
        let fx = orig_x1.max(0.0).min(img_w as f32);
        let fy = orig_y1.max(0.0).min(img_h as f32);
        let fw = (orig_x2.min(img_w as f32) - fx).max(0.0);
        let fh = (orig_y2.min(img_h as f32) - fy).max(0.0);

        if fw > 0.0 && fh > 0.0 {
            candidates.push(Face { x: fx, y: fy, w: fw, h: fh, score: conf });
        }
    }

    // Sort by score descending for NMS
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy NMS
    let mut keep = Vec::new();
    let mut suppressed = vec![false; candidates.len()];
    for i in 0..candidates.len() {
        if suppressed.get(i).copied().unwrap_or(true) {
            continue;
        }
        keep.push(candidates[i].clone());
        for j in (i + 1)..candidates.len() {
            if suppressed.get(j).copied().unwrap_or(true) {
                continue;
            }
            if iou(&candidates[i], &candidates[j]) > nms_threshold
                && let Some(s) = suppressed.get_mut(j)
            {
                *s = true;
            }
        }
    }

    Ok(keep)
}

/// Intersection over Union.
fn iou(a: &Face, b: &Face) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = a.w * a.h;
    let area_b = b.w * b.h;
    let union = area_a + area_b - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_identical() {
        let a = Face { x: 10.0, y: 20.0, w: 100.0, h: 100.0, score: 0.9 };
        let result = iou(&a, &a);
        assert!((result - 1.0).abs() < 1e-5, "identical boxes should have IoU 1.0");
    }

    #[test]
    fn iou_disjoint() {
        let a = Face { x: 0.0, y: 0.0, w: 10.0, h: 10.0, score: 0.9 };
        let b = Face { x: 100.0, y: 100.0, w: 10.0, h: 10.0, score: 0.8 };
        assert!(iou(&a, &b) < 1e-5, "disjoint boxes should have IoU 0");
    }

    #[test]
    fn iou_half_overlap() {
        let a = Face { x: 0.0, y: 0.0, w: 10.0, h: 10.0, score: 0.9 };
        let b = Face { x: 5.0, y: 0.0, w: 10.0, h: 10.0, score: 0.8 };
        // Intersection: 5×10 = 50, Union: 100 + 100 - 50 = 150
        let result = iou(&a, &b);
        assert!((result - 50.0 / 150.0).abs() < 1e-5, "expected IoU ≈ 0.333, got {result}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn anchor_generation() {
        let anchors = generate_anchors(640, 640).unwrap_or_default();
        // Stride 8: 80×80×2 = 12800
        // Stride 16: 40×40×2 = 3200
        // Stride 32: 20×20×2 = 800
        // Total: 16800
        assert_eq!(anchors.len(), 16800);

        // First anchor should be at (4, 4) for stride 8
        if let Some(a) = anchors.first() {
            assert!((a.cx - 4.0).abs() < 1e-5);
            assert!((a.cy - 4.0).abs() < 1e-5);
            assert_eq!(a.stride, 8);
        }
    }

    #[test]
    fn is_model_dir_false_for_empty() {
        let dir = std::env::temp_dir().join("scrfd-test-empty");
        let _ = std::fs::create_dir_all(&dir);
        assert!(!is_model_dir(&dir));
        let _ = std::fs::remove_dir(&dir);
    }
}
