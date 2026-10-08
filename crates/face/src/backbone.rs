//! MobileNetV1-style backbone for SCRFD-500M.
//!
//! Architecture from `insightface/detection/scrfd/mmdet/models/backbones/mobilenet.py`:
//! - Stem: conv3×3 stride 2 + BN + ReLU, then depthwise-separable conv stride 1
//! - 4 stages of depthwise-separable convolution blocks (first block stride 2, rest stride 1)
//!
//! SCRFD-500M config (`scrfd_500m.py`):
//!   stage_planes = [16, 16, 40, 72, 152, 288]
//!   stage_blocks = (2, 3, 2, 6)
//!
//! Output: 4 feature maps at strides 4, 8, 16, 32 (C2, C3, C4, C5).

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// A Conv2d + BatchNorm2d + ReLU block.
struct ConvBn {
    weight: Tensor,
    bn_weight: Tensor,
    bn_bias: Tensor,
    bn_mean: Tensor,
    bn_var: Tensor,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl ConvBn {
    fn load(vb: &VarBuilder, prefix: &str, conv_idx: usize, bn_idx: usize, groups: usize, stride: usize, padding: usize) -> Result<Self> {
        let weight = vb.get_unchecked(&format!("{prefix}.{conv_idx}.weight"))?;
        let bn_weight = vb.get_unchecked(&format!("{prefix}.{bn_idx}.weight"))?;
        let bn_bias = vb.get_unchecked(&format!("{prefix}.{bn_idx}.bias"))?;
        let bn_mean = vb.get_unchecked(&format!("{prefix}.{bn_idx}.running_mean"))?;
        let bn_var = vb.get_unchecked(&format!("{prefix}.{bn_idx}.running_var"))?;
        Ok(Self { weight, bn_weight, bn_bias, bn_mean, bn_var, stride, padding, groups })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv2d(&self.weight, self.padding, self.stride, 1, self.groups)?;
        let x = x.broadcast_sub(&self.bn_mean.reshape((1, (), 1, 1))?)?;
        let denom = (&self.bn_var + 1e-5f64)?.sqrt()?;
        let x = x.broadcast_div(&denom.reshape((1, (), 1, 1))?)?;
        let x = x.broadcast_mul(&self.bn_weight.reshape((1, (), 1, 1))?)?.broadcast_add(&self.bn_bias.reshape((1, (), 1, 1))?)?;
        x.relu()
    }
}

/// A depthwise-separable convolution block: depthwise conv + BN + ReLU, then pointwise conv + BN + ReLU.
struct ConvDw {
    dw: ConvBn,
    pw: ConvBn,
}

impl ConvDw {
    fn load(vb: &VarBuilder, prefix: &str, inp: usize, stride: usize) -> Result<Self> {
        // Depthwise: groups=inp, 3×3, indices 0 (conv) and 1 (bn)
        let dw = ConvBn::load(vb, prefix, 0, 1, inp, stride, 1)?;
        // Pointwise: groups=1, 1×1, indices 3 (conv) and 4 (bn)
        let pw = ConvBn::load(vb, prefix, 3, 4, 1, 1, 0)?;
        Ok(Self { dw, pw })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dw.forward(x)?;
        self.pw.forward(&x)
    }
}

/// The MobileNetV1 backbone for SCRFD-500M.
pub struct Backbone {
    /// Stem: conv3×3(3→planes[0], stride=2) + conv_dw(planes[0]→planes[1], stride=1)
    stem_conv: ConvBn,
    stem_dw: ConvDw,
    /// 4 stages of depthwise-separable blocks.
    stages: Vec<Vec<ConvDw>>,
}

impl Backbone {
    /// Load the backbone from safetensors with the given prefix.
    ///
    /// The SCRFD-500M config:
    ///   stage_planes = [16, 16, 40, 72, 152, 288]
    ///   stage_blocks = [2, 3, 2, 6]
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let stage_planes: [usize; 6] = [16, 16, 40, 72, 152, 288];
        let stage_blocks: [usize; 4] = [2, 3, 2, 6];

        // Stem: stem.0 is conv_bn(3→planes[0], stride=2), stem.1 is conv_dw(planes[0]→planes[1], stride=1)
        let stem_conv = ConvBn::load(vb, "stem.0", 0, 1, 1, 2, 1)?;
        let stem_dw = ConvDw::load(vb, "stem.1", stage_planes[0], 1)?;

        let mut stages = Vec::new();
        for (i, &num_blocks) in stage_blocks.iter().enumerate() {
            let mut blocks = Vec::new();
            for n in 0..num_blocks {
                let prefix = format!("layer{}.{n}", i + 1);
                let (inp, stride) = if n == 0 { (stage_planes[i + 1], 2usize) } else { (stage_planes[i + 2], 1usize) };
                blocks.push(ConvDw::load(vb, &prefix, inp, stride)?);
            }
            stages.push(blocks);
        }

        Ok(Self { stem_conv, stem_dw, stages })
    }

    /// Forward: `[1, 3, H, W]` → 4 feature maps C2..C5 at strides 4, 8, 16, 32.
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut x = self.stem_conv.forward(x)?;
        x = self.stem_dw.forward(&x)?;

        let mut outputs = Vec::with_capacity(4);
        for stage in &self.stages {
            for block in stage {
                x = block.forward(&x)?;
            }
            outputs.push(x.clone());
        }
        Ok(outputs)
    }
}
