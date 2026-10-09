//! IResNet-18 backbone for ArcFace/AdaFace face embedding.
//!
//! Architecture from insightface `arcface_torch/backbones/iresnet.py` (MIT license):
//! - Pre-activation residual blocks (BN -> conv -> BN -> PReLU -> conv -> BN + residual)
//! - Stem: conv3x3 (3->64, stride 1, pad 1) + BN + PReLU
//! - 4 stages: [2, 2, 2, 2] blocks, channels [64, 128, 256, 512], stride 2 on first block
//! - Head: BN + flatten + Linear(512*7*7, 512) + BN1d
//!
//! Input: 112x112x3 RGB, normalized (pixel - 127.5) / 128.0
//! Output: 512-d embedding vector (L2-normalizable)

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// PReLU activation: max(0,x) + weight*min(0,x).
/// Weight shape is either [1] (shared) or [channels].
struct Prelu {
    weight: Tensor,
}

impl Prelu {
    fn load(vb: &VarBuilder, name: &str) -> Result<Self> {
        let weight = vb.get_unchecked(name)?;
        Ok(Self { weight })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = if self.weight.dims().len() == 1 && self.weight.dim(0)? > 1 {
            // Per-channel: reshape to [1, C, 1, 1] for broadcasting
            self.weight.reshape((1, (), 1, 1))?
        } else {
            // Shared scalar
            self.weight.clone()
        };
        // PReLU(x) = max(0,x) + w * min(0,x)
        let pos = x.relu()?;
        let neg = x.minimum(&x.zeros_like()?)?;
        let neg_scaled = neg.broadcast_mul(&w)?;
        pos.add(&neg_scaled)
    }
}

/// IBasicBlock: pre-activation residual block.
///
/// Layout (following insightface iresnet.py):
///   bn1 -> conv1 (3x3, stride 1, pad 1) -> bn2 -> prelu -> conv2 (3x3, stride, pad 1) -> bn3
///   residual = downsample(input) if needed
///   output = block_output + residual
struct IBasicBlock {
    bn1: BatchNorm,
    conv1: Tensor,
    bn2: BatchNorm,
    prelu: Prelu,
    conv2: Tensor,
    bn3: BatchNorm,
    downsample: Option<Downsample>,
    stride: usize,
}

/// Downsample: 1x1 conv + BN to match dimensions.
struct Downsample {
    conv: Tensor,
    bn_weight: Tensor,
    bn_bias: Tensor,
    bn_mean: Tensor,
    bn_var: Tensor,
    stride: usize,
}

impl Downsample {
    fn load(vb: &VarBuilder, prefix: &str, stride: usize) -> Result<Self> {
        let conv = vb.get_unchecked(&format!("{prefix}.0.weight"))?;
        let bn_weight = vb.get_unchecked(&format!("{prefix}.1.weight"))?;
        let bn_bias = vb.get_unchecked(&format!("{prefix}.1.bias"))?;
        let bn_mean = vb.get_unchecked(&format!("{prefix}.1.running_mean"))?;
        let bn_var = vb.get_unchecked(&format!("{prefix}.1.running_var"))?;
        Ok(Self { conv, bn_weight, bn_bias, bn_mean, bn_var, stride })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv2d(&self.conv, 0, self.stride, 1, 1)?;
        batch_norm(&x, &self.bn_weight, &self.bn_bias, &self.bn_mean, &self.bn_var)
    }
}

/// Standalone batch norm parameters.
struct BatchNorm {
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
}

impl BatchNorm {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let weight = vb.get_unchecked(&format!("{prefix}.weight"))?;
        let bias = vb.get_unchecked(&format!("{prefix}.bias"))?;
        let mean = vb.get_unchecked(&format!("{prefix}.running_mean"))?;
        let var = vb.get_unchecked(&format!("{prefix}.running_var"))?;
        Ok(Self { weight, bias, mean, var })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        batch_norm(x, &self.weight, &self.bias, &self.mean, &self.var)
    }
}

impl IBasicBlock {
    fn load(vb: &VarBuilder, prefix: &str, stride: usize, has_downsample: bool) -> Result<Self> {
        let bn1 = BatchNorm::load(vb, &format!("{prefix}.bn1"))?;
        let conv1 = vb.get_unchecked(&format!("{prefix}.conv1.weight"))?;
        let bn2 = BatchNorm::load(vb, &format!("{prefix}.bn2"))?;
        let prelu = Prelu::load(vb, &format!("{prefix}.prelu.weight"))?;
        let conv2 = vb.get_unchecked(&format!("{prefix}.conv2.weight"))?;
        let bn3 = BatchNorm::load(vb, &format!("{prefix}.bn3"))?;
        let downsample = if has_downsample { Some(Downsample::load(vb, &format!("{prefix}.downsample"), stride)?) } else { None };
        Ok(Self { bn1, conv1, bn2, prelu, conv2, bn3, downsample, stride })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = match &self.downsample {
            Some(ds) => ds.forward(x)?,
            None => x.clone(),
        };
        // Pre-activation: BN -> conv -> BN -> PReLU -> conv -> BN
        let out = self.bn1.forward(x)?;
        let out = out.conv2d(&self.conv1, 1, 1, 1, 1)?; // conv1: stride 1 always
        let out = self.bn2.forward(&out)?;
        let out = self.prelu.forward(&out)?;
        let out = out.conv2d(&self.conv2, 1, self.stride, 1, 1)?; // conv2: carries the stride
        let out = self.bn3.forward(&out)?;
        out.add(&residual)
    }
}

/// A stage: a sequence of IBasicBlocks.
struct Stage {
    blocks: Vec<IBasicBlock>,
}

impl Stage {
    fn load(vb: &VarBuilder, prefix: &str, num_blocks: usize, stride: usize, in_planes: usize, out_planes: usize) -> Result<Self> {
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let s = if i == 0 { stride } else { 1 };
            let needs_downsample = i == 0 && (stride != 1 || in_planes != out_planes);
            blocks.push(IBasicBlock::load(vb, &format!("{prefix}.{i}"), s, needs_downsample)?);
        }
        Ok(Self { blocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

/// BatchNorm1d for the final embedding normalization.
struct BatchNorm1d {
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
}

impl BatchNorm1d {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let weight = vb.get_unchecked(&format!("{prefix}.weight"))?;
        let bias = vb.get_unchecked(&format!("{prefix}.bias"))?;
        let mean = vb.get_unchecked(&format!("{prefix}.running_mean"))?;
        let var = vb.get_unchecked(&format!("{prefix}.running_var"))?;
        Ok(Self { weight, bias, mean, var })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, features] (2D)
        let x = x.broadcast_sub(&self.mean)?;
        let denom = (&self.var + 1e-5f64)?.sqrt()?;
        let x = x.broadcast_div(&denom)?;
        x.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

/// IResNet-18 face embedding backbone.
///
/// Produces a 512-d embedding from a 112x112 RGB face crop.
pub struct IResNet {
    // Stem
    conv1: Tensor,
    bn1: BatchNorm,
    prelu: Prelu,
    // Stages
    layer1: Stage, // 64  -> 64,  stride 2
    layer2: Stage, // 64  -> 128, stride 2
    layer3: Stage, // 128 -> 256, stride 2
    layer4: Stage, // 256 -> 512, stride 2
    // Head
    bn2: BatchNorm,
    fc: Tensor,
    fc_bias: Tensor,
    features: BatchNorm1d,
}

impl IResNet {
    /// Load IResNet-18 from a VarBuilder.
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        // Stem: conv1 (3->64, 3x3, stride 1, pad 1) + bn1 + prelu
        let conv1 = vb.get_unchecked("conv1.weight")?;
        let bn1 = BatchNorm::load(vb, "bn1")?;
        let prelu = Prelu::load(vb, "prelu.weight")?;

        // Stages: IResNet-18 is [2, 2, 2, 2]
        let layer1 = Stage::load(vb, "layer1", 2, 2, 64, 64)?;
        let layer2 = Stage::load(vb, "layer2", 2, 2, 64, 128)?;
        let layer3 = Stage::load(vb, "layer3", 2, 2, 128, 256)?;
        let layer4 = Stage::load(vb, "layer4", 2, 2, 256, 512)?;

        // Head: bn2 (over 512 channels) + fc (512*7*7 -> 512) + features (BN1d)
        let bn2 = BatchNorm::load(vb, "bn2")?;
        let fc = vb.get_unchecked("fc.weight")?;
        let fc_bias = vb.get_unchecked("fc.bias")?;
        let features = BatchNorm1d::load(vb, "features")?;

        Ok(Self { conv1, bn1, prelu, layer1, layer2, layer3, layer4, bn2, fc, fc_bias, features })
    }

    /// Run the backbone on preprocessed input tensor [1, 3, 112, 112].
    /// Returns a 512-d embedding vector (not L2-normalized).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Stem
        let x = x.conv2d(&self.conv1, 1, 1, 1, 1)?;
        let x = self.bn1.forward(&x)?;
        let x = self.prelu.forward(&x)?;

        // Stages
        let x = self.layer1.forward(&x)?;
        let x = self.layer2.forward(&x)?;
        let x = self.layer3.forward(&x)?;
        let x = self.layer4.forward(&x)?;

        // Head: BN -> flatten -> FC -> BN1d
        let x = self.bn2.forward(&x)?;
        // Flatten spatial dims: [1, 512, 7, 7] -> [1, 512*7*7] = [1, 25088]
        let x = x.flatten_from(1)?;
        // Linear: x @ fc^T + bias
        let x = x.matmul(&self.fc.t()?)?.broadcast_add(&self.fc_bias)?;
        // Final BN1d on the embedding
        self.features.forward(&x)
    }
}

/// Batch normalization (inference mode): (x - mean) / sqrt(var + eps) * weight + bias.
fn batch_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, mean: &Tensor, var: &Tensor) -> Result<Tensor> {
    let x = x.broadcast_sub(&mean.reshape((1, (), 1, 1))?)?;
    let denom = (var + 1e-5f64)?.sqrt()?;
    let x = x.broadcast_div(&denom.reshape((1, (), 1, 1))?)?;
    x.broadcast_mul(&weight.reshape((1, (), 1, 1))?)?.broadcast_add(&bias.reshape((1, (), 1, 1))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelu_positive_passthrough() {
        let dev = candle_core::Device::Cpu;
        let w = Tensor::from_vec(vec![0.25f32], (1,), &dev).unwrap();
        let prelu = Prelu { weight: w };
        let x = Tensor::from_vec(vec![1.0f32, -2.0, 0.0, 3.0], (1, 1, 2, 2), &dev).unwrap();
        let y = prelu.forward(&x).unwrap();
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        // positive passes through, negative scaled by 0.25
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - (-0.5)).abs() < 1e-6);
        assert!((v[2] - 0.0).abs() < 1e-6);
        assert!((v[3] - 3.0).abs() < 1e-6);
    }
}
