//! PAFPN (Path Aggregation Feature Pyramid Network) neck for SCRFD-500M.
//!
//! SCRFD-500M config:
//!   in_channels=[40, 72, 152, 288], out_channels=16, start_level=1, num_outs=3,
//!   add_extra_convs='on_output', no norm/act on neck convs.
//!
//! Takes backbone outputs C2, C3, C4, C5 and produces 3 feature maps at strides 8, 16, 32.
//! Uses start_level=1, so only C3, C4, C5 (indices 1, 2, 3 of the 4 backbone outputs) feed in.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// A plain Conv2d (no batch norm, no activation) used in the FPN/PAFPN neck.
struct Conv {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    padding: usize,
}

impl Conv {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        Self::load_with_stride(vb, prefix, 1)
    }

    fn load_with_stride(vb: &VarBuilder, prefix: &str, stride: usize) -> Result<Self> {
        let weight = vb.get_unchecked(&format!("{prefix}.conv.weight"))?;
        let bias = vb.get_unchecked(&format!("{prefix}.conv.bias"))?;
        let k = weight.dim(2)?;
        let padding = if k == 1 { 0 } else { 1 };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv2d(&self.weight, self.padding, self.stride, 1, 1)?;
        x.broadcast_add(&self.bias.reshape((1, (), 1, 1))?)
    }
}

/// The PAFPN neck for SCRFD-500M.
pub struct Neck {
    /// 1×1 lateral convs: project backbone channels to out_channels (3 convs for C3, C4, C5)
    lateral_convs: Vec<Conv>,
    /// 3×3 FPN convs: refine laterals (3 convs)
    fpn_convs: Vec<Conv>,
    /// 3×3 stride-2 downsample convs for bottom-up path (2 convs)
    downsample_convs: Vec<Conv>,
    /// 3×3 PAFPN convs after bottom-up path (2 convs)
    pafpn_convs: Vec<Conv>,
}

impl Neck {
    /// Load the PAFPN neck.
    ///
    /// SCRFD-500M: in_channels=[40,72,152,288], out_channels=16, start_level=1, num_outs=3.
    /// With start_level=1, the lateral/fpn convs process backbone levels 1..4 (C3, C4, C5):
    ///   lateral_convs: 3 × Conv2d(in_ch→16, 1×1)
    ///   fpn_convs: 3 × Conv2d(16→16, 3×3)
    /// PAFPN bottom-up path (start_level+1 to backbone_end_level-1 → 2 sets):
    ///   downsample_convs: 2 × Conv2d(16→16, 3×3, stride=2)
    ///   pafpn_convs: 2 × Conv2d(16→16, 3×3)
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let mut lateral_convs = Vec::new();
        for i in 0..3 {
            lateral_convs.push(Conv::load(vb, &format!("lateral_convs.{i}"))?);
        }

        let mut fpn_convs = Vec::new();
        for i in 0..3 {
            fpn_convs.push(Conv::load(vb, &format!("fpn_convs.{i}"))?);
        }

        let mut downsample_convs = Vec::new();
        for i in 0..2 {
            downsample_convs.push(Conv::load_with_stride(
                vb,
                &format!("downsample_convs.{i}"),
                2,
            )?);
        }

        let mut pafpn_convs = Vec::new();
        for i in 0..2 {
            pafpn_convs.push(Conv::load(vb, &format!("pafpn_convs.{i}"))?);
        }

        Ok(Self {
            lateral_convs,
            fpn_convs,
            downsample_convs,
            pafpn_convs,
        })
    }

    /// Forward: takes 4 backbone outputs [C2, C3, C4, C5], produces 3 feature maps at strides 8, 16, 32.
    ///
    /// With start_level=1, only C3, C4, C5 (indices 1, 2, 3) are used.
    pub fn forward(&self, backbone_outs: &[Tensor]) -> Result<Vec<Tensor>> {
        // Step 1: lateral connections (1×1 conv to project to out_channels)
        // We use backbone outputs at indices 1, 2, 3 (start_level=1)
        let mut laterals = Vec::new();
        for (i, lat_conv) in self.lateral_convs.iter().enumerate() {
            let idx = i + 1; // start_level=1
            let feat = backbone_outs.get(idx).ok_or_else(|| {
                candle_core::Error::Msg(format!("missing backbone output at index {idx}"))
            })?;
            laterals.push(lat_conv.forward(feat)?);
        }

        // Step 2: top-down path (add upsampled higher-level to lower-level)
        let n = laterals.len();
        for i in (1..n).rev() {
            let target_h = laterals[i - 1].dim(2)?;
            let target_w = laterals[i - 1].dim(3)?;
            let upsampled = laterals[i].upsample_nearest2d(target_h, target_w)?;
            laterals[i - 1] = (&laterals[i - 1] + &upsampled)?;
        }

        // Step 3: apply FPN convs
        let mut inter_outs = Vec::new();
        for (i, fpn_conv) in self.fpn_convs.iter().enumerate() {
            inter_outs.push(fpn_conv.forward(&laterals[i])?);
        }

        // Step 4: bottom-up path (PAFPN)
        for i in 0..(n - 1) {
            let down = self.downsample_convs[i].forward(&inter_outs[i])?;
            inter_outs[i + 1] = (&inter_outs[i + 1] + &down)?;
        }

        // Step 5: apply PAFPN convs (only to levels 1..n)
        let mut outs = Vec::new();
        outs.push(inter_outs[0].clone());
        for i in 1..n {
            outs.push(self.pafpn_convs[i - 1].forward(&inter_outs[i])?);
        }

        Ok(outs)
    }
}
