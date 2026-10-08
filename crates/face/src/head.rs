//! SCRFDHead — detection head for SCRFD-500M.
//!
//! SCRFD-500M head config:
//!   num_classes=1, in_channels=16, stacked_convs=2, feat_channels=64,
//!   norm_cfg=GN(16), cls_reg_share=True, strides_share=True, dw_conv=True,
//!   scale_mode=2, num_anchors=2 (scales=[1,2], ratios=[1.0]),
//!   use_dfl=False, use_kps=False.
//!
//! Architecture:
//!   2 depthwise-separable convs (shared across strides, shared between cls/reg).
//!   Final cls conv: Conv2d(64→2, 3×3)  (1 class × 2 anchors)
//!   Final reg conv: Conv2d(64→8, 3×3)  (4 coords × 2 anchors, no DFL)
//!   Learnable Scale per stride level (3 scales).
//!
//! GroupNorm(16 groups) is used inside the depthwise-separable convs.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// GroupNorm: groups fixed at 16 for SCRFD-500M head.
struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    num_groups: usize,
}

impl GroupNorm {
    fn load(vb: &VarBuilder, prefix: &str, num_groups: usize) -> Result<Self> {
        let weight = vb.get_unchecked(&format!("{prefix}.weight"))?;
        let bias = vb.get_unchecked(&format!("{prefix}.bias"))?;
        Ok(Self {
            weight,
            bias,
            num_groups,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = (x.dim(0)?, x.dim(1)?, x.dim(2)?, x.dim(3)?);
        let g = self.num_groups;
        let cpg = c / g; // channels per group

        // Reshape to [B, G, C/G, H, W]
        let x = x.reshape((b, g, cpg, h, w))?;

        // Mean and variance over (C/G, H, W) dims
        let mean = x.mean_keepdim((2, 3, 4))?; // [B, G, 1, 1, 1]
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim((2, 3, 4))?; // [B, G, 1, 1, 1]
        let std = (var + 1e-5f64)?.sqrt()?;
        let normalized = centered.broadcast_div(&std)?;

        // Reshape back to [B, C, H, W]
        let normalized = normalized.reshape((b, c, h, w))?;

        // Apply affine: weight * normalized + bias
        normalized
            .broadcast_mul(&self.weight.reshape((1, (), 1, 1))?)?
            .broadcast_add(&self.bias.reshape((1, (), 1, 1))?)
    }
}

/// A depthwise conv + GroupNorm + ReLU layer (part of DepthwiseSeparableConvModule).
struct DwConvGn {
    weight: Tensor,
    gn: GroupNorm,
    padding: usize,
}

impl DwConvGn {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let weight = vb.get_unchecked(&format!("{prefix}.conv.weight"))?;
        let gn = GroupNorm::load(vb, &format!("{prefix}.gn"), 16)?;
        let k = weight.dim(2)?;
        let padding = if k == 1 { 0 } else { 1 };
        Ok(Self {
            weight,
            gn,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let groups = self.weight.dim(0)?;
        let x = x.conv2d(&self.weight, self.padding, 1, 1, groups)?;
        let x = self.gn.forward(&x)?;
        x.relu()
    }
}

/// A pointwise conv + GroupNorm + ReLU layer (part of DepthwiseSeparableConvModule).
struct PwConvGn {
    weight: Tensor,
    gn: GroupNorm,
}

impl PwConvGn {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let weight = vb.get_unchecked(&format!("{prefix}.conv.weight"))?;
        let gn = GroupNorm::load(vb, &format!("{prefix}.gn"), 16)?;
        Ok(Self { weight, gn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv2d(&self.weight, 0, 1, 1, 1)?;
        let x = self.gn.forward(&x)?;
        x.relu()
    }
}

/// One DepthwiseSeparableConvModule: dw 3×3 + GN + ReLU → pw 1×1 + GN + ReLU.
struct DwSepConv {
    depthwise_conv: DwConvGn,
    pointwise_conv: PwConvGn,
}

impl DwSepConv {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let depthwise_conv = DwConvGn::load(vb, &format!("{prefix}.depthwise_conv"))?;
        let pointwise_conv = PwConvGn::load(vb, &format!("{prefix}.pointwise_conv"))?;
        Ok(Self {
            depthwise_conv,
            pointwise_conv,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.depthwise_conv.forward(x)?;
        self.pointwise_conv.forward(&x)
    }
}

/// A plain Conv2d (weight + bias) for final cls/reg prediction.
struct ConvBias {
    weight: Tensor,
    bias: Tensor,
}

impl ConvBias {
    fn load(vb: &VarBuilder, prefix: &str) -> Result<Self> {
        let weight = vb.get_unchecked(&format!("{prefix}.weight"))?;
        let bias = vb.get_unchecked(&format!("{prefix}.bias"))?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv2d(&self.weight, 1, 1, 1, 1)?;
        x.broadcast_add(&self.bias.reshape((1, (), 1, 1))?)
    }
}

/// The SCRFDHead for SCRFD-500M.
pub struct Head {
    /// 2 depthwise-separable convs (shared across strides and cls/reg via cls_reg_share+strides_share).
    cls_convs: Vec<DwSepConv>,
    /// Final classification conv: 64→2 (1 class × 2 anchors).
    cls_pred: ConvBias,
    /// Final regression conv: 64→8 (4 coords × 2 anchors, no DFL).
    reg_pred: ConvBias,
    /// Learnable scale per stride level (3 scales for strides 8, 16, 32).
    scales: Vec<Tensor>,
}

impl Head {
    /// Load the SCRFDHead.
    ///
    /// SCRFD-500M: stacked_convs=2, feat_channels=64, GN(16), dw_conv=True,
    /// cls_reg_share=True, strides_share=True (key "0"), scale_mode=2.
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let mut cls_convs = Vec::new();
        for i in 0..2 {
            cls_convs.push(DwSepConv::load(vb, &format!("cls_stride_convs.0.{i}"))?);
        }

        let cls_pred = ConvBias::load(vb, "stride_cls.0")?;
        let reg_pred = ConvBias::load(vb, "stride_reg.0")?;

        let mut scales = Vec::new();
        for i in 0..3 {
            let s = vb.get_unchecked(&format!("scales.{i}.scale"))?;
            scales.push(s);
        }

        Ok(Self {
            cls_convs,
            cls_pred,
            reg_pred,
            scales,
        })
    }

    /// Forward: process one feature level and return (cls_scores, bbox_preds).
    ///
    /// cls_scores: [B, H*W*num_anchors, 1] after sigmoid
    /// bbox_preds: [B, H*W*num_anchors, 4] scaled distances
    fn forward_single(&self, x: &Tensor, scale_idx: usize) -> Result<(Tensor, Tensor)> {
        let mut feat = x.clone();
        for conv in &self.cls_convs {
            feat = conv.forward(&feat)?;
        }
        // cls_reg_share: reg uses same features
        let cls_out = self.cls_pred.forward(&feat)?;
        let reg_out = self.reg_pred.forward(&feat)?;

        // Apply learnable scale to reg output
        let scale = self.scales.get(scale_idx).ok_or_else(|| {
            candle_core::Error::Msg(format!("missing scale at index {scale_idx}"))
        })?;
        let reg_scaled = reg_out.broadcast_mul(scale)?;

        let b = cls_out.dim(0)?;
        let h = cls_out.dim(2)?;
        let w = cls_out.dim(3)?;
        let num_classes = 1usize;
        let num_anchors = 2usize;

        // cls_out: [B, num_anchors * num_classes, H, W] → [B, H*W*num_anchors, num_classes]
        let cls_score = cls_out
            .reshape((b, num_anchors, num_classes, h, w))?
            .permute((0, 3, 4, 1, 2))?
            .reshape((b, h * w * num_anchors, num_classes))?;

        // reg_scaled: [B, num_anchors * 4, H, W] → [B, H*W*num_anchors, 4]
        let bbox_pred = reg_scaled
            .reshape((b, num_anchors, 4, h, w))?
            .permute((0, 3, 4, 1, 2))?
            .reshape((b, h * w * num_anchors, 4))?;

        Ok((cls_score, bbox_pred))
    }

    /// Forward: process all 3 feature levels from the neck.
    ///
    /// Returns (all_cls_scores, all_bbox_preds) concatenated across levels.
    /// all_cls_scores: [B, total_anchors, 1] (raw logits, call sigmoid in post-processing)
    /// all_bbox_preds: [B, total_anchors, 4] (scaled distances)
    pub fn forward(&self, features: &[Tensor]) -> Result<(Tensor, Tensor)> {
        let mut all_cls = Vec::new();
        let mut all_reg = Vec::new();
        for (i, feat) in features.iter().enumerate() {
            let (cls, reg) = self.forward_single(feat, i)?;
            all_cls.push(cls);
            all_reg.push(reg);
        }
        let cls = Tensor::cat(&all_cls, 1)?;
        let reg = Tensor::cat(&all_reg, 1)?;
        Ok((cls, reg))
    }
}
