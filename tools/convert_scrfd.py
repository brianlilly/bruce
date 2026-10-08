#!/usr/bin/env python3
"""Convert an SCRFD-500M mmdet checkpoint to safetensors for Bruce.

Usage:
    pip install torch safetensors
    python tools/convert_scrfd.py model.pth -o scrfd_500m.safetensors

The input is the mmdet checkpoint (.pth) from the insightface SCRFD model zoo:
    https://github.com/deepinsight/insightface/tree/master/detection/scrfd

The output is a flat safetensors file whose keys match the VarBuilder prefixes
used in crates/face/ (backbone.*, neck.*, bbox_head.*).

SCRFD-500M specifics:
  - MobileNetV1 backbone: stem + 4 stages, BatchNorm with running_mean/var
  - PAFPN neck: lateral, fpn, downsample, pafpn convs (no norm)
  - SCRFDHead: cls_reg_share=True, strides_share=True, dw_conv=True,
    GroupNorm(16), scale_mode=2, use_dfl=False, use_kps=False

Keys that are not needed by the Rust inference code (num_batches_tracked,
integral.project, optimizer state) are dropped.
"""

import argparse
import sys
from pathlib import Path


def convert(src: Path, dst: Path, verbose: bool = False) -> None:
    try:
        import torch
    except ImportError:
        print("error: PyTorch is required. Install with: pip install torch", file=sys.stderr)
        sys.exit(1)
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("error: safetensors is required. Install with: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {src}")
    checkpoint = torch.load(str(src), map_location="cpu", weights_only=False)

    # mmdet checkpoints wrap the state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if "meta" in checkpoint:
            meta = checkpoint["meta"]
            if verbose:
                print(f"  mmdet epoch: {meta.get('epoch', '?')}, iter: {meta.get('iter', '?')}")
    elif isinstance(checkpoint, dict):
        # Some checkpoints are just the raw state_dict
        state_dict = checkpoint
    else:
        print(f"error: unexpected checkpoint type: {type(checkpoint)}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"  Source keys: {len(state_dict)}")

    # Filter: keep only the keys the Rust code loads, drop the rest.
    # The Rust VarBuilder uses get_unchecked (loads by name), so only keys that
    # are actually requested need to be present. Extra keys are harmless but
    # waste space.
    kept = {}
    dropped = []
    for key, tensor in state_dict.items():
        # Drop num_batches_tracked (BatchNorm bookkeeping, not used in inference)
        if key.endswith(".num_batches_tracked"):
            dropped.append(key)
            continue
        # Drop integral.project (DFL integral, not used in SCRFD-500M)
        if "integral.project" in key:
            dropped.append(key)
            continue
        # Keep backbone, neck, bbox_head keys
        if key.startswith(("backbone.", "neck.", "bbox_head.")):
            kept[key] = tensor.contiguous().float()
        else:
            dropped.append(key)

    if verbose:
        print(f"  Kept:    {len(kept)}")
        print(f"  Dropped: {len(dropped)}")
        if dropped:
            for d in sorted(dropped):
                print(f"    - {d}")

    if not kept:
        print("error: no backbone/neck/bbox_head keys found in checkpoint", file=sys.stderr)
        sys.exit(1)

    # Validate expected key families exist
    prefixes = {k.split(".")[0] for k in kept}
    for required in ("backbone", "neck", "bbox_head"):
        if required not in prefixes:
            print(f"warning: no '{required}.*' keys found", file=sys.stderr)

    # Validate expected counts for SCRFD-500M
    backbone_keys = [k for k in kept if k.startswith("backbone.")]
    neck_keys = [k for k in kept if k.startswith("neck.")]
    head_keys = [k for k in kept if k.startswith("bbox_head.")]

    if verbose:
        print(f"  backbone: {len(backbone_keys)} tensors")
        print(f"  neck:     {len(neck_keys)} tensors")
        print(f"  bbox_head: {len(head_keys)} tensors")

    # Spot-check a few critical keys
    critical = [
        "backbone.stem.0.0.weight",       # stem conv
        "backbone.stem.0.1.weight",       # stem bn
        "backbone.layer1.0.0.weight",     # first stage depthwise
        "neck.lateral_convs.0.conv.weight",
        "neck.fpn_convs.0.conv.weight",
        "bbox_head.cls_stride_convs.0.0.depthwise_conv.conv.weight",
        "bbox_head.stride_cls.0.weight",
        "bbox_head.stride_reg.0.weight",
        "bbox_head.scales.0.scale",
    ]
    missing_critical = [k for k in critical if k not in kept]
    if missing_critical:
        print("warning: missing expected keys:", file=sys.stderr)
        for k in missing_critical:
            print(f"  - {k}", file=sys.stderr)
        print("The checkpoint may use a different key naming convention.", file=sys.stderr)
        print("Continuing anyway; the Rust loader will report missing weights at load time.", file=sys.stderr)

    # Save
    print(f"Saving safetensors: {dst} ({len(kept)} tensors)")
    save_file(kept, str(dst))

    # Report size
    size = dst.stat().st_size
    if size < 1024 * 1024:
        print(f"  Size: {size / 1024:.1f} KB")
    else:
        print(f"  Size: {size / 1024 / 1024:.1f} MB")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SCRFD-500M mmdet checkpoint to safetensors for Bruce"
    )
    parser.add_argument("checkpoint", type=Path, help="Input .pth checkpoint file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .safetensors file (default: scrfd_500m.safetensors in the same directory)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed key info")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if output is None:
        output = args.checkpoint.parent / "scrfd_500m.safetensors"

    convert(args.checkpoint, output, verbose=args.verbose)


if __name__ == "__main__":
    main()
