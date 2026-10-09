#!/usr/bin/env python3
"""Convert an AdaFace/ArcFace IResNet-18 checkpoint to safetensors for Bruce.

Usage:
    pip install torch safetensors
    python tools/convert_adaface.py model.ckpt -o adaface_ir18.safetensors

The input is a PyTorch checkpoint (.ckpt or .pth) from AdaFace or ArcFace
model zoos.

AdaFace checkpoints (MIT license code, research-only weights):
    https://github.com/mk-minchul/AdaFace
    Recommended: adaface_ir18_webface4m.ckpt (IResNet-18, trained on WebFace4M)

The output is a flat safetensors file whose keys match the VarBuilder names
used in crates/face-embed/src/iresnet.rs.

IResNet-18 architecture:
  - Stem: conv1 (3->64, 3x3) + bn1 + PReLU
  - 4 stages: [2, 2, 2, 2] blocks, channels [64, 128, 256, 512], stride 2
  - Head: bn2 + fc (25088->512) + features (BN1d)

Keys that are not needed by the Rust inference code (num_batches_tracked,
optimizer state, head.kernel) are dropped.
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

    # AdaFace checkpoints may wrap state_dict in various ways
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        print(f"error: unexpected checkpoint type: {type(checkpoint)}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"  Source keys: {len(state_dict)}")
        for k in sorted(state_dict.keys()):
            print(f"    {k}: {list(state_dict[k].shape)}")

    # Strip common prefixes from AdaFace checkpoints
    # Some checkpoints use "model." prefix, some use "backbone." prefix
    cleaned = {}
    for key, tensor in state_dict.items():
        # Strip "model." prefix (AdaFace lightning format)
        k = key
        if k.startswith("model."):
            k = k[len("model."):]
        # Strip "backbone." prefix if present
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        cleaned[k] = tensor

    # Filter: keep only the keys the Rust code loads
    kept = {}
    dropped = []
    for key, tensor in cleaned.items():
        # Drop num_batches_tracked (BatchNorm bookkeeping, not used in inference)
        if key.endswith(".num_batches_tracked"):
            dropped.append(key)
            continue
        # Drop ArcFace/AdaFace classification head kernel (not needed for embedding)
        if key.startswith("head.") or key.startswith("kernel") or key.startswith("loss."):
            dropped.append(key)
            continue
        # Drop optimizer/scheduler state
        if key.startswith("optimizer") or key.startswith("scheduler"):
            dropped.append(key)
            continue
        # Drop AdaFace-specific head components we don't need
        if key.startswith("head_") or key.startswith("norm."):
            dropped.append(key)
            continue

        # Check this is a key the IResNet loader expects
        valid_prefixes = (
            "conv1.", "bn1.", "prelu.",          # stem
            "layer1.", "layer2.", "layer3.", "layer4.",  # stages
            "bn2.", "fc.", "features.",           # head
        )
        if any(key.startswith(p) for p in valid_prefixes):
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
        print("error: no IResNet keys found in checkpoint", file=sys.stderr)
        print("  Expected keys like conv1.weight, bn1.weight, layer1.0.bn1.weight, etc.", file=sys.stderr)
        print("  Found keys:", file=sys.stderr)
        for k in sorted(cleaned.keys())[:20]:
            print(f"    {k}", file=sys.stderr)
        if len(cleaned) > 20:
            print(f"    ... and {len(cleaned) - 20} more", file=sys.stderr)
        sys.exit(1)

    # Spot-check critical keys
    critical = [
        "conv1.weight",          # stem conv
        "bn1.weight",            # stem bn
        "prelu.weight",          # stem prelu
        "layer1.0.bn1.weight",   # first block
        "layer1.0.conv1.weight",
        "layer2.0.downsample.0.weight",  # first downsample conv
        "layer2.0.downsample.1.weight",  # first downsample bn
        "layer4.1.conv2.weight", # last block
        "bn2.weight",            # head bn
        "fc.weight",             # head fc
        "fc.bias",
        "features.weight",       # head bn1d
    ]
    missing_critical = [k for k in critical if k not in kept]
    if missing_critical:
        print("warning: missing expected keys:", file=sys.stderr)
        for k in missing_critical:
            print(f"  - {k}", file=sys.stderr)
        print("The checkpoint may use a different key naming convention.", file=sys.stderr)
        print("Continuing anyway; the Rust loader will report missing weights at load time.", file=sys.stderr)

    # Validate shapes for key tensors
    shape_checks = {
        "conv1.weight": (64, 3, 3, 3),
        "fc.weight": (512, 25088),
        "fc.bias": (512,),
    }
    for key, expected_shape in shape_checks.items():
        if key in kept:
            actual = tuple(kept[key].shape)
            if actual != expected_shape:
                print(f"warning: {key} shape {actual} != expected {expected_shape}", file=sys.stderr)
                print("  This may not be an IResNet-18 checkpoint.", file=sys.stderr)

    # Report tensor counts per section
    sections = {
        "stem": ["conv1.", "bn1.", "prelu."],
        "layer1": ["layer1."],
        "layer2": ["layer2."],
        "layer3": ["layer3."],
        "layer4": ["layer4."],
        "head": ["bn2.", "fc.", "features."],
    }
    if verbose:
        for section, prefixes in sections.items():
            count = sum(1 for k in kept if any(k.startswith(p) for p in prefixes))
            print(f"  {section}: {count} tensors")

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
        description="Convert AdaFace/ArcFace IResNet-18 checkpoint to safetensors for Bruce"
    )
    parser.add_argument("checkpoint", type=Path, help="Input .ckpt or .pth checkpoint file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .safetensors file (default: adaface_ir18.safetensors in the same directory)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed key info")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if output is None:
        output = args.checkpoint.parent / "adaface_ir18.safetensors"

    convert(args.checkpoint, output, verbose=args.verbose)


if __name__ == "__main__":
    main()
