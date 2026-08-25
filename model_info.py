#!/usr/bin/env python3
"""
Combined FLOPs & Parameters calculator for all pansharpening models.

Models:
  1. Net_ms2pan (SDE)         -- SDE baseline
  2. FusionNet (mymodel)      -- original fusion network
  3. SDNetFusionNet_All       -- trainself self-supervised model (CSC layers)
     - LR training input:   128x128 (Wald protocol)
     - Full-size inference: 512x512

Usage:
  python model_info.py [--spectral_num 8]
"""

import argparse
import torch
import torch.nn as nn
from SDE import Net_ms2pan
from mymodel import FusionNet, SDNetFusionNet_All


def count_params(model: nn.Module, grad_only: bool = True) -> int:
    """Count parameters. grad_only=True: only trainable params."""
    return sum(p.numel() for p in model.parameters()
               if (not grad_only) or p.requires_grad)


def count_params_detail(model: nn.Module) -> dict:
    """Per-module parameter breakdown (deduplicated by param id)."""
    seen = set()
    leaf = {}
    total = 0
    modules_by_depth = []
    for name, mod in model.named_modules():
        depth = name.count(".") + 1 if name else 0
        modules_by_depth.append((depth, name, mod))
    for _, name, mod in sorted(modules_by_depth, key=lambda x: -x[0]):
        own = 0
        for p in mod.parameters(recurse=False):
            pid = id(p)
            if pid not in seen and p.requires_grad:
                seen.add(pid)
                own += p.numel()
        if own > 0:
            leaf[name] = own
            total += own
    leaf["__total__"] = total
    return leaf


def calc_flops_params(model: nn.Module, inputs: tuple, label: str):
    """Print FLOPs and param counts via thop.profile, fall back to manual."""
    try:
        from thop import profile
        macs, params = profile(model, inputs=inputs, verbose=False)
        print(f"{label}")
        print(f"  FLOPs (MACs): {macs/1e9:.4f} G")
        print(f"  Params:       {params/1e6:.4f} M  ({int(params):,})")
    except ImportError:
        print(f"{label}  [thop not available]")

    manual = count_params(model)
    print(f"  Params (manual): {manual/1e6:.4f} M  ({manual:,})")


def main():
    parser = argparse.ArgumentParser(description="Model FLOPs & params report")
    parser.add_argument("--spectral_num", type=int, default=8,
                        help="Spectral channels (default: 8 for WV3/WV2)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Full-size spatial dim (default: 512)")
    args = parser.parse_args()

    C = args.spectral_num
    H = args.resolution
    H_lr = H // 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  spectral_num={C}  |  resolution={H}x{H}  |  LR={H_lr}x{H_lr}")
    print("=" * 70)

    # 1. Net_ms2pan (SDE)
    print("\n[1] Net_ms2pan  (SDE baseline)")
    ms0 = torch.randn(1, C, H_lr, H_lr).to(device)
    model1 = Net_ms2pan().to(device)
    calc_flops_params(model1, (ms0,), "  input: (1,{}, {}, {})".format(C, H_lr, H_lr))

    # 2. FusionNet
    print("\n[2] FusionNet  (original)")
    ms = torch.randn(1, C, H, H).to(device)
    pan = torch.randn(1, 1, H, H).to(device)
    model2 = FusionNet().to(device)
    calc_flops_params(model2, (ms, pan), "  input: ms(1,{},H,W) + pan(1,1,H,W)".format(C))

    # 3. SDNetFusionNet_All (trainself) — separate instances per input size
    #    (DictBlock caches spatial dims on first forward, so thop needs fresh models)
    print("\n[3] SDNetFusionNet_All  (trainself -- CSC layers)")

    model3_lr = SDNetFusionNet_All(spectral_num=C).to(device)
    lms_lr = torch.randn(1, C, H_lr, H_lr).to(device)
    pan_lr = torch.randn(1, 1, H_lr, H_lr).to(device)
    calc_flops_params(model3_lr, (lms_lr, pan_lr),
                      "  LR training: (1,{},{},{}) + pan(1,1,{},{})".format(C, H_lr, H_lr, H_lr, H_lr))

    model3_full = SDNetFusionNet_All(spectral_num=C).to(device)
    lms_full = torch.randn(1, C, H, H).to(device)
    pan_full = torch.randn(1, 1, H, H).to(device)
    calc_flops_params(model3_full, (lms_full, pan_full),
                      "  Full-size:  (1,{},{},{}) + pan(1,1,{},{})".format(C, H, H, H, H))

    # Per-module breakdown (use one instance for display)
    print("\n" + "=" * 70)
    print("SDNetFusionNet_All -- per-module parameter breakdown:")
    detail = count_params_detail(model3_full)
    for k in sorted(detail, key=lambda x: detail[x], reverse=True):
        if k == "__total__":
            continue
        print(f"  {k:<50s} {detail[k]:>12,}")
    print(f"  {'TOTAL':<50s} {detail['__total__']:>12,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
