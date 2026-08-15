#!/usr/bin/env python3
"""ZUP: Two-phase self-supervised pansharpening — thin CLI entry point.

Phase 1: LR Wald pretrain → self-teacher
Phase 2: Full-res distillation with spatial + spectral fidelity

Usage:
  python train_self.py --mode full --dataset WV3                     # 20 images
  python train_self.py --mode full --dataset WV3 --data_id 0         # single image

Ablation examples:
  python train_self.py --mode full --dataset WV3 --fista_T 5
  python train_self.py --mode full --dataset WV3 --lmbd_weight 300
"""
import argparse, os, sys, time, random, h5py, traceback
import numpy as np, torch, torch.nn as nn
import torch.backends.cudnn as cudnn

import config as cfg
from trainer import phase1_pretrain, phase2_distill

# ── seeds ──────────────────────────────────────────────
SEED = cfg.TRAIN_DEFAULTS["seed"]
torch.manual_seed(SEED); torch.cuda.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ── CLI ────────────────────────────────────────────────
ap = argparse.ArgumentParser("ZUP Self-Supervised Pansharpening")
ap.add_argument("--mode",       default="full", choices=["lr","full"])
ap.add_argument("--dataset",    default="WV3",  choices=["WV3","WV2","QB","GF2"])
ap.add_argument("--data_id",    type=int, default=None)
ap.add_argument("--start_id",   type=int, default=0)
ap.add_argument("--end_id",     type=int, default=19)
ap.add_argument("--device",     default="cuda:0")
# Hyperparams (override cfg.TRAIN_DEFAULTS)
ap.add_argument("--lr_phase1",     type=float, default=None)
ap.add_argument("--epochs_phase1", type=int,   default=None)
ap.add_argument("--lr_phase2",     type=float, default=None)
ap.add_argument("--epochs_phase2", type=int,   default=None)
ap.add_argument("--batch_size",    type=int,   default=None)
ap.add_argument("--w_spa",         type=float, default=None)
ap.add_argument("--w_spec",        type=float, default=None)
ap.add_argument("--lmbd_weight",   type=float, default=None)
# Ablation
ap.add_argument("--fista_T",  type=int, default=None, help="Override FISTA steps")
ap.add_argument("--ablation", default="none",
                choices=["none","no_spatial","no_spectral","no_phase1","no_csc","fix_lmbd"])
args = ap.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# ── resolve dataset ────────────────────────────────────
preset = cfg.DATASET_PRESETS[args.dataset.upper()]
sensor = preset["sensor"]
data_path = preset["path"]
if args.mode == "lr":
    data_path = data_path.replace("_OrigScale", "")
with h5py.File(data_path, "r") as f:
    args.spectral_num = f["ms"].shape[1]
os.makedirs("result_self", exist_ok=True); os.makedirs("model_FUG", exist_ok=True)
save_dir = os.path.join("result_self", f"{args.dataset.upper()}_{args.mode}")
os.makedirs(save_dir, exist_ok=True)

# ── hyperparams (CLI overrides defaults) ───────────────
for k in ["lr_phase1","epochs_phase1","lr_phase2","epochs_phase2",
          "batch_size","w_spa","w_spec","lmbd_weight"]:
    cli_val = getattr(args, k, None)
    if cli_val is not None:
        setattr(args, k, cli_val)
    elif getattr(args, k, None) is None:
        setattr(args, k, cfg.TRAIN_DEFAULTS.get(k))

# Derived
args.sensor = sensor
args.data_path = data_path

# FISTA override → patch cfg
if args.fista_T is not None:
    cfg.CSC_CONFIG["NUM_LAYERS"] = args.fista_T
    print(f"[cfg] NUM_LAYERS = {args.fista_T}")

# ── process ────────────────────────────────────────────
def process_one(did):
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
    args.model_prefix = f"{args.dataset.upper()}_{did}_self_{args.mode}"

    teacher, lms, pan, ms, gt = phase1_pretrain(did, data_path, save_dir, args, device)
    phase2_distill(did, teacher, lms, pan, ms, data_path, save_dir, args, device, gt)

    print(f"  [{did}] done ({time.time()-t0:.0f}s)", flush=True)

print(f"ZUP | {args.dataset.upper()} | mode={args.mode} | {args.spectral_num}ch")
print(f"  P1: {args.epochs_phase1}ep lr={args.lr_phase1}  P2: {args.epochs_phase2}ep lr={args.lr_phase2}")
print(f"  lmbd_weight={args.lmbd_weight}  CSC steps={cfg.CSC_CONFIG['NUM_LAYERS']}")
print()

if args.data_id is not None:
    process_one(args.data_id)
else:
    total = args.end_id - args.start_id + 1
    print(f"Processing {total} images ({args.start_id}-{args.end_id})...\n")
    t_all = time.time()
    for did in range(args.start_id, args.end_id + 1):
        try:
            process_one(did)
        except Exception as e:
            print(f"  [{did}] ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
    print(f"\nDone. {total} images, {time.time()-t_all:.0f}s total")
