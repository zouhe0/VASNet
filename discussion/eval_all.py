#!/usr/bin/env python3
"""Evaluate all ablation experiments on WV3 full-res HQNR."""
import sys, os, h5py, numpy as np, scipy.io as sio
sys.path.insert(0, '/media/zouhe/Elements/baseline/baseline_test')
from metrics import evaluate_fs

DISC = '/media/zouhe/Elements/zspan/zup/discussion'
TEST_DATA = '/media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5'

# Load test data - pass raw float values (0-2047) to evaluate_fs
f = h5py.File(TEST_DATA, 'r')
pan_all = np.array(f['pan'][:], dtype=np.float64)
lms_all = np.array(f['lms'][:], dtype=np.float64)
ms_all = np.array(f['ms'][:], dtype=np.float64)
f.close()

experiments = sorted([
    d for d in os.listdir(DISC)
    if os.path.isdir(os.path.join(DISC, d))
    and (d.startswith('fista_T') or d.startswith('lmbd_weight_'))
])

print(f"{'Experiment':<20s} {'D_lambda':>12s} {'D_s':>12s} {'HQNR':>12s}")
print("-" * 56)

for exp in experiments:
    result_dir = os.path.join(DISC, exp, 'result_self', 'WV3_full')
    if not os.path.isdir(result_dir):
        continue

    dl_list, ds_list, hqnr_list = [], [], []

    for idx in range(20):
        mat = sio.loadmat(os.path.join(result_dir, f'{idx}_self_result.mat'))
        fused = mat['proposed'].astype(np.float64)  # (512, 512, 8)
        lms = lms_all[idx].transpose(1, 2, 0).astype(np.float64)  # (512, 512, 8)
        pan = pan_all[idx].squeeze().astype(np.float64)  # (512, 512)
        ms = ms_all[idx].transpose(1, 2, 0).astype(np.float64)  # (128, 128, 8)

        metrics = evaluate_fs(fused, lms, pan, ms, 'WV3', ratio=4)
        dl_list.append(metrics['D_lambda'])
        ds_list.append(metrics['D_s'])
        hqnr_list.append(metrics['HQNR'])

    print(f"{exp:<20s} {np.mean(dl_list):.4f}±{np.std(dl_list):.3f}  "
          f"{np.mean(ds_list):.4f}±{np.std(ds_list):.3f}  "
          f"{np.mean(hqnr_list):.4f}±{np.std(hqnr_list):.3f}")

print("-" * 56)
