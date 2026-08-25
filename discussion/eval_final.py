import sys, os, h5py, numpy as np, scipy.io as sio
sys.path.insert(0, '/media/zouhe/Elements/baseline/baseline_test')
from metrics import evaluate_fs

DISC = '/media/zouhe/Elements/zspan/zup/discussion'
TEST = '/media/zouhe/Elements/Data/PanCollection/test_data/test_wv3_OrigScale_multiExm1.h5'

# Load test data once
f = h5py.File(TEST, 'r')
pan_all = np.array(f['pan'][:], dtype=np.float64)
lms_all = np.array(f['lms'][:], dtype=np.float64)
ms_all  = np.array(f['ms'][:], dtype=np.float64)
f.close()

experiments = sorted([
    d for d in os.listdir(DISC)
    if os.path.isdir(os.path.join(DISC, d))
    and (d.startswith('fista_T') or d.startswith('lmbd_weight_'))
])

results = []
for exp in experiments:
    rdir = os.path.join(DISC, exp, 'result_self', 'WV3_full')
    if not os.path.isdir(rdir): continue
    
    dl, ds, hq = [], [], []
    for idx in range(20):
        mat = sio.loadmat(os.path.join(rdir, f'{idx}_self_result.mat'))
        fused = mat['proposed'].astype(np.float64)
        m = evaluate_fs(
            fused,
            lms_all[idx].transpose(1,2,0),
            pan_all[idx].squeeze(),
            ms_all[idx].transpose(1,2,0),
            'WV3', ratio=4
        )
        dl.append(m['D_lambda']); ds.append(m['D_s']); hq.append(m['HQNR'])
    
    results.append((exp, np.mean(dl), np.std(dl), np.mean(ds), np.std(ds), np.mean(hq), np.std(hq)))

# Print table
print(f"{'Experiment':<22s} {'D_lambda':>14s} {'D_s':>14s} {'HQNR':>14s}")
print("-" * 64)
for r in results:
    print(f"{r[0]:<22s} {r[1]:.4f}±{r[2]:<8.4f} {r[3]:.4f}±{r[4]:<8.4f} {r[5]:.4f}±{r[6]:<8.4f}")

# Save CSV
csv_path = os.path.join(DISC, 'ablation_results.csv')
with open(csv_path, 'w') as f:
    f.write("Experiment,D_lambda_mean,D_lambda_std,D_s_mean,D_s_std,HQNR_mean,HQNR_std\n")
    for r in results:
        f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f}\n")
print(f"\nSaved: {csv_path}")
