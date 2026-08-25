#!/usr/bin/env python3
"""消融实验: WV3 数据集上分别去掉空间约束/光谱约束/Phase1约束, 测指标汇总"""

import subprocess, re, csv, sys, time
from pathlib import Path

WORKDIR = Path("/media/zouhe/Elements1/zspan/zup")
PYTHON = "/home/zouhe/miniconda3/envs/zspan/bin/python"

# WV3 最优参数
WV3_PARAMS = {
    "--dataset": "WV3",
    "--mode": "full",
    "--epochs_phase1": "120",
    "--epochs_phase2": "2000",
    "--w_spa": "200",
    "--w_spec": "250",
    "--lmbd_weight": "500",
}

ABLATIONS = ["no_spatial", "no_spectral", "no_phase1"]
RESULTS = []

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"[{desc}]")
    print(f"  CMD: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(WORKDIR), capture_output=True, text=True)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s)")
        print(f"  STDERR:\n{r.stderr[-500:]}")
        return False, ""
    print(f"  OK ({elapsed:.0f}s)")
    return True, r.stdout

def parse_metrics(stdout):
    """从 test_toolbox.py 的输出里提取 D_lambda, D_S, QNRI 的 mean"""
    m = re.search(r"(\d+\.\d+)===\|===(\d+\.\d+)===\|===(\d+\.\d+)", stdout)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    # Try line-based fallback
    lines = stdout.split("\n")
    for i, line in enumerate(lines):
        if "D_lambda_mean" in line and i+1 < len(lines):
            parts = lines[i+1].split("===|===")
            if len(parts) >= 3:
                try:
                    return float(parts[0]), float(parts[1]), float(parts[2])
                except:
                    pass
    return None, None, None

# ---- 依次跑三个消融实验 ----
for ab in ABLATIONS:
    print(f"\n{'#'*60}")
    print(f"# ABLATION: {ab}")
    print(f"{'#'*60}")

    # Step 1: 训练
    train_args = [PYTHON, str(WORKDIR/"train_self.py"), "--ablation", ab,
                  "--start_id", "0", "--end_id", "19"]
    for k, v in WV3_PARAMS.items():
        train_args.extend([k, v])

    ok, _ = run_cmd(train_args, f"Train {ab}")
    if not ok:
        print(f"  SKIP {ab} — training failed")
        RESULTS.append({"ablation": ab, "D_lambda": "FAIL", "D_S": "FAIL", "HQNR": "FAIL"})
        continue

    time.sleep(5)

    # Step 2: 测试 (FS 模式)
    result_path = f"result_abla/{ab}/WV3_full/%d_self_result.mat"
    test_args = [PYTHON, str(WORKDIR/"test_toolbox.py"), "--mode", "fs",
                 "--path", result_path, "--sensor", "WV3", "--start", "0", "--count", "20"]

    ok, stdout = run_cmd(test_args, f"Test {ab}")
    if not ok:
        print(f"  SKIP {ab} — testing failed")
        RESULTS.append({"ablation": ab, "D_lambda": "FAIL", "D_S": "FAIL", "HQNR": "FAIL"})
        continue

    dl, ds, hqnr = parse_metrics(stdout)
    if dl is not None:
        RESULTS.append({"ablation": ab, "D_lambda": round(dl, 6), "D_S": round(ds, 6), "HQNR": round(hqnr, 6)})
        print(f"  Metrics: D_lambda={dl:.6f}  D_S={ds:.6f}  HQNR={hqnr:.6f}")
    else:
        print(f"  WARNING: could not parse metrics from output")
        RESULTS.append({"ablation": ab, "D_lambda": "PARSE_ERR", "D_S": "PARSE_ERR", "HQNR": "PARSE_ERR"})

# ---- 保存 CSV ----
csv_path = WORKDIR / "result_abla" / "ablation_metrics.csv"
Path(WORKDIR / "result_abla").mkdir(parents=True, exist_ok=True)
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ablation", "D_lambda", "D_S", "HQNR"])
    w.writeheader()
    w.writerows(RESULTS)
print(f"\nSaved: {csv_path}")
for r in RESULTS:
    print(f"  {r['ablation']:15s}  D_lambda={r['D_lambda']}  D_S={r['D_S']}  HQNR={r['HQNR']}")

print("\nDone!")
