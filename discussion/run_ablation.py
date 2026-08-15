#!/usr/bin/env python3
"""
Master ablation script: FISTA steps (T=3,5,7) and lmbd (10..500).
Each experiment runs train_self.py on all 20 WV3 full images.
Results saved to discussion/{exp_name}/.
"""

import subprocess, sys, os, shutil, time

BASE = "/media/zouhe/Elements/zspan/zup"
PYTHON = "/home/zouhe/miniconda3/envs/zspan/bin/python"
DISC = os.path.join(BASE, "discussion")
SRC_FILES = ["train_self.py", "mymodel.py", "data.py", "loss.py", "wald_utilities.py"]

os.makedirs(DISC, exist_ok=True)

def prepare_exp(exp_name):
    """Copy source files to discussion/{exp_name}/"""
    exp_dir = os.path.join(DISC, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    for f in SRC_FILES:
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(exp_dir, f))
    return exp_dir

def patch_num_layers(exp_dir, T):
    """Set NUM_LAYERS in copied mymodel.py"""
    path = os.path.join(exp_dir, "mymodel.py")
    with open(path) as f: content = f.read()
    import re
    content = re.sub(r'"NUM_LAYERS":\s*\d+', f'"NUM_LAYERS": {T}', content)
    with open(path, 'w') as f: f.write(content)
    print(f"  Patched NUM_LAYERS -> {T}")

def patch_fixed_lmbd(exp_dir, lmbd_val):
    """Fix all DictBlock.lmbd to a constant value (non-trainable)"""
    path = os.path.join(exp_dir, "mymodel.py")
    with open(path) as f: content = f.read()
    # Add FIXED_LMBd class variable to DictBlock before __init__
    fix_line = f'\nDictBlock._fixed_lmbd = {lmbd_val}\n'
    content = fix_line + content
    with open(path, 'w') as f: f.write(content)
    print(f"  Fixed lmbd -> {lmbd_val} (non-trainable)")

def run_experiment(exp_dir, exp_name):
    """Run train_self.py from the experiment directory."""
    log_path = os.path.join(exp_dir, "run.log")
    train_path = os.path.join(exp_dir, "train_self.py")
    
    print(f"  Running {exp_name} ... ", end="", flush=True)
    t0 = time.time()
    
    with open(log_path, 'w') as log:
        result = subprocess.run(
            [PYTHON, train_path, "--mode", "full", "--dataset", "WV3",
             "--start_id", "0", "--end_id", "19"],
            cwd=exp_dir,  # run from exp_dir so result_self/ is local
            stdout=log, stderr=subprocess.STDOUT,
            timeout=7200
        )
    
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
    print(f"{status} ({elapsed:.0f}s)")
    return result.returncode

def move_results(exp_dir, exp_name):
    """Move result_self/ and model_FUG/ into the experiment directory if outside."""
    for sub in ["result_self", "model_FUG"]:
        src = os.path.join(exp_dir, sub)
        if not os.path.exists(src):
            # Maybe they ended up in BASE
            src2 = os.path.join(BASE, sub)
            if os.path.exists(src2):
                dst = os.path.join(exp_dir, sub)
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.move(src2, dst)
                print(f"  Moved {sub} -> {dst}")

def main():
    experiments = []
    
    # ---- FISTA steps: T=3,5,7 ----
    for T in [3, 5, 7]:
        experiments.append(("fista_T%d" % T, "fista", T))
    
    # ---- Fixed lmbd: 10,100,200,300,400,500 ----
    for lmbd in [10, 100, 200, 300, 400, 500]:
        experiments.append(("lmbd_%d" % lmbd, "lmbd", lmbd))
    
    print("=" * 60)
    print("ABLATION EXPERIMENTS — 9 groups × 20 images each")
    print("=" * 60)
    
    results = {}
    for exp_name, exp_type, exp_val in experiments:
        print(f"\n{'='*40}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*40}")
        
        exp_dir = prepare_exp(exp_name)
        
        if exp_type == "fista":
            patch_num_layers(exp_dir, exp_val)
        elif exp_type == "lmbd":
            patch_fixed_lmbd(exp_dir, exp_val)
        
        rc = run_experiment(exp_dir, exp_name)
        move_results(exp_dir, exp_name)
        results[exp_name] = rc
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, rc in results.items():
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  {name}: {status}")
    print("=" * 60)

if __name__ == "__main__":
    main()
