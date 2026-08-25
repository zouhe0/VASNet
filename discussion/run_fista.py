#!/usr/bin/env python3
"""
FISTA step (T) ablation: T = 3, 5, 7
Runs train_self.py on all 20 WV3 full images for each T.
Results saved to discussion/fista_T{T}/WV3_full/
"""
import subprocess, sys, os, shutil

T_VALUES = [3, 5, 7]
BASE_DIR = "/media/zouhe/Elements1/zspan/zup"
PYTHON = "/home/zouhe/miniconda3/envs/zspan/bin/python"
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_self.py")

for T in T_VALUES:
    out_dir = os.path.join(BASE_DIR, "discussion", f"fista_T{T}")
    
    # Patch mymodel.py: set NUM_LAYERS
    model_path = os.path.join(BASE_DIR, "mymodel.py")
    with open(model_path) as f:
        orig = f.read()
    
    # Modify NUM_LAYERS in _default_sdnet_cfg
    import re
    patched = re.sub(r'"NUM_LAYERS":\s*\d+', f'"NUM_LAYERS": {T}', orig)
    with open(model_path, 'w') as f:
        f.write(patched)
    
    # Create output subdir
    os.makedirs(os.path.join(out_dir, "WV3_full"), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FISTA T={T} — processing 20 images")
    print(f"{'='*60}")
    
    for img_id in range(20):
        print(f"\n--- FISTA T={T}, image {img_id} ---")
        result = subprocess.run(
            [PYTHON, TRAIN_SCRIPT, "--mode", "full", "--dataset", "WV3",
             "--data_id", str(img_id)],
            cwd=BASE_DIR,
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"ERROR at img {img_id}: {result.stderr[-500:]}")
        else:
            # Show last line
            lines = result.stdout.strip().split('\n')
            for l in lines[-3:]:
                print(f"  {l}")
    
    # Move results to discussion dir
    src = os.path.join(BASE_DIR, "result_self", "WV3_full")
    dst = os.path.join(out_dir, "WV3_full")
    if os.path.exists(src):
        for f in os.listdir(src):
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
    
    # Move models
    src_m = os.path.join(BASE_DIR, "model_FUG")
    dst_m = os.path.join(out_dir, "model_FUG")
    if os.path.exists(src_m):
        os.makedirs(dst_m, exist_ok=True)
        for f in os.listdir(src_m):
            if os.path.isfile(os.path.join(src_m, f)):
                shutil.move(os.path.join(src_m, f), os.path.join(dst_m, f))
    
    # Restore original mymodel.py
    with open(model_path, 'w') as f:
        f.write(orig)
    
    print(f"FISTA T={T} DONE. Results in {out_dir}")

print("\nAll FISTA experiments complete!")
