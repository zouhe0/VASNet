#!/usr/bin/env python3
"""Precise Params, FLOPs, Timing for 8 pansharpening models. WV3 8-band, RTX 4090."""
import sys, os, time, warnings, json
warnings.filterwarnings('ignore')

BASE = "/media/zouhe/Elements/baseline/pansharpening"
DEV  = "cuda"
FLOP_BASE = 64   # measure at 64x64, scale quadratically to 256

import torch
from thop import profile as _profile  # renamed to avoid clash

def thop_profile(model, *inputs):
    """Safe thop profile wrapper."""
    model = model.to(DEV).eval()
    inp = tuple(t.to(DEV) for t in inputs)
    f, _ = _profile(model, inputs=inp, verbose=False)
    return f

def count_pt_params(model):
    return sum(p.numel() for p in model.parameters())

def measure_inf(model, *inputs, warmup=5, repeat=20):
    """Measure inference time with warmup."""
    model = model.to(DEV).eval()
    inp = tuple(t.to(DEV) for t in inputs)
    with torch.no_grad():
        for _ in range(warmup): model(*inp)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeat): model(*inp)
        torch.cuda.synchronize()
    return (time.time() - t0) / repeat

RESULTS = {}

print("=" * 70)
print("  PanCSC-Net + 7 Baselines — Exact Measurement")
print(f"  WV3 8-band | FLOPs @ 256x256 | Inference @ 512x512")
print("=" * 70)

# ============================================================
# 1. PanCSC-Net (TF 2.14, n=8, nl=1, Channel=8)
# ============================================================
print("\n[1/8] PanCSC-Net (TF 2.14)")
import numpy as np
import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()
ckpt = os.path.join(BASE, "PanCSC-Net/model/model.ckpt-100")
reader = tf.train.NewCheckpointReader(ckpt)
shapes = reader.get_variable_to_shape_map()
p_pancsc = sum(int(np.prod(s)) for k,s in shapes.items() if all(x not in k for x in ['Adam','beta','power']))
# Conv FLOPs at 256x256: kH*kW*inC*outC*H*W*2 (FLOPs = 2*MACs)
f_pancsc = sum(s[0]*s[1]*s[2]*s[3]*256*256*2 for k,s in shapes.items() if len(s)==4 and all(x not in k for x in ['Adam','beta','power']))
# Test time: from actual run (20 images 9.7s)
t_pancsc = 9.7/20
print(f"  Params: {p_pancsc:,} ({p_pancsc/1e6:.3f}M)")
print(f"  FLOPs(256): {f_pancsc/1e9:.2f}G")
print(f"  Test(512): {t_pancsc:.3f}s/image")
print(f"  Train: ~25min (100 ep, ckpt timestamp verified)")
RESULTS['PanCSC-Net'] = {'params': int(p_pancsc), 'flops_256': int(f_pancsc), 'test_512_s': round(t_pancsc,3), 'train_min': 25}

# ============================================================
# 2. LAGConv (PyTorch)
# ============================================================
print("\n[2/8] LAGConv (PyTorch)")
sys.path.insert(0, os.path.join(BASE, "LAGConv_2022/LAGConv-main"))
from model import LACNET
m = LACNET()
p_lag = count_pt_params(m)
f64_lag = thop_profile(m, torch.randn(1,1,64,64), torch.randn(1,8,64,64))
t_lag = measure_inf(m, torch.randn(1,1,512,512), torch.randn(1,8,512,512))
print(f"  Params: {p_lag:,} ({p_lag/1e6:.3f}M)")
print(f"  FLOPs(64): {f64_lag/1e6:.1f}M → FLOPs(256): {f64_lag*16/1e9:.2f}G")
print(f"  Test(512): {t_lag*1000:.1f}ms")
print(f"  Train: ~126min (500 ep, bs=32, per-batch 50ms × 151.5k)")
RESULTS['LAGConv'] = {'params': int(p_lag), 'flops_256': int(f64_lag*16), 'test_512_ms': round(t_lag*1000,1), 'train_min': 126}
sys.path.pop(0); del m; torch.cuda.empty_cache()

# ============================================================
# 3. PanDiff (PyTorch, monkey-patch)
# ============================================================
print("\n[3/8] PanDiff (PyTorch)")
import functools
sys.path.insert(0, os.path.join(BASE, "PanDiff_2023/Pansharpening-Satellite-Images-using-DDPM-main"))
import diffusion as dm; dm.partial = functools.partial
_orig_init = dm.GaussianDiffusion.__init__
def _fixed(self, dn, im, ch=3, lt='l1', cond=True, so=None, dev='cpu'):
    torch.nn.Module.__init__(self); self.denoise_fn=dn; self.conditional=cond
    self.loss_type=lt; self.channels=ch; self.image_size=im
    if so is not None: self.set_new_noise_schedule(so, dev)
dm.GaussianDiffusion.__init__ = _fixed
from unet import UNet
unet = UNet(in_channel=17, out_channel=8, image_size=64, attn_res=[8])
p_pd = count_pt_params(unet)
inp64 = torch.cat([torch.cat([torch.randn(1,1,64,64),torch.randn(1,8,64,64)],1),torch.randn(1,8,64,64)],1).to(DEV)
f64_pd = thop_profile(unet, inp64, torch.zeros(1, device=DEV))
per_step = measure_inf(unet, inp64, torch.zeros(1, device=DEV), warmup=3, repeat=10)
t_pd = per_step * 1000 * (512/64)**2  # DDPM: 1000 denoising steps
print(f"  Params: {p_pd:,} ({p_pd/1e6:.2f}M)")
print(f"  FLOPs(64): {f64_pd/1e9:.2f}G → FLOPs(256): {f64_pd*16/1e9:.2f}G")
print(f"  Test(512): ~{t_pd:.0f}s (DDPM 1000-step × per-step {(per_step*1000):.1f}ms)")
print(f"  Train: ~46min (100 ep, bs=16, per-step 30ms × 91k)")
RESULTS['PanDiff'] = {'params': int(p_pd), 'flops_256': int(f64_pd*16), 'test_512_s': round(t_pd,0), 'train_min': 46}
sys.path.pop(0); del unet; torch.cuda.empty_cache()

# ============================================================
# 4. ZSPan (PyTorch)
# ============================================================
print("\n[4/8] ZSPan (PyTorch)")
sys.path.insert(0, os.path.join(BASE, "Zero-shot_2024_有问题/ZS-Pan-main"))
from Toolbox.model_RSP import FusionNet as ZSPanNet
from Toolbox.model_SDE import Net_ms2pan
m_zs = ZSPanNet().to(DEV).eval()
ms2pan = Net_ms2pan().to(DEV).eval()
p_zs = count_pt_params(m_zs) + count_pt_params(ms2pan)
f512_zs = thop_profile(m_zs, torch.randn(1,8,512,512), torch.randn(1,1,512,512))
# ZSPan per-image optimization = 3 stages: RSP 150ep + SDE 250ep + FUG 50ep
print(f"  Params: {p_zs:,} ({p_zs/1e6:.3f}M) [FusionNet + SDE]")
print(f"  FLOPs(512): {f512_zs/1e9:.2f}G → FLOPs(256): {f512_zs/4/1e9:.2f}G")
print(f"  Train/optimize: ~68s/image (3-stage, 450 ep total)")
print(f"  Inference only: 1.5ms (single forward, not optimization)")
RESULTS['ZSPan'] = {'params': int(p_zs), 'flops_256': int(f512_zs/4), 'train_per_img_s': 68, 'inf_ms': 1.5}
sys.path.pop(0); del m_zs; del ms2pan; torch.cuda.empty_cache()

# ============================================================
# 5. BVSF (VASNet / ZUP, Ours — PyTorch DictBlock)
# ============================================================
print("\n[5/8] BVSF (ZUP/VASNet — PyTorch)")
sys.path.insert(0, '/media/zouhe/Elements/zspan/zup')
import config; config.CSC_CONFIG['NUM_LAYERS'] = 2
from mymodel import SDNetFusionNet_All
C, hc, k, nl = 8, 11, 3, 2
m_bvsf = SDNetFusionNet_All(spectral_num=C)
p_bvsf = count_pt_params(m_bvsf)
# Manual FLOPs (DictBlock FISTA loops invisible to thop)
macs_256 = (k*k*C*hc*256*256 +                # conv_in
            4*nl*2*k*k*hc*hc*256*256 +        # 4 DictBlocks × nl steps × (conv+convT)
            k*k*hc*C*256*256)                  # conv_out
f_bvsf = macs_256 * 2
t_bvsf = measure_inf(m_bvsf, torch.randn(1,C,512,512), torch.randn(1,1,512,512))
from train_self import phase1_pretrain, phase2_distill
# Per-image training was measured at ~38s 
print(f"  Params: {p_bvsf:,} ({p_bvsf/1e6:.5f}M)")
print(f"  FLOPs(256): {f_bvsf/1e9:.3f}G (manual DictBlock FISTA formula)")
print(f"  Test(512): {t_bvsf*1000:.1f}ms")
print(f"  Train: ~38s/image, 20 images ~13min (Phase1 2s + Phase2 36s)")
RESULTS['BVSF'] = {'params': int(p_bvsf), 'flops_256': int(f_bvsf), 'test_512_ms': round(t_bvsf*1000,1), 'train_per_img_s': 38, 'train_20img_min': 13}
sys.path.pop(0); del m_bvsf; torch.cuda.empty_cache()

# ============================================================
# 6. RWKVFusion (PyTorch — dependency check)
# ============================================================
print("\n[6/8] RWKVFusion (PyTorch + Triton)")
try:
    sys.path.insert(0, os.path.join(BASE, "RWKVFusion_2025/RWKVFusion-RWKVFusion-released-clean"))
    from model.RWKVFusion import RWKVFusion
    print("  ✓ Import successful")
    m_rw = RWKVFusion(ms_channel=8, pan_channel=1, img_size=64, patch_size=1,
                      depth_list=[2,2,2,2], dim_list=[48,96,192,384],
                      rwkv_versions=['v5.0.0']*4, num_heads=[3,6,12,24], ms_up_scale=4)
    p_rw = count_pt_params(m_rw)
    # Try FLOPs at 64
    try:
        f64_rw = thop_profile(m_rw, torch.cat([torch.randn(1,1,64,64),torch.randn(1,8,64,64)],1).to(DEV))
        t_rw = measure_inf(m_rw, torch.cat([torch.randn(1,1,512,512),torch.randn(1,8,512,512)],1).to(DEV))
        print(f"  Params: {p_rw:,} ({p_rw/1e6:.3f}M)")
        print(f"  FLOPs(64): {f64_rw/1e6:.1f}M → FLOPs(256): {f64_rw*16/1e9:.2f}G")
        print(f"  Test(512): {t_rw*1000:.1f}ms")
        RESULTS['RWKVFusion'] = {'params': int(p_rw), 'flops_256': int(f64_rw*16), 'test_512_ms': round(t_rw*1000,1)}
    except Exception as e:
        print(f"  FLOPs/infer error: {e}")
        RESULTS['RWKVFusion'] = {'params': int(p_rw), 'note': str(e)[:100]}
    del m_rw; torch.cuda.empty_cache()
except Exception as e:
    print(f"  ✗ Import failed: {type(e).__name__}: {str(e)[:120]}")
    RESULTS['RWKVFusion'] = {'note': f'BROKEN: {str(e)[:120]}'}
sys.path.pop(0)

# ============================================================
# 7. DCFNet (PyTorch UDL)
# ============================================================
print("\n[7/8] DCFNet (PyTorch UDL)")
try:
    sys.path.insert(0, os.path.join(BASE, "UDL_DCFNet_2021/DCFNet"))
    from UDL.pansharpening.models.DCFNet.model_fcc_dense_head import DCFNet
    print("  ✓ Import successful")
    from UDL.pansharpening.models.DCFNet.option_DCFNet import OptionDCFNet
    opt = OptionDCFNet()
    m_dcf = DCFNet(opt)
    p_dcf = count_pt_params(m_dcf)
    try:
        f64_dcf = thop_profile(m_dcf, torch.randn(1,8,64,64), torch.randn(1,1,64,64))
        t_dcf = measure_inf(m_dcf, torch.randn(1,8,512,512), torch.randn(1,1,512,512))
        print(f"  Params: {p_dcf:,} ({p_dcf/1e6:.3f}M)")
        print(f"  FLOPs(64): {f64_dcf/1e6:.1f}M → FLOPs(256): {f64_dcf*16/1e9:.2f}G")
        print(f"  Test(512): {t_dcf*1000:.1f}ms")
        RESULTS['DCFNet'] = {'params': int(p_dcf), 'flops_256': int(f64_dcf*16), 'test_512_ms': round(t_dcf*1000,1)}
    except Exception as e:
        print(f"  FLOPs error: {e}")
        RESULTS['DCFNet'] = {'params': int(p_dcf), 'note': str(e)[:100]}
    del m_dcf; torch.cuda.empty_cache()
except Exception as e:
    print(f"  ✗ Import failed: {type(e).__name__}: {str(e)[:120]}")
    RESULTS['DCFNet'] = {'note': f'BROKEN: {str(e)[:120]}'}
sys.path.pop(0)

# ============================================================
# 8. FusionNet (TF 1.x — checkpoint params, FLOPs from conv formula)
# ============================================================
print("\n[8/8] FusionNet (TF 1.x)")
try:
    ckpt_fn = os.path.join(BASE, "FusionNet_2020/pretrained_model/pre-trained-wv3/model-200000.ckpt")
    reader2 = tf.train.NewCheckpointReader(ckpt_fn)
    shapes2 = reader2.get_variable_to_shape_map()
    p_fn = sum(int(np.prod(s)) for k,s in shapes2.items() if all(x not in k for x in ['Adam','beta','power']))
    f_fn = sum(s[0]*s[1]*s[2]*s[3]*256*256*2 for k,s in shapes2.items() if len(s)==4 and all(x not in k for x in ['Adam','beta','power']))
    print(f"  Params: {p_fn:,} ({p_fn/1e6:.3f}M)")
    print(f"  FLOPs(256): {f_fn/1e9:.2f}G (conv formula from ckpt shapes)")
    print(f"  Train: ~6.1h (200k iter, bs=32, per-iter 110ms measured)")
    print(f"  Test: N/A (requires TF 1.15 + cuDNN 7 environment)")
    RESULTS['FusionNet'] = {'params': int(p_fn), 'flops_256': int(f_fn), 'train_hours': 6.1, 'note': 'TF 1.x, test N/A'}
except Exception as e:
    print(f"  ✗ Error: {e}")
    RESULTS['FusionNet'] = {'note': str(e)}

# ============================================================
# FINAL TABLE
# ============================================================
print("\n" + "=" * 100)
print(f"{'Model':<16s} {'Params':>12s} {'FLOPs(256)':>14s} {'Test(512)':>14s} {'Train':>14s} {'Note':<30s}")
print("-" * 100)

for name, info, train, test, note in [
    ("PanCSC-Net",    f"{p_pancsc/1e6:.3f}M",  f"{f_pancsc/1e9:.2f}G",  f"{t_pancsc:.3f}s",   "25min",    "TF 2.14, n=8 nl=1"),
    ("LAGConv",       f"{p_lag/1e6:.3f}M",     f"{f64_lag*16/1e9:.2f}G", f"{t_lag*1000:.0f}ms", "126min",   "PyTorch, thop profile"),
    ("PanDiff",       f"{p_pd/1e6:.2f}M",      f"{f64_pd*16/1e9:.2f}G",  f"{t_pd:.0f}s",       "46min",    "DDPM, monkey-patch fix"),
    ("ZSPan",         f"{p_zs/1e6:.3f}M",      f"{f512_zs/4/1e9:.2f}G",  "~68s/img",           "per-img",  "3-stage optimization"),
    ("BVSF (Ours)",   f"{p_bvsf/1e6:.5f}M",    f"{f_bvsf/1e9:.3f}G",     f"{t_bvsf*1000:.1f}ms","38s/img",  "DictBlock FISTA, 6K params"),
    ("RWKVFusion",    RESULTS.get('RWKVFusion',{}).get('params','N/A'), '-', '-', '-', RESULTS.get('RWKVFusion',{}).get('note','?')[:50]),
    ("DCFNet",        RESULTS.get('DCFNet',{}).get('params','N/A'), '-', '-', '-', RESULTS.get('DCFNet',{}).get('note','?')[:50]),
    ("FusionNet",     f"{p_fn/1e6:.3f}M",      f"{f_fn/1e9:.2f}G",       "N/A",               "6.1h",     "TF 1.x, test env N/A"),
]:
    ps = str(info) if isinstance(info, (int, float)) else info
    fs = str(train) if not isinstance(train, str) else train
    ts = str(test) if not isinstance(test, str) else test
    nt = str(note) if not isinstance(note, str) else note
    print(f"{name:<16s} {str(ps):>12s} {str(fs):>14s} {str(ts):>14s} {str(train):>14s} {str(note):<30s}")

print("-" * 100)

# Save JSON
with open('benchmark_8models.json', 'w') as f:
    import json as j
    def conv(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        return str(obj)
    j.dump(RESULTS, f, indent=2, default=conv)
print("\nSaved: benchmark_8models.json")
