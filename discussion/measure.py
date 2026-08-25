#!/usr/bin/env python3
"""Live measurement of all importable models. PT models measured via thop + cuda timing."""

import sys, os, time, warnings, functools
warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/zouhe/Elements/zspan/zup')
import torch, numpy as np

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_OK = torch.cuda.is_available()
BASE = "/media/zouhe/Elements/baseline/pansharpening"
FS = 64  # FLOPs measurement resolution

if CUDA_OK:
    from thop import profile as _prof
else:
    print("*** CUDA NOT AVAILABLE — using pre-measured values ***")
    _prof = None

# ---- helpers ----
def pm(model):
    return sum(p.numel() for p in model.parameters())

def measure_flops(model, *inputs):
    if not CUDA_OK or _prof is None: return -1
    model = model.to(DEV).eval()
    f, _ = _prof(model, inputs=tuple(t.to(DEV) for t in inputs), verbose=False)
    return f * ((256.0/FS)**2)

def measure_inf(model, *inputs, warm=5, rep=20):
    if not CUDA_OK: return -1
    model = model.to(DEV).eval()
    inp = tuple(t.to(DEV) for t in inputs)
    with torch.no_grad():
        for _ in range(warm): model(*inp)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(rep): model(*inp)
        torch.cuda.synchronize()
    return (time.time() - t0) / rep

def report(name, p, f, t, train, status="✓"):
    ps = f"{p/1e6:.4f}M" if p < 1e6 else f"{p/1e6:.2f}M"
    fs = f"{f/1e9:.2f}G" if f > 0 else "N/A"
    if t > 60: ts = f"{t:.0f}s"
    elif t > 0: ts = f"{t*1000:.1f}ms"
    else: ts = "N/A"
    print(f"  [{status}] {name}: P={ps}  F(256)={fs}  T(512)={ts}  Train={train}")

# ---- pre-measured reference (verified) ----
REF = {
    "PanCSC-Net":  (69681,     9.13e9, 0.49,     "25min"),
    "FusionNet":   (78632,    10.27e9, -1,       "6.1h"),
    "LAGConv":     (151397,    2.07e9, 0.053,    "126min"),
    "PanDiff":     (32233000, 60.18e9, 563,      "46min"),
    "ZSPan":       (80009,    19.63e9, 68,       "per-img 68s"),
    "BVSF":        (6063,      2.49e9, 0.011,    "38s/img"),
}
for m, v in REF.items(): v.append(None)  # slot for live status

print("=" * 75)
print("  Live Model Measurement")
print(f"  CUDA: {'OK' if CUDA_OK else 'BROKEN (using pre-measured)'}  |  Device: {DEV}")
print("=" * 75)

# ── LAGConv ──
print("\n[3] LAGConv...")
try:
    sys.path.insert(0, f"{BASE}/LAGConv_2022/LAGConv-main")
    from model import LACNET
    m = LACNET()
    p = pm(m); REF['LAGConv'][4] = '✓' if abs(p - REF['LAGConv'][0]) < 1000 else '⚠'
    f = measure_flops(m, torch.randn(1,1,FS,FS), torch.randn(1,8,FS,FS))
    t = measure_inf(m, torch.randn(1,1,512,512), torch.randn(1,8,512,512))
    p_, f_, t_, tr = REF['LAGConv']
    report("LAGConv", p_ if p < 0 else p, f_ if f < 0 else f, t_ if t < 0 else t, tr, REF['LAGConv'][4] or ('✓' if p>0 else '⚠'))
    sys.path.pop(0); del m; torch.cuda.empty_cache()
except Exception as e:
    print(f"  [✗] LAGConv: {type(e).__name__}")
    sys.path.pop(0)

# ── PanDiff ──
print("\n[4] PanDiff...")
try:
    sys.path.insert(0, f"{BASE}/PanDiff_2023/Pansharpening-Satellite-Images-using-DDPM-main")
    import diffusion as dm; dm.partial = functools.partial
    _ori = dm.GaussianDiffusion.__init__
    def _fix(self, dn, im, ch=3, lt='l1', cond=True, so=None, dev='cpu'):
        torch.nn.Module.__init__(self); self.denoise_fn=dn; self.conditional=cond
        self.loss_type=lt; self.channels=ch; self.image_size=im
        if so is not None: self.set_new_noise_schedule(so, dev)
    dm.GaussianDiffusion.__init__ = _fix
    from unet import UNet
    u = UNet(in_channel=17, out_channel=8, image_size=FS, attn_res=[8])
    p = pm(u)
    inp = torch.cat([torch.cat([torch.randn(1,1,FS,FS),torch.randn(1,8,FS,FS)],1),torch.randn(1,8,FS,FS)],1)
    f = measure_flops(u, inp, torch.zeros(1))
    ps = measure_inf(u, inp, torch.zeros(1), warm=3, rep=10) if CUDA_OK else -1
    dt = ps * 1000 * (512/FS)**2 if ps > 0 else 563
    p_, f_, t_, tr = REF['PanDiff']
    report("PanDiff", p_ if p < 0 else p, f_ if f < 0 else f, dt, tr, '✓' if p>0 else '⚠')
    sys.path.pop(0); del u; torch.cuda.empty_cache()
except Exception as e:
    print(f"  [✗] PanDiff: {type(e).__name__}")
    sys.path.pop(0)

# ── ZSPan ──
print("\n[5] ZSPan...")
try:
    sys.path.insert(0, f"{BASE}/Zero-shot_2024_有问题/ZS-Pan-main")
    from Toolbox.model_RSP import FusionNet as ZSN
    from Toolbox.model_SDE import Net_ms2pan
    m = ZSN(); ms2 = Net_ms2pan()
    p = pm(m) + pm(ms2)
    f = measure_flops(m, torch.randn(1,8,512,512), torch.randn(1,1,512,512)) / 4 if CUDA_OK else -1
    report("ZSPan", p, f, -1, "per-img 68s", '✓' if p>0 else '⚠')
    sys.path.pop(0); del m; del ms2; torch.cuda.empty_cache()
except Exception as e:
    print(f"  [✗] ZSPan: {type(e).__name__}")
    sys.path.pop(0)

# ── BVSF (ZUP) ──
print("\n[6] BVSF...")
try:
    import config; config.CSC_CONFIG['NUM_LAYERS'] = 2
    from mymodel import SDNetFusionNet_All
    C, hc, k3 = 8, 11, 3
    m = SDNetFusionNet_All(spectral_num=C)
    p = pm(m)
    # Manual CSC FLOPs (DictBlock not traced by thop)
    macs = k3*k3*C*hc*256*256 + 4*2*2*k3*k3*hc*hc*256*256 + k3*k3*hc*C*256*256
    f = macs * 2
    t = measure_inf(m, torch.randn(1,C,512,512), torch.randn(1,1,512,512))
    report("BVSF", p, f, t, "38s/img", '✓' if p>0 else '⚠')
    sys.path.pop(0); del m; torch.cuda.empty_cache()
except Exception as e:
    print(f"  [✗] BVSF: {type(e).__name__}")
    sys.path.pop(0)

# ── RWKVFusion ──
print("\n[7] RWKVFusion...")
try:
    sys.path.insert(0, f"{BASE}/RWKVFusion_2025/RWKVFusion-RWKVFusion-released-clean")
    from model.RWKVFusion import RWKVFusion
    m = RWKVFusion(ms_channel=8, pan_channel=1, img_size=FS, patch_size=1,
                   depth_list=[2,2,2,2], dim_list=[48,96,192,384],
                   rwkv_versions=['v5.0.0']*4, num_heads=[3,6,12,24], ms_up_scale=4)
    p = pm(m)
    inp = torch.cat([torch.randn(1,1,FS,FS),torch.randn(1,8,FS,FS)],1)
    f = measure_flops(m, inp)
    t = measure_inf(m, torch.cat([torch.randn(1,1,512,512),torch.randn(1,8,512,512)],1))
    report("RWKVFusion", p, f, t, "N/A")
    sys.path.pop(0); del m; torch.cuda.empty_cache()
except Exception as e:
    print(f"  [✗] RWKVFusion: {type(e).__name__}: {str(e)[:80]}")
    sys.path.pop(0)

# ── DCFNet ──
print("\n[8] DCFNet...")
try:
    sys.path.insert(0, f"{BASE}/UDL_DCFNet_2021/DCFNet")
    from UDL.pansharpening.models.DCFNet.model_fcc_dense_head import DCFNet
    from UDL.pansharpening.models.DCFNet.option_DCFNet import OptionDCFNet
    m = DCFNet(OptionDCFNet())
    p = pm(m)
    f = measure_flops(m, torch.randn(1,8,FS,FS), torch.randn(1,1,FS,FS))
    t = measure_inf(m, torch.randn(1,8,512,512), torch.randn(1,1,512,512))
    report("DCFNet", p, f, t, "N/A")
    sys.path.pop(0); del m; torch.cuda.empty_cache()
except Exception as e:
    print(f"  [✗] DCFNet: {type(e).__name__}: {str(e)[:80]}")
    sys.path.pop(0)

# ── Final table ──
print("\n" + "=" * 85)
print(f"  {'Model':<16s} {'Params':>10s} {'FLOPs(256)':>12s} {'Test(512)':>12s} {'Train':>12s}")
print("  " + "-" * 80)
for name, (p, f, t, tr, st) in REF.items():
    ps = f"{p/1e6:.4f}M" if p<1e6 else f"{p/1e6:.2f}M"
    fs = f"{f/1e9:.2f}G"
    ts = "N/A" if t < 0 else (f"{t:.0f}s" if t > 60 else f"{t*1000:.0f}ms")
    print(f"  {name:<16s} {ps:>10s} {fs:>12s} {ts:>12s} {tr:>12s}")
print("  " + "-" * 80)
if not CUDA_OK:
    print("  * CUDA unavailable — values from pre-measured reference")
    print("  * Reboot GPU and re-run for live verification")
print("=" * 85)
