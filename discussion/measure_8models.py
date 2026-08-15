#!/usr/bin/env python3
"""
8-Model: Params, FLOPs(256), Timing(512) — WV3 8-band, RTX 4090.
Follows baseline-size/measure.py register pattern.
Falls back to pre-measured reference when CUDA/deps broken.
"""
import sys, os, time, warnings, functools, argparse
from collections import OrderedDict
import numpy as np
warnings.filterwarnings('ignore')

BASE = "/media/zouhe/Elements/baseline/pansharpening"
ZUP  = "/media/zouhe/Elements/zspan/zup"
sys.path.insert(0, ZUP)
import torch, torch.nn as nn

HAS_THOP = False
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_OK = DEVICE == "cuda"
FS = 64  # FLOPs measurement base resolution

# ================================================================
# Pre-measured reference (all verified 2026-07-23~28, RTX 4090)
# ================================================================
# Each: (params, flops_256, inf_512_s, train_time)
REF = {
    "PanCSC-Net": (69681,    9.13e9,  0.490,  "25min"),
    "FusionNet":  (78632,   10.27e9,  -1,     "6.1h"),
    "LAGConv":    (151397,   2.07e9,  0.053,  "126min"),
    "PanDiff":    (32233000, 60.18e9, 563,    "46min"),
    "ZSPan":      (80009,   19.63e9,  68,     "~68s/img"),
    "BVSF":       (6063,     2.49e9,  0.011,  "38s/img"),
    "RWKVFusion": (830000,   2.34e9,  -1,     "?"),
    "DCFNet":     (2770000,  3.46e9,  -1,     "?"),
}

# ================================================================
# Helpers
# ================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def measure_flops(model, inputs):
    if not CUDA_OK or not HAS_THOP: return float("nan")
    model = model.to(DEVICE).eval()
    if isinstance(inputs, tuple):
        inp = tuple(t.to(DEVICE) for t in inputs)
    else:
        inp = (inputs.to(DEVICE),)
    try:
        f, _ = profile(model, inputs=inp, verbose=False)
        return f * ((256.0 / FS) ** 2)
    except Exception:
        return float("nan")

def measure_inf(model, inputs, warm=5, rep=20):
    if not CUDA_OK: return -1
    model = model.to(DEVICE).eval()
    if isinstance(inputs, tuple):
        inp = tuple(t.to(DEVICE) for t in inputs)
    else:
        inp = (inputs.to(DEVICE),)
    with torch.no_grad():
        for _ in range(warm): model(*inp)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(rep): model(*inp)
        torch.cuda.synchronize()
    return (time.time() - t0) / rep

def fmt_p(n): return f"{n/1e6:.4f}M" if n < 1e6 else f"{n/1e6:.2f}M"
def fmt_f(n): 
    if isinstance(n, str): return n
    if np.isnan(n) or n <= 0: return "N/A"
    return f"{n/1e9:.2f}G"
def fmt_t(t):
    if isinstance(t, str): return t
    if t < 0: return "N/A"
    if t > 60: return f"{t:.0f}s"
    return f"{t*1000:.0f}ms"

# ================================================================
# Dummy inputs
# ================================================================
def d64():  return (torch.randn(1,1,FS,FS), torch.randn(1,8,FS,FS))
def d512(): return (torch.randn(1,1,512,512), torch.randn(1,8,512,512))
def s64():  return torch.cat(list(d64()), dim=1)

# ================================================================
# Model registry
# ================================================================
MODELS = OrderedDict()
def register(name, note=""):
    def d(fn): MODELS[name] = (fn, note); return fn
    return d

@register("PanCSC-Net", "TF 2.14, n=8 nl=1")
def build_pancsc():
    raise ImportError("TF — params from ckpt")

@register("FusionNet", "TF 1.x, 9-layer ConvNet")
def build_fusionnet():
    raise ImportError("TF — params from ckpt")

@register("LAGConv", "PyTorch, LAConv2D")
def build_lagconv():
    sys.path.insert(0, os.path.join(BASE, "LAGConv_2022/LAGConv-main"))
    from model import LACNET; m = LACNET(); p, l = d64()
    sys.path.pop(0); return m, (p, l), (d512()[0], d512()[1])

@register("PanDiff", "PyTorch DDPM")
def build_pandiff():
    sys.path.insert(0, os.path.join(BASE, "PanDiff_2023/Pansharpening-Satellite-Images-using-DDPM-main"))
    import diffusion as dm; dm.partial = functools.partial
    _ori = dm.GaussianDiffusion.__init__
    def _f(self, dn, im, ch=3, lt='l1', cond=True, so=None, dev='cpu'):
        nn.Module.__init__(self); self.denoise_fn = dn; self.conditional = cond
        self.loss_type = lt; self.channels = ch; self.image_size = im
        if so is not None: self.set_new_noise_schedule(so, dev)
    dm.GaussianDiffusion.__init__ = _f
    from unet import UNet
    m = UNet(in_channel=17, out_channel=8, image_size=FS, attn_res=[8])
    inp = torch.cat([s64(), torch.randn(1,8,FS,FS)], dim=1)
    sys.path.pop(0); return m, (inp, torch.zeros(1)), None  # 512: DDPM estimate

@register("ZSPan", "PyTorch, 3-stage zero-shot")
def build_zspan():
    sys.path.insert(0, os.path.join(BASE, "Zero-shot_2024_有问题/ZS-Pan-main"))
    from Toolbox.model_RSP import FusionNet as ZSN
    m = ZSN(); lms = torch.randn(1,8,FS,FS); pan = torch.randn(1,1,FS,FS)
    sys.path.pop(0); return m, (lms, pan), None

@register("BVSF", "Ours, DictBlock FISTA CSC")
def build_bvsf():
    import config; config.CSC_CONFIG['NUM_LAYERS'] = 2
    from mymodel import SDNetFusionNet_All
    m = SDNetFusionNet_All(spectral_num=8)
    _, L = d64(); _, p512 = d512(); L512, _ = d512()
    return m, (L, _), (L512, p512)  # special: model(lms, pan)

@register("RWKVFusion", "PyTorch+Triton, import broken")
def build_rwkvfusion():
    raise ImportError("chain: hydra→loguru→accelerate→triton broken")

@register("DCFNet", "PyTorch UDL, import broken")
def build_dcfnet():
    raise ImportError("UDL framework: >10 missing deps")

# ================================================================
# BVSF manual FLOPs
# ================================================================
def bvsf_flops(res=256):
    C, hc, k, nl = 8, 11, 3, 2
    return ((k*k*C*hc + 4*nl*2*k*k*hc*hc + k*k*hc*C) * res*res) * 2

# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    if args.list:
        for k, (_, note) in MODELS.items():
            print(f"  {k:<16s} {note}")
        return

    names = [args.model] if args.model else list(MODELS.keys())
    print(); print("=" * 95)
    print(f"  {'Model':<15s} {'Params':>9s} {'FLOPs(256)':>11s} {'Test(512)':>11s} {'Train':>11s}  Source/Note")
    print(f"  {'-'*15} {'-'*9} {'-'*11} {'-'*11} {'-'*11}  {'-'*25}")

    for name in names:
        build_fn, note = MODELS[name]
        ref = REF.get(name, (0, 0, -1, "?"))
        rp, rf, rt, rtr = ref

        # Try live measurement
        p_live = float("nan"); f_live = float("nan"); t_live = float("nan")
        live_src = ""

        try:
            model, inp64, inp512 = build_fn()
            p_live = count_params(model)
            live_src = "model"

            if name == "BVSF":
                f_live = bvsf_flops(256)
                live_src += " [manual FISTA]"
            elif name == "PanDiff":
                f_live = measure_flops(model, inp64)
                if CUDA_OK: t_live = measure_inf(model, inp64, warm=3, rep=10) * 1000 * (512/FS)**2
            elif name in ("PanCSC-Net", "FusionNet", "RWKVFusion", "DCFNet"):
                pass
            else:
                f_live = measure_flops(model, inp64)
                if inp512 and CUDA_OK: t_live = measure_inf(model, inp512)

            del model; torch.cuda.empty_cache()
        except Exception as e:
            live_src = f"{type(e).__name__}"

        # Choose best value
        p = p_live if not np.isnan(p_live) and p_live > 0 else rp
        f = f_live if not np.isnan(f_live) and f_live > 0 else rf
        t = t_live if not np.isnan(t_live) and t_live > 0 else rt
        tr = rtr

        src = live_src
        if src == "model" and np.isnan(f_live): src = "model [FLOPs ref]"
        if np.isnan(p_live) or p_live <= 0: src = "ckpt/ref"

        print(f"  {name:<15s} {fmt_p(p):>9s} {fmt_f(f):>11s} {fmt_t(t):>11s} {tr:>11s}  {src}")

    print(f"  {'-'*15} {'-'*9} {'-'*11} {'-'*11} {'-'*11}  {'-'*25}")
    print(f"  CUDA: {'OK' if CUDA_OK else 'BROKEN (using reference)'}")
    print("=" * 95)
    print()

if __name__ == "__main__":
    main()
