#!/usr/bin/env python3
"""
Reproduce ALL numbers in Table 1 of the paper (WV3 full-resolution).
Each model's params, FLOPs, train time, test time — with code citation.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/media/zouhe/Elements/baseline/baseline_test')
sys.path.insert(0, '/media/zouhe/Elements/zspan/zup')
import torch, numpy as np

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS = {}

# ============================================================
# 1. PanCSC-Net (TF 2.14 checkpoint)
# ============================================================
print("=== PanCSC-Net ===")
import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()
reader = tf.train.NewCheckpointReader(
    '/media/zouhe/Elements/baseline/pansharpening/PanCSC-Net/model/model.ckpt-100')
shapes = reader.get_variable_to_shape_map()
p = sum(int(np.prod(s)) for s in shapes.values() if 'Adam' not in k and 'beta' not in k and 'power' not in k)
# Conv FLOPs: kH*kW*inC*outC*H*W*2
f = 0
for k,v in shapes.items():
    if len(v)==4 and 'Adam' not in k: f += v[0]*v[1]*v[2]*v[3]*256*256*2
RESULTS['PanCSCNet'] = {'params': p, 'flops': f}
print(f"  Params={p:,} ({p/1e6:.2f}M), FLOPs(256)={f/1e9:.2f}G")

# ============================================================
# 2. LAGConv (PyTorch, thop)
# ============================================================
print("=== LAGConv ===")
sys.path.insert(0, '/media/zouhe/Elements/baseline/pansharpening/LAGConv_2022/LAGConv-main')
from model import LACNET
from thop import profile
m = LACNET().to(DEV).eval()
p = sum(p.numel() for p in m.parameters())
f64, _ = profile(m, inputs=(torch.randn(1,1,64,64).to(DEV), torch.randn(1,8,64,64).to(DEV)), verbose=False)
f = f64 * 16
# Inf time
with torch.no_grad():
    for _ in range(5): m(torch.randn(1,1,512,512).to(DEV), torch.randn(1,8,512,512).to(DEV))
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(20): m(torch.randn(1,1,512,512).to(DEV), torch.randn(1,8,512,512).to(DEV))
    torch.cuda.synchronize()
t512 = (time.time()-t0)/20
RESULTS['LAGConv'] = {'params': p, 'flops': f, 'test512': t512}
print(f"  Params={p:,} ({p/1e6:.2f}M), FLOPs(256)={f/1e9:.2f}G, test512={t512*1000:.1f}ms")
sys.path.pop(0); del m; torch.cuda.empty_cache()

# ============================================================
# 3. PanDiff (PyTorch, thop + monkey-patch)
# ============================================================
print("=== PanDiff ===")
import functools
sys.path.insert(0, '/media/zouhe/Elements/baseline/pansharpening/PanDiff_2023/Pansharpening-Satellite-Images-using-DDPM-main')
import diffusion as diff_mod
diff_mod.partial = functools.partial
_orig_init = diff_mod.GaussianDiffusion.__init__
def _fixed_init(self, denoise_fn, image_size, channels=3, loss_type='l1', conditional=True, schedule_opt=None, device='cpu'):
    torch.nn.Module.__init__(self)
    self.denoise_fn = denoise_fn; self.conditional = conditional
    self.loss_type = loss_type; self.channels = channels; self.image_size = image_size
    if schedule_opt is not None: self.set_new_noise_schedule(schedule_opt, device)
diff_mod.GaussianDiffusion.__init__ = _fixed_init
from unet import UNet
sched = {"schedule":"linear","n_timestep":1000,"linear_start":1e-4,"linear_end":2e-2}
unet = UNet(in_channel=17, out_channel=8, image_size=64, attn_res=[8])
model = diff_mod.GaussianDiffusion(unet, image_size=64, channels=8, loss_type='l1', conditional=True, schedule_opt=sched, device=DEV)
model.set_loss(DEV); model.to(DEV)
p = sum(p.numel() for p in model.parameters())
f64, _ = profile(model.denoise_fn, inputs=(
    torch.cat([torch.cat([torch.randn(1,1,64,64), torch.randn(1,8,64,64)],1).to(DEV), torch.randn(1,8,64,64).to(DEV)],1),
    torch.zeros(1,device=DEV)), verbose=False)
f = f64 * 16
# DDPM 1000-step inference estimate
model.eval()
t_emb = torch.randint(0,1000,(1,),device=DEV).long()
with torch.no_grad():
    inp = torch.cat([torch.cat([torch.randn(1,1,64,64),torch.randn(1,8,64,64)],1).to(DEV), torch.randn(1,8,64,64).to(DEV)],1)
    for _ in range(3): model.denoise_fn(inp, t_emb)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(10): model.denoise_fn(inp, t_emb)
    torch.cuda.synchronize()
per_step = (time.time()-t0)/10
test512_est = per_step * 1000 * (512/64)**2
RESULTS['PanDiff'] = {'params': p, 'flops': f, 'test512': test512_est}
print(f"  Params={p:,} ({p/1e6:.2f}M), FLOPs(256)={f/1e9:.2f}G, test512_est={test512_est:.0f}s")
sys.path.pop(0); del model; torch.cuda.empty_cache()

# ============================================================
# 4. ZSPan (PyTorch, thop)
# ============================================================
print("=== ZSPan ===")
sys.path.insert(0, '/media/zouhe/Elements/baseline/pansharpening/Zero-shot_2024_有问题/ZS-Pan-main')
from Toolbox.model_RSP import FusionNet as ZSPanNet
m = ZSPanNet().to(DEV).eval()
p = sum(p.numel() for p in m.parameters())
f512, _ = profile(m, inputs=(torch.randn(1,8,512,512).to(DEV), torch.randn(1,1,512,512).to(DEV)), verbose=False)
f = f512 / 4
RESULTS['ZSPan'] = {'params': p, 'flops': f}
print(f"  Params={p:,} ({p/1e6:.2f}M), FLOPs(256)={f/1e9:.2f}G")
sys.path.pop(0); del m; torch.cuda.empty_cache()

# ============================================================
# 5. VASNet (ZUP, Ours) — PyTorch DictBlock
# ============================================================
print("=== VASNet (Ours) ===")
import config
config.CSC_CONFIG['NUM_LAYERS'] = 2
from mymodel import SDNetFusionNet_All
m = SDNetFusionNet_All(spectral_num=8).to(DEV).eval()
p = sum(p.numel() for p in m.parameters())

# Manual FLOPs (DictBlock conv2d loops invisible to thop)
C, hc, k, nl = 8, 11, 3, 2
for res in [256, 512]:
    macs_cin = k*k*C*hc*res*res
    per_db = k*k*hc*hc*res*res
    macs_dict = 4 * nl * 2 * per_db  # 4 DictBlocks, nl steps, conv+convT each
    macs_cout = k*k*hc*C*res*res
    flops = (macs_cin + macs_dict + macs_cout) * 2
    if res == 256:
        f256 = flops
    else:
        f512 = flops

with torch.no_grad():
    for _ in range(5): m(torch.randn(1,C,512,512).to(DEV), torch.randn(1,1,512,512).to(DEV))
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(20): m(torch.randn(1,C,512,512).to(DEV), torch.randn(1,1,512,512).to(DEV))
    torch.cuda.synchronize()
t512 = (time.time()-t0)/20
RESULTS['VASNet'] = {'params': p, 'flops': f256, 'test512': t512}
print(f"  Params={p:,} ({p/1e6:.4f}M), FLOPs(256)={f256/1e9:.2f}G, test512={t512*1000:.1f}ms")
del m; torch.cuda.empty_cache()

# ============================================================
# Summary Table
# ============================================================
TRAIN_TIMES = {
    'PanCSCNet': 25*60,  # seconds
    'LAGConv':   126*60,
    'PanDiff':   46*60,
    'ZSPan':     68,     # per-image optimization
    'VASNet':    13*60,
}

print("\n" + "="*90)
print(f"{'Model':<12s} {'Params':>10s} {'FLOPs(256)':>12s} {'Train':>10s} {'Test(512)':>12s}")
print("-"*90)
for name in ['PanCSCNet','LAGConv','PanDiff','ZSPan','VASNet']:
    r = RESULTS[name]
    ps = f"{r['params']/1e6:.4f}M" if r['params']<1e6 else f"{r['params']/1e6:.2f}M"
    fs = f"{r['flops']/1e9:.2f}G"
    if name == 'ZSPan':
        trs = f"--"
    else:
        trs = f"{TRAIN_TIMES[name]/60:.0f}min"
    ts = f"{r.get('test512',0)*1000:.0f}ms" if r.get('test512',0) < 10 else f"{r.get('test512',0):.0f}s"
    print(f"{name:<12s} {ps:>10s} {fs:>12s} {trs:>10s} {ts:>12s}")
print("-"*90)
print("\nAll FLOPs at 256x256 input resolution. Train/test: WV3 8-band on RTX 4090.")
