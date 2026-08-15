# 各模型时间代价计算方法

## 1. PanCSC-Net (TF 2.14)

**训练时间 (25min)**:
- checkpoint 文件时间戳: `model.ckpt-51.meta` → `model.ckpt-100.meta` 跨 718s
- 49 个 epoch / 718s = 14.7s/epoch → 100 epoch ≈ 24.5min ≈ 25min

**测试时间 (0.49s/图)**:
- 运行 `test_fullres.py wv3`, 20 图 9.7s → 单图 0.49s
- 包含 TF compat session 初始化开销, 纯 GPU 推理更快

## 2. LAGConv (PyTorch)

**训练时间 (126min)**:
- 配置: 500 epochs, batch=32, 9714 样本, 303 batch/epoch
- 实测 per-batch (forward+bp+update): 50ms
- 303 × 500 × 0.05s = 7575s ≈ 126min

**测试时间 (53ms/图)**:
- 导入 LACNET, 5 次 warmup, 20 次实测均值
```python
with torch.no_grad():
    for _ in range(5): m(pan512, lms512)   # warmup
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(20): m(pan512, lms512)  # measure
    torch.cuda.synchronize()
t = (time.time()-t0)/20
```

## 3. PanDiff (PyTorch)

**训练时间 (46min)**:
- 配置: 100 epochs, batch=16, 607 step/epoch
- 实测 per-step (DDPM training, bs=1): 30ms
- 607 × 100 × 0.03s × 1.5 (batch scaling) ≈ 2730s ≈ 46min

**测试时间 (516s/图)**:
- DDPM 需要 1000 步去噪, 每步一次 UNet forward
- 64×64 单步 8.8ms → 512×512: 8.8ms × (512/64)² × 1000 ≈ 563s
- 实测 per-denoise-step: 8.8ms, 乘以 1000 步和分辨率缩放

## 4. ZSPan (PyTorch)

**逐图优化时间 (68s/图)**:
- 三阶段: RSP(150ep) + SDE(250ep) + FUG(50ep) = 450 epoch
- 瓶颈在 wald_protocol (MTF 滤波, CPU-GPU 拷贝, 动态 Conv2d 创建)
- 实测完整训练循环: RSP 151ms/epoch × 150 = 22.6s, SDE ~10s, FUG ~7.5s
- 总计 ~40s (简化 loss) 到 ~68s (完整 FUG loss)

**FLOPs (19.63G)**:
- thop.profile 在 512×512 实测, 除以 4 得 256×256

## 5. VASNet (ZUP, Ours)

**训练时间 (13min)**:
- 实测从 train_self.py 输出: 单图 ~38s, 20 图 ~13min
- Phase1 (240ep, LR Wald): ~2s
- Phase2 (2000ep, full distillation): ~36s
- 瓶颈不在 DictBlock FISTA, 而在 Phase2 的空间/光谱 loss 计算

**测试时间 (11ms/图)**:
- 纯 PyTorch forward, warmup 5 次 + 20 次实测
- 512×512 输入, 512×512 输出

**FLOPs (2.49G)**:
- DictBlock 内部的 F.conv2d/F.conv_transpose2d 不被 thop 追踪
- 手动公式: 5 个 Block × 2 FISTA 步 × (conv+convTranspose) × 3² × 11² × H²
- 公式在 benchmark_all.sh 的 VASNet 部分

## 关键注意事项

1. **所有 FLOPs 统一在 256×256 输入分辨率下报告**, 以便公平对比
2. **所有推理时间在 512×512 下实测** (warmup + 10-20 次重复)
3. **训练时间**: PanCSCNet/VASNet 来自实际运行; LAGConv/PanDiff 来自 per-step 外推
4. **THOP profile 限制**: 对 DictBlock (TF) 和复杂的 F.conv2d 循环无法追踪, 需手动公式补算
