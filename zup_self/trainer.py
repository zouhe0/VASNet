"""Phase1 (Wald pretrain) and Phase2 (self-distillation) training logic."""
import os, sys, time, h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import scipy.io as sio
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import optim

from data import Dataset
from mymodel import SDNetFusionNet_All, SDNetFusionNet_Conv
from loss import LossCalculator
import config as cfg


def _norm_pan(pan):
    """Ensure pan is (1, 1, H, W) 4D."""
    if pan.dim() == 2:       return pan.unsqueeze(0).unsqueeze(0)
    elif pan.dim() == 3:     return pan.unsqueeze(0) if pan.shape[0] == 1 else pan.unsqueeze(1)
    elif pan.dim() == 4:     return pan
    raise ValueError(f"Unexpected pan dim: {pan.dim()}")


def _to_hwc(x, scale=2047.0):
    """(1, C, H, W) tensor -> (H, W, C) numpy, denormalized."""
    return x.squeeze(0).permute(1, 2, 0).cpu().numpy() * scale


def phase1_pretrain(data_id, data_path, save_dir, args, device):
    """Wald LR self-training. Returns (teacher_tensor, lms, pan_raw, ms, gt_hw)."""
    modelfix =     (SDNetFusionNet_Conv if args.ablation == "no_csc" else SDNetFusionNet_All)
    spectral_num = args.spectral_num
    best_path =    os.path.join("model_FUG", f"{args.model_prefix}_phase1_best.pth")

    print(f"  [P1] LR pretrain ({args.epochs_phase1} ep) ...", end="", flush=True)
    t0 = time.time()

    with h5py.File(data_path, "r") as f:
        ms  = torch.from_numpy(np.array(f["ms"][data_id],  dtype=np.float32) / cfg.TRAIN_DEFAULTS["max_value"]).to(device).clamp(0, 1)
        lms = torch.from_numpy(np.array(f["lms"][data_id], dtype=np.float32) / cfg.TRAIN_DEFAULTS["max_value"]).to(device).clamp(0, 1)
        pan_raw = torch.from_numpy(np.array(f["pan"][data_id], dtype=np.float32) / cfg.TRAIN_DEFAULTS["max_value"]).to(device).clamp(0, 1)
        gt_np = np.array(f["gt"][data_id], dtype=np.float32) if "gt" in f else None

    while pan_raw.dim() > 2: pan_raw = pan_raw.squeeze(0)
    if pan_raw.dim() == 1:
        s = int(np.sqrt(pan_raw.shape[0])); pan_raw = pan_raw.reshape(s, s)

    pan_4d = _norm_pan(pan_raw)
    lms_4d, ms_4d = lms.unsqueeze(0), ms.unsqueeze(0)
    ratio = cfg.TRAIN_DEFAULTS["ratio"]

    pan_lr = F.interpolate(pan_4d, size=ms.shape[1:], mode="bilinear", align_corners=False)
    ms_low = F.interpolate(ms_4d, scale_factor=1.0 / ratio, mode="bilinear", align_corners=False)
    lms_lr = F.interpolate(ms_low, size=ms.shape[1:], mode="bilinear", align_corners=False)

    model = modelfix(spectral_num=spectral_num).to(device)
    if args.ablation == "fix_lmbd":
        cfg.freeze_lmbd(model)
    if args.fista_T is not None:
        cfg.set_fista_steps(model, args.fista_T)

    opt = optim.Adam(model.parameters(), lr=args.lr_phase1, betas=(0.9, 0.999))
    sched = CosineAnnealingLR(opt, T_max=args.epochs_phase1, eta_min=args.lr_phase1 * 0.01)
    best_loss = float("inf")

    for ep in range(args.epochs_phase1):
        model.train(); opt.zero_grad()
        out = model(lms_lr, pan_lr) + lms_lr
        loss = torch.mean((out - ms_4d) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sched.step()
        if args.ablation not in ("no_csc", "fix_lmbd"):
            cfg.clamp_lmbd(model)
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        p1_lr = (model(lms_lr, pan_lr) + lms_lr).clamp(0, 1)

    d = {"I_MS_LR": ms.permute(1,2,0).cpu().numpy() * cfg.TRAIN_DEFAULTS["max_value"],
         "I_MS": lms.permute(1,2,0).cpu().numpy() * cfg.TRAIN_DEFAULTS["max_value"],
         "I_PAN": pan_raw.cpu().numpy() * cfg.TRAIN_DEFAULTS["max_value"],
         "proposed": _to_hwc(p1_lr)}
    if gt_np is not None: d["gt"] = gt_np.transpose(1, 2, 0)
    sio.savemat(os.path.join(save_dir, f"{data_id}_self_phase1_lr.mat"), d)

    del model

    model_full = modelfix(spectral_num=spectral_num).to(device)
    if args.ablation == "fix_lmbd": cfg.freeze_lmbd(model_full)
    if args.fista_T is not None: cfg.set_fista_steps(model_full, args.fista_T)
    model_full.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model_full.eval()
    with torch.no_grad():
        teacher = (model_full(lms_4d, pan_4d) + lms_4d).clamp(0, 1)

    sio.savemat(os.path.join(save_dir, f"{data_id}_self_teacher.mat"),
                {"sr": _to_hwc(teacher), **(dict(gt=gt_np.transpose(1,2,0)) if gt_np is not None else {})})

    sio.savemat(os.path.join(save_dir, f"{data_id}_self_phase1_full.mat"),
                {"I_MS_LR": ms.permute(1,2,0).cpu().numpy() * cfg.TRAIN_DEFAULTS["max_value"],
                 "I_MS": lms.permute(1,2,0).cpu().numpy() * cfg.TRAIN_DEFAULTS["max_value"],
                 "I_PAN": pan_raw.cpu().numpy() * cfg.TRAIN_DEFAULTS["max_value"],
                 "proposed": _to_hwc(teacher),
                 **(dict(gt=gt_np.transpose(1,2,0)) if gt_np is not None else {})})

    del model_full
    print(f" done ({time.time()-t0:.0f}s)", flush=True)
    return (teacher.squeeze(0).detach(), lms, pan_raw, ms,
            gt_np.transpose(1,2,0) if gt_np is not None else None)


def phase2_distill(data_id, teacher_tensor, lms, pan_raw, ms, data_path, save_dir, args, device, gt_hw=None):
    """Self-distillation with spatial/spectral fidelity."""
    modelfix =     (SDNetFusionNet_Conv if args.ablation == "no_csc" else SDNetFusionNet_All)
    spectral_num = args.spectral_num
    best_path =    os.path.join("model_FUG", f"{args.model_prefix}_best.pth")
    ratio =        cfg.TRAIN_DEFAULTS["ratio"]

    print(f"  [P2] Distillation ({args.epochs_phase2} ep) ...", end="", flush=True)
    t0 = time.time()
    train_set = Dataset(data_path, data_id)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    model = modelfix(spectral_num=spectral_num).to(device)
    if args.ablation == "fix_lmbd": cfg.freeze_lmbd(model)
    if args.fista_T is not None: cfg.set_fista_steps(model, args.fista_T)

    loss_calc = LossCalculator(sensor=args.sensor, ratio=ratio, N=41, device=device)
    opt = optim.Adam(model.parameters(), lr=args.lr_phase2, betas=(0.9, 0.999))
    sched = CosineAnnealingLR(opt, T_max=args.epochs_phase2, eta_min=args.lr_phase2 * 0.01)
    best_total = float("inf")
    wv = cfg.TRAIN_DEFAULTS["w_var"]

    for ep in range(args.epochs_phase2):
        model.train()
        for batch in loader:
            ms_b, lms_b, pan_b = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            if pan_b.dim() == 3: pan_b = pan_b.unsqueeze(1)
            opt.zero_grad()

            out = (model(lms_b, pan_b) + lms_b).squeeze(0)
            loss_var = torch.mean((out - teacher_tensor) ** 2)

            out_hwc = out.permute(1, 2, 0)
            _, H, _ = ms_b[0].shape
            loss_spa = loss_calc.compute_spatial_fidelity_loss(out_hwc, ms_b[0].permute(1,2,0), pan_b[0].squeeze(0), H // ratio, use_ergas=True)
            loss_spec = loss_calc.compute_spectral_loss(out_hwc, ms_b[0].permute(1,2,0))

            lmbd_reg = 0.0 if args.ablation in ("no_csc", "fix_lmbd") else cfg.compute_lmbd_regularization(model)

            if args.ablation == "no_spatial":
                total = wv * loss_var + args.w_spec * loss_spec - args.lmbd_weight * lmbd_reg
            elif args.ablation == "no_spectral":
                total = wv * loss_var + args.w_spa * loss_spa - args.lmbd_weight * lmbd_reg
            elif args.ablation == "no_phase1":
                total = args.w_spa * loss_spa + args.w_spec * loss_spec - args.lmbd_weight * lmbd_reg
            else:
                total = wv * loss_var + args.w_spa * loss_spa + args.w_spec * loss_spec - args.lmbd_weight * lmbd_reg

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sched.step()
            if args.ablation not in ("no_csc", "fix_lmbd"):
                cfg.clamp_lmbd(model)

        if ep % 50 == 0 or ep == args.epochs_phase2 - 1 or ep == 0:
            pass

    torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        final = (model(lms.unsqueeze(0), _norm_pan(pan_raw)) + lms.unsqueeze(0)).clamp(0, 1)

    d = {"I_MS_LR": ms.permute(1,2,0).cpu().numpy() * cfg.TRAIN_DEFAULTS["max_value"],
         "I_MS": lms.permute(1,2,0).cpu().numpy() * cfg.TRAIN_DEFAULTS["max_value"],
         "I_PAN": pan_raw.cpu().numpy() * cfg.TRAIN_DEFAULTS["max_value"],
         "proposed": _to_hwc(final)}
    if gt_hw is not None: d["gt"] = gt_hw
    sio.savemat(os.path.join(save_dir, f"{data_id}_self_result.mat"), d)

    del model
    print(f" done ({time.time()-t0:.0f}s)", flush=True)
