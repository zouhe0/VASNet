import argparse
import time
import os
import sys
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import scipy.io as sio
import h5py

from data import Dataset
from mymodel import SDNetFusionNet_All
from loss import LossCalculator
import warnings
warnings.filterwarnings("ignore")

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

parser = argparse.ArgumentParser(description="Self-supervised two-phase training (no external teacher)")
parser.add_argument("--lr_phase1", type=float, default=0.015, help="Phase1 LR pretrain learning rate")
parser.add_argument("--epochs_phase1", type=int, default=8, help="Phase1 LR pretrain epochs")
parser.add_argument("--lr_phase2", type=float, default=0.0028, help="Phase2 distillation learning rate")
parser.add_argument("--epochs_phase2", type=int, default=250, help="Phase2 distillation epochs")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--data_id", type=int, default=None, help="data ID (0-19), None=process all")
parser.add_argument("--start_id", type=int, default=0, help="start data ID")
parser.add_argument("--end_id", type=int, default=19, help="end data ID")
parser.add_argument("--sensor", type=str, default="WV3")
parser.add_argument("--ratio", type=int, default=4)
parser.add_argument("--alfa", type=float, default=0.15)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--data_path", type=str,
                    default=r"D:\DeepLearning\zspan\test_wv3_OrigScale_multiExm1.h5")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
sensor = args.sensor.upper()
ratio = args.ratio

ri = 185
alfa = args.alfa
w_var = 1850000.0
w_spa = alfa * ri
w_spec = (1 - alfa) * ri

save_dir = "result_self"
os.makedirs(save_dir, exist_ok=True)
os.makedirs("model_FUG", exist_ok=True)


def norm_pan_to_4d(pan_tensor):
    """Ensure pan is (1, 1, H, W) 4D for model input."""
    if pan_tensor.dim() == 2:
        return pan_tensor.unsqueeze(0).unsqueeze(0)
    elif pan_tensor.dim() == 3:
        if pan_tensor.shape[0] == 1:
            return pan_tensor.unsqueeze(1)
        else:
            return pan_tensor.unsqueeze(0)
    elif pan_tensor.dim() == 4:
        return pan_tensor
    else:
        raise ValueError(f"Unexpected pan dim: {pan_tensor.dim()}")


def norm_lms_to_4d(lms_tensor):
    """Ensure lms is (1, C, H, W) 4D for model input."""
    if lms_tensor.dim() == 3:
        return lms_tensor.unsqueeze(0)
    elif lms_tensor.dim() == 4:
        return lms_tensor
    else:
        raise ValueError(f"Unexpected lms dim: {lms_tensor.dim()}")


def hwc_save(tensor_4d, scale=2047.0):
    """Convert (1, C, H, W) tensor → (H, W, C) numpy, denormalized."""
    return tensor_4d.squeeze(0).permute(1, 2, 0).cpu().numpy() * scale


# ================== Phase 1: LR self-training (Wald protocol) ==================
def phase1_pretrain(data_id):
    """Train on Wald-degraded LR data, generate self-teacher, save Phase1 result."""
    print(f"\n{'='*50}")
    print(f"Phase 1 [{data_id}]: LR Wald-protocol pretraining")
    print(f"{'='*50}")
    t1 = time.time()

    # Load single-image data
    with h5py.File(args.data_path, "r") as f:
        ms_np  = np.array(f["ms"][data_id],  dtype=np.float32) / 2047.0
        lms_np = np.array(f["lms"][data_id], dtype=np.float32) / 2047.0
        pan_np = np.array(f["pan"][data_id], dtype=np.float32) / 2047.0

    ms  = torch.from_numpy(ms_np).to(device)   # (8, 128, 128)
    lms = torch.from_numpy(lms_np).to(device)  # (8, 512, 512)
    pan_raw = torch.from_numpy(pan_np).to(device)  # (512, 512)

    # Normalize to standard shapes
    pan_4d = norm_pan_to_4d(pan_raw)           # (1, 1, 512, 512)
    lms_4d = lms.unsqueeze(0)                   # (1, 8, 512, 512)
    ms_4d  = ms.unsqueeze(0)                    # (1, 8, 128, 128)

    # Wald LR inputs
    pan_lr = F.interpolate(pan_4d, size=ms.shape[1:],
                           mode="bilinear", align_corners=False)  # (1, 1, 128, 128)
    ms_low = F.interpolate(ms_4d, scale_factor=1.0 / ratio,
                           mode="bilinear", align_corners=False)  # (1, 8, 32, 32)
    lms_lr = F.interpolate(ms_low, size=ms.shape[1:],
                           mode="bilinear", align_corners=False)  # (1, 8, 128, 128)

    model = SDNetFusionNet_All().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr_phase1, betas=(0.9, 0.999))
    sched = CosineAnnealingLR(opt, T_max=args.epochs_phase1, eta_min=args.lr_phase1 * 0.01)
    min_loss = float("inf")
    best_path = os.path.join("model_FUG", f"{sensor}_{data_id}_self_phase1_best.pth")

    for epoch in range(1, args.epochs_phase1 + 1):
        model.train()
        opt.zero_grad()

        res = model(lms_lr, pan_lr)
        output = res + lms_lr
        loss = torch.mean((output - ms_4d) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), best_path)

        if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs_phase1:
            print(f"  P1 [{epoch}/{args.epochs_phase1}] MSE: {loss.item():.8f}")

    print(f"Phase 1 train done: {time.time()-t1:.1f}s, best MSE: {min_loss:.8f}")

    # ---- Save Phase 1 LR result (reload best weights) ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    with torch.no_grad():
        p1_res_lr = model(lms_lr, pan_lr)
        p1_out_lr = p1_res_lr + lms_lr   # (1, 8, 128, 128)

    I_p1_lr = hwc_save(p1_out_lr)
    p1_lr_path = os.path.join(save_dir, f"{data_id}_self_phase1_lr.mat")
    sio.savemat(p1_lr_path, {
        "I_MS_LR": ms.permute(1, 2, 0).cpu().numpy() * 2047.0,
        "I_MS": lms.permute(1, 2, 0).cpu().numpy() * 2047.0,
        "I_PAN": pan_raw.cpu().numpy(),
        "proposed": I_p1_lr,
    })
    print(f"Phase 1 LR result saved: {p1_lr_path}")

    # ---- Generate self-teacher at full resolution ----
    # CRITICAL: DictConv2d caches xsize from LR; fresh model needed for 512
    del model
    model_full = SDNetFusionNet_All().to(device)
    model_full.load_state_dict(torch.load(best_path, map_location=device))
    model_full.eval()
    with torch.no_grad():
        teacher_res = model_full(lms_4d, pan_4d)
        teacher_out = teacher_res + lms_4d       # (1, 8, 512, 512)

    # Save teacher as .mat
    I_teacher = hwc_save(teacher_out)
    teacher_mat_path = os.path.join(save_dir, f"{data_id}_self_teacher.mat")
    sio.savemat(teacher_mat_path, {"sr": I_teacher})
    print(f"Self-teacher saved: {teacher_mat_path}  shape: {I_teacher.shape}")

    # ---- Save Phase 1 full-res result ----
    I_p1_full = hwc_save(teacher_out)
    p1_full_path = os.path.join(save_dir, f"{data_id}_self_phase1_full.mat")
    sio.savemat(p1_full_path, {
        "I_MS_LR": ms.permute(1, 2, 0).cpu().numpy() * 2047.0,
        "I_MS": lms.permute(1, 2, 0).cpu().numpy() * 2047.0,
        "I_PAN": pan_raw.cpu().numpy(),
        "proposed": I_p1_full,
    })
    print(f"Phase 1 full-res result saved: {p1_full_path}")

    teacher_tensor = teacher_out.squeeze(0).detach()  # (8, 512, 512)
    del model_full
    return teacher_tensor, lms, pan_raw, ms


# ================== Phase 2: Distillation (EXACT trainba logic) ==================
def phase2_distill(data_id, teacher_tensor, lms, pan_raw, ms):
    """Train a fresh model with self-teacher, identical to trainba.py."""
    print(f"\n{'='*50}")
    print(f"Phase 2 [{data_id}]: Self-distillation (trainba-style)")
    print(f"{'='*50}")
    t2 = time.time()
    print(f"  teacher_tensor: {teacher_tensor.shape} on {teacher_tensor.device}")

    train_set = Dataset(args.data_path, data_id)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    model = SDNetFusionNet_All().to(device)
    print("  Student model (SDNetFusionNet_All) initialized")
    loss_calculator = LossCalculator(sensor=sensor, ratio=ratio, N=41, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_phase2, betas=(0.9, 0.999))

    min_total_loss = float("inf")
    identifier = f"{sensor}_{data_id}_self"
    best_path = os.path.join("model_FUG", f"{identifier}_best.pth")

    for epoch in range(1, args.epochs_phase2 + 1):
        model.train()
        epoch_loss_var, epoch_loss_spa, epoch_loss_spec = [], [], []

        for i, batch in enumerate(train_loader):
            ms_b, lms_b, pan_b = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()

            if len(pan_b.shape) == 3:
                pan_b = pan_b.unsqueeze(1)

            # EXACT trainba forward path
            res_student = model(lms_b, pan_b)
            fusion_out = res_student + lms_b
            fusion_out = fusion_out.squeeze(0)

            # Loss1: variance against self-teacher
            loss_var = torch.mean((fusion_out - teacher_tensor) ** 2)

            # Loss2: spatial fidelity
            fusion_out_hw_c = fusion_out.permute(1, 2, 0)
            _, H, _ = ms_b[0].shape
            block_size = H // ratio
            loss_spa = loss_calculator.compute_spatial_fidelity_loss(
                fusion_out_hw_c, ms_b[0].permute(1, 2, 0), pan_b[0].squeeze(0), block_size)

            # Loss3: spectral fidelity
            loss_spec = loss_calculator.compute_spectral_loss(
                fusion_out_hw_c, ms_b[0].permute(1, 2, 0))

            total_loss = w_var * loss_var + w_spa * loss_spa + w_spec * loss_spec

            epoch_loss_var.append(loss_var.item())
            epoch_loss_spa.append(loss_spa.item())
            epoch_loss_spec.append(loss_spec.item())

            total_loss.backward()
            optimizer.step()

        avg_loss_var = np.mean(epoch_loss_var)
        avg_loss_spa = np.mean(epoch_loss_spa)
        avg_loss_spec = np.mean(epoch_loss_spec)
        avg_total_loss = w_var * avg_loss_var + w_spa * avg_loss_spa + w_spec * avg_loss_spec

        if epoch % 50 == 0 or epoch == args.epochs_phase2:
            print(f"  Epoch [{epoch}/{args.epochs_phase2}] - "
                  f"var:{avg_loss_var:.6f} spa:{avg_loss_spa:.6f} "
                  f"spec:{avg_loss_spec:.6f} total:{avg_total_loss:.6f}")

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            torch.save(model.state_dict(), best_path)

    t2_end = time.time()
    print(f"Phase 2 train done: {t2_end-t2:.1f}s, best loss: {min_total_loss:.6f}")

    # ---- Final inference (testba-style) ----
    print("  Running final inference...")
    # (model already has best weights from training)
    model.eval()
    with torch.no_grad():
        lms_in = lms.unsqueeze(0)  # (1, 8, 512, 512)
        pan_in = norm_pan_to_4d(pan_raw)  # (1, 1, 512, 512)
        final_res = model(lms_in, pan_in)
        final_out = final_res + lms_in

    I_final = hwc_save(final_out)
    I_ms_out = ms.permute(1, 2, 0).cpu().numpy() * 2047.0
    I_pan_out = pan_raw.cpu().numpy()
    I_lms_out = lms.permute(1, 2, 0).cpu().numpy() * 2047.0

    result_path = os.path.join(save_dir, f"{data_id}_self_result.mat")
    sio.savemat(result_path, {
        "I_MS_LR": I_ms_out,
        "I_MS": I_lms_out,
        "I_PAN": I_pan_out,
        "proposed": I_final,
    })
    print(f"  Final result saved: {result_path}")
    del model


# ================== Process single image ==================
def process_single_image(data_id):
    img_start = time.time()
    print(f"\n{'#'*60}")
    print(f"Processing image {data_id}")
    print(f"{'#'*60}")

    teacher_tensor, lms, pan_raw, ms = phase1_pretrain(data_id)
    phase2_distill(data_id, teacher_tensor, lms, pan_raw, ms)

    print(f"Image {data_id} complete! Total: {time.time()-img_start:.1f}s")
    return True


# ================== Main ==================
def main():
    total_start = time.time()

    if args.data_id is None:
        start_id, end_id = args.start_id, args.end_id
        total_images = end_id - start_id + 1
        print(f"\nBatch processing images {start_id}-{end_id} ({total_images} total)")

        success = 0
        for did in range(start_id, end_id + 1):
            print(f"\n>> Image {did} ({did-start_id+1}/{total_images})")
            try:
                if process_single_image(did):
                    success += 1
            except Exception as e:
                print(f"\n[ERROR] Image {did} FAILED: {e}", file=sys.stderr)
                traceback.print_exc()

            elapsed = time.time() - total_start
            if did > start_id:
                eta = elapsed / (did - start_id + 1) * total_images - elapsed
                print(f"Progress: {(did-start_id+1)/total_images*100:.0f}% | "
                      f"elapsed:{elapsed:.0f}s | ETA:{eta:.0f}s")

        total_time = time.time() - total_start
        print(f"\nDone! Success:{success}/{total_images} | "
              f"Total:{total_time:.0f}s | Avg:{total_time/total_images:.0f}s/img")
    else:
        try:
            process_single_image(args.data_id)
        except Exception as e:
            print(f"\n[ERROR] Image {args.data_id} FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
        print(f"\nDone! Total:{time.time()-total_start:.0f}s")


if __name__ == "__main__":
    main()


