import argparse
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import scipy.io as sio
from torch.cuda.amp import autocast, GradScaler

from data import Dataset
from mymodel import SDNetFusionNet_All, FusionNet
from SDE import Net_ms2pan
from loss import LossCalculator
import warnings
warnings.filterwarnings("ignore")

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--epochs", type=int, default=250, help="training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--device", type=str, default="cuda:0", help="training device")
parser.add_argument("--data_id", type=int, default=0, help="data ID (0-19)")
parser.add_argument("--sensor", type=str, default="wv3", help="sensor type")
parser.add_argument("--ratio", type=int, default=4, help="downsample ratio")
parser.add_argument("--temperature", type=float, default=1.0, help="distillation temperature")
parser.add_argument("--amp", default=True, action="store_true", help="enable AMP mixed precision training")
parser.add_argument("--alfa", type=float, default=0, help="loss weight")
parser.add_argument("--pretrain", type=int, default=1, help="load pretrained weights")
parser.add_argument("--no_sde", action="store_true", default=False, help="disable SDE model and SDE loss (self-consistent mode)")
parser.add_argument("--data_path", type=str, default="HardDisk/HeZou/test_wv3_OrigScale_multiExm1.h5", help="data file path")
parser.add_argument("--teacher_mat_path", type=str, default=r"D:\\BaiduNetdiskDownload\\Fusionmamba\\output_mulExm_{data_id}.mat",
                    help="Teacher fusion .mat file path pattern, use {data_id} placeholder")
parser.add_argument("--teacher_mat_key", type=str, default="sr", help="Key name in .mat file for teacher fusion result")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
data_id = args.data_id
sensor = args.sensor.upper()
ratio = args.ratio
use_amp = args.amp and (device.type == "cuda")
data_path = args.data_path
teacher_mat_path = args.teacher_mat_path
teacher_mat_key = args.teacher_mat_key
use_sde = not args.no_sde

ri = 180      #185
alfa = args.alfa
w_var = 1850000.0 * 2.0
w_spa = alfa * ri
w_spec = (1 - alfa) * ri

#model_student = FusionNet().to(device)
model_student = SDNetFusionNet_All().to(device)
print("Student model (SDNetFusionNet_All) initialized")

# Load pretrained weights if requested
if args.pretrain:
    pretrain_path = os.path.join("model_FUG", f"{sensor}_{data_id}_FusionNet_SDNet_All_SDE_pretrain.pth")
    if os.path.exists(pretrain_path):
        model_student.load_state_dict(torch.load(pretrain_path, map_location=device))
        print(f"Loaded pretrained weights from: {pretrain_path}")
    else:
        print(f"Warning: pretrain weights not found at {pretrain_path}, training from scratch")

loss_calculator = LossCalculator(sensor=sensor, ratio=ratio, N=41, device=device)
if use_sde:
    F_ms2pan = Net_ms2pan().to(device)
    F_ms2pan.load_state_dict(torch.load(f"model_SDE/{sensor}/{data_id}_Net_ms2pan.pth"))
    F_ms2pan.eval()
else:
    F_ms2pan = None
    print("SDE mode disabled, using spatial+spectral loss only")

optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
scaler = GradScaler() if use_amp else None

if use_amp:
    print("AMP mixed precision training enabled")
elif args.amp and device.type != "cuda":
    print("Warning: AMP only supports CUDA, disabled")


def save_checkpoint(model, identifier):
    os.makedirs("model_FUG", exist_ok=True)
    model_out_path = os.path.join("model_FUG", f"{identifier}.pth")
    torch.save(model.state_dict(), model_out_path)


def load_teacher_from_mat(data_id, device, mat_path, mat_key):
    file_path = mat_path.format(data_id=data_id)
    print(f"Loading teacher fusion from: {file_path}")
    mat_data = sio.loadmat(file_path)
    if mat_key not in mat_data:
        available_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{mat_key}' not found, available: {available_keys}")
    teacher_result = mat_data[mat_key]
    teacher_result = np.array(teacher_result, dtype=np.float32) / 2047.0
    teacher_result = torch.from_numpy(teacher_result)
    teacher_result = teacher_result.permute(2, 0, 1)
    teacher_result = teacher_result.to(device)
    print(f"Teacher fusion shape: {teacher_result.shape}")
    return teacher_result



def get_all_lmbd_values(model):
    """Collect all lmbd (lambda) parameter values from DictBlock modules."""
    values = []
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock':
            values.append(m.lmbd.item())
    return values


def compute_lmbd_regularization(model):
    """Compute L2 regularization over all lmbd parameters (sum of squares)."""
    total = 0.0
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock':
            total = total + m.lmbd.pow(2).sum()
    return total


def clamp_lmbd_nonneg(model):
    """Clamp all DictBlock lmbd parameters to be non-negative."""
    for m in model.modules():
        if m.__class__.__name__ == 'DictBlock':
            m.lmbd.data.clamp_(min=0.0)


def train(training_data_loader, identifier):
    print("Starting knowledge distillation training (mat-based teacher + SDE, SDNetFusionNet)...")
    start_time = time.time()
    teacher_result = load_teacher_from_mat(data_id, device, teacher_mat_path, teacher_mat_key)
    min_total_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model_student.train()
        epoch_loss_var, epoch_loss_spa, epoch_loss_spec = [], [], []

        for i, batch in enumerate(training_data_loader):
            ms, lms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            if len(pan.shape) == 3:
                pan = pan.unsqueeze(1)

            if use_amp:
                with autocast():
                    res_student = model_student(lms, pan)
                    fusion_out_ori = res_student + lms
                    fusion_out = fusion_out_ori.squeeze(0)
                    fusion_out_teacher = teacher_result
                    loss_var = torch.mean((fusion_out - fusion_out_teacher) ** 2)
                    fusion_out_hw_c = fusion_out.permute(1, 2, 0)
                    if use_sde:
                        with autocast():
                            sde_out = F_ms2pan(fusion_out_ori).squeeze(1).squeeze(0)
                        loss_spa = loss_calculator.SDE_Loss(sde_out, pan[0].squeeze(0))
                    else:
                        _, H, _ = ms[0].shape
                        block_size = H // ratio
                        loss_spa = loss_calculator.compute_spatial_fidelity_loss(
                            fusion_out_hw_c, ms[0].permute(1, 2, 0), pan[0].squeeze(0), block_size)
                    loss_spec = loss_calculator.compute_spectral_loss(fusion_out_hw_c, ms[0].permute(1, 2, 0))
                    lmbd_reg = compute_lmbd_regularization(model_student)
                    total_loss = w_var * loss_var + w_spa * loss_spa + w_spec * loss_spec - 1000.0 * lmbd_reg
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                clamp_lmbd_nonneg(model_student)
            else:
                res_student = model_student(lms, pan)
                fusion_out_ori = res_student + lms
                fusion_out = fusion_out_ori.squeeze(0)
                fusion_out_teacher = teacher_result
                loss_var = torch.mean((fusion_out - fusion_out_teacher) ** 2)
                fusion_out_hw_c = fusion_out.permute(1, 2, 0)
                _, H, _ = ms[0].shape
                if use_sde:
                    with torch.no_grad():
                        sde_out = F_ms2pan(fusion_out_ori).squeeze(1).squeeze(0)
                    loss_spa = loss_calculator.SDE_Loss(sde_out, pan[0].squeeze(0))
                else:
                    _, H, _ = ms[0].shape
                    block_size = H // ratio
                    loss_spa = loss_calculator.compute_spatial_fidelity_loss(
                        fusion_out_hw_c, ms[0].permute(1, 2, 0), pan[0].squeeze(0), block_size)
                loss_spec = loss_calculator.compute_spectral_loss(fusion_out_hw_c, ms[0].permute(1, 2, 0))
                lmbd_reg = compute_lmbd_regularization(model_student)
                total_loss = w_var * loss_var + w_spa * loss_spa + w_spec * loss_spec - 100.0 * lmbd_reg
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0)
                optimizer.step()
                clamp_lmbd_nonneg(model_student)

            epoch_loss_var.append(loss_var.item())
            epoch_loss_spa.append(loss_spa.item())
            epoch_loss_spec.append(loss_spec.item())

        avg_loss_var = np.mean(epoch_loss_var)
        avg_loss_spa = np.mean(epoch_loss_spa)
        avg_loss_spec = np.mean(epoch_loss_spec)
        avg_total_loss = w_var * avg_loss_var + w_spa * avg_loss_spa + w_spec * avg_loss_spec

        scheduler.step()

        # Print lmbd values every 20 epochs
        if epoch % 50 == 0 or epoch == 1:
            lmbd_vals = get_all_lmbd_values(model_student)
            if lmbd_vals:
                vals_str = ', '.join(f'{v:.6f}' for v in lmbd_vals)
                print(f"Epoch [{epoch}/{epochs}] - lmbd values: [{vals_str}]")

        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss1(var): {avg_loss_var:.6f}, Loss2(fspa): {avg_loss_spa:.6f}, Loss3(fspec): {avg_loss_spec:.6f}, Total Loss: {avg_total_loss:.6f}")

        if avg_total_loss < min_total_loss:
            min_total_loss = avg_total_loss
            save_checkpoint(model_student, f"{identifier}_best")

    total_time = time.time() - start_time
    print(f"Training complete, total time {total_time:.2f} s")
    print(f"Final best loss: {min_total_loss:.6f}")
    print(f"Model saved to: {identifier}_best.pth")


def main():
    train_set = Dataset(data_path, data_id)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    identifier = f"{sensor}_{data_id}_FusionNet_SDNet_All_nosde" if args.no_sde else f"{sensor}_{data_id}_FusionNet_SDNet_All_SDE"
    train(train_loader, identifier)


if __name__ == "__main__":
    main()
