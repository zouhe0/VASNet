import argparse
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Dataset
from mymodel import SDNetFusionNet_All
from mymodel import FusionNet


SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="pretrain learning rate")
parser.add_argument("--epochs", type=int, default=8, help="pretrain epochs")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--device", type=str, default="cuda:0", help="training device")
parser.add_argument("--data_id", type=int, default=0, help="data ID (0-19)")
parser.add_argument("--sensor", type=str, default="wv3", help="sensor type")
parser.add_argument("--ratio", type=int, default=4, help="downsample ratio")
parser.add_argument("--identifier", type=str, default=None, help="model identifier for saving")
parser.add_argument("--data_path", type=str,
                    default=r"D:\DeepLearning\zspan\test_wv3_OrigScale_multiExm1.h5",
                    help="data file path")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
data_id = args.data_id
sensor = args.sensor.upper()
ratio = args.ratio

#model = FusionNet().to(device)
model = SDNetFusionNet_All().to(device)
print("SDNetFusionNet initialized for LR pretraining")

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))


def save_checkpoint(model, identifier):
    os.makedirs("model_FUG", exist_ok=True)
    model_out_path = os.path.join("model_FUG", f"{identifier}_pretrain.pth")
    torch.save(model.state_dict(), model_out_path)



def train(train_loader, identifier):
    print(f"Starting LR pretraining ({args.epochs} epochs)...")
    start_time = time.time()
    min_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = []

        for i, batch in enumerate(train_loader):
            ms, _, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # ---------- 统一维度为 4D ----------
            # ms 应该已经是 (1, C, H, W)
            if ms.dim() == 3:
                ms = ms.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

            # pan 处理成 (1, 1, H_pan, W_pan)
            if pan.dim() == 2:
                pan = pan.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
            elif pan.dim() == 3:
                pan = pan.unsqueeze(1)  # (1,H,W) -> (1,1,H,W)
            # 此时 pan 为 (1, 1, H_pan, W_pan)

            # ---------- Wald protocol：制作低分辨率训练对 ----------
            # 下采样 ms 得到 LR 版本
            ms_lr = F.interpolate(ms, scale_factor=1.0 / ratio,
                                  mode='bilinear', align_corners=False)  # (1, C, H/ratio, W/ratio)

            # 上采样回原始 ms 尺寸，作为低分辨率上采样的输入
            lms_lr = F.interpolate(ms_lr, size=ms.shape[-2:],
                                   mode='bilinear', align_corners=False)  # (1, C, H, W)

            # 下采样 PAN 到 ms 的空间尺寸
            pan_lr = F.interpolate(pan, size=ms.shape[-2:],
                                   mode='bilinear', align_corners=False)  # (1, 1, H, W)

            # ---------- 模型前向 ----------
            optimizer.zero_grad()
            # 模型要求 x: (1,8,H,W), y: (1,1,H,W)
            res = model(lms_lr, pan_lr)  # (1, 8, H, W)

            output = res + lms_lr  # 残差 + 上采样图像
            loss = torch.mean((output - ms) ** 2)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        if epoch % 2 == 0 or epoch == args.epochs:
            print(f"Pretrain Epoch [{epoch}/{args.epochs}] - MSE Loss: {avg_loss:.8f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            save_checkpoint(model, identifier)

    total_time = time.time() - start_time
    print(f"Pretraining complete, total time {total_time:.2f} s, best loss: {min_loss:.8f}")
    print("save complete")


def main():
    train_set = Dataset(args.data_path, args.data_id)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    identifier = args.identifier if args.identifier else f"{sensor}_{data_id}_FusionNet_SDNet_All_SDE"
    train(train_loader, identifier)


if __name__ == "__main__":
    main()