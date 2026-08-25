import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio

from mymodel import SDNetFusionNet_All, FusionNet
import warnings
warnings.filterwarnings("ignore")


def tensor_to_image(tensor):
    tensor = tensor.cpu().numpy()
    if tensor.shape[0] >= 3:
        img = tensor[:3, :, :]
    else:
        img = np.repeat(tensor, 3, axis=0)
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    return img


def main():
    parser = argparse.ArgumentParser(description="Test SDNetFusionNet model (mat-based teacher)")
    parser.add_argument("--process_model", type=int, default=1, help="model type: 0=trainba, 1=SDE_train")
    parser.add_argument("--data_id", type=int, default=None, help="test sample index (0 ~ N-1)")
    parser.add_argument("--data_path", type=str, default=r"/HardDisk/HeZou/test_wv3_OrigScale_multiExm1.h5", help="h5 data file path")
    parser.add_argument("--satellite", type=str, default="WV3/", help="satellite type")
    parser.add_argument("--show_results", action="store_true", help="show fusion result")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "reduce"], help="output mode")
    parser.add_argument("--gt_path", type=str, default=None, help="ground truth path (for reduce mode)")
    parser.add_argument("--sensor_type", type=str, default="WV3", help="sensor type")
    parser.add_argument("--alfa", type=float, default=0.5, help="fusion weight")
    parser.add_argument("--device", type=str, default="cuda:0", help="test device")
    parser.add_argument("--teacher_mat_path", type=str, default=r"D:\BaiduNetdiskDownload\Fusionmamba\output_mulExm_{data_id}.mat",
                        help="Teacher fusion .mat path pattern")
    parser.add_argument("--teacher_mat_key", type=str, default="sr", help="Key name in .mat file")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    teacher_mat_path = args.teacher_mat_path
    teacher_mat_key = args.teacher_mat_key

    # --------------------- Load data ---------------------
    print("Loading data...")
    with h5py.File(args.data_path, "r") as data:
        lms_np = np.array(data["lms"][args.data_id], dtype=np.float32) / 2047.0
        pan_np = np.array(data["pan"][args.data_id], dtype=np.float32) / 2047.0
        ms_np = np.array(data["ms"][args.data_id], dtype=np.float32) / 2047.0

    lms = torch.from_numpy(lms_np).to(device)
    pan = torch.from_numpy(pan_np).to(device)
    ms = torch.from_numpy(ms_np).to(device)

    if pan.dim() == 2:
        pan = pan.unsqueeze(0)

    lms_batch = lms.unsqueeze(0)
    pan_batch = pan.unsqueeze(0)
    ms_batch = ms.unsqueeze(0)

    input_height, input_width = pan.shape[1], pan.shape[2]
    print(f"Input image size: {input_height}x{input_width}")

    # --------------------- Load student model ---------------------
    print("Loading student model (SDNetFusionNet)...")
    model_student = SDNetFusionNet_All().to(device)
    #model_student = FusionNet().to(device)
    if args.process_model == 0:
        best_checkpoint = os.path.join("model_FUG", args.sensor_type.upper() + f"_{args.data_id}_FusionNet_SDNet_All_best.pth")
    elif args.process_model == 2:
        best_checkpoint = os.path.join("model_FUG", args.sensor_type.upper() + f"_{args.data_id}_FusionNet_SDNet_All_nosde_best.pth")
    else:
        best_checkpoint = os.path.join("model_FUG", args.sensor_type.upper() + f"_{args.data_id}_FusionNet_SDNet_All_SDE_best.pth")
    if os.path.exists(best_checkpoint):
        model_student.load_state_dict(torch.load(best_checkpoint, map_location=device))
        print(f"Successfully loaded student model: {best_checkpoint}")
    else:
        print(f"Student model weight file not found: {best_checkpoint}")
        return
    model_student.eval()

    # --------------------- Load teacher fusion from .mat ---------------------
    print("Loading teacher fusion from .mat...")
    file_path = teacher_mat_path.format(data_id=args.data_id)
    print(f"Loading from: {file_path}")
    mat_data = sio.loadmat(file_path)
    if teacher_mat_key not in mat_data:
        available_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{teacher_mat_key}' not found, available: {available_keys}")
    teacher_result = mat_data[teacher_mat_key]
    teacher_result_np = np.array(teacher_result, dtype=np.float32) / 2047.0
    I_teacher_raw = torch.from_numpy(teacher_result_np).permute(2, 0, 1).to(device)
    fused_teacher = I_teacher_raw.unsqueeze(0)
    print(f"Teacher fusion shape: {fused_teacher.shape}")

    # --------------------- Student model inference ---------------------
    print("Running student model inference...")
    with torch.no_grad():
        res_student = model_student(lms_batch, pan_batch)
        fused_student = res_student + lms_batch
        fused_student = fused_student.squeeze(0)

    # --------------------- Convert to image format ---------------------
    # Convert to HWC format and denormalize (values stored as 0-2047)
    I_MS_LR = ms.permute(1, 2, 0).cpu().detach().numpy() * 2047     # 128x128x8 low-res MS
    I_MS = lms.permute(1, 2, 0).cpu().detach().numpy() * 2047        # 512x512x8 upsampled LMS
    I_PAN = pan.squeeze(0).cpu().detach().numpy() * 2047
    # For visualization only (3-channel RGB display)
    lms_img = tensor_to_image(lms)

    if args.mode == "normal":
        I_student = torch.squeeze(fused_student).permute(1, 2, 0).cpu().detach().numpy() * 2047
        I_teacher = torch.squeeze(fused_teacher).permute(1, 2, 0).cpu().detach().numpy() * 2047

        student_dict = {"I_MS_LR": I_MS_LR, "I_MS": I_MS, "I_PAN": I_PAN, "proposed": I_student}
        os.makedirs(os.path.join("result", args.satellite), exist_ok=True)
        student_save_path = os.path.join("result", args.satellite, f"{args.data_id}_student_SDNet_All_{args.alfa}.mat")
        sio.savemat(student_save_path, student_dict)
        print(f"Student result saved to: {student_save_path}")

        teacher_dict = {"I_MS_LR": I_MS_LR, "I_MS": I_MS, "I_PAN": I_PAN, "proposed": I_teacher}
        teacher_save_path = os.path.join("result", args.satellite, f"{args.data_id}_teacher.mat")
        sio.savemat(teacher_save_path, teacher_dict)
        print(f"Teacher result saved to: {teacher_save_path}")

        if args.show_results:
            fused_student_img = tensor_to_image(fused_student)
            fused_teacher_img = tensor_to_image(fused_teacher)
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 4, 1); plt.imshow(lms_img); plt.title("LMS"); plt.axis("off")
            plt.subplot(1, 4, 2); plt.imshow(I_PAN, cmap="gray"); plt.title("PAN"); plt.axis("off")
            plt.subplot(1, 4, 3); plt.imshow(fused_student_img); plt.title("Student Fusion (SDNet)"); plt.axis("off")
            plt.subplot(1, 4, 4); plt.imshow(fused_teacher_img); plt.title("Teacher Fusion (mat)"); plt.axis("off")
            plt.tight_layout(); plt.show()

    elif args.mode == "reduce":
        I_student = torch.squeeze(fused_student).permute(1, 2, 0).cpu().detach().numpy() * 2047
        I_teacher = torch.squeeze(fused_teacher).permute(1, 2, 0).cpu().detach().numpy() * 2047

        try:
            with h5py.File(args.data_path, "r") as data:
                if "gt" in data:
                    gt_data = np.array(data["gt"][args.data_id], dtype=np.float32)
                    if gt_data.shape[0] < 10:
                        gt_data = np.transpose(gt_data, (1, 2, 0))
                elif "GT" in data:
                    gt_data = np.array(data["GT"][args.data_id], dtype=np.float32)
                    if gt_data.shape[0] < 10:
                        gt_data = np.transpose(gt_data, (1, 2, 0))
                else:
                    gt_data = lms.permute(1, 2, 0).cpu().detach().numpy()
        except Exception:
            gt_data = lms.permute(1, 2, 0).cpu().detach().numpy()

        student_reduce_dict = {"gt": gt_data, "proposed": I_student, "ms": I_MS_LR, "pan": I_PAN, "ratio": 4, "sensor_type": args.sensor_type}
        os.makedirs(os.path.join("reduce_result", args.satellite), exist_ok=True)
        student_reduce_path = os.path.join("reduce_result", args.satellite, f"{args.data_id}_student_SDNet_All.mat")
        sio.savemat(student_reduce_path, student_reduce_dict)
        print(f"Student reduce data saved to: {student_reduce_path}")

        teacher_reduce_dict = {"gt": gt_data, "proposed": I_teacher, "ms": I_MS_LR, "pan": I_PAN, "ratio": 4, "sensor_type": args.sensor_type}
        teacher_reduce_path = os.path.join("reduce_result", args.satellite, f"{args.data_id}_teacher.mat")
        sio.savemat(teacher_reduce_path, teacher_reduce_dict)
        print(f"Teacher reduce data saved to: {teacher_reduce_path}")

    print("Processing complete!")


if __name__ == "__main__":
    main()
