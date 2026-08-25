import argparse
import subprocess
import os
import time
import sys
import warnings
warnings.filterwarnings("ignore")

def process_single_image(data_id, args):
    image_start_time = time.time()

    # 1. Training phase
    if not args.skip_train:
        if args.process_model == 0:
            # Pretrain at LR resolution first
            if not args.skip_pretrain:
                print("Phase 1: LR pretraining fusion model...")
                pretrain_cmd = [
                    sys.executable,
                    "pretrain.py",
                    f"--data_id={data_id}",
                    f"--lr={args.pretrain_lr}",
                    f"--epochs={args.pretrain_epochs}",
                    f"--device={args.device}",
                    f"--sensor={args.sensor}",
                    f"--ratio={args.ratio}",
                    f"--data_path={args.data_path}",
                    f"--identifier={args.sensor.upper()}_{data_id}_FusionNet_SDNet",
                ]
                print(f"Running: {' '.join(pretrain_cmd)}")
                try:
                    subprocess.run(pretrain_cmd, check=True)
                except subprocess.CalledProcessError:
                    print(f"Image {data_id} pretraining failed, continuing without pretrain...")

            train_cmd = [
                sys.executable,
                "trainba.py",
                f"--data_id={data_id}",
                "--pretrain",
                f"--lr={args.lr}",
                f"--epochs={args.epochs}",
                f"--batch_size={args.batch_size}",
                f"--device={args.device}",
                f"--sensor={args.sensor}",
                f"--ratio={args.ratio}",
                f"--temperature={args.temperature}",
                f"--alfa={args.alfa}",
                f"--data_path={args.data_path}",
                f"--teacher_mat_path={args.teacher_mat_path}",
                f"--teacher_mat_key={args.teacher_mat_key}",
            ]
        elif args.process_model == 2:
            # Self-consistent mode: skip SDE, skip pretrain, train_SDE without SDE loss
            print("Phase 1: Direct distillation training (self-consistent, train_SDE --no_sde)...")
            train_cmd = [
                sys.executable,
                "train_SDE.py",
                f"--data_id={data_id}",
                "--no_sde",
                f"--lr={args.lr}",
                f"--epochs={args.epochs}",
                f"--batch_size={args.batch_size}",
                f"--device={args.device}",
                f"--sensor={args.sensor}",
                f"--ratio={args.ratio}",
                f"--temperature={args.temperature}",
                f"--alfa={args.alfa}",
                f"--data_path={args.data_path}",
                f"--teacher_mat_path={args.teacher_mat_path}",
                f"--teacher_mat_key={args.teacher_mat_key}",
            ]

        elif args.process_model == 1:
            print("Phase 1: SDE network training...")
            SDE_train_cmd = [
                sys.executable,
                "main_SDE_amp.py",
                f"--lr={args.SDE_lr}",
                f"--epochs={args.SDE_epochs}",
                f"--batch_size={args.batch_size}",
                f"--device={args.SDE_device}",
                f"--satellite={args.sensor}",
                f"--name={data_id}",
                f"--data_path={args.data_path}",
            ]

            print(f"Running: {' '.join(SDE_train_cmd)}")
            try:
                SDE_train_result = subprocess.run(SDE_train_cmd, check=True)
                SDE_train_time = time.time() - image_start_time
                print(f"SDE network training complete! Time: {SDE_train_time:.2f} s")
            except subprocess.CalledProcessError:
                print(f"Image {data_id} SDE training failed, skipping")
                return False

            # Phase 2: Pretrain fusion model at LR resolution
            if not args.skip_pretrain:
                print("Phase 2: LR pretraining fusion model...")
                pretrain_cmd = [
                    sys.executable,
                    "pretrain.py",
                    f"--data_id={data_id}",
                    f"--lr={args.pretrain_lr}",
                    f"--epochs={args.pretrain_epochs}",
                    f"--device={args.device}",
                    f"--sensor={args.sensor}",
                    f"--ratio={args.ratio}",
                    f"--data_path={args.data_path}",
                    f"--identifier={args.sensor.upper()}_{data_id}_FusionNet_SDNet_SDE",
                ]
                print(f"Running: {' '.join(pretrain_cmd)}")
                try:
                    pretrain_result = subprocess.run(pretrain_cmd, check=True)
                    pretrain_time = time.time() - image_start_time
                    print(f"Pretraining complete! Time: {pretrain_time:.2f} s")
                except subprocess.CalledProcessError:
                    print(f"Image {data_id} pretraining failed, continuing without pretrain...")

            train_cmd = [
                sys.executable,
                "train_SDE.py",
                f"--data_id={data_id}",
                f"--pretrain={args.pretrain}",
                f"--lr={args.lr}",
                f"--epochs={args.epochs}",
                f"--batch_size={args.batch_size}",
                f"--device={args.device}",
                f"--sensor={args.sensor}",
                f"--ratio={args.ratio}",
                f"--temperature={args.temperature}",
                f"--alfa={args.alfa}",
                f"--data_path={args.data_path}",
                f"--teacher_mat_path={args.teacher_mat_path}",
                f"--teacher_mat_key={args.teacher_mat_key}",
            ]

        print(f"Running: {' '.join(train_cmd)}")
        try:
            train_result = subprocess.run(train_cmd, check=True)
            train_time = time.time() - image_start_time
            print(f"Distillation training complete! Time: {train_time:.2f} s")
        except subprocess.CalledProcessError:
            print(f"Image {data_id} distillation training failed, skipping")
            return False
    else:
        print(f"Skipping training for image {data_id}, directly testing...")

    # 2. Testing phase
    print(f"\n{'='*20} Testing image {data_id} {'='*20}")

    test_cmd = [
        sys.executable,
        "testba.py",
        f"--data_id={data_id}",
        f"--data_path={args.data_path}",
        f"--satellite={args.satellite}",
        f"--process_model={args.process_model}",
        f"--alfa={args.alfa}",
        f"--device={args.test_device}",
        f"--teacher_mat_path={args.teacher_mat_path}",
        f"--teacher_mat_key={args.teacher_mat_key}",
    ]

    if args.show_results:
        test_cmd.append("--show_results")

    print(f"Running: {' '.join(test_cmd)}")
    try:
        test_result = subprocess.run(test_cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Image {data_id} testing failed")
        return False

    if args.process_model == 2:
        result_path = os.path.join("result", args.satellite, f"{data_id}_student_SDNet_All_nosde_{args.alfa}.mat")
    else:
        result_path = os.path.join("result", args.satellite, f"{data_id}_student_SDNet_All_{args.alfa}.mat")
    print(f"\nImage {data_id} result files:")
    print(f"- Result: {result_path}")

    image_time = time.time() - image_start_time
    print(f"Image {data_id} processing complete! Total time: {image_time:.2f} s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Mat-based teacher distillation training and testing (SDNetFusionNet)")
    parser.add_argument("--process_model", type=int, default=2, help="model type: 0=trainba, 1=train_SDE, 2=trainba(no pretrain+SDE)")
    parser.add_argument("--pretrain",type=int, default= 0, help="pretarin or not")
    parser.add_argument("--SDE_lr", type=float, default=0.00075, help="SDE learning rate")
    parser.add_argument("--SDE_epochs", type=int, default=45, help="SDE training epochs")
    parser.add_argument("--SDE_device", type=str, default="cuda:0", help="SDE training device")
    parser.add_argument("--pretrain_lr", type=float, default=0.015, help="pretrain learning rate")
    parser.add_argument("--pretrain_epochs", type=int, default=8, help="pretrain epochs")
    parser.add_argument("--skip_pretrain", action="store_true", default=True, help="skip pretrain phase")
    parser.add_argument("--data_id", type=int, default=None, help="data ID (0-19), if not set process all")
    parser.add_argument("--process_all", action="store_true", help="process all images (0-19)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="training device")
    parser.add_argument("--sensor", type=str, default="WV3", help="sensor type")
    parser.add_argument("--ratio", type=int, default=4, help="downsample ratio")
    parser.add_argument("--temperature", type=float, default=1.0, help="distillation temperature")
    parser.add_argument("--data_path", type=str,
                        default=r"D:\DeepLearning\zspan\test_wv3_OrigScale_multiExm1.h5",
                        help="data file path")
    parser.add_argument("--satellite", type=str, default="WV3/", help="satellite type (for result saving)")
    parser.add_argument("--skip_train", action="store_true", help="skip training phase")
    parser.add_argument("--show_results", action="store_true", help="show fusion results")
    parser.add_argument("--start_id", type=int, default=0, help="start data ID")
    parser.add_argument("--end_id", type=int, default=19, help="end data ID")
    parser.add_argument("--alfa", type=float, default=0.15, help="fusion weight")
    parser.add_argument("--test_device", type=str, default="cuda:0", help="test device")
    parser.add_argument("--teacher_mat_path", type=str,
                        default=r"D:\BaiduNetdiskDownload\results\output_mulExm_{data_id}.mat",
                        help="Teacher fusion .mat path pattern, use {data_id} placeholder")
    parser.add_argument("--teacher_mat_key", type=str, default="sr", help="Key name in .mat file")

    args = parser.parse_args()

    total_start_time = time.time()

    if args.data_id is None:
        args.process_all = True

    if args.process_all:
        start_id = args.start_id
        end_id = args.end_id
        total_images = end_id - start_id + 1

        print(f"\n{'='*20} Batch processing {'='*20}")
        print(f"Processing images {start_id} to {end_id}, total {total_images} images")

        success_count = 0
        for data_id in range(start_id, end_id + 1):
            print(f"\n{'#'*30}")
            print(f"Processing image {data_id} ({data_id - start_id + 1}/{total_images})")
            print(f"{'#'*30}")

            if process_single_image(data_id, args):
                success_count += 1

            progress = (data_id - start_id + 1) / total_images * 100
            elapsed = time.time() - total_start_time
            if data_id > start_id:
                estimated_total = elapsed / (data_id - start_id + 1) * total_images
                remaining = estimated_total - elapsed
                print(f"\nProgress: {progress:.1f}% | Completed: {data_id - start_id + 1}/{total_images}")
                print(f"Elapsed: {elapsed:.2f}s | Estimated remaining: {remaining:.2f}s")

        total_time = time.time() - total_start_time
        print(f"\n{'='*20} Batch processing complete {'='*20}")
        print(f"Successfully processed: {success_count}/{total_images} images")
        print(f"Total time: {total_time:.2f} s (avg {total_time/total_images:.2f} s per image)")

    else:
        data_id = args.data_id
        process_single_image(data_id, args)
        total_time = time.time() - total_start_time
        print(f"\n{'='*20} Complete {'='*20}")
        print(f"Data ID: {data_id} processing complete")
        print(f"Total time: {total_time:.2f} s")


if __name__ == "__main__":
    main()
