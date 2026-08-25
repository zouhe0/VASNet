"""
一键读取 .mat 文件信息
默认扫描当前目录下的 ./result_self 文件夹（可修改）
双击运行即可，无需任何命令行参数。
"""

import sys
import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path

# ====== 默认配置（可在此修改） ======
DEFAULT_PATH = "./result_self"      # 默认搜索路径
RECURSIVE = False                   # 是否递归子目录
SHOW_STATS = True                   # 是否显示统计信息（min/max/mean/std）
SHOW_SAMPLE = False                 # 是否显示小数组的样例值（元素≤10个）
TARGET_VARS = None                  # 只显示指定变量，None 表示全部显示
# ====================================

def print_array_info(name, arr, show_stats=True, show_sample=False):
    """打印数组信息"""
    print(f"\n  [{name}]")
    if not isinstance(arr, np.ndarray):
        print(f"    type  : {type(arr).__name__}")
        print(f"    value : {arr}")
        return
    print(f"    shape : {arr.shape}")
    print(f"    dtype : {arr.dtype}")
    if arr.size == 0:
        print("    (empty array)")
        return
    if np.issubdtype(arr.dtype, np.number):
        flat = arr.flatten()
        if show_stats:
            print(f"    min   : {np.min(flat):.6f}")
            print(f"    max   : {np.max(flat):.6f}")
            print(f"    mean  : {np.mean(flat):.6f}")
            print(f"    std   : {np.std(flat):.6f}")
        if show_sample and arr.size <= 10:
            print(f"    values: {flat}")
    else:
        print("    (non-numeric array)")

def process_file(file_path, show_stats=True, show_sample=False, target_vars=None):
    """加载并显示一个.mat文件的内容"""
    try:
        data = sio.loadmat(file_path, squeeze_me=True)
    except Exception as e:
        print(f"❌ 读取失败: {file_path}\n   {e}")
        return

    var_names = [k for k in data.keys() if not k.startswith('__')]
    if target_vars is not None:
        var_names = [v for v in var_names if v in target_vars]
    if not var_names:
        print(f"⚠️  {file_path} 中没有目标变量")
        return

    print(f"\n{'='*60}")
    print(f"📁 {file_path}")
    print(f"   包含 {len(var_names)} 个变量:")
    for name in var_names:
        print_array_info(name, data[name], show_stats, show_sample)
    print(f"{'='*60}")

def main():
    # 如果用户通过命令行传参，则使用命令行参数（覆盖默认）
    parser = argparse.ArgumentParser(description="读取 .mat 文件信息（默认扫描 ./result_self）")
    parser.add_argument("path",default="D:/DeepLearning/zspan/zup/result_self/0_self_result.mat", nargs='?',
                        help=f"路径（文件或目录），默认: {DEFAULT_PATH}")
    parser.add_argument("-r", "--recursive", action="store_true", default=RECURSIVE,
                        help="是否递归子目录，默认: {RECURSIVE}")
    parser.add_argument("--no-stats", action="store_false", dest="show_stats",
                        default=SHOW_STATS, help="不显示统计信息，默认显示")
    parser.add_argument("--sample", action="store_true", default=SHOW_SAMPLE,
                        help="显示小数组的样例值，默认不显示")
    parser.add_argument("-v", "--var", nargs="*", default=TARGET_VARS,
                        help="只显示指定变量名，默认显示全部")
    args = parser.parse_args()

    # 应用配置
    path = Path(args.path)
    show_stats = args.show_stats
    show_sample = args.sample
    target_vars = set(args.var) if args.var else None

    if not path.exists():
        print(f"❌ 路径不存在: {path}")
        input("按 Enter 键退出...")
        sys.exit(1)

    # 收集所有.mat文件
    if path.is_file():
        mat_files = [path] if path.suffix.lower() == '.mat' else []
    else:
        if args.recursive:
            mat_files = list(path.rglob("*.mat"))
        else:
            mat_files = list(path.glob("*.mat"))

    if not mat_files:
        print(f"⚠️ 在 {path} 下未找到 .mat 文件")
        input("按 Enter 键退出...")
        sys.exit(0)

    print(f"找到 {len(mat_files)} 个 .mat 文件")
    for f in mat_files:
        process_file(f, show_stats, show_sample, target_vars)

    print("\n✅ 完成！")
    # 如果没有任何命令行参数（即双击运行），则暂停等待按键
    if len(sys.argv) == 1:
        input("按 Enter 键退出...")

if __name__ == "__main__":
    main()