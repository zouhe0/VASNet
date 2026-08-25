#!/usr/bin/env python3
"""
visualize.py — DLPan-Toolbox 可视化 Python 复现
严格对齐 /media/zouhe/Elements/zspan/DLPan-Toolbox/02-Test-toolbox/Tools/

核心函数:
  linstretch       — 直方图百分位线性拉伸 (对齐 linstretch.m)
  viewimage        — 单图拉伸显示 (对齐 viewimage.m, BGR通道翻转)
  viewimage2       — 多图统一拉伸 (对齐 viewimage2.m)
  rectangleonimage — 矩形框 + 角部放大 (对齐 rectangleonimage.m)
  show_image_4lr   — 4波段MS显示 (对齐 showImage4LR.m)
  show_pan         — 全色图像显示 (对齐 showPan.m)
  show_images_all  — 多算法网格对比 (对齐 showImagesAll.m)

MATLAB特有无法平移的:
  - iptsetpref('ImshowBorder','tight') — 用 plt.subplots_adjust 近似
  - print('-depsc',...) EPS导出 — matplotlib savefig 替代
  - imresize(...,'nearest') — PIL.Image.resize(Image.NEAREST) 完全等价
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
#  linstretch — 直方图百分位线性拉伸
#  对齐 linstretch.m:
#    b = double(uint16(img(:,:,i)))  -> clip[0,65535]+取整
#    hist(b, max(b)-min(b)) + cumsum -> 累积直方图
#    t(1)=下百分位  t(2)=上百分位
#    clip到[t(1),t(2)]后归一化[0,1]
# ============================================================
def linstretch(img, tol=None):
    """
    img: (H, W, 3) float64, 任意范围
    tol: (3,2) 或 (2,) — 默认每通道 [0.01, 0.99]
    返回: (H, W, 3) float64, [0,1]
    """
    if tol is None:
        tol = np.array([[0.01, 0.99], [0.01, 0.99], [0.01, 0.99]])
    elif tol.ndim == 1:
        tol = np.tile(tol, (3, 1))

    H, W, C = img.shape
    result = np.zeros_like(img, dtype=np.float64)

    for i in range(min(C, 3)):
        # MATLAB: b = double(uint16(img(:,:,i)))
        b = np.clip(np.round(img[:, :, i]), 0, 65535)
        b = b.astype(np.uint16).ravel().astype(np.float64)
        NM = len(b)

        b_min, b_max = int(b.min()), int(b.max())
        if b_max <= b_min:
            result[:, :, i] = 0.0
            continue

        bins = np.arange(b_min, b_max + 2)
        hb, _ = np.histogram(b, bins=bins)
        chb = np.cumsum(hb)
        levelb = np.arange(b_min, b_max + 1)

        idx_low = np.where(chb > NM * tol[i, 0])[0]
        t_low = levelb[idx_low[0]] if len(idx_low) > 0 else b_min

        idx_high = np.where(chb < NM * tol[i, 1])[0]
        t_high = levelb[idx_high[-1]] if len(idx_high) > 0 else b_max

        band = img[:, :, i].copy()
        band[band < t_low] = t_low
        band[band > t_high] = t_high
        band = (band - t_low) / (t_high - t_low) if t_high > t_low else np.zeros_like(band)
        result[:, :, i] = band

    return result


# ============================================================
#  viewimage — 单图显示 (对齐 viewimage.m)
# ============================================================
def viewimage(img, tol=None, ax=None):
    """img: (H,W,B) -> BGR可视化 (B<3 复制, B>=3 取前3)"""
    img = img.astype(np.float64)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] < 3:
        img = np.concatenate([img, img[:, :, :1], img[:, :, :1]], axis=-1)[:, :, :3]

    if tol is None:
        tol = np.array([0.01, 0.99])

    stretched = linstretch(img[:, :, :3], tol)
    bgr = stretched[:, :, ::-1]  # MATLAB: ImageToView(:,:,3:-1:1)

    if ax is not None:
        ax.imshow(bgr)
        ax.axis('off')

    return bgr


# ============================================================
#  viewimage2 — 统一拉伸 (对齐 viewimage2.m, 不翻转, 不imshow)
# ============================================================
def viewimage2(img, tol=None):
    """img: (H,W,3) -> 统一拉伸, 用于多图拼接场景"""
    img = img.astype(np.float64)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] < 3:
        img = np.concatenate([img, img[:, :, :1], img[:, :, :1]], axis=-1)[:, :, :3]

    if tol is None:
        tol = np.array([0.01, 0.99])

    return linstretch(img[:, :, :3], tol)


# ============================================================
#  rectangleonimage — 矩形框 + 角部放大 (对齐 rectangleonimage.m)
# ============================================================
def rectangleonimage(pic, sw, n=3, ch=3, c=1, scale=3, rect_type=1):
    """
    pic: (H,W,3) 或 (H,W), [0,1]
    sw: [x0, x1, y0, y1] 矩形框坐标(行列)
    n: 线宽, ch: 1灰度/3彩色, c: 1红/2绿/3蓝/other反色
    scale: 放大倍数, rect_type: 1左下/2右下/3右上/4左上
    """
    x0, x1, y0, y1 = int(sw[0]), int(sw[1]), int(sw[2]), int(sw[3])
    ent = pic.copy()
    max_val = 1.0

    if ch == 1:
        if c == 1:
            ent[x0:x1, y0:y0+n] = max_val
            ent[x0:x1, y1-n:y1] = max_val
            ent[x0:x0+n, y0:y1] = max_val
            ent[x1-n:x1, y0:y1] = max_val
        elif c == 2:
            for sl in [(slice(x0,x1), slice(y0,y0+n)), (slice(x0,x1), slice(y1-n,y1)),
                       (slice(x0,x0+n), slice(y0,y1)), (slice(x1-n,x1), slice(y0,y1))]:
                ent[sl] = 0
        else:
            for sl in [(slice(x0,x1), slice(y0,y0+n)), (slice(x0,x1), slice(y1-n,y1)),
                       (slice(x0,x0+n), slice(y0,y1)), (slice(x1-n,x1), slice(y0,y1))]:
                ent[sl] = max_val - ent[sl]
    elif ch == 3:
        if c == 1:  # red
            for sl in [(slice(x0,x1), slice(y0,y0+n)), (slice(x0,x1), slice(y1-n,y1)),
                       (slice(x0,x0+n), slice(y0,y1)), (slice(x1-n,x1), slice(y0,y1))]:
                ent[sl + (0,)] = max_val
                ent[sl + (1,)] = 0
                ent[sl + (2,)] = 0
        elif c == 2:  # green
            for sl in [(slice(x0,x1), slice(y0,y0+n)), (slice(x0,x1), slice(y1-n,y1)),
                       (slice(x0,x0+n), slice(y0,y1)), (slice(x1-n,x1), slice(y0,y1))]:
                ent[sl + (0,)] = 0
                ent[sl + (1,)] = max_val
                ent[sl + (2,)] = 0
        elif c == 3:  # blue
            for sl in [(slice(x0,x1), slice(y0,y0+n)), (slice(x0,x1), slice(y1-n,y1)),
                       (slice(x0,x0+n), slice(y0,y1)), (slice(x1-n,x1), slice(y0,y1))]:
                ent[sl + (0,)] = 0
                ent[sl + (1,)] = 0
                ent[sl + (2,)] = max_val
        else:  # inverse
            for sl in [(slice(x0,x1), slice(y0,y0+n)), (slice(x0,x1), slice(y1-n,y1)),
                       (slice(x0,x0+n), slice(y0,y1)), (slice(x1-n,x1), slice(y0,y1))]:
                ent[sl] = max_val - ent[sl]

    # crop + nearest resize
    samp_im = ent[x0:x1, y0:y1]
    if samp_im.size == 0:
        return ent
    h_s, w_s = samp_im.shape[:2]
    samp_pil = Image.fromarray((np.clip(samp_im, 0, 1) * 255).astype(np.uint8))
    new_w, new_h = int(w_s * scale), int(h_s * scale)
    samp_resized = np.array(samp_pil.resize((new_w, new_h), Image.NEAREST)) / 255.0

    p, q = ent.shape[:2]
    a, b = samp_resized.shape[:2]

    if rect_type == 1:   ent[(p-a):p, :b] = samp_resized
    elif rect_type == 2: ent[(p-a):p, (q-b):q] = samp_resized
    elif rect_type == 3: ent[:a, (q-b):q] = samp_resized
    elif rect_type == 4: ent[:a, :b] = samp_resized

    return ent


# ============================================================
#  show_image_4lr — 4波段MS显示 (对齐 showImage4LR.m)
# ============================================================
def show_image_4lr(I_MS, ax=None, flag_cut_bounds=0, dim_cut=1,
                   th_values=0, L=11, ratio=4):
    """I_MS: (H,W,4) -> BGR可视化"""
    I_MS = I_MS.astype(np.float64)

    if flag_cut_bounds:
        d = int(np.round(dim_cut / ratio))
        if I_MS.shape[0] > 2*d and I_MS.shape[1] > 2*d:
            I_MS = I_MS[d:-d, d:-d, :]

    if th_values:
        I_MS[I_MS > 2**L] = 2**L
        I_MS[I_MS < 0] = 0

    rgb = I_MS[:, :, :3]
    imn = linstretch(rgb, np.array([0.01, 0.99]))
    bgr = imn[:, :, ::-1]

    if ax is not None:
        ax.imshow(bgr)
        ax.axis('off')

    return bgr


# ============================================================
#  show_pan — PAN显示 (对齐 showPan.m)
# ============================================================
def show_pan(I_PAN, ax=None, flag_cut_bounds=0, dim_cut=1):
    I_PAN = I_PAN.astype(np.float64)
    if I_PAN.ndim == 3:
        I_PAN = I_PAN[:, :, 0]

    if flag_cut_bounds:
        I_PAN = I_PAN[dim_cut:-dim_cut, dim_cut:-dim_cut]

    gray = np.stack([I_PAN, I_PAN, I_PAN], axis=-1)
    stretched = linstretch(gray, np.array([0.01, 0.99]))
    bgr = stretched[:, :, ::-1]

    if ax is not None:
        ax.imshow(bgr)
        ax.axis('off')

    return bgr


# ============================================================
#  show_images_all — 多图网格对比 (对齐 showImagesAll.m)
#  核心: 所有图水平拼接后统一linstretch, 再拆分, 确保公平对比
# ============================================================
def show_images_all(matrix_image, titles, vect_index_rgb=None,
                    flag_cut_bounds=0, dim_cut=1, flag_pan=0,
                    ncols=7, figsize=(20, 12)):
    """
    matrix_image: (H, W, B, Z) Z个图像
    titles: [str]*Z
    vect_index_rgb: RGB波段索引, 默认 [2,1,0]
    flag_pan: 1=第一张是PAN图
    """
    if vect_index_rgb is None:
        vect_index_rgb = [2, 1, 0]

    H, W, B, Z = matrix_image.shape

    if flag_cut_bounds:
        matrix_image_cat = matrix_image[dim_cut:-dim_cut, dim_cut:-dim_cut]
    else:
        matrix_image_cat = matrix_image

    Hc, Wc = matrix_image_cat.shape[:2]

    if flag_pan:
        T_parts = [matrix_image_cat[:, :, vect_index_rgb, ii] for ii in range(1, Z)]
        T = np.concatenate(T_parts, axis=1)
        pan_rgb = matrix_image_cat[:, :, vect_index_rgb, 0]
    else:
        T_parts = [matrix_image_cat[:, :, vect_index_rgb, ii] for ii in range(Z)]
        T = np.concatenate(T_parts, axis=1)

    T_stretched = viewimage2(T)
    matrix_print = np.zeros((Hc, Wc, 3, Z), dtype=np.float64)

    if flag_pan:
        matrix_print[:, :, :, 0] = viewimage2(pan_rgb)
        for ii in range(1, Z):
            matrix_print[:, :, :, ii] = T_stretched[:, (ii-1)*Wc:ii*Wc, :]
    else:
        for ii in range(Z):
            matrix_print[:, :, :, ii] = T_stretched[:, ii*Wc:(ii+1)*Wc, :]

    nrows = int(np.ceil(Z / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ii in range(Z):
        axes[ii].imshow(matrix_print[:, :, :, ii])
        axes[ii].set_title(titles[ii], fontsize=8)
        axes[ii].axis('off')

    for ii in range(Z, len(axes)):
        axes[ii].axis('off')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                        wspace=0.02, hspace=0.05)

    return fig, matrix_print


# ============================================================
#  Demo: RR结果可视化
# ============================================================
def visualize_mat(mat_path, save_path='viz_output.png',
                  flag_cut_bounds=1, dim_cut=1, th_values=0, L=11,
                  vect_rgb=None):
    """
    通用 .mat 可视化 — 自动检测 key, 适配各种 pipeline 输出.
    mat_path: .mat 文件路径
    save_path: 保存路径
    vect_rgb: RGB波段索引, None=自动 (8波段[4,2,1], 4波段[2,1,0])
    """
    import scipy.io as sio

    data = sio.loadmat(mat_path)
    keys = [k for k in data.keys() if not k.startswith('__')]
    print(f"  File: {mat_path}")
    print(f"  Keys: {keys}")

    proposed = data.get('proposed', None)
    gt       = data.get('gt', None)
    I_MS     = data.get('I_MS', None)
    I_MS_LR  = data.get('I_MS_LR', None)
    I_PAN    = data.get('I_PAN', None)

    if proposed is None:
        raise KeyError(f"No 'proposed' key. Available: {keys}")

    proposed = proposed.astype(np.float64)
    nbands = proposed.shape[2]
    print(f"  Bands: {nbands}  shape: {proposed.shape}")

    if vect_rgb is None:
        vect_rgb = [4, 2, 1] if nbands == 8 else [2, 1, 0]
    print(f"  RGB bands: {vect_rgb}")

    def _cut(arr):
        if arr is None: return None
        if flag_cut_bounds and arr.shape[0] > 2*dim_cut and arr.shape[1] > 2*dim_cut:
            arr = arr[dim_cut:-dim_cut, dim_cut:-dim_cut]
        if th_values and arr.ndim >= 2:
            arr = np.clip(arr, 0, 2**L)
        return arr

    def _cut_pan(arr):
        if arr is None: return None
        if arr.ndim == 3: arr = arr[:, :, 0]
        return _cut(arr)

    proposed = _cut(proposed)
    gt       = _cut(gt)
    I_MS     = _cut(I_MS)
    I_MS_LR  = _cut(I_MS_LR)
    I_PAN    = _cut_pan(I_PAN)

    H, W = proposed.shape[:2]

    # 上采样 LR MS
    if I_MS_LR is not None and I_MS_LR.shape[0] != H:
        lr_up = np.zeros((H, W, I_MS_LR.shape[2]), dtype=np.float64)
        for b in range(I_MS_LR.shape[2]):
            pil_b = Image.fromarray(I_MS_LR[:, :, b].astype(np.float32))
            lr_up[:, :, b] = np.array(pil_b.resize((W, H), Image.NEAREST))
        I_MS_LR = lr_up

    # PAN 转3通道
    if I_PAN is not None:
        if I_PAN.ndim == 2:
            I_PAN = np.stack([I_PAN, I_PAN, I_PAN], axis=-1)
        elif I_PAN.shape[2] < 3:
            I_PAN = np.stack([I_PAN[:,:,0]]*3, axis=-1)
        else:
            I_PAN = I_PAN[:, :, :3]

    # 构建面板
    panels, titles = [], []
    for key, label in [('gt', 'GT'), ('I_MS', 'I_MS (EXP)'), ('I_PAN', 'PAN'), ('I_MS_LR', 'I_MS_LR')]:
        arr = locals().get(key)
        if arr is not None:
            panels.append(arr[:, :, vect_rgb] if arr.ndim == 3 else np.stack([arr]*3, axis=-1))
            titles.append(label)

    panels.append(proposed[:, :, vect_rgb])
    titles.append('Proposed')

    matrix_image = np.stack(panels, axis=-1)
    ncols = len(panels)
    fig, _ = show_images_all(matrix_image, titles, flag_cut_bounds=0,
                             ncols=ncols, figsize=(4*ncols, 5))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {save_path}')
    return fig


# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DLPan-Toolbox .mat 可视化')
    parser.add_argument('path', nargs='?', default='result_self/QB_full/0_self_result.mat',
                        help='.mat 文件路径 或 结果目录')
    parser.add_argument('--id', type=int, default=0, help='数据编号')
    parser.add_argument('--pattern', type=str, default='{id}_self_result.mat',
                        help='文件名模板')
    parser.add_argument('--save', type=str, default='viz_output.png')
    parser.add_argument('--rgb', type=int, nargs=3, default=None,
                        help='RGB波段索引, 如 --rgb 4 2 1')
    args = parser.parse_args()

    # 智能判断: path 是文件还是目录
    import os
    if os.path.isfile(args.path):
        mat_path = args.path
    elif os.path.isdir(args.path):
        mat_path = os.path.join(args.path, args.pattern.replace('{id}', str(args.id)))
    else:
        mat_path = args.path  # 直接尝试

    print(f'Loading: {mat_path}')
    fig = visualize_mat(mat_path, save_path=args.save, vect_rgb=args.rgb)

