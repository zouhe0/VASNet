#!/usr/bin/env python3
"""
CVPR / NeurIPS / TPAMI  —  polished publication-quality bubble chart.
Cohesive muted palette, BVSF in burnt orange, Segoe UI typography.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  Style  —  Segoe UI, clean typography
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size":        13,
    "axes.labelsize":   17,
    "axes.labelweight": "bold",
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

# ═══════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════
records = [
    ("PanCSCNet",    9.13,   0.9524,  0.070),
    ("FusionNet",   10.27,   0.9356,  0.079),
    ("LAGConv",      2.07,   0.9230,  0.151),
    ("RWKVFusion",   2.34,   0.9491,  1.210),
    ("DCFNet",       3.46,   0.9483,  2.770),
    ("PanDiff",     60.18,   0.9203, 32.233),
    ("ZSPan",       19.63,   0.9505,  0.080),
    ("BVSF (Ours)",  2.49,   0.9654,  0.006),
]

names  = [r[0] for r in records]
flops  = np.array([r[1] for r in records])
hqnr   = np.array([r[2] for r in records])
params = np.array([r[3] for r in records])
n      = len(records)
idx_o  = names.index("BVSF (Ours)")

# ═══════════════════════════════════════════════════════════════════════
#  Colours  —  cohesive muted palette, BVSF in burnt orange
# ═══════════════════════════════════════════════════════════════════════
palette = [
    "#5B9BD5",   # PanCSCNet    medium blue
    "#7EA7C4",   # FusionNet    muted steel blue
    "#8FBC8F",   # LAGConv      muted sage
    "#D4B872",   # RWKVFusion   muted gold
    "#C9957B",   # DCFNet       muted terracotta
    "#B09DB8",   # PanDiff      muted lavender
    "#9EA7AD",   # ZSPan        muted slate
    "#D35400",   # BVSF (Ours)  rich burnt orange
]

# ═══════════════════════════════════════════════════════════════════════
#  Bubble sizing
# ═══════════════════════════════════════════════════════════════════════
SCALE = 900
areas = SCALE * np.sqrt(params)

# ═══════════════════════════════════════════════════════════════════════
#  Figure
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.8, 5.5))
ax.set_facecolor("white")

# ═══════════════════════════════════════════════════════════════════════
#  Scatter  —  subtle darker borders, smooth anti-aliased edges
# ═══════════════════════════════════════════════════════════════════════
for i in range(n):
    ax.scatter(flops[i], hqnr[i], s=areas[i],
               c=palette[i], marker="o", alpha=0.82,
               edgecolors="#aaaaaa", linewidth=0.5,
               zorder=3, rasterized=False)

# ═══════════════════════════════════════════════════════════════════════
#  Labels  —  dark grey, hand-tuned offsets
# ═══════════════════════════════════════════════════════════════════════
labels_manual = [
    (names.index("PanCSCNet"),     0,  16,  "center", "bottom"),
    (names.index("FusionNet"),    14,   6,  "left",   "center"),
    (names.index("LAGConv"),      14,   0,  "left",   "center"),
    (names.index("RWKVFusion"),    0,  24,  "center", "bottom"),
    (names.index("DCFNet"),       24,   0,  "left",   "center"),
    (names.index("PanDiff"),     -52,   0,  "right",  "center"),
    (names.index("ZSPan"),        14,   0,  "left",   "center"),
]

for i, dx, dy, ha, va in labels_manual:
    ax.annotate(names[i],
                xy=(flops[i], hqnr[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=15, fontweight="bold",
                color="#555555", ha=ha, va=va,
                zorder=8)

# ═══════════════════════════════════════════════════════════════════════
#  BVSF (Ours)  —  burnt orange, bold, curved arrow, white bbox
# ═══════════════════════════════════════════════════════════════════════
ax.annotate("Proposed",
            xy=(flops[idx_o], hqnr[idx_o]),
            xytext=(28, 12),
            textcoords="offset points",
            fontsize=15, fontweight="bold",
            color="#D35400",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="none", alpha=0.72),
            arrowprops=dict(arrowstyle="->", color="#999999",
                            lw=0.7, connectionstyle="arc3,rad=0.15"),
            zorder=10)

# ═══════════════════════════════════════════════════════════════════════
#  Size guide  —  lower-left, clean aligned circles + text
# ═══════════════════════════════════════════════════════════════════════
ref_params  = [10.0, 1.0, 0.01]
ref_labels  = ["10", "1", "0.01"]
GUIDE_SCALE = 0.20

gx_circle  = 0.055
gx_label   = 0.110
gy_start   = 0.190

for j, (rp, rl) in enumerate(zip(ref_params, ref_labels)):
    r_area = GUIDE_SCALE * SCALE * np.sqrt(rp)
    yc = gy_start - j * 0.070
    ax.scatter([gx_circle], [yc], s=r_area,
               c="#999999", marker="o", alpha=0.70,
               edgecolors="#666666", linewidth=0.50,
               transform=ax.transAxes, zorder=15)
    ax.text(gx_label, yc, rl + " M",
            transform=ax.transAxes, fontsize=11,
            fontweight="bold", color="#333333",
            ha="left", va="center", zorder=15)

ax.text(0.046, gy_start + 0.095, "Bubble Size",
        transform=ax.transAxes, fontsize=10,
        fontweight="bold", color="#333333",
        ha="left", va="bottom", zorder=15)
ax.text(0.046, gy_start + 0.050, "Parameters (M)",
        transform=ax.transAxes, fontsize=9,
        fontweight="normal", color="#555555",
        ha="left", va="bottom", zorder=15)

# ═══════════════════════════════════════════════════════════════════════
#  Axes  —  dotted grid, clean ticks, generous label padding
# ═══════════════════════════════════════════════════════════════════════
ax.set_xscale("log")
ax.set_xlim(0.5, 90)
ax.set_ylim(0.910, 0.976)
ax.set_xlabel("FLOPs (G)  —  log scale", labelpad=14)
ax.set_ylabel("HQNR", labelpad=18)

ax.set_xticks([0.5, 1, 3, 10, 30])
ax.set_xticklabels(["0.5","1","3","10","30"])
ax.xaxis.set_minor_locator(ticker.NullLocator())

ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.45, color="#cccccc")
ax.set_axisbelow(True)

for sp in ax.spines.values():
    sp.set_edgecolor("#cccccc")
    sp.set_linewidth(0.5)

# ═══════════════════════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════════════════════
fig.tight_layout(pad=0.8)

out = "/media/zouhe/Elements/zspan/zup/bubble_chart_flops_hqnr"
fig.savefig(out + ".pdf")
fig.savefig(out + ".png")
print(f"Saved: {out}.pdf  &  .png")
