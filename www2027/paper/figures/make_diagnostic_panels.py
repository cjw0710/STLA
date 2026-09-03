from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "diagnostic_panels.pdf"
OUT_PNG = ROOT / "diagnostic_panels.png"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE = "#58AFC9"
ORANGE = "#F28C52"
GREEN = "#97B85A"
GOLD = "#E0AC23"
INK = "#333333"
GRID = "#D9DEE2"
BG = "#FAFAF8"


def style_axis(ax, grid_axis="x"):
    ax.set_facecolor(BG)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(length=2.2, width=0.6, colors=INK)


fig = plt.figure(figsize=(7.15, 1.84), constrained_layout=False)
gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[1.08, 1.02, 1.20],
    left=0.075,
    right=0.985,
    top=0.84,
    bottom=0.25,
    wspace=0.42,
)

# (a) Validation ablation gains over each frozen anchor.
ax = fig.add_subplot(gs[0, 0])
configs = ["Historical only", "No environment", "No prefix", "Full STLA"]
dy_gain = np.array([0.025, 0.282, 0.302, 0.297])
dis_gain = np.array([0.244, 0.484, 0.457, 0.488])
y = np.arange(len(configs))
h = 0.32
ax.barh(y - h / 2, dy_gain, h, color=ORANGE, label="DyHGCN", edgecolor="white", linewidth=0.35)
ax.barh(y + h / 2, dis_gain, h, color=BLUE, label="DisenIDP", edgecolor="white", linewidth=0.35)
for yi, value in zip(y - h / 2, dy_gain):
    ax.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=5.8, color=INK)
for yi, value in zip(y + h / 2, dis_gain):
    ax.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=5.8, color=INK)
ax.set_yticks(y, configs)
ax.invert_yaxis()
ax.set_xlim(0, 0.56)
ax.set_xticks([0.0, 0.2, 0.4])
ax.set_xlabel("Validation gain (percentage points)")
ax.set_title("(a) Component gain over anchor", fontweight="bold", loc="left")
ax.legend(frameon=False, ncol=1, loc="upper right", handlelength=1.2, borderaxespad=0.3)
style_axis(ax, "x")

# (b) Protected opportunities retained by the fused ranking.
ax = fig.add_subplot(gs[0, 1])
cutoffs = np.array([10, 50, 100])
dy_hits = np.array([764, 2431, 4327])
dis_hits = np.array([1026, 3624, 6227])
x = np.arange(len(cutoffs))
w = 0.34
b1 = ax.bar(x - w / 2, dy_hits, w, color=ORANGE, label="DyHGCN", edgecolor="white", linewidth=0.35)
b2 = ax.bar(x + w / 2, dis_hits, w, color=BLUE, label="DisenIDP", edgecolor="white", linewidth=0.35)
for bars in (b1, b2):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 135,
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
            fontsize=5.6,
            rotation=0,
        )
ax.set_xticks(x, ["10", "50", "100"])
ax.set_xlabel("Cutoff $K$")
ax.set_ylabel("Protected anchor hits")
ax.set_ylim(0, 7200)
ax.set_yticks([0, 2000, 4000, 6000], ["0", "2k", "4k", "6k"])
ax.set_title("(b) Exact preservation audit", fontweight="bold", loc="left")
ax.text(
    0.5,
    0.97,
    "0 violations at every cutoff",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=6.0,
    fontweight="bold",
    color="#715300",
    bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFF5D6", edgecolor=GOLD, linewidth=0.65),
)
style_axis(ax, "y")

# (c) Accuracy gain before and after deterministic safety fusion.
ax = fig.add_subplot(gs[0, 2])
dy_adaptive = np.array([0.836, 0.842, 0.843])
dy_fused = np.array([0.705, 0.756, 0.763])
dis_adaptive = np.array([0.987, 0.978, 0.975])
dis_fused = np.array([0.896, 0.935, 0.935])
ax.plot(cutoffs, dy_adaptive, color=ORANGE, marker="o", markersize=3.4, linewidth=1.25, label="DyHGCN adaptive")
ax.plot(cutoffs, dy_fused, color=ORANGE, marker="o", markersize=3.4, linewidth=1.25, linestyle="--", label="DyHGCN fused")
ax.plot(cutoffs, dis_adaptive, color=BLUE, marker="s", markersize=3.2, linewidth=1.25, label="DisenIDP adaptive")
ax.plot(cutoffs, dis_fused, color=BLUE, marker="s", markersize=3.2, linewidth=1.25, linestyle="--", label="DisenIDP fused")
ax.set_xticks(cutoffs)
ax.set_xlim(5, 132)
ax.set_xlabel("Cutoff $K$")
ax.set_ylabel("Gain over anchor (percentage points)")
ax.set_ylim(0.64, 1.04)
ax.set_yticks([0.7, 0.8, 0.9, 1.0])
ax.set_title("(c) Accuracy-safety trade-off", fontweight="bold", loc="left")
for value, label, color in [
    (dis_adaptive[-1], "DisenIDP adaptive", BLUE),
    (dis_fused[-1], "DisenIDP fused", BLUE),
    (dy_adaptive[-1], "DyHGCN adaptive", ORANGE),
    (dy_fused[-1], "DyHGCN fused", ORANGE),
]:
    ax.text(104, value, label, va="center", ha="left", fontsize=5.6, color=color, fontweight="bold")
style_axis(ax, "y")

fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.015)
fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", pad_inches=0.015)
print(OUT_PDF)
print(OUT_PNG)
