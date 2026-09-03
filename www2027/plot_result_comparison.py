from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Libertinus Serif",
            "Linux Libertine O",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "font.size": 8.0,
        "axes.titlesize": 9.4,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

anchor_gray = "#AEB4BA"
stla_orange = "#F28B57"
dy_green = "#91B45B"
disen_blue = "#62ADC7"
grid = "#D9DDE1"
ink = "#25282B"

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(10.8, 2.55),
    gridspec_kw={"width_ratios": [1.30, 1.0], "wspace": 0.27},
)
fig.patch.set_facecolor("white")

# (a) The sealed test is shown on a zero-based axis to avoid exaggerating gains.
labels = ["DyHGCN\nMean", "DyHGCN\nWorst", "DisenIDP\nMean", "DisenIDP\nWorst"]
anchor = np.array([0.09868, 0.09344, 0.08846, 0.08563])
adapted = np.array([0.10631, 0.10109, 0.09781, 0.09477])
anchor_sd = np.array([0.00236, 0.00164, 0.00296, 0.00290])
adapted_sd = np.array([0.00242, 0.00206, 0.00353, 0.00368])
x = np.arange(len(labels))
w = 0.34

ax1.set_facecolor("#FFF5EE")
b1 = ax1.bar(
    x - w / 2,
    anchor,
    w,
    yerr=anchor_sd,
    capsize=2.2,
    color=anchor_gray,
    edgecolor="white",
    linewidth=0.7,
    error_kw={"elinewidth": 0.8, "ecolor": "#63686D", "capthick": 0.8},
    label="Frozen anchor",
    zorder=3,
)
b2 = ax1.bar(
    x + w / 2,
    adapted,
    w,
    yerr=adapted_sd,
    capsize=2.2,
    color=stla_orange,
    edgecolor="white",
    linewidth=0.7,
    error_kw={"elinewidth": 0.8, "ecolor": "#8A4827", "capthick": 0.8},
    label="STLA",
    zorder=3,
)
ax1.set_ylim(0, 0.122)
ax1.set_yticks(np.arange(0, 0.121, 0.02))
ax1.set_ylabel("MAP@100")
ax1.set_xticks(x, labels)
ax1.set_title("(a) Sealed MemeTracker test", loc="left", fontweight="bold", pad=5)
ax1.grid(axis="y", color=grid, linewidth=0.65, alpha=0.85, zorder=0)
ax1.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.1, columnspacing=1.0)
ax1.tick_params(axis="x", length=0)
ax1.spines["left"].set_color("#8A8F94")
ax1.spines["bottom"].set_color("#8A8F94")

for rect, value in zip(b1, anchor):
    ax1.text(
        rect.get_x() + rect.get_width() / 2,
        value - 0.0070,
        f"{value:.4f}",
        ha="center",
        va="center",
        fontsize=6.25,
        fontweight="bold",
        color="white",
    )
for rect, value, err in zip(b2, adapted, adapted_sd):
    ax1.text(
        rect.get_x() + rect.get_width() / 2,
        value - 0.0070,
        f"{value:.4f}*",
        ha="center",
        va="center",
        fontsize=6.25,
        fontweight="bold",
        color="white",
    )

# (b) Development evidence is separated and explicitly marked validation-only.
datasets = ["Christianity", "Android", "Douban", "Twitter"]
dy = np.array([0.00302, 0.00063, 0.00156, 0.00463])
disen = np.array([0.00328, 0.00065, 0.01080, 0.00526])
y = np.arange(len(datasets))
h = 0.31

ax2.set_facecolor("#F2F8FA")
d1 = ax2.barh(
    y - h / 2,
    dy,
    h,
    color=dy_green,
    edgecolor="white",
    linewidth=0.7,
    label="DyHGCN + STLA",
    zorder=3,
)
d2 = ax2.barh(
    y + h / 2,
    disen,
    h,
    color=disen_blue,
    edgecolor="white",
    linewidth=0.7,
    label="DisenIDP + STLA",
    zorder=3,
)
ax2.set_xlim(0, 0.0124)
ax2.set_xticks(np.arange(0, 0.0121, 0.002), [f"{v:.3f}" for v in np.arange(0, 0.0121, 0.002)])
ax2.set_yticks(y, datasets)
ax2.invert_yaxis()
ax2.set_xlabel(r"Validation $\Delta$MAP@100 over frozen anchor")
ax2.set_title("(b) Cross-dataset gains (validation only)", loc="left", fontweight="bold", pad=5)
ax2.grid(axis="x", color=grid, linewidth=0.65, alpha=0.85, zorder=0)
ax2.legend(loc="lower right", frameon=False, handlelength=1.2)
ax2.tick_params(axis="y", length=0)
ax2.spines["left"].set_color("#8A8F94")
ax2.spines["bottom"].set_color("#8A8F94")

for i, (rect, value) in enumerate(zip(d1, dy)):
    star = "" if i == 1 else "*"
    ax2.text(
        value + 0.00016,
        rect.get_y() + rect.get_height() / 2,
        f"{value:.4f}{star}",
        ha="left",
        va="center",
        fontsize=6.5,
        fontweight="bold" if star else "normal",
        color="#486321",
    )
for rect, value in zip(d2, disen):
    ax2.text(
        value + 0.00016,
        rect.get_y() + rect.get_height() / 2,
        f"{value:.4f}*",
        ha="left",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color="#245D70",
    )

ax2.text(
    0.99,
    0.965,
    "* exact paired $p=.03125$\nDyHGCN--Android: $p=.25$",
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=6.4,
    color=ink,
    bbox={"boxstyle": "round,pad=0.23", "facecolor": "white", "edgecolor": "#C9DCE2", "linewidth": 0.6},
)

fig.savefig(OUT / "result_comparison.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "result_comparison.png", dpi=240, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
