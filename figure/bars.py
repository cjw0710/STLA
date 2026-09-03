import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_bars():
    # Style: Times New Roman, thicker fonts
    mpl.rcParams.update({
        'font.size': 24,
        'axes.titlesize': 28,
        'axes.labelsize': 28,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 13,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
    })

    # Data (order: twitter, douban, Android, Christianity)
    datasets = ['Twitter', 'Douban', 'Android', 'Christianity']
    memory_mb = [14119, 9603, 2399, 1913]
    samples_per_sec = [95.2411, 102.8244, 202.4546, 217.9007]
    latency_ms = [1000.0 / v if v != 0 else float('inf') for v in samples_per_sec]

    x = range(len(datasets))

    # ===== Manual Y-axis controls (set by you) =====
    # Set limits as (min, max) or leave as None to auto-scale
    MEM_LIM = (0, 16000)       # e.g., (0, 16000)
    MEM_TICKS = [3000, 6000, 9000, 12000, 15000]     # e.g., [0, 2000, 4000, ..., 16000]
    LAT_LIM = (0, 13)       # e.g., (3, 12)
    LAT_TICKS = [2, 4, 6, 8, 10,12]     # e.g., [4, 6, 8, 10, 12]
    # ===============================================

    # -------- Combined figure: two equal subplots side-by-side --------
    fig, (ax_mem, ax_lat) = plt.subplots(
        1, 2, figsize=(14, 4), constrained_layout=True,
        gridspec_kw={'width_ratios': [1, 1]}
    )

    # Memory (red)
    ax_mem.bar(x, memory_mb, color='#d62728', width=0.5)
    ax_mem.set_xticks(list(x))
    ax_mem.set_xticklabels(datasets)
    ax_mem.set_ylabel('Memory (MB)')
    # ax_mem.set_title('Memory')
    ax_mem.grid(True, axis='y', linestyle='--', color='#9e9e9e', alpha=0.4)
    if MEM_LIM is not None:
        ax_mem.set_ylim(*MEM_LIM)
    if MEM_TICKS is not None:
        ax_mem.set_yticks(MEM_TICKS)

    # Latency (blue)
    ax_lat.bar(x, latency_ms, color='#1f77b4', width=0.5)
    ax_lat.set_xticks(list(x))
    ax_lat.set_xticklabels(datasets)
    ax_lat.set_ylabel('Latency (ms/sample)')
    # ax_lat.set_title('Latency')
    ax_lat.grid(True, axis='y', linestyle='--', color='#9e9e9e', alpha=0.4)
    if LAT_LIM is not None:
        ax_lat.set_ylim(*LAT_LIM)
    if LAT_TICKS is not None:
        ax_lat.set_yticks(LAT_TICKS)

    # Thicker spines
    for ax in (ax_mem, ax_lat):
        for spine in ['left', 'bottom', 'right', 'top']:
            ax.spines[spine].set_linewidth(1.5)

    out_path = os.path.join(os.path.dirname(__file__), 'efficiency.pdf')
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    out = plot_bars()
    print(f'Saved figure to: {out}')
