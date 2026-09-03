import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_efficiency():
    # Global style: larger fonts and thicker spines
    mpl.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        # Use Times New Roman
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
    })
    datasets = ['twitter', 'douban', 'Android', 'Christianity']
    memory_mb = [14119, 9603, 2399, 1913]
    samples_per_sec = [95.2411, 102.8244, 202.4546, 217.9007]
    # Convert to latency in ms per sample
    sec_per_sample = [1.0 / v if v != 0 else float('inf') for v in samples_per_sec]
    lat_ms_per_sample = [s * 1000.0 for s in sec_per_sample]
    x = list(range(len(datasets)))

    # ===== Manual Y-axis controls (set by you) =====
    # Set limits as a tuple (min, max) or leave as None to auto-place into bands
    # Left axis (sec/sample)
    Y1_LIM = (3, 12)        # e.g., (0.004, 0.012)
    Y1_TICKS = [4, 6, 8, 10, 12]
    # Right axis (Memory MB)
    Y2_LIM = (0, 16000)        # e.g., (0, 16000)
    Y2_TICKS = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
    # ===============================================

    # Single axes with twin y, but vertically separated bands
    fig, ax1 = plt.subplots(figsize=(8, 4))
    for spine in ['left', 'bottom', 'right', 'top']:
        ax1.spines[spine].set_linewidth(1.5)

    # Helper: compute ylim so that data spans a specific normalized band [b0, b1]
    def band_ylim(dmin, dmax, b0, b1, nonneg=False):
        rng = max(dmax - dmin, 1e-12)
        d = rng / (b1 - b0)
        low = dmin - b0 * d
        high = low + d
        if nonneg and low < 0:
            shift = -low
            low += shift
            high += shift
        return low, high

    # Left axis: sec/sample in lower band
    color1 = '#1f77b4'
    ax1.plot(
        x, lat_ms_per_sample,
        marker='o', markersize=8, markeredgewidth=0,
        linewidth=3.0, color=color1, label='Latency (ms/sample)'
    )
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Latency (ms/sample)', color=color1, labelpad=6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.grid(True, axis='y', linestyle='--', color='#9e9e9e', alpha=0.5)

    lmin, lmax = min(lat_ms_per_sample), max(lat_ms_per_sample)
    if Y1_LIM is not None:
        ax1.set_ylim(*Y1_LIM)
    else:
        ax1.set_ylim(*band_ylim(lmin, lmax, 0.08, 0.48))
    if Y1_TICKS is not None:
        ax1.set_yticks(Y1_TICKS)

    # Right axis: memory in upper band
    ax2 = ax1.twinx()
    for spine in ['right']:
        ax2.spines[spine].set_linewidth(1.5)
    color2 = '#d62728'
    ax2.plot(
        x, memory_mb,
        marker='s', markersize=8, markeredgewidth=0,
        linewidth=3.0, color=color2, label='memory (MB)'
    )
    ax2.set_ylabel('Memory (MB)', color=color2, labelpad=6)
    ax2.tick_params(axis='y', labelcolor=color2)
    rmin, rmax = min(memory_mb), max(memory_mb)
    if Y2_LIM is not None:
        ax2.set_ylim(*Y2_LIM)
    else:
        ax2.set_ylim(*band_ylim(rmin, rmax, 0.56, 0.96, nonneg=True))
    if Y2_TICKS is not None:
        ax2.set_yticks(Y2_TICKS)

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), 'efficency.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    return out_path


if __name__ == '__main__':
    path = plot_efficiency()
    print(f'Saved figure to: {path}')
