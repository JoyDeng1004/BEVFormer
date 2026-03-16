"""
Experiment 1: Visualize TSA Sampling Points on BEV plane.

Usage:
    python tools/visualize_exp1.py

Reads from: debug_outputs/exp1_sampling_locations/frame*_layer*.pt
Outputs to:  debug_outputs/exp1_sampling_locations/figures/
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import glob

DATA_DIR = 'debug_outputs/exp1_sampling_locations'
FIG_DIR = os.path.join(DATA_DIR, 'figures')
BEV_H, BEV_W = 200, 200

# 8 distinct colors for 8 heads
HEAD_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]
HEAD_MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']

# Representative query indices
QUERY_INDICES = {
    'center': 20100,       # row=100, col=100
    'top_left': 0,         # row=0, col=0
    'top_right': 199,      # row=0, col=199
    'bottom_left': 39800,  # row=199, col=0
    'bottom_right': 39999, # row=199, col=199
    'mid_left': 20000,     # row=100, col=0
    'mid_right': 20199,    # row=100, col=199
}


def load_data(frame_idx, layer_idx):
    path = os.path.join(DATA_DIR, f'frame{frame_idx}_layer{layer_idx}.pt')
    if not os.path.exists(path):
        print(f'  [WARN] {path} not found, skipping.')
        return None
    return torch.load(path, map_location='cpu')


def query_idx_to_rc(idx, bev_w=200):
    return idx // bev_w, idx % bev_w


def plot_single_query(ax, sampling_locs, ref_pts, query_idx, title):
    """Plot sampling points for one query on one BEV side (history or current)."""
    ax.set_xlim(0, BEV_W)
    ax.set_ylim(BEV_H, 0)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x (col)')
    ax.set_ylabel('y (row)')
    ax.set_xticks(np.arange(0, BEV_W + 1, 50))
    ax.set_yticks(np.arange(0, BEV_H + 1, 50))
    ax.grid(True, alpha=0.2)

    # Reference point (red X)
    ref = ref_pts[query_idx, 0, :]  # (2,) normalized [0,1]
    ref_x = ref[0].item() * BEV_W
    ref_y = ref[1].item() * BEV_H
    ax.plot(ref_x, ref_y, 'rx', markersize=14, markeredgewidth=3, zorder=10)

    # Query's own grid position (green +)
    q_row, q_col = query_idx_to_rc(query_idx)
    ax.plot(q_col + 0.5, q_row + 0.5, 'g+', markersize=12, markeredgewidth=2, zorder=10)

    # Sampling points: 8 heads x 4 points
    sl = sampling_locs[query_idx]  # (num_heads, 1, num_points, 2)
    for h in range(sl.shape[0]):
        for p in range(sl.shape[2]):
            sx = sl[h, 0, p, 0].item() * BEV_W
            sy = sl[h, 0, p, 1].item() * BEV_H
            ax.plot(sx, sy, marker=HEAD_MARKERS[h], color=HEAD_COLORS[h],
                    markersize=6, markeredgewidth=1, markeredgecolor='black',
                    alpha=0.8, zorder=5)


def make_legend():
    elements = [
        plt.Line2D([0], [0], marker='x', color='red', linestyle='None',
                   markersize=10, markeredgewidth=2, label='Reference Point'),
        plt.Line2D([0], [0], marker='+', color='green', linestyle='None',
                   markersize=10, markeredgewidth=2, label='Query Grid Pos'),
    ]
    for h in range(8):
        elements.append(
            plt.Line2D([0], [0], marker=HEAD_MARKERS[h], color=HEAD_COLORS[h],
                       linestyle='None', markersize=6, markeredgecolor='black',
                       label=f'Head {h}'))
    return elements


# ========================================
# Experiment 1a: Single query, history vs current
# ========================================
def visualize_query_comparison(frame_idx, layer_idx, query_name, query_idx):
    data = load_data(frame_idx, layer_idx)
    if data is None:
        return

    sl = data['sampling_locations']   # (bs*2, 40000, 8, 1, 4, 2)
    rp = data['reference_points']     # (bs*2, 40000, 1, 2)
    has_hist = data['has_history']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    hist_label = 'History BEV (self-copy)' if not has_hist else 'History BEV (real prev)'
    plot_single_query(ax1, sl[0], rp[0], query_idx, hist_label)
    plot_single_query(ax2, sl[1], rp[1], query_idx, 'Current BEV')

    fig.legend(handles=make_legend(), loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    q_row, q_col = query_idx_to_rc(query_idx)
    fig.suptitle(
        f'Exp1: Sampling Points | Frame {frame_idx} (has_history={has_hist}) | '
        f'Layer {layer_idx} | Query "{query_name}" idx={query_idx} (r={q_row},c={q_col})',
        fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    fname = f'frame{frame_idx}_layer{layer_idx}_{query_name}.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ========================================
# Experiment 1b: All queries overview
# ========================================
def visualize_all_queries_overview(frame_idx, layer_idx):
    data = load_data(frame_idx, layer_idx)
    if data is None:
        return

    sl = data['sampling_locations']
    rp = data['reference_points']
    has_hist = data['has_history']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    query_colors = plt.cm.tab10(np.linspace(0, 1, len(QUERY_INDICES)))

    for ax, side_idx, side_name in [(ax1, 0, 'History'), (ax2, 1, 'Current')]:
        sl_side = sl[side_idx]
        rp_side = rp[side_idx]

        ax.set_xlim(0, BEV_W)
        ax.set_ylim(BEV_H, 0)
        ax.set_aspect('equal')
        ax.set_title(f'{side_name} BEV', fontsize=11)
        ax.set_xlabel('x (col)')
        ax.set_ylabel('y (row)')
        ax.grid(True, alpha=0.15)

        for i, (qname, qidx) in enumerate(QUERY_INDICES.items()):
            ref = rp_side[qidx, 0, :]
            ref_x = ref[0].item() * BEV_W
            ref_y = ref[1].item() * BEV_H
            ax.plot(ref_x, ref_y, 'x', color=query_colors[i], markersize=12,
                    markeredgewidth=3, zorder=10)

            sq = sl_side[qidx]  # (8, 1, 4, 2)
            for h in range(sq.shape[0]):
                for p in range(sq.shape[2]):
                    sx = sq[h, 0, p, 0].item() * BEV_W
                    sy = sq[h, 0, p, 1].item() * BEV_H
                    ax.plot(sx, sy, '.', color=query_colors[i], markersize=3,
                            alpha=0.6, zorder=5)
                    ax.plot([ref_x, sx], [ref_y, sy], '-', color=query_colors[i],
                            alpha=0.15, linewidth=0.5, zorder=3)

    legend_elements = []
    for i, (qname, qidx) in enumerate(QUERY_INDICES.items()):
        r, c = query_idx_to_rc(qidx)
        legend_elements.append(
            mpatches.Patch(color=query_colors[i], label=f'{qname} (r={r},c={c})'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'Exp1: All Queries Overview | Frame {frame_idx} (has_history={has_hist}) | Layer {layer_idx}',
        fontsize=11)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    fname = f'frame{frame_idx}_layer{layer_idx}_overview.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ========================================
# Experiment 1c: Layer 0 vs Layer 5 comparison
# ========================================
def visualize_layer_comparison(query_name, query_idx, frame_idx=1):
    data_l0 = load_data(frame_idx, 0)
    data_l5 = load_data(frame_idx, 5)
    if data_l0 is None or data_l5 is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for row, (data, lidx) in enumerate([(data_l0, 0), (data_l5, 5)]):
        sl = data['sampling_locations']
        rp = data['reference_points']
        for col, (side_idx, side_name) in enumerate([(0, 'History'), (1, 'Current')]):
            plot_single_query(axes[row, col], sl[side_idx], rp[side_idx],
                            query_idx, f'Layer {lidx} - {side_name} BEV')

    fig.suptitle(
        f'Exp1c: Layer Comparison | Frame {frame_idx} | '
        f'Query "{query_name}" idx={query_idx}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f'layer_compare_frame{frame_idx}_{query_name}.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ========================================
# Numerical statistics
# ========================================
def print_stats(frame_idx, layer_idx):
    data = load_data(frame_idx, layer_idx)
    if data is None:
        return

    sl = data['sampling_locations']
    rp = data['reference_points']
    has_hist = data['has_history']

    print(f'\n--- Frame {frame_idx}, Layer {layer_idx}, has_history={has_hist} ---')
    print(f'  sampling_locations shape: {list(sl.shape)}, range: [{sl.min():.4f}, {sl.max():.4f}]')
    print(f'  reference_points shape:   {list(rp.shape)}, range: [{rp.min():.4f}, {rp.max():.4f}]')

    rp_b = rp[:, :, None, :, None, :]
    offsets = sl - rp_b
    offset_dist = offsets.norm(dim=-1)

    for side_idx, side_name in [(0, 'History'), (1, 'Current')]:
        d = offset_dist[side_idx]
        d_px = d * 200
        print(f'  [{side_name}] offset: mean={d_px.mean():.2f}px, '
              f'std={d_px.std():.2f}px, max={d_px.max():.2f}px')

    rp_diff = (rp[0] - rp[1]).norm(dim=-1)
    print(f'  ref_point diff (hist-curr): mean={rp_diff.mean()*200:.2f}px, '
          f'max={rp_diff.max()*200:.2f}px')


# ========================================
# Main
# ========================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'frame*_layer*.pt')))
    if not files:
        print(f'No data found in {DATA_DIR}/.')
        print('Run: TSA_DEBUG=1 bash tools/debug_tsa.sh')
        return

    print(f'Found {len(files)} data files.')

    available = set()
    for f in files:
        base = os.path.basename(f).replace('.pt', '')
        parts = base.split('_')
        frame = int(parts[0].replace('frame', ''))
        layer = int(parts[1].replace('layer', ''))
        available.add((frame, layer))

    frames = sorted(set(f for f, l in available))
    layers = sorted(set(l for f, l in available))
    print(f'Frames: {frames}, Layers: {layers}')

    # --- Stats ---
    print('\n' + '=' * 60)
    print('STATISTICS')
    print('=' * 60)
    for fi in frames:
        for li in [layers[0], layers[-1]]:
            if (fi, li) in available:
                print_stats(fi, li)

    # --- Per-query detailed plots ---
    print('\n' + '=' * 60)
    print('PER-QUERY PLOTS')
    print('=' * 60)
    key_queries = ['center', 'top_left', 'bottom_right']
    for fi in frames:
        for li in [0, 5]:
            if (fi, li) not in available:
                continue
            for qname in key_queries:
                visualize_query_comparison(fi, li, qname, QUERY_INDICES[qname])

    # --- Overview plots ---
    print('\n' + '=' * 60)
    print('OVERVIEW PLOTS')
    print('=' * 60)
    for fi in frames:
        for li in [0, 5]:
            if (fi, li) in available:
                visualize_all_queries_overview(fi, li)

    # --- Layer comparison ---
    print('\n' + '=' * 60)
    print('LAYER COMPARISON')
    print('=' * 60)
    for fi in frames:
        if (fi, 0) in available and (fi, 5) in available:
            for qname in ['center', 'top_left']:
                visualize_layer_comparison(qname, QUERY_INDICES[qname], fi)

    print(f'\nAll figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
