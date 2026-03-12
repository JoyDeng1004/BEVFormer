"""
Visualization 2: TSA — Temporal alignment between current and history BEV.

For the same set of representative BEV queries:
  1. Alignment panel: ref_2d vs shift_ref_2d + TSA sampling_locations on BEV
  2. Temporal comparison: current vs prev BEV feature PCA side-by-side with
     connecting lines showing TSA correspondence
  3. Fusion weights: per-queue attention concentration + bar chart per query

Supports --compare for warp before/after comparison.

Usage:
    python tools/debug_visualize/visualize_tsa_alignment.py [--frame 1] [--layer 0]
    python tools/debug_visualize/visualize_tsa_alignment.py --warp_dx 1.0 --compare

Reads from:
    debug_outputs/exp1_sampling_locations/
    debug_outputs/exp2_attention_weights/
    debug_outputs/exp3_reference_points/
    debug_outputs/exp4_fusion_output/
    debug_outputs/scene_meta/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob

from vis_utils import (
    BEV_H, BEV_W, PC_RANGE,
    HEAD_COLORS, HEAD_MARKERS, QUERY_COLORS,
    physical_to_bev, query_idx_to_rc,
    get_representative_query_indices,
    get_gt_and_images, style_bev_ax, mark_queries_on_bev,
    make_se2_warp, apply_warp_to_bev_points, add_warp_argparse,
)

EXP1_DIR = 'debug_outputs/exp1_sampling_locations'
EXP2_DIR = 'debug_outputs/exp2_attention_weights'
EXP3_DIR = 'debug_outputs/exp3_reference_points'
EXP4_DIR = 'debug_outputs/exp4_fusion_output'
FIG_DIR = 'debug_outputs/tsa_alignment/figures'


# ============================================================
# Data Loading
# ============================================================
def load_exp1(frame_idx, layer_idx):
    path = os.path.join(EXP1_DIR, f'frame{frame_idx}_layer{layer_idx}.pt')
    return torch.load(path, map_location='cpu') if os.path.exists(path) else None


def load_exp2(frame_idx, layer_idx):
    path = os.path.join(EXP2_DIR, f'frame{frame_idx}_layer{layer_idx}.pt')
    return torch.load(path, map_location='cpu') if os.path.exists(path) else None


def load_exp3(frame_idx):
    path = os.path.join(EXP3_DIR, f'frame{frame_idx}.pt')
    return torch.load(path, map_location='cpu') if os.path.exists(path) else None


def load_exp4(frame_idx, layer_idx):
    path = os.path.join(EXP4_DIR, f'frame{frame_idx}_layer{layer_idx}.pt')
    return torch.load(path, map_location='cpu') if os.path.exists(path) else None


# ============================================================
# Vis 2a: Alignment process — ref_2d vs shift_ref_2d + sampling
# ============================================================
def plot_alignment(frame_idx, layer_idx):
    """BEV plane showing ref_2d, shift_ref_2d, and TSA sampling locations."""
    exp3 = load_exp3(frame_idx)
    exp1 = load_exp1(frame_idx, layer_idx)
    if exp3 is None or exp1 is None:
        print(f'  [WARN] Missing data for frame {frame_idx}, layer {layer_idx}')
        return

    ref_2d = exp3['ref_2d'][0]           # (40000, 1, 2)
    shift_ref_2d = exp3['shift_ref_2d'][0]  # (40000, 1, 2)
    shift = exp3['shift'][0]             # (2,)
    has_prev = exp3['has_prev_bev']

    sl = exp1['sampling_locations']      # (bs*2, 40000, 8, 1, 4, 2)
    rp = exp1['reference_points']        # (bs*2, 40000, 1, 2)

    gt_boxes, _, _ = get_gt_and_images(frame_idx)
    queries = get_representative_query_indices()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: ref_2d vs shift_ref_2d with arrows
    style_bev_ax(ax1, gt_boxes,
                 title=f'Ego Motion Shift\nshift=({shift[0]:.4f}, {shift[1]:.4f})')

    for i, (qname, q) in enumerate(queries.items()):
        qidx = q['idx']
        color = QUERY_COLORS[i % len(QUERY_COLORS)]

        # ref_2d (blue circle)
        r2d = ref_2d[qidx, 0]
        rx, ry = r2d[0].item() * BEV_W, r2d[1].item() * BEV_H
        ax1.plot(rx, ry, 'o', color=color, markersize=10,
                 markeredgecolor='white', markeredgewidth=1.5, zorder=15)

        # shift_ref_2d (diamond)
        sr2d = shift_ref_2d[qidx, 0]
        sx, sy = sr2d[0].item() * BEV_W, sr2d[1].item() * BEV_H
        ax1.plot(sx, sy, 'D', color=color, markersize=8,
                 markeredgecolor='black', markeredgewidth=1, zorder=15)

        # Arrow from ref_2d to shift_ref_2d
        ax1.annotate('', xy=(sx, sy), xytext=(rx, ry),
                     arrowprops=dict(arrowstyle='->', color=color,
                                     lw=2, alpha=0.8))
        ax1.annotate(f'{qname}', (rx, ry),
                     textcoords='offset points', xytext=(8, -10),
                     fontsize=6, color=color, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                               alpha=0.6))

    ax1.legend(['○ ref_2d', '◇ shift_ref_2d'], fontsize=8, loc='upper right')

    # Right: TSA sampling locations on history BEV
    style_bev_ax(ax2, gt_boxes,
                 title=f'TSA Sampling on History BEV\n'
                       f'has_prev={has_prev}, Layer {layer_idx}')

    for i, (qname, q) in enumerate(queries.items()):
        qidx = q['idx']
        color = QUERY_COLORS[i % len(QUERY_COLORS)]

        # History side reference point
        rp_hist = rp[0, qidx, 0]  # (2,)
        hrx = rp_hist[0].item() * BEV_W
        hry = rp_hist[1].item() * BEV_H
        ax2.plot(hrx, hry, 'x', color=color, markersize=12,
                 markeredgewidth=2.5, zorder=15)

        # Sampling points (history side)
        sl_hist = sl[0, qidx]  # (8, 1, 4, 2)
        for h in range(sl_hist.shape[0]):
            for p in range(sl_hist.shape[2]):
                spx = sl_hist[h, 0, p, 0].item() * BEV_W
                spy = sl_hist[h, 0, p, 1].item() * BEV_H
                ax2.plot(spx, spy, marker=HEAD_MARKERS[h % 8],
                         color=HEAD_COLORS[h % 8], markersize=5,
                         markeredgecolor='black', markeredgewidth=0.5,
                         alpha=0.7, zorder=10)

    fig.suptitle(
        f'TSA Alignment | Frame {frame_idx}, Layer {layer_idx}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f'frame{frame_idx}_layer{layer_idx}_tsa_alignment.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')

# PLACEHOLDER_TSA_TEMPORAL


# ============================================================
# Vis 2b: Temporal comparison — current vs prev BEV feature PCA
# ============================================================
def _pca_rgb(features_flat):
    """(N, C) -> (N, 3) RGB via PCA."""
    centered = features_flat - features_flat.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    pca_3 = U[:, :3] * S[:3]
    for i in range(3):
        lo, hi = pca_3[:, i].min(), pca_3[:, i].max()
        pca_3[:, i] = (pca_3[:, i] - lo) / (hi - lo + 1e-8)
    return pca_3


def plot_temporal_comparison(frame_idx, layer_idx):
    """Side-by-side PCA of history vs current BEV features with TSA lines."""
    exp4 = load_exp4(frame_idx, layer_idx)
    exp1 = load_exp1(frame_idx, layer_idx)
    if exp4 is None:
        return

    # output_before_fusion: (num_query, embed_dims, bs, 2)
    obf = exp4['output_before_fusion']  # (40000, 256, 1, 2)
    hist_feat = obf[:, :, 0, 0].numpy()   # (40000, 256) — history
    curr_feat = obf[:, :, 0, 1].numpy()   # (40000, 256) — current

    hist_rgb = _pca_rgb(hist_feat).reshape(BEV_H, BEV_W, 3)
    curr_rgb = _pca_rgb(curr_feat).reshape(BEV_H, BEV_W, 3)

    gt_boxes, _, _ = get_gt_and_images(frame_idx)
    queries = get_representative_query_indices()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.imshow(hist_rgb, origin='upper', extent=[0, BEV_W, BEV_H, 0])
    style_bev_ax(ax1, gt_boxes, title='History BEV Features (PCA)')

    ax2.imshow(curr_rgb, origin='upper', extent=[0, BEV_W, BEV_H, 0])
    style_bev_ax(ax2, gt_boxes, title='Current BEV Features (PCA)')

    # Draw connecting lines for representative queries
    if exp1 is not None:
        rp = exp1['reference_points']  # (2, 40000, 1, 2)
        for i, (qname, q) in enumerate(queries.items()):
            qidx = q['idx']
            color = QUERY_COLORS[i % len(QUERY_COLORS)]

            # History ref point
            h_rp = rp[0, qidx, 0]
            hx, hy = h_rp[0].item() * BEV_W, h_rp[1].item() * BEV_H
            ax1.plot(hx, hy, 'o', color=color, markersize=10,
                     markeredgecolor='white', markeredgewidth=1.5, zorder=15)

            # Current ref point
            c_rp = rp[1, qidx, 0]
            cx, cy = c_rp[0].item() * BEV_W, c_rp[1].item() * BEV_H
            ax2.plot(cx, cy, 'o', color=color, markersize=10,
                     markeredgecolor='white', markeredgewidth=1.5, zorder=15)

            # Cross-panel connecting line (using figure coords)
            con = matplotlib.patches.ConnectionPatch(
                xyA=(hx, hy), xyB=(cx, cy),
                coordsA='data', coordsB='data',
                axesA=ax1, axesB=ax2,
                color=color, linewidth=1.5, alpha=0.6,
                linestyle='--')
            fig.add_artist(con)

    mark_queries_on_bev(ax1, queries, with_labels=False)
    mark_queries_on_bev(ax2, queries)

    fig.suptitle(
        f'TSA Temporal Comparison | Frame {frame_idx}, Layer {layer_idx}',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f'frame{frame_idx}_layer{layer_idx}_tsa_temporal.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')

# PLACEHOLDER_TSA_FUSION


# ============================================================
# Vis 2c: Fusion weights — attention concentration per queue
# ============================================================
def plot_fusion_weights(frame_idx, layer_idx):
    """
    Attention concentration maps + per-query bar chart.

    TSA uses per-queue softmax (each queue sums to 1.0) + hard mean fusion.
    So the meaningful metric is concentration (how sharp the attention is).
    """
    exp2 = load_exp2(frame_idx, layer_idx)
    if exp2 is None:
        return

    aw = exp2['attention_weights']  # (bs, 40000, 8, 2, 4)
    has_hist = exp2['has_history']
    aw0 = aw[0]  # (40000, 8, 2, 4)

    # Concentration: avg max weight per queue
    hist_conc = aw0[:, :, 0, :].max(dim=-1).values.mean(dim=1)  # (40000,)
    curr_conc = aw0[:, :, 1, :].max(dim=-1).values.mean(dim=1)

    hist_map = hist_conc.numpy().reshape(BEV_H, BEV_W)
    curr_map = curr_conc.numpy().reshape(BEV_H, BEV_W)

    gt_boxes, _, _ = get_gt_and_images(frame_idx)
    queries = get_representative_query_indices()

    fig = plt.figure(figsize=(22, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.2)

    # History concentration
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(hist_map, cmap='inferno', vmin=0.25, vmax=1.0,
                     origin='upper', extent=[0, BEV_W, BEV_H, 0])
    style_bev_ax(ax1, gt_boxes, title=f'History Queue Concentration')
    mark_queries_on_bev(ax1, queries, with_labels=False)
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    # Current concentration
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(curr_map, cmap='inferno', vmin=0.25, vmax=1.0,
                     origin='upper', extent=[0, BEV_W, BEV_H, 0])
    style_bev_ax(ax2, gt_boxes, title=f'Current Queue Concentration')
    mark_queries_on_bev(ax2, queries, with_labels=False)
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    # Bar chart: per-query, per-head weights
    ax3 = fig.add_subplot(gs[2])
    query_names = list(queries.keys())[:5]  # limit to 5 for readability
    x = np.arange(len(query_names))
    width = 0.35

    hist_vals = []
    curr_vals = []
    for qname in query_names:
        qidx = queries[qname]['idx']
        hist_vals.append(aw0[qidx, :, 0, :].max(dim=-1).values.mean().item())
        curr_vals.append(aw0[qidx, :, 1, :].max(dim=-1).values.mean().item())

    bars1 = ax3.bar(x - width/2, hist_vals, width, label='History',
                    color='#FF6B6B', alpha=0.8)
    bars2 = ax3.bar(x + width/2, curr_vals, width, label='Current',
                    color='#4ECDC4', alpha=0.8)

    ax3.set_ylabel('Avg Max Weight')
    ax3.set_title('Per-Query Attention Concentration')
    ax3.set_xticks(x)
    ax3.set_xticklabels(query_names, rotation=30, ha='right', fontsize=8)
    ax3.legend()
    ax3.set_ylim(0.2, 1.0)
    ax3.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5,
                label='uniform baseline')
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    hist_tag = 'self-copy' if not has_hist else 'real'
    fig.suptitle(
        f'TSA Fusion Weights | Frame {frame_idx} (history={hist_tag}), '
        f'Layer {layer_idx}\n'
        f'NOTE: per-queue softmax + hard mean → history_ratio always 0.5',
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    fname = f'frame{frame_idx}_layer{layer_idx}_tsa_fusion.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Warp comparison
# ============================================================
def plot_tsa_warp_compare(frame_idx, layer_idx, warp):
    """Compare shift_ref_2d with and without warp perturbation."""
    exp3 = load_exp3(frame_idx)
    if exp3 is None:
        return

    ref_2d = exp3['ref_2d'][0]              # (40000, 1, 2)
    shift_ref_2d = exp3['shift_ref_2d'][0]  # (40000, 1, 2)

    # Apply warp to shift_ref_2d
    warped_shift = apply_warp_to_bev_points(shift_ref_2d, warp)

    gt_boxes, _, _ = get_gt_and_images(frame_idx)
    queries = get_representative_query_indices()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Original
    style_bev_ax(ax1, gt_boxes, title='Original shift_ref_2d')
    for i, (qname, q) in enumerate(queries.items()):
        qidx = q['idx']
        color = QUERY_COLORS[i % len(QUERY_COLORS)]
        r = ref_2d[qidx, 0]
        s = shift_ref_2d[qidx, 0]
        ax1.plot(r[0].item()*BEV_W, r[1].item()*BEV_H, 'o', color=color,
                 markersize=8, markeredgecolor='white', zorder=15)
        ax1.plot(s[0].item()*BEV_W, s[1].item()*BEV_H, 'D', color=color,
                 markersize=6, markeredgecolor='black', zorder=15)
        ax1.annotate('', xy=(s[0].item()*BEV_W, s[1].item()*BEV_H),
                     xytext=(r[0].item()*BEV_W, r[1].item()*BEV_H),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Warped
    style_bev_ax(ax2, gt_boxes,
                 title=f'Warped shift_ref_2d\n'
                       f'(dx={warp["dx_m"]:.1f}m, dy={warp["dy_m"]:.1f}m, '
                       f'dθ={warp["dtheta_deg"]:.1f}°)')
    for i, (qname, q) in enumerate(queries.items()):
        qidx = q['idx']
        color = QUERY_COLORS[i % len(QUERY_COLORS)]
        r = ref_2d[qidx, 0]
        w = warped_shift[qidx, 0]
        ax2.plot(r[0].item()*BEV_W, r[1].item()*BEV_H, 'o', color=color,
                 markersize=8, markeredgecolor='white', zorder=15)
        ax2.plot(w[0].item()*BEV_W, w[1].item()*BEV_H, 'D', color=color,
                 markersize=6, markeredgecolor='black', zorder=15)
        ax2.annotate('', xy=(w[0].item()*BEV_W, w[1].item()*BEV_H),
                     xytext=(r[0].item()*BEV_W, r[1].item()*BEV_H),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    fig.suptitle(
        f'TSA Warp Comparison | Frame {frame_idx}, Layer {layer_idx}',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = (f'frame{frame_idx}_layer{layer_idx}_tsa_warp_compare_'
             f'dx{warp["dx_m"]:.1f}_dy{warp["dy_m"]:.1f}.png')
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='TSA Temporal Alignment Visualization')
    parser.add_argument('--frame', type=int, default=None)
    parser.add_argument('--layer', type=int, default=None)
    add_warp_argparse(parser)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    # Discover available data
    exp1_files = sorted(glob.glob(os.path.join(EXP1_DIR, 'frame*_layer*.pt')))
    if not exp1_files:
        print(f'No TSA data found. Run: TSA_DEBUG=1 python tools/test.py ...')
        return

    available = set()
    for f in exp1_files:
        base = os.path.basename(f).replace('.pt', '')
        parts = base.split('_')
        frame = int(parts[0].replace('frame', ''))
        layer = int(parts[1].replace('layer', ''))
        available.add((frame, layer))

    frames = [args.frame] if args.frame is not None else sorted(set(f for f, l in available))
    layers_all = sorted(set(l for f, l in available))
    layers = [args.layer] if args.layer is not None else [l for l in [0, 5] if l in layers_all]

    warp = None
    if args.warp_dx != 0 or args.warp_dy != 0 or args.warp_dtheta != 0:
        warp = make_se2_warp(args.warp_dx, args.warp_dy, args.warp_dtheta)

    print(f'Frames: {frames}, Layers: {layers}')

    for fi in frames:
        for li in layers:
            if (fi, li) not in available:
                continue
            print(f'\n--- Frame {fi}, Layer {li} ---')
            plot_alignment(fi, li)
            plot_temporal_comparison(fi, li)
            plot_fusion_weights(fi, li)

            if args.compare and warp is not None:
                plot_tsa_warp_compare(fi, li, warp)

    print(f'\nAll figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
