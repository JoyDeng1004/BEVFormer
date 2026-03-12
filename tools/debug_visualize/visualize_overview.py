"""
Visualization 3: End-to-end overview — unified view of SCA + TSA pipeline.

Layout:
  Left:   6 camera images (2x3) + GT 3D bbox projection + SCA sampling points
  Center: Current frame BEV + GT boxes + query positions + TSA sampling traces
  Right:  Previous frame BEV + TSA history sampling positions

All selected queries use consistent color coding across panels.

Supports --compare for warp before/after comparison.

Usage:
    python tools/debug_visualize/visualize_overview.py [--frame 1] [--layer 0]
    python tools/debug_visualize/visualize_overview.py --warp_dx 1.0 --compare

Reads from:
    debug_outputs/exp1_sampling_locations/
    debug_outputs/exp3_reference_points/
    debug_outputs/exp6_sca/
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
    BEV_H, BEV_W, PC_RANGE, CAM_NAMES, CAM_LAYOUT,
    HEAD_COLORS, HEAD_MARKERS, QUERY_COLORS,
    physical_to_bev, query_idx_to_rc,
    get_representative_query_indices,
    get_gt_and_images, load_calib, build_lidar2img,
    project_boxes_to_image, style_bev_ax, draw_bev_boxes,
    mark_queries_on_bev,
    make_se2_warp, apply_warp_to_bev_points, add_warp_argparse,
)

EXP1_DIR = 'debug_outputs/exp1_sampling_locations'
EXP3_DIR = 'debug_outputs/exp3_reference_points'
SCA_DIR = 'debug_outputs/exp6_sca'
FIG_DIR = 'debug_outputs/overview/figures'


# ============================================================
# Data Loading
# ============================================================
def load_exp1(frame_idx, layer_idx):
    path = os.path.join(EXP1_DIR, f'frame{frame_idx}_layer{layer_idx}.pt')
    return torch.load(path, map_location='cpu') if os.path.exists(path) else None


def load_exp3(frame_idx):
    path = os.path.join(EXP3_DIR, f'frame{frame_idx}.pt')
    return torch.load(path, map_location='cpu') if os.path.exists(path) else None


def load_sca(frame_idx, layer_idx):
    path = os.path.join(SCA_DIR, f'frame{frame_idx}_layer{layer_idx}.pt')
    return torch.load(path, map_location='cpu') if os.path.exists(path) else None


# ============================================================
# Main overview figure
# ============================================================
def plot_overview(frame_idx, layer_idx, warp=None):
    """
    Full pipeline overview:
      Left (2x3):  Camera images + GT 3D bbox + SCA ref points
      Center:      Current BEV + GT boxes + queries + TSA current sampling
      Right:       History BEV + TSA history sampling
    """
    gt_boxes, sample_info, cam_images = get_gt_and_images(frame_idx)
    queries = get_representative_query_indices()
    exp1 = load_exp1(frame_idx, layer_idx)
    exp3 = load_exp3(frame_idx)
    sca = load_sca(frame_idx, layer_idx)

    if exp1 is None:
        print(f'  [WARN] No exp1 data for frame {frame_idx}, layer {layer_idx}')
        return

    # --- Figure layout ---
    # Left: 2x3 cameras | Center: BEV current | Right: BEV history
    fig = plt.figure(figsize=(32, 12))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1.3, 1.3],
                           hspace=0.15, wspace=0.12)

    # ---- LEFT: Camera panels (2x3) ----
    ref_cam = sca['reference_points_cam'] if sca is not None else None
    bev_mask = sca['bev_mask'] if sca is not None else None

    for ri, cam_row in enumerate(CAM_LAYOUT):
        for ci, cam_name in enumerate(cam_row):
            ax = fig.add_subplot(gs[ri, ci])
            cam_idx = CAM_NAMES.index(cam_name)
            img = cam_images.get(cam_name)
            if img is None:
                ax.axis('off')
                ax.set_title(cam_name.replace('CAM_', ''), fontsize=8,
                             color='gray')
                continue

            ax.imshow(img)
            h_img, w_img = img.shape[:2]

            # GT 3D bbox projection
            if gt_boxes is not None and sample_info is not None:
                cam_info = sample_info['cams'].get(cam_name)
                if cam_info is not None:
                    l2i = build_lidar2img(cam_info)
                    for p1, p2 in project_boxes_to_image(
                            gt_boxes, l2i, img.shape[:2]):
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                color='lime', linewidth=0.7, alpha=0.7)

            # SCA reference points for each query
            if ref_cam is not None and bev_mask is not None:
                for i, (qname, q) in enumerate(queries.items()):
                    qidx = q['idx']
                    color = QUERY_COLORS[i % len(QUERY_COLORS)]
                    mask_q = bev_mask[cam_idx, 0, qidx]
                    if mask_q.sum() == 0:
                        continue
                    rpts = ref_cam[cam_idx, 0, qidx]
                    for d in range(rpts.shape[0]):
                        if not mask_q[d]:
                            continue
                        rx = rpts[d, 0].item() * w_img
                        ry = rpts[d, 1].item() * h_img
                        ax.plot(rx, ry, 'o', color=color, markersize=5,
                                markeredgecolor='white', markeredgewidth=0.8,
                                zorder=10, alpha=0.9)

            ax.set_xlim(0, w_img)
            ax.set_ylim(h_img, 0)
            ax.axis('off')
            ax.set_title(cam_name.replace('CAM_', '').replace('_', ' '),
                         fontsize=9)

    # ---- CENTER: Current BEV ----
    ax_curr = fig.add_subplot(gs[:, 3])
    style_bev_ax(ax_curr, gt_boxes, title='Current Frame BEV')
    mark_queries_on_bev(ax_curr, queries)

    # TSA current-side sampling locations
    sl = exp1['sampling_locations']  # (2, 40000, 8, 1, 4, 2)
    rp = exp1['reference_points']    # (2, 40000, 1, 2)

    for i, (qname, q) in enumerate(queries.items()):
        qidx = q['idx']
        color = QUERY_COLORS[i % len(QUERY_COLORS)]

        # Current side (index 1)
        sl_curr = sl[1, qidx]  # (8, 1, 4, 2)
        for h in range(min(sl_curr.shape[0], 8)):
            for p in range(sl_curr.shape[2]):
                sx = sl_curr[h, 0, p, 0].item() * BEV_W
                sy = sl_curr[h, 0, p, 1].item() * BEV_H
                ax_curr.plot(sx, sy, marker='.', color=color,
                             markersize=3, alpha=0.5, zorder=8)

    # Show ego motion shift arrows
    if exp3 is not None:
        ref_2d = exp3['ref_2d'][0]
        shift_ref_2d = exp3['shift_ref_2d'][0]
        for i, (qname, q) in enumerate(queries.items()):
            qidx = q['idx']
            color = QUERY_COLORS[i % len(QUERY_COLORS)]
            r = ref_2d[qidx, 0]
            s = shift_ref_2d[qidx, 0]
            rx, ry = r[0].item()*BEV_W, r[1].item()*BEV_H
            sx, sy = s[0].item()*BEV_W, s[1].item()*BEV_H
            if abs(rx - sx) > 0.1 or abs(ry - sy) > 0.1:
                ax_curr.annotate(
                    '', xy=(sx, sy), xytext=(rx, ry),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5, alpha=0.6))

    # ---- RIGHT: History BEV ----
    ax_hist = fig.add_subplot(gs[:, 4])
    has_hist = exp1['has_history']
    hist_tag = 'real prev' if has_hist else 'self-copy'
    style_bev_ax(ax_hist, gt_boxes,
                 title=f'History BEV ({hist_tag})')

    for i, (qname, q) in enumerate(queries.items()):
        qidx = q['idx']
        color = QUERY_COLORS[i % len(QUERY_COLORS)]

        # History ref point
        h_rp = rp[0, qidx, 0]
        hx, hy = h_rp[0].item() * BEV_W, h_rp[1].item() * BEV_H
        ax_hist.plot(hx, hy, 'x', color=color, markersize=12,
                     markeredgewidth=2, zorder=15)

        # History side sampling
        sl_hist = sl[0, qidx]  # (8, 1, 4, 2)
        for h in range(min(sl_hist.shape[0], 8)):
            for p in range(sl_hist.shape[2]):
                sx = sl_hist[h, 0, p, 0].item() * BEV_W
                sy = sl_hist[h, 0, p, 1].item() * BEV_H
                ax_hist.plot(sx, sy, marker=HEAD_MARKERS[h % 8],
                             color=HEAD_COLORS[h % 8], markersize=4,
                             markeredgecolor='black', markeredgewidth=0.3,
                             alpha=0.7, zorder=10)

        # Cross-panel line: current query -> history sampling center
        con = matplotlib.patches.ConnectionPatch(
            xyA=(q['col'], q['row']), xyB=(hx, hy),
            coordsA='data', coordsB='data',
            axesA=ax_curr, axesB=ax_hist,
            color=color, linewidth=1.2, alpha=0.4, linestyle='--')
        fig.add_artist(con)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, (qname, q) in enumerate(queries.items()):
        color = QUERY_COLORS[i % len(QUERY_COLORS)]
        legend_elements.append(
            Line2D([0], [0], marker='o', color=color, linestyle='None',
                   markersize=6, label=f'{qname} ({q["x_m"]:.0f},{q["y_m"]:.0f})m'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))

    warp_tag = ''
    if warp is not None:
        warp_tag = (f' | Warp: dx={warp["dx_m"]:.1f}m, '
                    f'dy={warp["dy_m"]:.1f}m, dθ={warp["dtheta_deg"]:.1f}°')
    fig.suptitle(
        f'End-to-End Overview | Frame {frame_idx}, Layer {layer_idx}{warp_tag}',
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    warp_fname = '' if warp is None else f'_warp{warp["dx_m"]:.1f}_{warp["dy_m"]:.1f}'
    fname = f'frame{frame_idx}_layer{layer_idx}_overview{warp_fname}.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')

# PLACEHOLDER_OVERVIEW_MAIN


# ============================================================
# Warp comparison: side-by-side original vs warped
# ============================================================
def plot_overview_warp_compare(frame_idx, layer_idx, warp):
    """Two overview figures stacked: original (top) vs warped (bottom)."""
    # Just generate both and let the user compare files
    print(f'  Generating original overview...')
    plot_overview(frame_idx, layer_idx, warp=None)
    print(f'  Generating warped overview...')
    plot_overview(frame_idx, layer_idx, warp=warp)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='End-to-End Overview Visualization')
    parser.add_argument('--frame', type=int, default=None)
    parser.add_argument('--layer', type=int, default=None)
    add_warp_argparse(parser)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    exp1_files = sorted(glob.glob(os.path.join(EXP1_DIR, 'frame*_layer*.pt')))
    if not exp1_files:
        print('No data found. Run: TSA_DEBUG=1 python tools/test.py ...')
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
            plot_overview(fi, li, warp=None)

            if args.compare and warp is not None:
                plot_overview_warp_compare(fi, li, warp)

    print(f'\nAll figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
