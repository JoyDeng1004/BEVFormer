"""
Visualization 1: SCA — BEV query sampling on camera images.

For each representative BEV query:
  - BEV panel: query position + physical coords + GT boxes
  - Camera panels: reference_points_cam (2D projection) + sampling_locations
    (with learned offsets), arrows showing offset direction
  - Feature change heatmap: L2 norm of query_after - query_before

Supports --compare for warp before/after comparison.

Usage:
    python tools/debug_visualize/visualize_sca.py [--frame 1] [--layer 0]
    python tools/debug_visualize/visualize_sca.py --warp_dx 1.0 --compare

Reads from:
    debug_outputs/exp6_sca/frame*_layer*.pt
    debug_outputs/scene_meta/
    data/nuscenes/nuscenes_infos_temporal_val.pkl
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
    physical_to_bev, bev_to_physical, query_idx_to_rc,
    get_representative_query_indices, bev_to_display, grid_to_display,
    get_gt_and_images, load_calib, build_lidar2img,
    project_boxes_to_image, style_bev_ax, draw_bev_boxes,
    mark_queries_on_bev,
    make_se2_warp, apply_warp_to_bev_points, add_warp_argparse,
)

SCA_DIR = 'debug_outputs/exp6_sca'
FIG_DIR = os.path.join(SCA_DIR, 'figures')


# ============================================================
# Data Loading
# ============================================================
def load_sca_data(frame_idx, layer_idx):
    path = os.path.join(SCA_DIR, f'frame{frame_idx}_layer{layer_idx}.pt')
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu')


# ============================================================
# Per-query SCA visualization
# ============================================================
def plot_sca_query(frame_idx, layer_idx, query_name, query_info, color,
                   warp=None):
    """Full SCA visualization for one query.

    Layout: BEV panel (left) + 6 camera panels (right 2x3)
    """
    data = load_sca_data(frame_idx, layer_idx)
    if data is None:
        print(f'  [WARN] No SCA data for frame{frame_idx}_layer{layer_idx}')
        return

    gt_boxes, sample_info, cam_images = get_gt_and_images(frame_idx)
    if not cam_images:
        print(f'  [WARN] No camera images for frame {frame_idx}')
        return

    qidx = query_info['idx']
    ref_cam = data['reference_points_cam']   # (num_cams, bs, num_query, D, 2)
    bev_mask = data['bev_mask']              # (num_cams, bs, num_query, D)
    indexes = data['indexes']                # list of per-cam index tensors
    sampling_locs = data['sampling_locations']  # (bs*num_cams, max_len, H, L, P, 2)
    sampling_offs = data['sampling_offsets']

    num_cams = ref_cam.shape[0]
    calib = load_calib(frame_idx)
    img_shape = calib['img_shape'] if calib else None

    # --- Figure layout ---
    fig = plt.figure(figsize=(28, 10))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1.5, 1, 1, 1],
                           hspace=0.2, wspace=0.15)

    # --- BEV panel ---
    ax_bev = fig.add_subplot(gs[:, 0])
    style_bev_ax(ax_bev, gt_boxes,
                 title=f'BEV — Query "{query_name}"\n'
                       f'({query_info["x_m"]:.1f}, {query_info["y_m"]:.1f}) m')
    ax_bev.plot(query_info['plot_x'], query_info['plot_y'], 'o', color=color,
                markersize=14, markeredgecolor='white', markeredgewidth=2,
                zorder=20)

    # Mark ref_3d pillar heights on BEV (they all project to same BEV cell)
    ax_bev.annotate(
        f'idx={qidx}\n({query_info["x_m"]:.0f},{query_info["y_m"]:.0f})m',
        (query_info['plot_x'], query_info['plot_y']),
        textcoords='offset points', xytext=(12, 8),
        fontsize=7, color=color, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # --- Camera panels ---
    for ri, cam_row in enumerate(CAM_LAYOUT):
        for ci, cam_name in enumerate(cam_row):
            ax = fig.add_subplot(gs[ri, 1 + ci])
            cam_idx = CAM_NAMES.index(cam_name)
            img = cam_images.get(cam_name)
            if img is None:
                ax.axis('off')
                continue

            ax.imshow(img)

            # Draw GT 3D boxes
            if gt_boxes is not None and sample_info is not None:
                cam_info = sample_info['cams'].get(cam_name)
                if cam_info is not None:
                    l2i = build_lidar2img(cam_info)
                    for p1, p2 in project_boxes_to_image(
                            gt_boxes, l2i, img.shape[:2]):
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                color='lime', linewidth=0.6, alpha=0.7)

            # Check if this query is visible in this camera
            mask_q = bev_mask[cam_idx, 0, qidx]  # (D,)
            if mask_q.sum() == 0:
                ax.set_title(f'{cam_name.replace("CAM_", "")}\n(not visible)',
                             fontsize=8, color='gray')
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(img.shape[0], 0)
                ax.axis('off')
                continue

            # Reference points (circles) — D z-anchors
            ref_pts = ref_cam[cam_idx, 0, qidx]  # (D, 2) normalized
            h_img, w_img = img.shape[:2]
            for d in range(ref_pts.shape[0]):
                if not mask_q[d]:
                    continue
                rx = ref_pts[d, 0].item() * w_img
                ry = ref_pts[d, 1].item() * h_img
                ax.plot(rx, ry, 'o', color=color, markersize=8,
                        markeredgecolor='white', markeredgewidth=1.5,
                        zorder=10, alpha=0.9)

            # Sampling locations — find this query in the rebatched index
            idx_tensor = indexes[cam_idx]  # valid query indices for this cam
            pos_in_rebatch = (idx_tensor == qidx).nonzero(as_tuple=True)
            if len(pos_in_rebatch[0]) > 0:
                rb_pos = pos_in_rebatch[0][0].item()
                # sampling_locs: (bs*num_cams, max_len, num_heads, num_levels, num_points, 2)
                sl = sampling_locs[cam_idx, rb_pos]  # (H, L, P, 2)
                num_heads = sl.shape[0]
                for h in range(min(num_heads, 8)):
                    for lvl in range(sl.shape[1]):
                        for p in range(sl.shape[2]):
                            sx = sl[h, lvl, p, 0].item() * w_img
                            sy = sl[h, lvl, p, 1].item() * h_img
                            ax.plot(sx, sy, marker=HEAD_MARKERS[h % 8],
                                    color=HEAD_COLORS[h % 8],
                                    markersize=4, markeredgewidth=0.5,
                                    markeredgecolor='black', alpha=0.7,
                                    zorder=8)

                # Draw arrows from ref_point to sampling points (first head only)
                for d in range(ref_pts.shape[0]):
                    if not mask_q[d]:
                        continue
                    rx = ref_pts[d, 0].item() * w_img
                    ry = ref_pts[d, 1].item() * h_img
                    for p in range(sl.shape[2]):
                        sx = sl[0, 0, p, 0].item() * w_img
                        sy = sl[0, 0, p, 1].item() * h_img
                        ax.annotate('', xy=(sx, sy), xytext=(rx, ry),
                                    arrowprops=dict(arrowstyle='->', color=color,
                                                    lw=0.8, alpha=0.5))

            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            ax.axis('off')
            ax.set_title(cam_name.replace('CAM_', '').replace('_', ' '),
                         fontsize=8)

    warp_tag = '' if warp is None else f'_warp{warp["dx_m"]:.1f}_{warp["dy_m"]:.1f}'
    fig.suptitle(
        f'SCA Sampling | Frame {frame_idx}, Layer {layer_idx} | '
        f'Query "{query_name}" ({query_info["x_m"]:.1f}, {query_info["y_m"]:.1f})m',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f'frame{frame_idx}_layer{layer_idx}_sca_{query_name}{warp_tag}.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')

# PLACEHOLDER_SCA_HEATMAP


# ============================================================
# Feature change heatmap
# ============================================================
def plot_feature_change(frame_idx, layer_idx):
    """Heatmap of L2 norm of (query_after - query_before) over BEV grid."""
    data = load_sca_data(frame_idx, layer_idx)
    if data is None:
        return

    q_before = data['query_before'][0]  # (40000, 256)
    q_after = data['query_after'][0]
    diff = (q_after - q_before).norm(dim=-1).numpy().reshape(BEV_H, BEV_W)
    diff_disp = bev_to_display(diff)

    gt_boxes, _, _ = get_gt_and_images(frame_idx)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(diff_disp, cmap='hot', origin='upper',
                   extent=[0, BEV_W, BEV_H, 0])
    style_bev_ax(ax, gt_boxes,
                 title=f'SCA Feature Change (L2 norm)\n'
                       f'Frame {frame_idx}, Layer {layer_idx}')
    queries = mark_queries_on_bev(ax)
    plt.colorbar(im, ax=ax, shrink=0.8, label='||after - before||₂')

    plt.tight_layout()
    fname = f'frame{frame_idx}_layer{layer_idx}_sca_feature_change.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Overview: all queries on BEV + cameras
# ============================================================
def plot_sca_overview(frame_idx, layer_idx):
    """All representative queries on one BEV + camera overview."""
    data = load_sca_data(frame_idx, layer_idx)
    if data is None:
        return

    gt_boxes, sample_info, cam_images = get_gt_and_images(frame_idx)
    queries = get_representative_query_indices()

    ref_cam = data['reference_points_cam']
    bev_mask = data['bev_mask']

    fig = plt.figure(figsize=(28, 10))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1.5, 1, 1, 1],
                           hspace=0.2, wspace=0.15)

    # BEV panel
    ax_bev = fig.add_subplot(gs[:, 0])
    style_bev_ax(ax_bev, gt_boxes,
                 title=f'BEV Overview — Frame {frame_idx}, Layer {layer_idx}')
    mark_queries_on_bev(ax_bev, queries)

    # Camera panels
    for ri, cam_row in enumerate(CAM_LAYOUT):
        for ci, cam_name in enumerate(cam_row):
            ax = fig.add_subplot(gs[ri, 1 + ci])
            cam_idx = CAM_NAMES.index(cam_name)
            img = cam_images.get(cam_name)
            if img is None:
                ax.axis('off')
                continue
            ax.imshow(img)

            if gt_boxes is not None and sample_info is not None:
                cam_info = sample_info['cams'].get(cam_name)
                if cam_info is not None:
                    l2i = build_lidar2img(cam_info)
                    for p1, p2 in project_boxes_to_image(
                            gt_boxes, l2i, img.shape[:2]):
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                color='lime', linewidth=0.6, alpha=0.7)

            h_img, w_img = img.shape[:2]
            for i, (qname, q) in enumerate(queries.items()):
                qidx = q['idx']
                color = QUERY_COLORS[i % len(QUERY_COLORS)]
                mask_q = bev_mask[cam_idx, 0, qidx]
                if mask_q.sum() == 0:
                    continue
                ref_pts = ref_cam[cam_idx, 0, qidx]
                for d in range(ref_pts.shape[0]):
                    if not mask_q[d]:
                        continue
                    rx = ref_pts[d, 0].item() * w_img
                    ry = ref_pts[d, 1].item() * h_img
                    ax.plot(rx, ry, 'o', color=color, markersize=6,
                            markeredgecolor='white', markeredgewidth=1,
                            zorder=10, alpha=0.9)

            ax.set_xlim(0, w_img)
            ax.set_ylim(h_img, 0)
            ax.axis('off')
            ax.set_title(cam_name.replace('CAM_', '').replace('_', ' '),
                         fontsize=8)

    fig.suptitle(
        f'SCA Overview | Frame {frame_idx}, Layer {layer_idx}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f'frame{frame_idx}_layer{layer_idx}_sca_overview.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Warp comparison
# ============================================================
def plot_sca_warp_compare(frame_idx, layer_idx, warp):
    """Side-by-side: original vs warped feature change heatmap."""
    data = load_sca_data(frame_idx, layer_idx)
    if data is None:
        return

    q_before = data['query_before'][0]
    q_after = data['query_after'][0]
    diff_orig = (q_after - q_before).norm(dim=-1).numpy().reshape(BEV_H, BEV_W)
    diff_orig_disp = bev_to_display(diff_orig)

    # Simulate warp effect: shift the BEV grid
    queries = get_representative_query_indices()
    gt_boxes, _, _ = get_gt_and_images(frame_idx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.imshow(diff_orig_disp, cmap='hot', origin='upper',
                     extent=[0, BEV_W, BEV_H, 0])
    style_bev_ax(ax1, gt_boxes, title='Original')
    mark_queries_on_bev(ax1, queries)

    # Warped: shift the heatmap in grid space, then rotate for display
    from scipy.ndimage import shift as ndshift
    dx_px = warp['dx_norm'] * BEV_W
    dy_px = warp['dy_norm'] * BEV_H
    diff_warped = ndshift(diff_orig, [dy_px, dx_px], order=1, mode='constant')
    diff_warped_disp = bev_to_display(diff_warped)

    im2 = ax2.imshow(diff_warped_disp, cmap='hot', origin='upper',
                     extent=[0, BEV_W, BEV_H, 0])
    style_bev_ax(ax2, gt_boxes,
                 title=f'Warped (dx={warp["dx_m"]:.1f}m, '
                       f'dy={warp["dy_m"]:.1f}m, '
                       f'dθ={warp["dtheta_deg"]:.1f}°)')
    mark_queries_on_bev(ax2, queries)

    for ax, im in [(ax1, im1), (ax2, im2)]:
        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(
        f'SCA Warp Comparison | Frame {frame_idx}, Layer {layer_idx}',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = (f'frame{frame_idx}_layer{layer_idx}_sca_warp_compare_'
             f'dx{warp["dx_m"]:.1f}_dy{warp["dy_m"]:.1f}.png')
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='SCA Sampling Visualization')
    parser.add_argument('--frame', type=int, default=None,
                        help='Frame index (default: all available)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Layer index (default: 0 and 5)')
    add_warp_argparse(parser)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(SCA_DIR, 'frame*_layer*.pt')))
    if not files:
        print(f'No SCA data found in {SCA_DIR}/.')
        print('Run: TSA_DEBUG=1 python tools/test.py ...')
        return

    available = set()
    for f in files:
        base = os.path.basename(f).replace('.pt', '')
        parts = base.split('_')
        frame = int(parts[0].replace('frame', ''))
        layer = int(parts[1].replace('layer', ''))
        available.add((frame, layer))

    frames = [args.frame] if args.frame is not None else sorted(set(f for f, l in available))
    layers_all = sorted(set(l for f, l in available))
    layers = [args.layer] if args.layer is not None else [l for l in [0, 5] if l in layers_all]

    queries = get_representative_query_indices()
    warp = None
    if args.warp_dx != 0 or args.warp_dy != 0 or args.warp_dtheta != 0:
        warp = make_se2_warp(args.warp_dx, args.warp_dy, args.warp_dtheta)

    print(f'Frames: {frames}, Layers: {layers}')
    print(f'Queries: {list(queries.keys())}')

    # Per-query detailed plots
    for fi in frames:
        for li in layers:
            if (fi, li) not in available:
                continue
            print(f'\n--- Frame {fi}, Layer {li} ---')
            for i, (qname, qinfo) in enumerate(queries.items()):
                color = QUERY_COLORS[i % len(QUERY_COLORS)]
                plot_sca_query(fi, li, qname, qinfo, color, warp=warp)

            plot_feature_change(fi, li)
            plot_sca_overview(fi, li)

            if args.compare and warp is not None:
                plot_sca_warp_compare(fi, li, warp)

    print(f'\nAll figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
