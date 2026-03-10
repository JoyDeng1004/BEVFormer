"""
Experiment 3: Visualize Reference Points Ego Motion Shift on BEV plane.

Shows how ego motion compensation shifts history reference points relative
to current reference points via arrow fields on the BEV grid.

Usage:
    python tools/debug_visualize/visualize_exp3.py

Reads from:
    debug_outputs/exp3_reference_points/frame*.pt
    debug_outputs/scene_meta/frame*.json           (optional, for GT overlay)
    data/nuscenes/nuscenes_infos_temporal_val.pkl   (optional, for GT overlay)

Outputs to:
    debug_outputs/exp3_reference_points/figures/
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import numpy as np
import os
import glob
import json
import pickle

# ============================================================
# Constants
# ============================================================
DATA_DIR = 'debug_outputs/exp3_reference_points'
SCENE_META_DIR = 'debug_outputs/scene_meta'
FIG_DIR = os.path.join(DATA_DIR, 'figures')
VAL_PKL = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
BEV_H, BEV_W = 200, 200
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Arrow subsampling: draw one arrow every STEP queries in each row/col
ARROW_STEP = 10

# BEV axis tick labels (pixel -> meters)
_BEV_TICKS = [0, 50, 100, 150, 200]
_BEV_TICK_LABELS = ['-51.2', '-25.6', '0', '25.6', '51.2']


# ============================================================
# Data Loading
# ============================================================
def load_data(frame_idx):
    path = os.path.join(DATA_DIR, f'frame{frame_idx}.pt')
    if not os.path.exists(path):
        print(f'  [WARN] {path} not found')
        return None
    return torch.load(path, map_location='cpu')


def load_scene_meta(frame_idx):
    path = os.path.join(SCENE_META_DIR, f'frame{frame_idx}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


_val_infos_cache = None

def load_nuscenes_info(sample_idx):
    global _val_infos_cache
    if sample_idx is None:
        return None
    if not os.path.exists(VAL_PKL):
        return None
    if _val_infos_cache is None:
        with open(VAL_PKL, 'rb') as f:
            _val_infos_cache = pickle.load(f)['infos']
        print(f'  Loaded {len(_val_infos_cache)} samples from val pkl')
    for info in _val_infos_cache:
        if info['token'] == sample_idx:
            return info
    return None


# ============================================================
# BEV GT Boxes
# ============================================================
def lidar_to_bev(x, y):
    col = (x - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * BEV_W
    row = (y - PC_RANGE[1]) / (PC_RANGE[4] - PC_RANGE[1]) * BEV_H
    return col, row


def get_bev_box_corners(gt_boxes):
    corners_list = []
    for box in gt_boxes:
        x, y, _, dx, dy, _, yaw = box[:7]
        hdx, hdy = dx / 2, dy / 2
        local = np.array([
            [+hdx, +hdy], [-hdx, +hdy],
            [-hdx, -hdy], [+hdx, -hdy],
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        world = (R @ local.T).T + np.array([x, y])
        cols, rows = lidar_to_bev(world[:, 0], world[:, 1])
        corners_list.append(np.stack([cols, rows], axis=1))
    return corners_list


def draw_bev_boxes(ax, gt_boxes, color='white', linewidth=1.2):
    if gt_boxes is None or len(gt_boxes) == 0:
        return
    for corners in get_bev_box_corners(gt_boxes):
        poly = Polygon(corners, closed=True, fill=False,
                       edgecolor=color, linewidth=linewidth, alpha=0.85)
        ax.add_patch(poly)


def _get_gt_boxes(frame_idx):
    meta = load_scene_meta(frame_idx)
    if meta is None:
        return None
    info = load_nuscenes_info(meta.get('sample_idx'))
    if info is None:
        return None
    return info.get('gt_boxes')


def _style_bev_ax(ax, gt_boxes=None):
    ax.plot(BEV_W / 2, BEV_H / 2, marker='+', color='lime',
            markersize=10, markeredgewidth=2, zorder=10)
    draw_bev_boxes(ax, gt_boxes)
    ax.set_xticks(_BEV_TICKS)
    ax.set_xticklabels(_BEV_TICK_LABELS, fontsize=8)
    ax.set_yticks(_BEV_TICKS)
    ax.set_yticklabels(_BEV_TICK_LABELS, fontsize=8)
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('y (m)', fontsize=9)


def _add_stats_text(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7,
            va='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))


# ============================================================
# Core: compute arrow field
# ============================================================
def compute_arrow_field(ref_2d, shift_ref_2d, step=ARROW_STEP):
    """
    Compute arrow origins and displacements in BEV pixel coordinates.

    Args:
        ref_2d:       (bs, 40000, 1, 2) — current reference points, normalized [0,1]
        shift_ref_2d: (bs, 40000, 1, 2) — shifted reference points for history
        step: subsample rate along each BEV axis

    Returns:
        origins: (N, 2)   — arrow start positions (x_px, y_px) on BEV grid
        deltas:  (N, 2)   — arrow displacements (dx_px, dy_px)
    """
    # Take batch 0
    ref = ref_2d[0, :, 0, :].numpy()        # (40000, 2) normalized
    sref = shift_ref_2d[0, :, 0, :].numpy()  # (40000, 2) normalized

    # Select subsampled grid positions
    rows = np.arange(0, BEV_H, step)
    cols = np.arange(0, BEV_W, step)
    indices = []
    for r in rows:
        for c in cols:
            indices.append(r * BEV_W + c)
    indices = np.array(indices)

    # Current ref in pixel coords
    ref_sub = ref[indices]     # (N, 2)  — (x_norm, y_norm)
    sref_sub = sref[indices]

    # Normalized -> pixel
    orig_x = ref_sub[:, 0] * BEV_W
    orig_y = ref_sub[:, 1] * BEV_H
    shift_x = sref_sub[:, 0] * BEV_W
    shift_y = sref_sub[:, 1] * BEV_H

    origins = np.stack([orig_x, orig_y], axis=1)
    # Arrow: from current position to where history looks
    deltas = np.stack([shift_x - orig_x, shift_y - orig_y], axis=1)

    return origins, deltas


# ============================================================
# Exp 3a: Arrow field (quiver plot)
# ============================================================
def plot_arrow_field(frame_idx):
    """
    Draw ego motion shift as an arrow field on the BEV grid.

    Each arrow points from the current reference point to the shifted
    (history-aligned) reference point. Arrow direction = ego motion direction,
    arrow length = ego motion magnitude.
    """
    data = load_data(frame_idx)
    if data is None:
        return

    ref_2d = data['ref_2d']
    shift_ref_2d = data['shift_ref_2d']
    shift = data['shift']
    has_prev = data['has_prev_bev']

    origins, deltas = compute_arrow_field(ref_2d, shift_ref_2d)
    mag = np.linalg.norm(deltas, axis=1)

    gt_boxes = _get_gt_boxes(frame_idx)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(0, BEV_W)
    ax.set_ylim(BEV_H, 0)  # y-axis inverted (row 0 at top)
    ax.set_aspect('equal')

    if mag.max() < 1e-4:
        # No shift (frame 0): just show the grid points
        ax.scatter(origins[:, 0], origins[:, 1], s=4, c='steelblue',
                   alpha=0.6, zorder=5)
        ax.text(0.5, 0.5, 'No ego motion shift\n(first frame or zero motion)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='gray', alpha=0.6)
    else:
        mag_mean = mag.mean()        
        q = ax.quiver(
            origins[:, 0], origins[:, 1],
            deltas[:, 0], deltas[:, 1],
            mag,
            cmap='coolwarm', scale=1, scale_units='xy', angles='xy',
            width=0.003, headwidth=3.5, headlength=4,
            zorder=5, alpha=0.85,
            clim=(0, mag_mean * 2),
        )
        cb = fig.colorbar(q, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label('Shift magnitude (pixels)', fontsize=9)

    _style_bev_ax(ax, gt_boxes)

    shift_val = shift[0].numpy()
    shift_px = shift_val * np.array([BEV_W, BEV_H])
    stats = (f'shift (norm): dx={shift_val[0]:.4f}, dy={shift_val[1]:.4f}\n'
             f'shift (px):   dx={shift_px[0]:.1f}, dy={shift_px[1]:.1f}\n'
             f'has_prev_bev: {has_prev}')
    _add_stats_text(ax, stats)

    prev_tag = 'no history' if not has_prev else 'has history'
    ax.set_title(
        f'Exp3: Ego Motion Shift Arrow Field\n'
        f'Frame {frame_idx} ({prev_tag}) | '
        f'Arrows: current ref -> history-aligned ref | step={ARROW_STEP}',
        fontsize=11)

    plt.tight_layout()
    fname = f'frame{frame_idx}_arrow_field.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 3b: Shift magnitude heatmap (per-query)
# ============================================================
def plot_shift_magnitude_heatmap(frame_idx):
    """
    200x200 heatmap of per-query shift magnitude (in pixels).

    Useful to see if shift is truly uniform (rigid ego motion)
    or has some spatial variation.
    """
    data = load_data(frame_idx)
    if data is None:
        return

    ref_2d = data['ref_2d']
    shift_ref_2d = data['shift_ref_2d']
    has_prev = data['has_prev_bev']

    # Compute per-query shift in pixel space
    ref = ref_2d[0, :, 0, :]       # (40000, 2)
    sref = shift_ref_2d[0, :, 0, :]  # (40000, 2)
    diff_px = (sref - ref) * torch.tensor([BEV_W, BEV_H], dtype=torch.float32)
    magnitude = diff_px.norm(dim=-1).numpy().reshape(BEV_H, BEV_W)

    gt_boxes = _get_gt_boxes(frame_idx)

    fig, ax = plt.subplots(figsize=(9, 8))

    if magnitude.max() < 1e-4:
        ax.imshow(np.zeros((BEV_H, BEV_W)), cmap='Blues', vmin=0, vmax=1,
                  origin='lower', extent=[0, BEV_W, 0, BEV_H])
        ax.text(0.5, 0.5, 'No shift (magnitude = 0 everywhere)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='gray', alpha=0.6)
    else:
        # colorbar range to [0, ceil(max)]
        vmax = np.ceil(magnitude.max())
        im = ax.imshow(magnitude, cmap='hot', origin='lower',
                       extent=[0, BEV_W, 0, BEV_H],
                       vmin=0, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label('Shift magnitude (pixels)', fontsize=9)

    _style_bev_ax(ax, gt_boxes)
    stats = (f'magnitude: mean={magnitude.mean():.2f}px, '
             f'std={magnitude.std():.4f}px\n'
             f'min={magnitude.min():.2f}px, max={magnitude.max():.2f}px')
    _add_stats_text(ax, stats)

    prev_tag = 'no history' if not has_prev else 'has history'
    ax.set_title(
        f'Exp3: Per-Query Shift Magnitude\n'
        f'Frame {frame_idx} ({prev_tag})',
        fontsize=11)

    plt.tight_layout()
    fname = f'frame{frame_idx}_shift_magnitude.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 3c: Frame 0 vs Frame 1 comparison
# ============================================================
def plot_frame_comparison():
    """Side-by-side arrow fields for frame 0 (no history) vs frame 1 (real history)."""
    d0 = load_data(0)
    d1 = load_data(1)
    if d0 is None or d1 is None:
        return

    gt0 = _get_gt_boxes(0)
    gt1 = _get_gt_boxes(1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, data, fi, gt in [(ax0, d0, 0, gt0), (ax1, d1, 1, gt1)]:
        origins, deltas = compute_arrow_field(data['ref_2d'], data['shift_ref_2d'])
        mag = np.linalg.norm(deltas, axis=1)
        has_prev = data['has_prev_bev']

        ax.set_xlim(0, BEV_W)
        ax.set_ylim(BEV_H, 0)
        ax.set_aspect('equal')

        if mag.max() < 1e-4:
            ax.scatter(origins[:, 0], origins[:, 1], s=4, c='steelblue',
                       alpha=0.6, zorder=5)
            ax.text(0.5, 0.5, 'No shift', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14,
                    color='gray', alpha=0.6)
        else:
            # Use fixed scale so both panels are comparable
            ax.quiver(
                origins[:, 0], origins[:, 1],
                deltas[:, 0], deltas[:, 1],
                mag,
                cmap='coolwarm', scale=1, scale_units='xy', angles='xy',
                width=0.003, headwidth=3.5, headlength=4,
                zorder=5, alpha=0.85,
            )

        _style_bev_ax(ax, gt)
        shift_val = data['shift'][0].numpy()
        shift_px = shift_val * np.array([BEV_W, BEV_H])
        prev_tag = 'no history' if not has_prev else 'has history'
        ax.set_title(
            f'Frame {fi} ({prev_tag})\n'
            f'shift: dx={shift_px[0]:.1f}px, dy={shift_px[1]:.1f}px',
            fontsize=10)

    fig.suptitle(
        'Exp3: Frame 0 vs Frame 1 — Ego Motion Shift Arrows',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fname = 'compare_frame0_vs_frame1.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 3d: Zoomed-in region (center of BEV)
# ============================================================
def plot_zoomed_center(frame_idx, zoom_size=40):
    """
    Zoomed-in arrow field around BEV center, with denser arrows (step=2).
    Makes individual arrow directions much easier to see.
    """
    data = load_data(frame_idx)
    if data is None:
        return

    has_prev = data['has_prev_bev']
    if not has_prev:
        print(f'  Frame {frame_idx}: no history, skipping zoom plot')
        return

    ref = data['ref_2d'][0, :, 0, :].numpy()      # (40000, 2)
    sref = data['shift_ref_2d'][0, :, 0, :].numpy()

    # Compute dense arrows for center region
    center_r, center_c = BEV_H // 2, BEV_W // 2
    r_lo = center_r - zoom_size // 2
    r_hi = center_r + zoom_size // 2
    c_lo = center_c - zoom_size // 2
    c_hi = center_c + zoom_size // 2

    dense_step = 2
    origins, deltas = [], []
    for r in range(r_lo, r_hi, dense_step):
        for c in range(c_lo, c_hi, dense_step):
            idx = r * BEV_W + c
            rx = ref[idx, 0] * BEV_W
            ry = ref[idx, 1] * BEV_H
            sx = sref[idx, 0] * BEV_W
            sy = sref[idx, 1] * BEV_H
            origins.append([rx, ry])
            deltas.append([sx - rx, sy - ry])

    origins = np.array(origins)
    deltas = np.array(deltas)
    mag = np.linalg.norm(deltas, axis=1)

    gt_boxes = _get_gt_boxes(frame_idx)

    fig, ax = plt.subplots(figsize=(9, 9))
    # Zoom to center region
    ax.set_xlim(c_lo, c_hi)
    ax.set_ylim(r_hi, r_lo)  # inverted y
    ax.set_aspect('equal')

    ax.quiver(
        origins[:, 0], origins[:, 1],
        deltas[:, 0], deltas[:, 1],
        mag,
        cmap='coolwarm', scale=1, scale_units='xy', angles='xy',
        width=0.004, headwidth=3.5, headlength=4,
        zorder=5, alpha=0.85,
    )

    # Draw GT boxes (they'll show if any happen to be in center region)
    if gt_boxes is not None:
        draw_bev_boxes(ax, gt_boxes, color='lime', linewidth=1.5)

    ax.plot(BEV_W / 2, BEV_H / 2, marker='+', color='lime',
            markersize=14, markeredgewidth=2, zorder=10)
    ax.grid(True, alpha=0.2)

    shift_val = data['shift'][0].numpy()
    shift_px = shift_val * np.array([BEV_W, BEV_H])
    stats = (f'Zoom: [{r_lo}:{r_hi}, {c_lo}:{c_hi}] | step={dense_step}\n'
             f'shift: dx={shift_px[0]:.1f}px, dy={shift_px[1]:.1f}px')
    _add_stats_text(ax, stats)

    ax.set_title(
        f'Exp3: Zoomed Center Region — Ego Motion Arrows\n'
        f'Frame {frame_idx} (has history) | {zoom_size}x{zoom_size} BEV pixels',
        fontsize=11)

    plt.tight_layout()
    fname = f'frame{frame_idx}_zoom_center.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Statistics
# ============================================================
def print_stats(frame_idx):
    data = load_data(frame_idx)
    if data is None:
        return

    ref = data['ref_2d']       # (bs, 40000, 1, 2)
    sref = data['shift_ref_2d']
    shift = data['shift']
    has_prev = data['has_prev_bev']

    print(f'\n--- Frame {frame_idx}, has_prev_bev={has_prev} ---')
    print(f'  ref_2d shape: {list(ref.shape)}, range: [{ref.min():.4f}, {ref.max():.4f}]')
    print(f'  shift_ref_2d shape: {list(sref.shape)}, range: [{sref.min():.4f}, {sref.max():.4f}]')

    shift_val = shift[0].numpy()
    shift_px = shift_val * np.array([BEV_W, BEV_H])
    print(f'  shift (normalized): dx={shift_val[0]:.6f}, dy={shift_val[1]:.6f}')
    print(f'  shift (pixels):     dx={shift_px[0]:.2f}, dy={shift_px[1]:.2f}')
    shift_m = shift_val * np.array([PC_RANGE[3] - PC_RANGE[0],
                                     PC_RANGE[4] - PC_RANGE[1]])
    print(f'  shift (meters):     dx={shift_m[0]:.3f}m, dy={shift_m[1]:.3f}m')

    # Per-query magnitude stats
    diff = (sref[0, :, 0, :] - ref[0, :, 0, :])
    diff_px = diff * torch.tensor([BEV_W, BEV_H], dtype=torch.float32)
    mag_px = diff_px.norm(dim=-1)
    print(f'  Per-query shift magnitude (px): '
          f'mean={mag_px.mean():.2f}, std={mag_px.std():.4f}, '
          f'min={mag_px.min():.2f}, max={mag_px.max():.2f}')

    # Check if shift is uniform (rigid body motion)
    if mag_px.max() > 1e-4:
        uniformity = mag_px.std() / mag_px.mean()
        print(f'  Uniformity (std/mean): {uniformity:.6f} '
              f'({"uniform (rigid)" if uniformity < 0.01 else "non-uniform"})')

    # Direction consistency
    if diff_px.norm(dim=-1).mean() > 1e-4:
        angles = torch.atan2(diff_px[:, 1], diff_px[:, 0])
        angle_std = angles.std().item() * 180 / np.pi
        mean_angle = angles.mean().item() * 180 / np.pi
        print(f'  Shift direction: mean={mean_angle:.1f}deg, std={angle_std:.2f}deg '
              f'({"consistent" if angle_std < 1.0 else "varies"})')


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'frame*.pt')))
    if not files:
        print(f'No data found in {DATA_DIR}/.')
        print('Run: TSA_DEBUG=1 bash tools/debug_tsa.sh')
        return

    frames = []
    for f in files:
        base = os.path.basename(f).replace('.pt', '')
        fi = int(base.replace('frame', ''))
        frames.append(fi)
    frames = sorted(frames)
    print(f'Found {len(files)} data files. Frames: {frames}')

    # --- Statistics ---
    print('\n' + '=' * 60)
    print('STATISTICS')
    print('=' * 60)
    for fi in frames:
        print_stats(fi)

    # --- Exp 3a: Arrow field per frame ---
    print('\n' + '=' * 60)
    print('EXP 3a: ARROW FIELD')
    print('=' * 60)
    for fi in frames:
        plot_arrow_field(fi)

    # --- Exp 3b: Shift magnitude heatmap ---
    print('\n' + '=' * 60)
    print('EXP 3b: SHIFT MAGNITUDE HEATMAP')
    print('=' * 60)
    for fi in frames:
        plot_shift_magnitude_heatmap(fi)

    # --- Exp 3c: Frame comparison ---
    print('\n' + '=' * 60)
    print('EXP 3c: FRAME 0 vs FRAME 1')
    print('=' * 60)
    if 0 in frames and 1 in frames:
        plot_frame_comparison()
    else:
        print('  Need both frame0 and frame1 for comparison.')

    # --- Exp 3d: Zoomed center ---
    print('\n' + '=' * 60)
    print('EXP 3d: ZOOMED CENTER')
    print('=' * 60)
    for fi in frames:
        plot_zoomed_center(fi)

    print(f'\nAll figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
