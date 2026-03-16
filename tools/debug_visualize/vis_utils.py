"""
Common visualization utilities for BEVFormer debug scripts.

Provides:
- GT loading (scene_meta, nuscenes pkl, camera images)
- BEV <-> physical coordinate conversion
- 3D box projection (BEV + camera)
- BEV axis styling helpers
- Representative query definitions (physical coordinates)
- Warp (SE2 perturbation) utilities
"""
import os
import json
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ============================================================
# Constants
# ============================================================
BEV_H, BEV_W = 200, 200
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
GRID_LENGTH_X = (PC_RANGE[3] - PC_RANGE[0]) / BEV_W  # 0.512 m/cell
GRID_LENGTH_Y = (PC_RANGE[4] - PC_RANGE[1]) / BEV_H

DEBUG_ROOT = 'debug_outputs'
SCENE_META_DIR = os.path.join(DEBUG_ROOT, 'scene_meta')
VAL_PKL = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'

CAM_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
CAM_LAYOUT = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'],
]

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# 8 distinct colors for attention heads
HEAD_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
]
HEAD_MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']

# Colors for representative queries (consistent across all panels)
QUERY_COLORS = [
    '#FF4136',  # red
    '#2ECC40',  # green
    '#0074D9',  # blue
    '#FF851B',  # orange
    '#B10DC9',  # purple
    '#FFDC00',  # yellow
    '#01FF70',  # lime
    '#7FDBFF',  # aqua
]

# Representative BEV queries defined by physical coordinates (meters)
# Coordinate system: x = right, y = forward, z = up
REPRESENTATIVE_QUERIES = {
    'front_10m':       (0.0,   10.0),   # 正前方 10m
    'front_left_30m':  (-21.2, 21.2),   # 左前方 30m (≈45°)
    'front_right_20m': (10.0,  17.3),   # 右前方 20m
    'rear_15m':        (0.0,  -15.0),   # 正后方 15m
    'ego_center':      (0.0,    0.0),   # 自车位置
    'left_20m':        (-20.0,  0.0),   # 正左方 20m
    'right_20m':       (20.0,   0.0),   # 正右方 20m
}


# ============================================================
# Coordinate Conversion
# ============================================================
#
# BEVFormer uses mmdet3d LiDAR coordinate convention:
#   x = right, y = forward, z = up
#
# BEVFormer grid convention (get_reference_points in encoder.py):
#   col (W dim) -> x_norm -> x (right)
#   row (H dim) -> y_norm -> y (forward)
#   query index: idx = row * W + col  (row-major)
#
# Display convention (forward = up, driver's perspective):
#   plot_x = col                  (right of vehicle = right of image)
#   plot_y = BEV_H - 1 - row     (forward = top of image)
#
# For heatmaps shaped (BEV_H, BEV_W) = (row, col):
#   display_arr = bev_to_display(arr)  flips rows so forward=up
# ============================================================

def _physical_to_grid(x_m, y_m):
    """Physical coords (m) -> BEV grid coords (col, row).

    col corresponds to x (right), row to y (forward).
    Used internally for query index computation.
    """
    col = (x_m - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * BEV_W
    row = (y_m - PC_RANGE[1]) / (PC_RANGE[4] - PC_RANGE[1]) * BEV_H
    return col, row


def physical_to_bev(x_m, y_m):
    """Physical coords (m) -> BEV display coords (plot_x, plot_y).

    Returns coordinates for matplotlib plotting with forward=up:
      plot_x: horizontal, left of vehicle = left of image
      plot_y: vertical, forward = top (small value)
    """
    col, row = _physical_to_grid(x_m, y_m)
    plot_x = col                  # x(right) -> horizontal
    plot_y = BEV_H - 1 - row     # y(forward) -> up (small y)
    return plot_x, plot_y


def bev_to_physical(plot_x, plot_y):
    """BEV display coords -> physical coords (m)."""
    col = plot_x
    row = BEV_H - 1 - plot_y
    x_m = col / BEV_W * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
    y_m = row / BEV_H * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
    return x_m, y_m


def bev_to_display(arr):
    """Rotate BEV heatmap for forward=up display.

    Input arr[row, col] where row=y(forward), col=x(right).
    Output: rows flipped so row=0 (y=-51.2, back) goes to bottom.
    Supports 2D (H, W) and 3D (H, W, C) arrays.
    """
    return arr[::-1, :]


def physical_to_query_idx(x_m, y_m):
    """Physical coords -> flat query index (row-major, 200x200)."""
    col, row = _physical_to_grid(x_m, y_m)
    col = int(np.clip(round(col), 0, BEV_W - 1))
    row = int(np.clip(round(row), 0, BEV_H - 1))
    return row * BEV_W + col


def query_idx_to_rc(idx, bev_w=BEV_W):
    """Flat query index -> (row, col) in grid space."""
    return idx // bev_w, idx % bev_w


def grid_to_display(col, row):
    """Convert grid coords (col, row) to display coords (plot_x, plot_y)."""
    return col, BEV_H - 1 - row


def norm_to_display(x_norm, y_norm):
    """Convert BEVFormer normalized coords [0,1] to display coords.

    In BEVFormer: x_norm = col/W (right), y_norm = row/H (forward).
    Returns (plot_x, plot_y) for forward-up display.
    """
    col = x_norm * BEV_W
    row = y_norm * BEV_H
    plot_x = col
    plot_y = BEV_H - 1 - row
    return plot_x, plot_y


def get_representative_query_indices():
    """Return dict of {name: {idx, x_m, y_m, plot_x, plot_y}}."""
    result = {}
    for name, (x_m, y_m) in REPRESENTATIVE_QUERIES.items():
        idx = physical_to_query_idx(x_m, y_m)
        plot_x, plot_y = physical_to_bev(x_m, y_m)
        result[name] = {
            'idx': idx,
            'x_m': x_m, 'y_m': y_m,
            'plot_x': plot_x, 'plot_y': plot_y,
        }
    return result

# PLACEHOLDER_VIS_UTILS_GT


# ============================================================
# Data Loading
# ============================================================
def load_scene_meta(frame_idx):
    """Load scene metadata JSON for a frame."""
    path = os.path.join(SCENE_META_DIR, f'frame{frame_idx}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_calib(frame_idx):
    """Load camera calibration (lidar2img, img_shape) for a frame."""
    path = os.path.join(SCENE_META_DIR, f'frame{frame_idx}_calib.pt')
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu')


_val_infos_cache = None


def load_nuscenes_info(sample_idx):
    """Load sample info from val pkl by sample token."""
    global _val_infos_cache
    if sample_idx is None:
        return None
    if not os.path.exists(VAL_PKL):
        return None
    if _val_infos_cache is None:
        with open(VAL_PKL, 'rb') as f:
            _val_infos_cache = pickle.load(f)['infos']
    for info in _val_infos_cache:
        if info['token'] == sample_idx:
            return info
    return None


def load_cam_images(scene_meta):
    """Load camera images keyed by camera name."""
    images = {}
    for fpath in scene_meta.get('filename', []):
        cam = cam_name_from_path(fpath)
        if cam is None:
            continue
        img_path = fpath[2:] if fpath.startswith('./') else fpath
        if not os.path.exists(img_path):
            continue
        try:
            images[cam] = plt.imread(img_path)
        except Exception:
            pass
    return images


def cam_name_from_path(path):
    """Extract camera name (e.g. CAM_FRONT) from file path."""
    for part in path.replace('\\', '/').split('/'):
        if part.startswith('CAM_'):
            return part
    return None


def get_gt_and_images(frame_idx):
    """Load GT boxes, sample info, and camera images for a frame.

    Returns:
        (gt_boxes, sample_info, cam_images)
    """
    meta = load_scene_meta(frame_idx)
    if meta is None:
        return None, None, {}
    info = load_nuscenes_info(meta.get('sample_idx'))
    cam_images = load_cam_images(meta)
    gt_boxes = info.get('gt_boxes') if info else None
    return gt_boxes, info, cam_images

# PLACEHOLDER_VIS_UTILS_3DBOX


# ============================================================
# 3D Box Utilities
# ============================================================
def build_lidar2img(cam_info):
    """Build lidar -> image 4x4 projection matrix from nuscenes cam_info."""
    s2l_r = np.array(cam_info['sensor2lidar_rotation'])
    s2l_t = np.array(cam_info['sensor2lidar_translation'])
    l2c_r = np.linalg.inv(s2l_r)
    l2c_t = l2c_r @ (-s2l_t)
    l2c = np.eye(4)
    l2c[:3, :3] = l2c_r
    l2c[:3, 3] = l2c_t
    K = np.eye(4)
    K[:3, :3] = np.array(cam_info['cam_intrinsic'])
    return K @ l2c


def get_3d_box_corners(box):
    """Compute 8 corners of a 3D box: (8, 3)."""
    x, y, z, dx, dy, dz, yaw = box[:7]
    hdx, hdy, hdz = dx / 2, dy / 2, dz / 2
    local = np.array([
        [+hdx, +hdy, +hdz], [+hdx, -hdy, +hdz],
        [-hdx, -hdy, +hdz], [-hdx, +hdy, +hdz],
        [+hdx, +hdy, -hdz], [+hdx, -hdy, -hdz],
        [-hdx, -hdy, -hdz], [-hdx, +hdy, -hdz],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (R @ local.T).T + np.array([x, y, z])


def get_bev_box_corners(gt_boxes):
    """3D boxes -> BEV display corners. Returns list of (4, 2) arrays as (plot_x, plot_y)."""
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
        px, py = physical_to_bev(world[:, 0], world[:, 1])
        corners_list.append(np.stack([px, py], axis=1))
    return corners_list


def project_boxes_to_image(gt_boxes, lidar2img, img_hw):
    """Project 3D boxes and return drawable edge segments [(p1, p2), ...]."""
    h, w = img_hw
    all_edges = []
    for box in gt_boxes:
        corners = get_3d_box_corners(box)
        pts_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
        proj = lidar2img @ pts_h.T
        behind = proj[2, :] < 0.1
        proj[:2, :] /= np.clip(proj[2:3, :], 1e-5, None)
        pts_2d = proj[:2, :].T
        for i, j in BOX_EDGES:
            if behind[i] or behind[j]:
                continue
            p1, p2 = pts_2d[i], pts_2d[j]
            if max(p1[0], p2[0]) < 0 or min(p1[0], p2[0]) > w:
                continue
            if max(p1[1], p2[1]) < 0 or min(p1[1], p2[1]) > h:
                continue
            all_edges.append((p1, p2))
    return all_edges

# PLACEHOLDER_VIS_UTILS_DRAW


# ============================================================
# Drawing Helpers
# ============================================================
# Tick positions and labels for forward-up BEV display
# x-axis: left(-x) to right(+x), y-axis: forward(+y) at top to back(-y) at bottom
_BEV_TICKS = [0, 50, 100, 150, 200]
# x-axis: plot_x=col, col=0 is x=-51.2 (left), col=200 is x=+51.2 (right)
_BEV_XTICK_LABELS = ['-51.2', '-25.6', '0', '25.6', '51.2']
# y-axis: plot_y=0 is y=+51.2 (forward/top), plot_y=200 is y=-51.2 (back/bottom)
_BEV_YTICK_LABELS = ['51.2', '25.6', '0', '-25.6', '-51.2']


def draw_bev_boxes(ax, gt_boxes, color='white', linewidth=1.2):
    """Draw BEV box outlines on an axis (display coords)."""
    if gt_boxes is None or len(gt_boxes) == 0:
        return
    for corners in get_bev_box_corners(gt_boxes):
        poly = Polygon(corners, closed=True, fill=False,
                       edgecolor=color, linewidth=linewidth, alpha=0.85)
        ax.add_patch(poly)


def style_bev_ax(ax, gt_boxes=None, title=None):
    """Apply common BEV axis styling: ego marker, GT boxes, physical ticks.

    Display convention: forward=up, left=left (driver's perspective).
    """
    ax.set_xlim(0, BEV_W)
    ax.set_ylim(BEV_H, 0)  # y=0 at top (forward)
    ax.set_aspect('equal')
    # Ego vehicle at center
    ego_px, ego_py = physical_to_bev(0.0, 0.0)
    ax.plot(ego_px, ego_py, marker='+', color='lime',
            markersize=10, markeredgewidth=2, zorder=10)
    draw_bev_boxes(ax, gt_boxes)
    ax.set_xticks(_BEV_TICKS)
    ax.set_xticklabels(_BEV_XTICK_LABELS, fontsize=8)
    ax.set_yticks(_BEV_TICKS)
    ax.set_yticklabels(_BEV_YTICK_LABELS, fontsize=8)
    ax.set_xlabel('x (m) ← left | right →', fontsize=9)
    ax.set_ylabel('y (m) ↑ forward', fontsize=9)
    ax.grid(True, alpha=0.15)
    if title:
        ax.set_title(title, fontsize=10)


def add_stats_text(ax, data, fmt='.3f'):
    """Add min/max/mean/std stats overlay."""
    txt = (f'mean={data.mean():{fmt}}  std={data.std():{fmt}}\n'
           f'min={data.min():{fmt}}  max={data.max():{fmt}}')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7,
            va='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))


def draw_cameras(fig, gs_slice, sample_info, cam_images, gt_boxes,
                 extra_draw_fn=None):
    """Draw 6 camera panels in a 2x3 grid.

    Args:
        fig: matplotlib figure
        gs_slice: GridSpec region for cameras (2 rows x 3 cols)
        sample_info: nuscenes info dict (with 'cams')
        cam_images: dict {cam_name: ndarray}
        gt_boxes: GT boxes array or None
        extra_draw_fn: optional callable(ax, cam_name, lidar2img, img_hw)
            for drawing additional overlays per camera
    """
    if not cam_images or sample_info is None:
        return
    for ri, cam_row in enumerate(CAM_LAYOUT):
        for ci, cam_name in enumerate(cam_row):
            ax = fig.add_subplot(gs_slice[ri, ci])
            img = cam_images.get(cam_name)
            if img is not None:
                ax.imshow(img)
                cam_info = sample_info['cams'].get(cam_name)
                if gt_boxes is not None and cam_info is not None:
                    l2i = build_lidar2img(cam_info)
                    for p1, p2 in project_boxes_to_image(
                            gt_boxes, l2i, img.shape[:2]):
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                color='lime', linewidth=0.8, alpha=0.8)
                    if extra_draw_fn is not None:
                        extra_draw_fn(ax, cam_name, l2i, img.shape[:2])
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(img.shape[0], 0)
            ax.axis('off')
            ax.set_title(cam_name.replace('CAM_', '').replace('_', ' '),
                         fontsize=8)


def mark_queries_on_bev(ax, queries=None, with_labels=True):
    """Mark representative queries on a BEV axis with consistent colors.

    Args:
        ax: matplotlib axis
        queries: dict from get_representative_query_indices(), or None for default
        with_labels: whether to add text labels
    Returns:
        queries dict
    """
    if queries is None:
        queries = get_representative_query_indices()
    for i, (name, q) in enumerate(queries.items()):
        color = QUERY_COLORS[i % len(QUERY_COLORS)]
        ax.plot(q['plot_x'], q['plot_y'], 'o', color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=20)
        if with_labels:
            ax.annotate(
                f"{name}\n({q['x_m']:.0f},{q['y_m']:.0f})m",
                (q['plot_x'], q['plot_y']),
                textcoords='offset points', xytext=(8, -8),
                fontsize=6, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                          alpha=0.6, edgecolor=color, linewidth=0.5))
    return queries

# PLACEHOLDER_VIS_UTILS_WARP


# ============================================================
# SE(2) Warp Utilities
# ============================================================
def make_se2_warp(dx_m=0.0, dy_m=0.0, dtheta_deg=0.0):
    """Create an SE(2) warp dict from physical perturbation parameters.

    Args:
        dx_m: x translation in meters
        dy_m: y translation in meters
        dtheta_deg: rotation in degrees (counter-clockwise)
    Returns:
        dict with 'dx_m', 'dy_m', 'dtheta_deg', 'dx_norm', 'dy_norm',
             'dtheta_rad', 'R' (2x2 rotation matrix)
    """
    dtheta_rad = np.deg2rad(dtheta_deg)
    c, s = np.cos(dtheta_rad), np.sin(dtheta_rad)
    bev_range_x = PC_RANGE[3] - PC_RANGE[0]
    bev_range_y = PC_RANGE[4] - PC_RANGE[1]
    return {
        'dx_m': dx_m, 'dy_m': dy_m, 'dtheta_deg': dtheta_deg,
        'dx_norm': dx_m / bev_range_x,
        'dy_norm': dy_m / bev_range_y,
        'dtheta_rad': dtheta_rad,
        'R': np.array([[c, -s], [s, c]]),
    }


def apply_warp_to_bev_points(points_norm, warp, center=(0.5, 0.5)):
    """Apply SE(2) warp to normalized BEV points.

    Args:
        points_norm: (..., 2) tensor or ndarray, values in [0, 1]
        warp: dict from make_se2_warp
        center: rotation center in normalized coords
    Returns:
        warped points, same shape
    """
    is_tensor = isinstance(points_norm, torch.Tensor)
    if is_tensor:
        pts = points_norm.numpy()
    else:
        pts = np.array(points_norm)

    shape = pts.shape
    pts_flat = pts.reshape(-1, 2)

    # Translate to center, rotate, translate back
    cx, cy = center
    pts_centered = pts_flat - np.array([cx, cy])
    R = warp['R']
    pts_rotated = (R @ pts_centered.T).T
    pts_warped = pts_rotated + np.array([cx, cy])

    # Apply translation
    pts_warped[:, 0] += warp['dx_norm']
    pts_warped[:, 1] += warp['dy_norm']

    result = pts_warped.reshape(shape)
    if is_tensor:
        return torch.from_numpy(result).float()
    return result


def add_warp_argparse(parser):
    """Add warp-related CLI arguments to an argparse parser."""
    parser.add_argument('--warp_dx', type=float, default=0.0,
                        help='Warp x translation (meters)')
    parser.add_argument('--warp_dy', type=float, default=0.0,
                        help='Warp y translation (meters)')
    parser.add_argument('--warp_dtheta', type=float, default=0.0,
                        help='Warp rotation (degrees)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate warp before/after comparison')
    return parser

