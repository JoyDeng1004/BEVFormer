"""
Experiment 2: Visualize TSA Attention Weights.

Exp 2a: Single-query head x point heatmap (8 heads x 8 points)
Exp 2b: Spatial distribution maps (200x200)
        + BEV GT bounding box overlay
        + 6 camera views with projected 3D boxes
        + PCA pseudo-color attention structure

ARCHITECTURE INSIGHT:
    TSA applies softmax independently per queue (history / current),
    each summing to 1.0 per head. The fusion is a hard-coded mean
    (output.mean(-1)), NOT a learned weighted sum.
    => "history_ratio" is always 0.5 everywhere by design.
    => The meaningful spatial metric is attention CONCENTRATION
       (how sharply each queue attends to its 4 sampling points).

Usage:
    python tools/debug_visualize/visualize_exp2.py

Reads from:
    debug_outputs/exp2_attention_weights/frame*_layer*.pt
    debug_outputs/scene_meta/frame*.json           (optional, for GT overlay)
    data/nuscenes/nuscenes_infos_temporal_val.pkl   (optional, for GT overlay)

Outputs to:
    debug_outputs/exp2_attention_weights/figures/
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
import numpy as np
import os
import glob
import json
import pickle

# ============================================================
# Constants
# ============================================================
DATA_DIR = 'debug_outputs/exp2_attention_weights'
SCENE_META_DIR = 'debug_outputs/scene_meta'
FIG_DIR = os.path.join(DATA_DIR, 'figures')
VAL_PKL = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
BEV_H, BEV_W = 200, 200
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

QUERY_INDICES = {
    'center': 20100,       # row=100, col=100
    'top_left': 0,
    'bottom_right': 39999,
}

# Camera display layout: 2 rows x 3 cols, spatially arranged
CAM_LAYOUT = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'],
]

# 3D box edges (corners 0-3 = top face, 4-7 = bottom face)
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


# ============================================================
# Data Loading
# ============================================================
def load_attention_data(frame_idx, layer_idx):
    path = os.path.join(DATA_DIR, f'frame{frame_idx}_layer{layer_idx}.pt')
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
    """Load sample info from val pkl by sample token."""
    global _val_infos_cache
    if sample_idx is None:
        return None
    if not os.path.exists(VAL_PKL):
        print(f'  [WARN] {VAL_PKL} not found, skipping GT overlay')
        return None
    if _val_infos_cache is None:
        with open(VAL_PKL, 'rb') as f:
            _val_infos_cache = pickle.load(f)['infos']
        print(f'  Loaded {len(_val_infos_cache)} samples from val pkl')
    for info in _val_infos_cache:
        if info['token'] == sample_idx:
            return info
    print(f'  [WARN] sample_idx={sample_idx} not found in val pkl')
    return None


# ============================================================
# Core Computation
# ============================================================
def compute_history_ratio(attn_weights):
    """
    Per-query history attention ratio.

    NOTE: TSA softmax is per-queue, so this is always 0.5 everywhere.
    Kept for verification purposes.

    Args:
        attn_weights: (bs, 40000, 8, 2, 4)
    Returns:
        (200, 200) numpy array.
    """
    aw = attn_weights[0]  # (40000, 8, 2, 4)
    hist_sum = aw[:, :, 0, :].sum(dim=(1, 2))
    total_sum = aw.sum(dim=(1, 2, 3))
    ratio = hist_sum / (total_sum + 1e-8)
    return ratio.numpy().reshape(BEV_H, BEV_W)


def compute_concentration_map(attn_weights, queue='both'):
    """
    Per-query attention concentration: average max weight across heads.

    High value (->1.0) = one-hot attention (sharp focus on 1 of 4 points).
    Low value  (->0.25) = uniform attention (no preference).

    Args:
        attn_weights: (bs, 40000, 8, 2, 4)
        queue: 'history', 'current', or 'both'
    Returns:
        (200, 200) numpy array, values in [0.25, 1.0].
    """
    aw = attn_weights[0]  # (40000, 8, 2, 4)
    if queue == 'history':
        aw = aw[:, :, 0:1, :]  # (40000, 8, 1, 4)
    elif queue == 'current':
        aw = aw[:, :, 1:2, :]
    # else 'both': use all
    # max over 4 points, then mean over heads and queues
    max_w = aw.max(dim=-1).values   # (40000, 8, Q)
    conc = max_w.mean(dim=(1, 2))   # (40000,)
    return conc.numpy().reshape(BEV_H, BEV_W)


def compute_pca_rgb(attn_weights):
    """
    PCA on attention channel dim -> RGB pseudo-color.

    (bs, 40000, 8, 2, 4) -> flatten to (40000, 64) -> PCA(3) -> (200, 200, 3)
    """
    aw = attn_weights[0].reshape(40000, -1).numpy()  # (40000, 64)
    centered = aw - aw.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    pca_3 = U[:, :3] * S[:3]  # (40000, 3)
    for i in range(3):
        lo, hi = pca_3[:, i].min(), pca_3[:, i].max()
        pca_3[:, i] = (pca_3[:, i] - lo) / (hi - lo + 1e-8)
    return pca_3.reshape(BEV_H, BEV_W, 3)


# ============================================================
# BEV GT Boxes
# ============================================================
def lidar_to_bev(x, y):
    """LiDAR coords -> BEV pixel coords."""
    col = (x - PC_RANGE[0]) / (PC_RANGE[3] - PC_RANGE[0]) * BEV_W
    row = (y - PC_RANGE[1]) / (PC_RANGE[4] - PC_RANGE[1]) * BEV_H
    return col, row


def get_bev_box_corners(gt_boxes):
    """3D boxes -> BEV pixel corners. Returns list of (4, 2) arrays."""
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
    """Draw BEV box outlines."""
    if gt_boxes is None or len(gt_boxes) == 0:
        return
    for corners in get_bev_box_corners(gt_boxes):
        poly = Polygon(corners, closed=True, fill=False,
                       edgecolor=color, linewidth=linewidth, alpha=0.85)
        ax.add_patch(poly)


# ============================================================
# Camera 3D Box Projection
# ============================================================
def build_lidar2img(cam_info):
    """Build lidar -> image 4x4 projection matrix."""
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
    """8 corners of a 3D box: (8, 3)."""
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


def project_boxes_to_image(gt_boxes, lidar2img, img_hw):
    """Project 3D boxes and return drawable edge segments."""
    h, w = img_hw
    all_edges = []
    for box in gt_boxes:
        corners = get_3d_box_corners(box)
        pts_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
        proj = (lidar2img @ pts_h.T)
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


def cam_name_from_path(path):
    for part in path.replace('\\', '/').split('/'):
        if part.startswith('CAM_'):
            return part
    return None


def load_cam_images(scene_meta):
    """Load camera images, keyed by camera name."""
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


# ============================================================
# Helper: BEV axis styling
# ============================================================
_BEV_TICKS = [0, 50, 100, 150, 200]
_BEV_TICK_LABELS = ['-51.2', '-25.6', '0', '25.6', '51.2']

def _style_bev_ax(ax, gt_boxes=None):
    """Apply common BEV axis styling: ego marker, GT boxes, tick labels."""
    ax.plot(BEV_W / 2, BEV_H / 2, marker='+', color='lime',
            markersize=10, markeredgewidth=2, zorder=10)
    draw_bev_boxes(ax, gt_boxes)
    ax.set_xticks(_BEV_TICKS)
    ax.set_xticklabels(_BEV_TICK_LABELS, fontsize=8)
    ax.set_yticks(_BEV_TICKS)
    ax.set_yticklabels(_BEV_TICK_LABELS, fontsize=8)
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('y (m)', fontsize=9)


def _add_stats_text(ax, data, fmt='.3f'):
    txt = (f'mean={data.mean():{fmt}}  std={data.std():{fmt}}\n'
           f'min={data.min():{fmt}}  max={data.max():{fmt}}')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7,
            va='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))


def _draw_cameras(fig, gs, sample_info, cam_images, gt_boxes):
    """Draw 6 camera panels in a 2x3 grid on the right side of gs."""
    if not cam_images or sample_info is None:
        return
    for ri, cam_row in enumerate(CAM_LAYOUT):
        for ci, cam_name in enumerate(cam_row):
            ax = fig.add_subplot(gs[ri, 1 + ci])
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
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(img.shape[0], 0)
            ax.axis('off')
            ax.set_title(cam_name.replace('CAM_', '').replace('_', ' '),
                         fontsize=8)


# ============================================================
# GT loading helper
# ============================================================
def _get_gt_boxes(frame_idx):
    """Load GT boxes for a frame via scene_meta -> val pkl."""
    meta = load_scene_meta(frame_idx)
    if meta is None:
        return None, None, {}
    info = load_nuscenes_info(meta.get('sample_idx'))
    cam_images = load_cam_images(meta)
    gt_boxes = info.get('gt_boxes') if info else None
    return gt_boxes, info, cam_images


# ============================================================
# Exp 2a: Single Query Head x Point Heatmap
# ============================================================
def plot_exp2a(frame_idx, layer_idx, query_name, query_idx):
    """(8 heads) x (8 points) heatmap for one query."""
    data = load_attention_data(frame_idx, layer_idx)
    if data is None:
        return

    q_aw = data['attention_weights'][0, query_idx].numpy()  # (8, 2, 4)
    heatmap = q_aw.reshape(8, 8)  # rows=heads, cols=[h_p0..p3, c_p0..p3]
    has_hist = data['has_history']

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0)

    col_labels = [f'H_p{i}' for i in range(4)] + [f'C_p{i}' for i in range(4)]
    ax.set_xticks(range(8))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'Head {i}' for i in range(8)], fontsize=9)
    ax.set_xlabel('Sampling Points  (H=History, C=Current)')
    ax.set_ylabel('Attention Head')

    for i in range(8):
        for j in range(8):
            v = heatmap[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if v > 0.5 else 'black')

    # Separator between history and current queues
    ax.axvline(x=3.5, color='white', linewidth=2, linestyle='--')
    plt.colorbar(im, ax=ax, shrink=0.8)

    q_r, q_c = query_idx // BEV_W, query_idx % BEV_W
    hist_tag = 'self-copy' if not has_hist else 'real'
    ax.set_title(
        f'Exp2a: Attn Weights | Frame {frame_idx} (history={hist_tag}) '
        f'| Layer {layer_idx}\n'
        f'Query "{query_name}" idx={query_idx} (r={q_r}, c={q_c})\n'
        f'NOTE: H and C are independently softmax-normalized (each sums to 1)',
        fontsize=9)

    plt.tight_layout()
    fname = f'frame{frame_idx}_layer{layer_idx}_exp2a_{query_name}.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_exp2a_comparison(layer_idx, query_name, query_idx):
    """Frame 0 vs Frame 1 side-by-side for one query."""
    d0 = load_attention_data(0, layer_idx)
    d1 = load_attention_data(1, layer_idx)
    if d0 is None or d1 is None:
        return

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
    col_labels = [f'H_p{i}' for i in range(4)] + [f'C_p{i}' for i in range(4)]

    for ax, data, fi in [(ax0, d0, 0), (ax1, d1, 1)]:
        hm = data['attention_weights'][0, query_idx].numpy().reshape(8, 8)
        has_hist = data['has_history']
        im = ax.imshow(hm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(8))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(8))
        ax.set_yticklabels([f'Head {i}' for i in range(8)], fontsize=8)
        for i in range(8):
            for j in range(8):
                v = hm[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6, color='white' if v > 0.5 else 'black')
        ax.axvline(x=3.5, color='white', linewidth=2, linestyle='--')
        hist_tag = 'self-copy' if not has_hist else 'real'
        ax.set_title(f'Frame {fi} (history={hist_tag})', fontsize=10)

    plt.colorbar(im, ax=[ax0, ax1], shrink=0.8)
    q_r, q_c = query_idx // BEV_W, query_idx % BEV_W
    fig.suptitle(
        f'Exp2a Compare: Frame 0 vs 1 | Layer {layer_idx} | '
        f'Query "{query_name}" (r={q_r}, c={q_c})', fontsize=11)

    fname = f'exp2a_compare_f0vs1_layer{layer_idx}_{query_name}.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 2b: Attention Concentration Spatial Map
# ============================================================
def plot_exp2b(frame_idx, layer_idx):
    """
    Main figure: 200x200 attention concentration map
    + BEV GT box overlay + 6 camera views with 3D bbox.

    Shows concentration (avg max weight, range [0.25, 1.0]):
    - bright/yellow = sharp one-hot attention
    - dark/purple = diffuse uniform attention
    """
    data = load_attention_data(frame_idx, layer_idx)
    if data is None:
        return
    has_hist = data['has_history']
    aw = data['attention_weights']

    conc_h = compute_concentration_map(aw, queue='history')
    conc_c = compute_concentration_map(aw, queue='current')

    gt_boxes, sample_info, cam_images = _get_gt_boxes(frame_idx)
    has_cams = len(cam_images) > 0

    # --- Figure layout ---
    if has_cams:
        fig = plt.figure(figsize=(24, 8))
        gs = gridspec.GridSpec(
            2, 5, width_ratios=[1.5, 1.5, 0.8, 0.8, 0.8],
            hspace=0.15, wspace=0.15)
        ax_h = fig.add_subplot(gs[:, 0])
        ax_c = fig.add_subplot(gs[:, 1])
    else:
        fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(16, 7))

    hist_tag = 'self-copy' if not has_hist else 'real'

    # --- History concentration ---
    im_h = ax_h.imshow(conc_h, cmap='inferno', vmin=0.25, vmax=1.0,
                       origin='lower', extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax_h, gt_boxes)
    _add_stats_text(ax_h, conc_h)
    ax_h.set_title(f'History Queue Concentration\n'
                   f'Frame {frame_idx} ({hist_tag}), Layer {layer_idx}',
                   fontsize=10)

    # --- Current concentration ---
    im_c = ax_c.imshow(conc_c, cmap='inferno', vmin=0.25, vmax=1.0,
                       origin='lower', extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax_c, gt_boxes)
    _add_stats_text(ax_c, conc_c)
    ax_c.set_title(f'Current Queue Concentration\n'
                   f'Frame {frame_idx} ({hist_tag}), Layer {layer_idx}',
                   fontsize=10)

    # Shared colorbar
    cb = fig.colorbar(im_c, ax=[ax_h, ax_c], shrink=0.7, pad=0.02)
    cb.set_label('Avg Max Weight (0.25=uniform, 1.0=one-hot)', fontsize=8)

    # --- Camera panels ---
    if has_cams:
        _draw_cameras(fig, gs, sample_info, cam_images, gt_boxes)

    fname = f'frame{frame_idx}_layer{layer_idx}_exp2b.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_exp2b_comparison(layer_idx):
    """Frame 0 vs Frame 1 concentration maps side by side (current queue)."""
    d0 = load_attention_data(0, layer_idx)
    d1 = load_attention_data(1, layer_idx)
    if d0 is None or d1 is None:
        return

    c0 = compute_concentration_map(d0['attention_weights'])
    c1 = compute_concentration_map(d1['attention_weights'])
    gt0, _, _ = _get_gt_boxes(0)
    gt1, _, _ = _get_gt_boxes(1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 7))
    h0_tag = 'self-copy' if not d0['has_history'] else 'real'
    h1_tag = 'self-copy' if not d1['has_history'] else 'real'

    ax0.imshow(c0, cmap='inferno', vmin=0.25, vmax=1.0,
               origin='lower', extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax0, gt0)
    _add_stats_text(ax0, c0)
    ax0.set_title(f'Frame 0 (history={h0_tag})', fontsize=10)

    im = ax1.imshow(c1, cmap='inferno', vmin=0.25, vmax=1.0,
                    origin='lower', extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax1, gt1)
    _add_stats_text(ax1, c1)
    ax1.set_title(f'Frame 1 (history={h1_tag})', fontsize=10)

    cb = fig.colorbar(im, ax=[ax0, ax1], shrink=0.7, pad=0.02)
    cb.set_label('Avg Max Weight (both queues)')

    fig.suptitle(
        f'Exp2b Compare: Attention Concentration | Layer {layer_idx}',
        fontsize=12)

    fname = f'exp2b_compare_f0vs1_layer{layer_idx}.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 2b PCA
# ============================================================
def plot_exp2b_pca(frame_idx, layer_idx):
    """PCA pseudo-color 200x200 showing spatial cluster structure."""
    data = load_attention_data(frame_idx, layer_idx)
    if data is None:
        return
    has_hist = data['has_history']
    rgb = compute_pca_rgb(data['attention_weights'])

    gt_boxes, _, _ = _get_gt_boxes(frame_idx)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, origin='lower', extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax, gt_boxes)

    hist_tag = 'self-copy' if not has_hist else 'real'
    ax.set_title(
        f'Exp2b PCA: Attention Clusters (RGB)\n'
        f'Frame {frame_idx} (history={hist_tag}), Layer {layer_idx}',
        fontsize=11)

    plt.tight_layout()
    fname = f'frame{frame_idx}_layer{layer_idx}_exp2b_pca.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Statistics
# ============================================================
def print_stats(frame_idx, layer_idx):
    data = load_attention_data(frame_idx, layer_idx)
    if data is None:
        return

    aw = data['attention_weights']  # (bs, 40000, 8, 2, 4)
    has_hist = data['has_history']

    print(f'\n--- Frame {frame_idx}, Layer {layer_idx}, '
          f'has_history={has_hist} ---')
    print(f'  attention_weights shape: {list(aw.shape)}')

    # Verify per-queue normalization
    aw0 = aw[0]  # (40000, 8, 2, 4)
    hist_sums = aw0[:, :, 0, :].sum(dim=-1)  # (40000, 8) — should all be 1.0
    curr_sums = aw0[:, :, 1, :].sum(dim=-1)
    print(f'  Per-queue sum check: '
          f'history={hist_sums.mean():.4f}, current={curr_sums.mean():.4f} '
          f'(should be 1.0 each — softmax is per-queue)')

    # History ratio (expect 0.5 everywhere)
    ratio = compute_history_ratio(aw)
    print(f'  History ratio: mean={ratio.mean():.4f}, std={ratio.std():.6f} '
          f'(uniform 0.5 = expected, fusion is output.mean(-1))')

    # Concentration stats
    conc_h = compute_concentration_map(aw, queue='history')
    conc_c = compute_concentration_map(aw, queue='current')
    conc_b = compute_concentration_map(aw, queue='both')
    print(f'  Concentration (avg max weight, 0.25=uniform, 1.0=one-hot):')
    print(f'    history: mean={conc_h.mean():.4f}, std={conc_h.std():.4f}')
    print(f'    current: mean={conc_c.mean():.4f}, std={conc_c.std():.4f}')
    print(f'    both:    mean={conc_b.mean():.4f}, std={conc_b.std():.4f}')

    # Per-head concentration
    for h in range(8):
        h_max = aw0[:, h, :, :].max(dim=-1).values.mean()
        print(f'    Head {h}: avg_max_weight={h_max:.4f}')


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'frame*_layer*.pt')))
    if not files:
        print(f'No data found in {DATA_DIR}/.')
        print('Run: TSA_DEBUG=1 bash tools/debug_tsa.sh')
        return

    print(f'Found {len(files)} data files.')
    print()
    print('NOTE: TSA architecture insight —')
    print('  softmax is applied per-queue (4 points), NOT jointly (8 points).')
    print('  History/current fusion is a hard-coded mean, not learned weights.')
    print('  => history_ratio is always 0.5 everywhere.')
    print('  => The meaningful metric is attention CONCENTRATION.')

    available = set()
    for f in files:
        base = os.path.basename(f).replace('.pt', '')
        parts = base.split('_')
        frame = int(parts[0].replace('frame', ''))
        layer = int(parts[1].replace('layer', ''))
        available.add((frame, layer))

    frames = sorted(set(f for f, l in available))
    layers = sorted(set(l for f, l in available))
    print(f'\nFrames: {frames}, Layers: {layers}')

    key_layers = [l for l in [0, 5] if l in layers]
    if not key_layers:
        key_layers = [layers[0], layers[-1]]

    # --- Stats ---
    print('\n' + '=' * 60)
    print('STATISTICS')
    print('=' * 60)
    for fi in frames:
        for li in key_layers:
            if (fi, li) in available:
                print_stats(fi, li)

    # --- Exp 2a ---
    print('\n' + '=' * 60)
    print('EXP 2a: PER-QUERY HEAD x POINT HEATMAPS')
    print('=' * 60)
    for fi in frames:
        for li in key_layers:
            if (fi, li) not in available:
                continue
            for qname, qidx in QUERY_INDICES.items():
                plot_exp2a(fi, li, qname, qidx)

    for li in key_layers:
        if (0, li) in available and (1, li) in available:
            for qname, qidx in QUERY_INDICES.items():
                plot_exp2a_comparison(li, qname, qidx)

    # --- Exp 2b: Concentration ---
    print('\n' + '=' * 60)
    print('EXP 2b: ATTENTION CONCENTRATION SPATIAL MAP')
    print('=' * 60)
    for fi in frames:
        for li in key_layers:
            if (fi, li) in available:
                plot_exp2b(fi, li)

    for li in key_layers:
        if (0, li) in available and (1, li) in available:
            plot_exp2b_comparison(li)

    # --- Exp 2b PCA ---
    print('\n' + '=' * 60)
    print('EXP 2b PCA: ATTENTION CLUSTERS')
    print('=' * 60)
    for fi in frames:
        for li in key_layers:
            if (fi, li) in available:
                plot_exp2b_pca(fi, li)

    print(f'\nAll figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
