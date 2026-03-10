"""
Experiment 4: Visualize History-Current Fusion Output of TSA.

Exp 4a: Feature map comparison (history / current / fused)
         using PCA pseudo-color and first-3-channel views.
Exp 4b: Per-pixel cosine similarity between history and current branches.
         High similarity = static background, low similarity = dynamic objects.

ARCHITECTURE INSIGHT:
    TSA output shape before fusion: (num_query, embed_dims, bs, num_bev_queue=2)
    Fusion is a simple mean: output.mean(-1)
    dim=-1 index 0 = history branch, index 1 = current branch.
    First frame (no real history): history branch = current branch (self-copy),
    so cosine similarity = 1.0 everywhere.

Usage:
    python tools/debug_visualize/visualize_exp4.py

Reads from:
    debug_outputs/exp4_fusion_output/frame*_layer*.pt
    debug_outputs/scene_meta/frame*.json           (optional, for GT overlay)
    data/nuscenes/nuscenes_infos_temporal_val.pkl   (optional, for GT overlay)

Outputs to:
    debug_outputs/exp4_fusion_output/figures/
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
DATA_DIR = 'debug_outputs/exp4_fusion_output'
SCENE_META_DIR = 'debug_outputs/scene_meta'
FIG_DIR = os.path.join(DATA_DIR, 'figures')
VAL_PKL = 'data/nuscenes/nuscenes_infos_temporal_val.pkl'
BEV_H, BEV_W = 200, 200
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Camera display layout
CAM_LAYOUT = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'],
]

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


# ============================================================
# Data Loading
# ============================================================
def load_fusion_data(frame_idx, layer_idx):
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


# ============================================================
# Camera 3D Box Projection
# ============================================================
def build_lidar2img(cam_info):
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
    meta = load_scene_meta(frame_idx)
    if meta is None:
        return None, None, {}
    info = load_nuscenes_info(meta.get('sample_idx'))
    cam_images = load_cam_images(meta)
    gt_boxes = info.get('gt_boxes') if info else None
    return gt_boxes, info, cam_images


# ============================================================
# Core Computation
# ============================================================
def extract_branches(data):
    """
    Extract history, current, and fused feature maps from saved data.

    Returns:
        history: (200, 200, 256) numpy array
        current: (200, 200, 256) numpy array
        fused:   (200, 200, 256) numpy array
    """
    before = data['output_before_fusion']  # (40000, 256, bs, 2)
    after = data['output_after_fusion']    # (40000, 256, bs)

    # Take batch 0
    history = before[:, :, 0, 0].numpy()   # (40000, 256)
    current = before[:, :, 0, 1].numpy()   # (40000, 256)
    fused = after[:, :, 0].numpy()         # (40000, 256)

    history = history.reshape(BEV_H, BEV_W, -1)
    current = current.reshape(BEV_H, BEV_W, -1)
    fused = fused.reshape(BEV_H, BEV_W, -1)

    return history, current, fused


def compute_cosine_similarity(history, current):
    """
    Per-pixel cosine similarity between history and current branches.

    Args:
        history: (200, 200, 256)
        current: (200, 200, 256)
    Returns:
        (200, 200) numpy array, values in [-1, 1].
    """
    # Flatten spatial dims for vectorized computation
    h_flat = history.reshape(-1, history.shape[-1])  # (40000, 256)
    c_flat = current.reshape(-1, current.shape[-1])  # (40000, 256)

    h_norm = np.linalg.norm(h_flat, axis=-1, keepdims=True) + 1e-8
    c_norm = np.linalg.norm(c_flat, axis=-1, keepdims=True) + 1e-8

    cos_sim = (h_flat * c_flat).sum(axis=-1) / (h_norm[:, 0] * c_norm[:, 0])
    return cos_sim.reshape(BEV_H, BEV_W)


def compute_pca_rgb(feature_map):
    """
    PCA on (200, 200, 256) -> (200, 200, 3) RGB pseudo-color.
    """
    flat = feature_map.reshape(-1, feature_map.shape[-1])  # (40000, 256)
    centered = flat - flat.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    pca_3 = U[:, :3] * S[:3]  # (40000, 3)
    for i in range(3):
        lo, hi = pca_3[:, i].min(), pca_3[:, i].max()
        pca_3[:, i] = (pca_3[:, i] - lo) / (hi - lo + 1e-8)
    return pca_3.reshape(BEV_H, BEV_W, 3)


def compute_l2_norm(feature_map):
    """Per-pixel L2 norm: (200, 200, 256) -> (200, 200)."""
    return np.linalg.norm(feature_map, axis=-1)


def compute_channel_rgb(feature_map, channels=(0, 1, 2)):
    """
    Take 3 channels from the feature map and normalize to [0,1] RGB.

    Args:
        feature_map: (200, 200, 256)
        channels: tuple of 3 channel indices
    Returns:
        (200, 200, 3) numpy array
    """
    rgb = np.stack([feature_map[:, :, c] for c in channels], axis=-1)
    for i in range(3):
        lo, hi = rgb[:, :, i].min(), rgb[:, :, i].max()
        rgb[:, :, i] = (rgb[:, :, i] - lo) / (hi - lo + 1e-8)
    return rgb


# ============================================================
# Exp 4a: Feature Map Comparison (PCA pseudo-color)
# ============================================================
def plot_exp4a_pca(frame_idx, layer_idx):
    """
    Three PCA pseudo-color maps side by side:
    history_output | current_output | fused_output
    """
    data = load_fusion_data(frame_idx, layer_idx)
    if data is None:
        return

    has_hist = data['has_history']
    history, current, fused = extract_branches(data)

    gt_boxes, _, _ = _get_gt_boxes(frame_idx)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Compute PCA jointly so color space is shared
    all_features = np.concatenate([
        history.reshape(-1, 256),
        current.reshape(-1, 256),
        fused.reshape(-1, 256),
    ], axis=0)  # (120000, 256)
    centered = all_features - all_features.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    pca_all = U[:, :3] * S[:3]  # (120000, 3)
    for i in range(3):
        lo, hi = pca_all[:, i].min(), pca_all[:, i].max()
        pca_all[:, i] = (pca_all[:, i] - lo) / (hi - lo + 1e-8)

    rgb_h = pca_all[:40000].reshape(BEV_H, BEV_W, 3)
    rgb_c = pca_all[40000:80000].reshape(BEV_H, BEV_W, 3)
    rgb_f = pca_all[80000:].reshape(BEV_H, BEV_W, 3)

    titles = ['History Branch', 'Current Branch', 'Fused (mean)']
    rgbs = [rgb_h, rgb_c, rgb_f]

    for ax, rgb, title in zip(axes, rgbs, titles):
        ax.imshow(rgb, origin='lower', extent=[0, BEV_W, 0, BEV_H])
        _style_bev_ax(ax, gt_boxes)
        ax.set_title(title, fontsize=11)

    hist_tag = 'self-copy' if not has_hist else 'real'
    fig.suptitle(
        f'Exp4a: Feature Map PCA Pseudo-Color (joint color space)\n'
        f'Frame {frame_idx} (history={hist_tag}), Layer {layer_idx}',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    fname = f'frame{frame_idx}_layer{layer_idx}_exp4a_pca.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_exp4a_channels(frame_idx, layer_idx, channels=(0, 1, 2)):
    """
    Three feature maps using raw channel values as RGB.
    """
    data = load_fusion_data(frame_idx, layer_idx)
    if data is None:
        return

    has_hist = data['has_history']
    history, current, fused = extract_branches(data)

    gt_boxes, _, _ = _get_gt_boxes(frame_idx)

    # Normalize jointly so colors are comparable
    all_vals = np.stack([
        np.stack([history[:, :, c] for c in channels], axis=-1),
        np.stack([current[:, :, c] for c in channels], axis=-1),
        np.stack([fused[:, :, c] for c in channels], axis=-1),
    ])  # (3, 200, 200, 3)
    lo = all_vals.min(axis=(0, 1, 2), keepdims=True)
    hi = all_vals.max(axis=(0, 1, 2), keepdims=True)
    all_vals = (all_vals - lo) / (hi - lo + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    titles = ['History Branch', 'Current Branch', 'Fused (mean)']

    for ax, rgb, title in zip(axes, all_vals, titles):
        ax.imshow(rgb, origin='lower', extent=[0, BEV_W, 0, BEV_H])
        _style_bev_ax(ax, gt_boxes)
        ax.set_title(title, fontsize=11)

    hist_tag = 'self-copy' if not has_hist else 'real'
    ch_str = ','.join(str(c) for c in channels)
    fig.suptitle(
        f'Exp4a: Feature Map Channels [{ch_str}] as RGB\n'
        f'Frame {frame_idx} (history={hist_tag}), Layer {layer_idx}',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    fname = f'frame{frame_idx}_layer{layer_idx}_exp4a_ch{ch_str.replace(",","_")}.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 4b: Cosine Similarity Heatmap
# ============================================================
def plot_exp4b(frame_idx, layer_idx):
    """
    200x200 per-pixel cosine similarity between history and current branches.
    + BEV GT box overlay + 6 camera views.

    High sim (->1.0) = static background (features unchanged across frames).
    Low sim (->0 or negative) = dynamic objects, occlusion, new areas.
    """
    data = load_fusion_data(frame_idx, layer_idx)
    if data is None:
        return

    has_hist = data['has_history']
    history, current, _ = extract_branches(data)
    cos_sim = compute_cosine_similarity(history, current)

    gt_boxes, sample_info, cam_images = _get_gt_boxes(frame_idx)
    has_cams = len(cam_images) > 0

    # --- Figure layout ---
    if has_cams:
        fig = plt.figure(figsize=(22, 8))
        gs = gridspec.GridSpec(
            2, 4, width_ratios=[2.0, 0.8, 0.8, 0.8],
            hspace=0.15, wspace=0.15)
        ax_main = fig.add_subplot(gs[:, 0])
    else:
        fig, ax_main = plt.subplots(figsize=(9, 8))

    hist_tag = 'self-copy' if not has_hist else 'real'

    im = ax_main.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1,
                        origin='lower', extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax_main, gt_boxes)
    _add_stats_text(ax_main, cos_sim)
    ax_main.set_title(
        f'Exp4b: Cosine Similarity (History vs Current)\n'
        f'Frame {frame_idx} (history={hist_tag}), Layer {layer_idx}\n'
        f'Red=similar (static), Blue=dissimilar (dynamic)',
        fontsize=10)

    cb = fig.colorbar(im, ax=ax_main, shrink=0.7, pad=0.02)
    cb.set_label('Cosine Similarity', fontsize=9)

    # --- Camera panels ---
    if has_cams:
        _draw_cameras(fig, gs, sample_info, cam_images, gt_boxes)

    fname = f'frame{frame_idx}_layer{layer_idx}_exp4b.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_exp4b_comparison(layer_idx):
    """Frame 0 vs Frame 1 cosine similarity side by side."""
    d0 = load_fusion_data(0, layer_idx)
    d1 = load_fusion_data(1, layer_idx)
    if d0 is None or d1 is None:
        return

    h0, c0, _ = extract_branches(d0)
    h1, c1, _ = extract_branches(d1)
    cos0 = compute_cosine_similarity(h0, c0)
    cos1 = compute_cosine_similarity(h1, c1)

    gt0, _, _ = _get_gt_boxes(0)
    gt1, _, _ = _get_gt_boxes(1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 8))
    h0_tag = 'self-copy' if not d0['has_history'] else 'real'
    h1_tag = 'self-copy' if not d1['has_history'] else 'real'

    ax0.imshow(cos0, cmap='RdBu_r', vmin=-1, vmax=1,
               origin='lower', extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax0, gt0)
    _add_stats_text(ax0, cos0)
    ax0.set_title(f'Frame 0 (history={h0_tag})', fontsize=10)

    im = ax1.imshow(cos1, cmap='RdBu_r', vmin=-1, vmax=1,
                    origin='lower', extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax1, gt1)
    _add_stats_text(ax1, cos1)
    ax1.set_title(f'Frame 1 (history={h1_tag})', fontsize=10)

    cb = fig.colorbar(im, ax=[ax0, ax1], shrink=0.7, pad=0.02)
    cb.set_label('Cosine Similarity')

    fig.suptitle(
        f'Exp4b Compare: History-Current Cosine Similarity | Layer {layer_idx}\n'
        f'Frame 0 (self-copy, expect ~1.0) vs Frame 1 (real history)',
        fontsize=12)

    fname = f'exp4b_compare_f0vs1_layer{layer_idx}.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_exp4b_layer_comparison(frame_idx):
    """
    Cosine similarity across all available layers for one frame.
    Shows how similarity evolves through the encoder stack.
    """
    data_layers = {}
    for li in range(6):
        d = load_fusion_data(frame_idx, li)
        if d is not None:
            data_layers[li] = d

    if len(data_layers) < 2:
        return

    layers = sorted(data_layers.keys())
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    gt_boxes, _, _ = _get_gt_boxes(frame_idx)

    for ax, li in zip(axes, layers):
        h, c, _ = extract_branches(data_layers[li])
        cos_sim = compute_cosine_similarity(h, c)

        ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1,
                  origin='lower', extent=[0, BEV_W, 0, BEV_H])
        _style_bev_ax(ax, gt_boxes)
        _add_stats_text(ax, cos_sim)
        ax.set_title(f'Layer {li}', fontsize=10)

    has_hist = data_layers[layers[0]]['has_history']
    hist_tag = 'self-copy' if not has_hist else 'real'
    fig.suptitle(
        f'Exp4b: Cosine Similarity Across Layers | '
        f'Frame {frame_idx} (history={hist_tag})',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    fname = f'frame{frame_idx}_exp4b_layer_evolution.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 4c: Feature Norm Comparison
# ============================================================
def plot_exp4c_norms(frame_idx, layer_idx):
    """
    Per-pixel L2 norm of history / current / fused features.
    Reveals spatial activation patterns and relative magnitudes.
    """
    data = load_fusion_data(frame_idx, layer_idx)
    if data is None:
        return

    has_hist = data['has_history']
    history, current, fused = extract_branches(data)

    norm_h = compute_l2_norm(history)
    norm_c = compute_l2_norm(current)
    norm_f = compute_l2_norm(fused)

    gt_boxes, _, _ = _get_gt_boxes(frame_idx)

    # Shared colorbar range
    vmax = max(norm_h.max(), norm_c.max(), norm_f.max())
    vmin = 0

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    titles = ['History Branch', 'Current Branch', 'Fused (mean)']
    norms = [norm_h, norm_c, norm_f]

    for ax, norm, title in zip(axes, norms, titles):
        im = ax.imshow(norm, cmap='viridis', vmin=vmin, vmax=vmax,
                       origin='lower', extent=[0, BEV_W, 0, BEV_H])
        _style_bev_ax(ax, gt_boxes)
        _add_stats_text(ax, norm, fmt='.1f')
        ax.set_title(title, fontsize=11)

    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.7, pad=0.02)
    cb.set_label('L2 Norm', fontsize=9)

    hist_tag = 'self-copy' if not has_hist else 'real'
    fig.suptitle(
        f'Exp4c: Feature L2 Norm | Frame {frame_idx} (history={hist_tag}), '
        f'Layer {layer_idx}',
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    fname = f'frame{frame_idx}_layer{layer_idx}_exp4c_norms.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 4d: History-Current Difference Map
# ============================================================
def plot_exp4d_difference(frame_idx, layer_idx):
    """
    Per-pixel L2 distance between history and current branches.
    Large difference = area changed between frames (dynamic objects, ego motion).
    """
    data = load_fusion_data(frame_idx, layer_idx)
    if data is None:
        return

    has_hist = data['has_history']
    history, current, _ = extract_branches(data)

    diff = np.linalg.norm(history - current, axis=-1)  # (200, 200)

    gt_boxes, _, _ = _get_gt_boxes(frame_idx)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(diff, cmap='hot', origin='lower',
                   extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax, gt_boxes)
    _add_stats_text(ax, diff, fmt='.1f')

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('L2 Distance', fontsize=9)

    hist_tag = 'self-copy' if not has_hist else 'real'
    ax.set_title(
        f'Exp4d: History-Current L2 Difference\n'
        f'Frame {frame_idx} (history={hist_tag}), Layer {layer_idx}\n'
        f'Bright = large difference (dynamic), Dark = similar (static)',
        fontsize=10)
    plt.tight_layout()

    fname = f'frame{frame_idx}_layer{layer_idx}_exp4d_diff.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Statistics
# ============================================================
def print_stats(frame_idx, layer_idx):
    data = load_fusion_data(frame_idx, layer_idx)
    if data is None:
        return

    has_hist = data['has_history']
    before = data['output_before_fusion']  # (40000, 256, bs, 2)
    after = data['output_after_fusion']    # (40000, 256, bs)

    print(f'\n--- Frame {frame_idx}, Layer {layer_idx}, '
          f'has_history={has_hist} ---')
    print(f'  output_before_fusion shape: {list(before.shape)}')
    print(f'  output_after_fusion shape:  {list(after.shape)}')

    history, current, fused = extract_branches(data)

    # Feature norms
    norm_h = compute_l2_norm(history)
    norm_c = compute_l2_norm(current)
    norm_f = compute_l2_norm(fused)
    print(f'  L2 norm:')
    print(f'    history: mean={norm_h.mean():.2f}, std={norm_h.std():.2f}')
    print(f'    current: mean={norm_c.mean():.2f}, std={norm_c.std():.2f}')
    print(f'    fused:   mean={norm_f.mean():.2f}, std={norm_f.std():.2f}')

    # Cosine similarity
    cos_sim = compute_cosine_similarity(history, current)
    print(f'  Cosine similarity (history vs current):')
    print(f'    mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}')
    print(f'    min={cos_sim.min():.4f}, max={cos_sim.max():.4f}')

    # L2 difference
    diff = np.linalg.norm(history - current, axis=-1)
    print(f'  L2 difference (history vs current):')
    print(f'    mean={diff.mean():.2f}, std={diff.std():.2f}')
    print(f'    min={diff.min():.2f}, max={diff.max():.2f}')

    # Verify fusion = mean
    expected_fused = (history + current) / 2.0
    fusion_err = np.abs(fused - expected_fused).max()
    print(f'  Fusion verification: max|fused - mean(h,c)| = {fusion_err:.6f} '
          f'(should be ~0)')


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

    # --- Exp 4a: PCA Feature Maps ---
    print('\n' + '=' * 60)
    print('EXP 4a: FEATURE MAP PCA PSEUDO-COLOR')
    print('=' * 60)
    for fi in frames:
        for li in key_layers:
            if (fi, li) in available:
                plot_exp4a_pca(fi, li)

    # --- Exp 4a: Channel RGB ---
    print('\n' + '=' * 60)
    print('EXP 4a: FEATURE MAP CHANNEL RGB')
    print('=' * 60)
    for fi in frames:
        for li in key_layers:
            if (fi, li) in available:
                plot_exp4a_channels(fi, li, channels=(0, 1, 2))

    # --- Exp 4b: Cosine Similarity ---
    print('\n' + '=' * 60)
    print('EXP 4b: COSINE SIMILARITY')
    print('=' * 60)
    for fi in frames:
        for li in key_layers:
            if (fi, li) in available:
                plot_exp4b(fi, li)

    # --- Exp 4b: Frame comparison ---
    for li in key_layers:
        if (0, li) in available and (1, li) in available:
            plot_exp4b_comparison(li)

    # --- Exp 4b: Layer evolution ---
    print('\n' + '=' * 60)
    print('EXP 4b: LAYER EVOLUTION')
    print('=' * 60)
    for fi in frames:
        plot_exp4b_layer_comparison(fi)

    # --- Exp 4c: Feature Norms ---
    print('\n' + '=' * 60)
    print('EXP 4c: FEATURE L2 NORMS')
    print('=' * 60)
    for fi in frames:
        for li in key_layers:
            if (fi, li) in available:
                plot_exp4c_norms(fi, li)

    # --- Exp 4d: Difference Map ---
    print('\n' + '=' * 60)
    print('EXP 4d: HISTORY-CURRENT DIFFERENCE')
    print('=' * 60)
    for fi in frames:
        for li in key_layers:
            if (fi, li) in available:
                plot_exp4d_difference(fi, li)

    print(f'\nAll figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
