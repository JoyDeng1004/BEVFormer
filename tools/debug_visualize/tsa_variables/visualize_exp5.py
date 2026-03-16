"""
Experiment 5: Visualize BEV Positional Encoding (Learned PE).

Exp 5a: Individual channel heatmaps.
        First 128 channels = col_embed (should vary only along x / columns).
        Last  128 channels = row_embed (should vary only along y / rows).

Exp 5b: PCA dimensionality reduction to 3D -> RGB pseudo-color.
        Spatially close positions should have similar colors.

Usage:
    python tools/debug_visualize/visualize_exp5.py

Reads from:
    debug_outputs/exp5_bev_pos/bev_pos.pt

Outputs to:
    debug_outputs/exp5_bev_pos/figures/
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# Constants
# ============================================================
DATA_DIR = 'debug_outputs/exp5_bev_pos'
FIG_DIR = os.path.join(DATA_DIR, 'figures')
BEV_H, BEV_W = 200, 200

_BEV_TICKS = [0, 50, 100, 150, 200]
_BEV_TICK_LABELS = ['-51.2', '-25.6', '0', '25.6', '51.2']


# ============================================================
# Data Loading
# ============================================================
def load_bev_pos():
    path = os.path.join(DATA_DIR, 'bev_pos.pt')
    if not os.path.exists(path):
        print(f'  [ERROR] {path} not found')
        print('  Run: TSA_DEBUG=1 bash tools/debug_tsa.sh')
        return None
    data = torch.load(path, map_location='cpu')
    return data['bev_pos']  # (bs, 256, 200, 200)


# ============================================================
# Helpers
# ============================================================
def _style_bev_ax(ax):
    ax.plot(BEV_W / 2, BEV_H / 2, marker='+', color='lime',
            markersize=8, markeredgewidth=1.5, zorder=10)
    ax.set_xticks(_BEV_TICKS)
    ax.set_xticklabels(_BEV_TICK_LABELS, fontsize=7)
    ax.set_yticks(_BEV_TICKS)
    ax.set_yticklabels(_BEV_TICK_LABELS, fontsize=7)
    ax.set_xlabel('x (m)', fontsize=8)
    ax.set_ylabel('y (m)', fontsize=8)


def _add_stats_text(ax, data, fmt='.3f'):
    txt = (f'mean={data.mean():{fmt}}  std={data.std():{fmt}}\n'
           f'min={data.min():{fmt}}  max={data.max():{fmt}}')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7,
            va='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))


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


# ============================================================
# Exp 5a: Individual Channel Heatmaps
# ============================================================
def plot_exp5a_channels(pe):
    """
    Visualize individual PE channels as 200x200 heatmaps.
    Channels 0,64 are from col_embed -> should vary along x (columns).
    Channels 128,192 are from row_embed -> should vary along y (rows).
    """
    channels = [0, 64, 128, 192]
    labels = [
        'Ch 0 (col_embed)',
        'Ch 64 (col_embed)',
        'Ch 128 (row_embed)',
        'Ch 192 (row_embed)',
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    for ax, ch, label in zip(axes, channels, labels):
        ch_data = pe[ch].numpy()  # (200, 200)
        im = ax.imshow(ch_data, cmap='RdBu_r', origin='lower',
                       extent=[0, BEV_W, 0, BEV_H])
        _style_bev_ax(ax)
        _add_stats_text(ax, ch_data)
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

    fig.suptitle(
        'Exp5a: Learned BEV Positional Encoding - Individual Channels\n'
        'Col_embed (ch 0-127): expect horizontal variation | '
        'Row_embed (ch 128-255): expect vertical variation',
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    fname = 'exp5a_channels.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_exp5a_channels_extended(pe):
    """
    More channels to confirm the col_embed / row_embed split.
    8 channels from col_embed (0-127) and 8 from row_embed (128-255).
    """
    col_chs = [0, 16, 32, 48, 64, 80, 96, 112]
    row_chs = [128, 144, 160, 176, 192, 208, 224, 240]

    fig, axes = plt.subplots(2, 8, figsize=(28, 7))

    for i, ch in enumerate(col_chs):
        ax = axes[0, i]
        ch_data = pe[ch].numpy()
        ax.imshow(ch_data, cmap='RdBu_r', origin='lower',
                  extent=[0, BEV_W, 0, BEV_H])
        ax.set_title(f'Ch {ch}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ch in enumerate(row_chs):
        ax = axes[1, i]
        ch_data = pe[ch].numpy()
        ax.imshow(ch_data, cmap='RdBu_r', origin='lower',
                  extent=[0, BEV_W, 0, BEV_H])
        ax.set_title(f'Ch {ch}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel('col_embed\n(ch 0-127)', fontsize=9)
    axes[1, 0].set_ylabel('row_embed\n(ch 128-255)', fontsize=9)

    fig.suptitle(
        'Exp5a: Col_embed vs Row_embed Channels\n'
        'Top row: col_embed channels (should show column/x patterns)\n'
        'Bottom row: row_embed channels (should show row/y patterns)',
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    fname = 'exp5a_channels_extended.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_exp5a_variance_profile(pe):
    """
    For each channel, compute variance along rows vs columns.
    Col_embed channels should have high column-variance and low row-variance.
    Row_embed channels should have the opposite.
    """
    pe_np = pe.numpy()  # (256, 200, 200)

    row_var = np.var(pe_np, axis=1).mean(axis=1)  # variance across rows, mean over cols -> (256,)
    col_var = np.var(pe_np, axis=2).mean(axis=1)  # variance across cols, mean over rows -> (256,)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    x = np.arange(256)

    ax1.bar(x, row_var, width=1.0, color='steelblue', alpha=0.8, label='Var across rows (axis=1)')
    ax1.bar(x, col_var, width=1.0, color='coral', alpha=0.6, label='Var across cols (axis=2)')
    ax1.axvline(x=127.5, color='black', linestyle='--', linewidth=1, label='col/row embed split')
    ax1.set_ylabel('Variance', fontsize=9)
    ax1.set_xlabel('Channel index', fontsize=9)
    ax1.set_title('Per-Channel Variance Along Row vs Column Axes', fontsize=10)
    ax1.legend(fontsize=8)

    # Ratio: row_var / (col_var + eps)  -- high = varies more across rows
    ratio = row_var / (col_var + 1e-10)
    colors = ['steelblue' if i < 128 else 'coral' for i in range(256)]
    ax2.bar(x, np.log10(ratio + 1e-10), width=1.0, color=colors, alpha=0.8)
    ax2.axvline(x=127.5, color='black', linestyle='--', linewidth=1)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('log10(row_var / col_var)', fontsize=9)
    ax2.set_xlabel('Channel index', fontsize=9)
    ax2.set_title(
        'Variance Ratio: >0 = varies more across rows, <0 = varies more across columns\n'
        'Blue = col_embed (ch 0-127), Red = row_embed (ch 128-255)',
        fontsize=9)

    plt.tight_layout()

    fname = 'exp5a_variance_profile.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 5b: PCA Dimensionality Reduction
# ============================================================
def plot_exp5b_pca(pe):
    """
    PCA on (200, 200, 256) -> RGB pseudo-color.
    Spatially close positions should have similar colors.
    """
    pe_np = pe.numpy()  # (256, 200, 200)
    pe_hwc = pe_np.transpose(1, 2, 0)  # (200, 200, 256)

    pca_rgb = compute_pca_rgb(pe_hwc)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pca_rgb, origin='lower', extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(ax)
    ax.set_title(
        'Exp5b: BEV Positional Encoding - PCA to RGB\n'
        'Spatially close positions should have similar colors',
        fontsize=11)
    plt.tight_layout()

    fname = 'exp5b_pca_rgb.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_exp5b_pca_components(pe):
    """
    Show the first 3 PCA components individually as heatmaps,
    plus the combined RGB image.
    """
    pe_np = pe.numpy()  # (256, 200, 200)
    pe_hwc = pe_np.transpose(1, 2, 0)  # (200, 200, 256)

    flat = pe_hwc.reshape(-1, 256)  # (40000, 256)
    centered = flat - flat.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_scores = U[:, :3] * S[:3]  # (40000, 3)

    # Explained variance
    total_var = (S ** 2).sum()
    explained = [(S[i] ** 2) / total_var * 100 for i in range(3)]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    cmap_names = ['Reds', 'Greens', 'Blues']
    for i in range(3):
        comp = pca_scores[:, i].reshape(BEV_H, BEV_W)
        im = axes[i].imshow(comp, cmap=cmap_names[i], origin='lower',
                            extent=[0, BEV_W, 0, BEV_H])
        _style_bev_ax(axes[i])
        _add_stats_text(axes[i], comp)
        axes[i].set_title(f'PC{i+1} ({explained[i]:.1f}% var)', fontsize=10)
        fig.colorbar(im, ax=axes[i], shrink=0.75, pad=0.03)

    # Combined RGB
    pca_rgb = pca_scores.copy()
    for i in range(3):
        lo, hi = pca_rgb[:, i].min(), pca_rgb[:, i].max()
        pca_rgb[:, i] = (pca_rgb[:, i] - lo) / (hi - lo + 1e-8)
    axes[3].imshow(pca_rgb.reshape(BEV_H, BEV_W, 3), origin='lower',
                   extent=[0, BEV_W, 0, BEV_H])
    _style_bev_ax(axes[3])
    axes[3].set_title('Combined RGB', fontsize=10)

    fig.suptitle(
        f'Exp5b: PCA Components of BEV Positional Encoding\n'
        f'Top-3 PCs explain {sum(explained):.1f}% of total variance',
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    fname = 'exp5b_pca_components.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Exp 5c: Cosine Similarity Between BEV Positions
# ============================================================
def plot_exp5c_similarity(pe):
    """
    Pick a few reference positions on the BEV grid and show
    cosine similarity of their PE vector to all other positions.
    Reveals the spatial structure learned by the PE.
    """
    pe_np = pe.numpy()  # (256, 200, 200)
    pe_flat = pe_np.reshape(256, -1).T  # (40000, 256)

    # Normalize for cosine similarity
    norms = np.linalg.norm(pe_flat, axis=1, keepdims=True) + 1e-8
    pe_normed = pe_flat / norms

    ref_positions = {
        'Center (100,100)': (100, 100),
        'Top-Left (25,25)': (25, 25),
        'Bottom-Right (175,175)': (175, 175),
        'Mid-Right (100,175)': (100, 175),
    }

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for ax, (label, (r, c)) in zip(axes, ref_positions.items()):
        ref_idx = r * BEV_W + c
        ref_vec = pe_normed[ref_idx]  # (256,)
        cos_sim = (pe_normed @ ref_vec).reshape(BEV_H, BEV_W)

        im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1,
                       origin='lower', extent=[0, BEV_W, 0, BEV_H])
        ax.plot(c, r, 'k*', markersize=12, markeredgewidth=1.5, zorder=10)
        _style_bev_ax(ax)
        _add_stats_text(ax, cos_sim)
        ax.set_title(label, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

    fig.suptitle(
        'Exp5c: PE Cosine Similarity to Reference Positions\n'
        'Star = reference position. Red = similar PE, Blue = dissimilar',
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    fname = 'exp5c_similarity.png'
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


# ============================================================
# Statistics
# ============================================================
def print_stats(pe):
    pe_np = pe.numpy()  # (256, 200, 200)
    print(f'  Shape: {list(pe_np.shape)}')
    print(f'  Overall: mean={pe_np.mean():.4f}, std={pe_np.std():.4f}, '
          f'min={pe_np.min():.4f}, max={pe_np.max():.4f}')

    # Col_embed vs row_embed stats
    col_part = pe_np[:128]  # (128, 200, 200)
    row_part = pe_np[128:]  # (128, 200, 200)
    print(f'  Col_embed (ch 0-127): mean={col_part.mean():.4f}, '
          f'std={col_part.std():.4f}')
    print(f'  Row_embed (ch 128-255): mean={row_part.mean():.4f}, '
          f'std={row_part.std():.4f}')

    # Variance analysis
    row_var = np.var(pe_np, axis=1).mean()  # variance across rows
    col_var = np.var(pe_np, axis=2).mean()  # variance across cols
    print(f'  Avg variance across rows: {row_var:.6f}')
    print(f'  Avg variance across cols: {col_var:.6f}')

    # Check if col_embed channels really vary along columns
    col_part_col_var = np.var(col_part, axis=2).mean()  # should be high
    col_part_row_var = np.var(col_part, axis=1).mean()  # should be low
    print(f'  Col_embed: var_across_cols={col_part_col_var:.6f}, '
          f'var_across_rows={col_part_row_var:.6f}')

    row_part_col_var = np.var(row_part, axis=2).mean()  # should be low
    row_part_row_var = np.var(row_part, axis=1).mean()  # should be high
    print(f'  Row_embed: var_across_cols={row_part_col_var:.6f}, '
          f'var_across_rows={row_part_row_var:.6f}')


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    bev_pos = load_bev_pos()
    if bev_pos is None:
        return

    print(f'Loaded bev_pos: {list(bev_pos.shape)}')

    # Take batch 0
    pe = bev_pos[0]  # (256, 200, 200)

    # --- Stats ---
    print('\n' + '=' * 60)
    print('STATISTICS')
    print('=' * 60)
    print_stats(pe)

    # --- Exp 5a: Channel Heatmaps ---
    print('\n' + '=' * 60)
    print('EXP 5a: CHANNEL HEATMAPS')
    print('=' * 60)
    plot_exp5a_channels(pe)
    plot_exp5a_channels_extended(pe)
    plot_exp5a_variance_profile(pe)

    # --- Exp 5b: PCA ---
    print('\n' + '=' * 60)
    print('EXP 5b: PCA DIMENSIONALITY REDUCTION')
    print('=' * 60)
    plot_exp5b_pca(pe)
    plot_exp5b_pca_components(pe)

    # --- Exp 5c: Similarity ---
    print('\n' + '=' * 60)
    print('EXP 5c: COSINE SIMILARITY')
    print('=' * 60)
    plot_exp5c_similarity(pe)

    print(f'\nAll figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
