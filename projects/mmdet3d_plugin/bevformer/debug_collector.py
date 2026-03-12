"""
Unified data collector for TSA visualization experiments.

Usage:
    # Enable via environment variable before running:
    TSA_DEBUG=1 python tools/test.py ...

    # All data is saved to debug_outputs/<exp_name>/ with per-experiment subdirs.
    # A meta.json is generated after collection completes.

When TSA_DEBUG is not set (default), all collector methods are no-ops,
so this module can be safely imported in production code with zero overhead.
"""
import os
import json
import numpy as np
import torch
from datetime import datetime

# Toggle via environment variable — zero overhead when disabled
TSA_DEBUG = os.environ.get('TSA_DEBUG', '0') == '1'

_OUTPUT_ROOT = 'debug_outputs'
_MAX_FRAMES = 2
_NUM_LAYERS = 6


class _Collector:
    """Singleton data collector. Access via module-level `collector` instance."""

    def __init__(self):
        self._tsa_sample_count = 0
        self._tsa_fusion_count = 0
        self._encoder_count = 0
        self._head_count = 0
        self._scene_meta_count = 0
        self._sca_sample_count = 0
        self._done = False
        self._meta = {}

        if TSA_DEBUG:
            self._init_meta()
            print(f'[Collector] TSA_DEBUG enabled. Collecting {_MAX_FRAMES} frames '
                  f'x {_NUM_LAYERS} layers. Output: {_OUTPUT_ROOT}/')

    def _init_meta(self):
        self._meta = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': 'ckpts/bevformer_r101_dcn_24ep.pth',
            'config': 'projects/configs/bevformer/bevformer_base.py',
            'bev_h': 200, 'bev_w': 200,
            'embed_dims': 256, 'num_heads': 8,
            'num_points': 4, 'num_bev_queue': 2,
            'max_frames': _MAX_FRAMES,
            'num_layers': _NUM_LAYERS,
        }
        try:
            self._meta['git_hash'] = os.popen('git rev-parse HEAD').read().strip()
            self._meta['git_branch'] = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()
        except Exception:
            pass

    @property
    def is_done(self):
        return self._done

    def _ensure_dir(self, subdir):
        path = os.path.join(_OUTPUT_ROOT, subdir)
        os.makedirs(path, exist_ok=True)
        return path

    # ========================
    # TSA forward — hook 1: after sampling_locations (saves exp1 + exp2)
    # ========================
    def save_tsa_sampling(self, sampling_locations, reference_points,
                          sampling_offsets, attention_weights_pre_reshape,
                          has_history):
        """
        Called from TSA forward, after sampling_locations computation.
        Saves data for Experiment 1 (sampling points) and Experiment 2 (attention weights).

        Args:
            sampling_locations: (bs*2, 40000, num_heads, 1, num_points, 2)
            reference_points:   (bs*2, 40000, 1, 2)
            sampling_offsets:   (bs*2, 40000, num_heads, 1, num_points, 2)
            attention_weights_pre_reshape: (bs, 40000, 8, 2, 4) — after softmax, before permute
            has_history: bool
        """
        if not TSA_DEBUG or self._done:
            return
        if self._tsa_sample_count >= _MAX_FRAMES * _NUM_LAYERS:
            return

        fi = self._tsa_sample_count // _NUM_LAYERS
        li = self._tsa_sample_count % _NUM_LAYERS

        # --- Exp1: Sampling Locations ---
        exp1_dir = self._ensure_dir('exp1_sampling_locations')
        torch.save({
            'sampling_locations': sampling_locations.detach().cpu(),
            'reference_points': reference_points.detach().cpu(),
            'sampling_offsets': sampling_offsets.detach().cpu(),
            'bev_h': 200, 'bev_w': 200,
            'layer_idx': li, 'frame_idx': fi,
            'has_history': has_history,
        }, os.path.join(exp1_dir, f'frame{fi}_layer{li}.pt'))

        # --- Exp2: Attention Weights ---
        exp2_dir = self._ensure_dir('exp2_attention_weights')
        torch.save({
            'attention_weights': attention_weights_pre_reshape.detach().cpu(),
            # shape: (bs, 40000, 8, 2, 4) — [history_4pts, current_4pts]
            'layer_idx': li, 'frame_idx': fi,
            'has_history': has_history,
        }, os.path.join(exp2_dir, f'frame{fi}_layer{li}.pt'))

        print(f'[Collector] TSA exp1+exp2: frame={fi}, layer={li}, has_history={has_history}')
        self._tsa_sample_count += 1

    # ========================
    # TSA forward — hook 2: at fusion point (saves exp4)
    # ========================
    def save_tsa_fusion(self, output_before_fusion, output_after_fusion,
                        has_history):
        """
        Called from TSA forward, around the output.mean(-1) fusion point.
        Saves data for Experiment 4 (history-current fusion).

        Args:
            output_before_fusion: (num_query, embed_dims, bs, 2)
            output_after_fusion:  (num_query, embed_dims, bs)
            has_history: bool
        """
        if not TSA_DEBUG or self._done:
            return
        if self._tsa_fusion_count >= _MAX_FRAMES * _NUM_LAYERS:
            return

        fi = self._tsa_fusion_count // _NUM_LAYERS
        li = self._tsa_fusion_count % _NUM_LAYERS

        exp4_dir = self._ensure_dir('exp4_fusion_output')
        torch.save({
            'output_before_fusion': output_before_fusion.detach().cpu(),
            'output_after_fusion': output_after_fusion.detach().cpu(),
            'layer_idx': li, 'frame_idx': fi,
            'has_history': has_history,
        }, os.path.join(exp4_dir, f'frame{fi}_layer{li}.pt'))

        print(f'[Collector] TSA exp4:    frame={fi}, layer={li}')
        self._tsa_fusion_count += 1

        # Check completion after the last hook in the forward
        if self._tsa_fusion_count >= _MAX_FRAMES * _NUM_LAYERS:
            self._done = True
            self._save_meta()
            print('[Collector] All data collection complete! You can Ctrl+C now.')

    # ========================
    # SCA forward — sampling data (saves exp6)
    # ========================
    def save_sca_sampling(self, sampling_locations, sampling_offsets,
                          attention_weights, reference_points_cam, bev_mask,
                          indexes, query_before, query_after):
        """
        Called from SpatialCrossAttention.forward, after deformable attention.
        Saves data for Experiment 6 (SCA sampling process).

        Args:
            sampling_locations: (bs*num_cams, max_len, num_heads, num_levels, num_all_points, 2)
            sampling_offsets:   same shape
            attention_weights:  (bs*num_cams, max_len, num_heads, num_levels, num_all_points)
            reference_points_cam: (num_cams, bs, num_query, D, 2) — normalized
            bev_mask:           (num_cams, bs, num_query, D)
            indexes:            list[Tensor] — per-cam valid query indices
            query_before:       (bs, num_query, embed_dims)
            query_after:        (bs, num_query, embed_dims)
        """
        if not TSA_DEBUG or self._done:
            return
        if self._sca_sample_count >= _MAX_FRAMES * _NUM_LAYERS:
            return

        fi = self._sca_sample_count // _NUM_LAYERS
        li = self._sca_sample_count % _NUM_LAYERS

        exp6_dir = self._ensure_dir('exp6_sca')
        torch.save({
            'sampling_locations': sampling_locations.detach().cpu(),
            'sampling_offsets': sampling_offsets.detach().cpu(),
            'attention_weights': attention_weights.detach().cpu(),
            'reference_points_cam': reference_points_cam.detach().cpu(),
            'bev_mask': bev_mask.detach().cpu(),
            'indexes': [idx.detach().cpu() for idx in indexes],
            'query_before': query_before.detach().cpu(),
            'query_after': query_after.detach().cpu(),
            'layer_idx': li, 'frame_idx': fi,
        }, os.path.join(exp6_dir, f'frame{fi}_layer{li}.pt'))

        print(f'[Collector] SCA exp6:    frame={fi}, layer={li}')
        self._sca_sample_count += 1

    # ========================
    # Encoder forward (saves exp3)
    # ========================
    def save_encoder(self, ref_2d, shift_ref_2d, shift, has_prev_bev):
        """
        Called from BEVFormerEncoder forward, after shift_ref_2d computation.
        Saves data for Experiment 3 (ego motion shift).

        Args:
            ref_2d:        (bs, 40000, 1, 2) — original BEV grid
            shift_ref_2d:  (bs, 40000, 1, 2) — shifted for history alignment
            shift:         (bs, 2) — ego motion shift in normalized coords
            has_prev_bev:  bool
        """
        if not TSA_DEBUG or self._done:
            return
        if self._encoder_count >= _MAX_FRAMES:
            return

        fi = self._encoder_count
        exp3_dir = self._ensure_dir('exp3_reference_points')
        torch.save({
            'ref_2d': ref_2d.detach().cpu(),
            'shift_ref_2d': shift_ref_2d.detach().cpu(),
            'shift': shift.detach().cpu(),
            'has_prev_bev': has_prev_bev,
            'frame_idx': fi,
        }, os.path.join(exp3_dir, f'frame{fi}.pt'))

        print(f'[Collector] Encoder exp3: frame={fi}, has_prev_bev={has_prev_bev}')
        self._encoder_count += 1

    # ========================
    # BEVFormerHead forward (saves exp5)
    # ========================
    def save_head(self, bev_pos):
        """
        Called from BEVFormerHead forward, after positional encoding.
        Saves data for Experiment 5 (BEV positional encoding).

        Args:
            bev_pos: (bs, 256, 200, 200)
        """
        if not TSA_DEBUG or self._done:
            return
        if self._head_count >= 1:  # PE is learned & static, only need one sample
            return

        exp5_dir = self._ensure_dir('exp5_bev_pos')
        torch.save({
            'bev_pos': bev_pos.detach().cpu(),
        }, os.path.join(exp5_dir, 'bev_pos.pt'))

        print('[Collector] Head exp5: bev_pos saved')
        self._head_count += 1

    # ========================
    # Scene metadata (sample token, camera paths)
    # ========================
    def save_scene_meta(self, img_metas):
        """
        Save scene metadata for linking debug outputs to nuScenes samples.
        Called from BEVFormerEncoder.forward, once per frame.

        Args:
            img_metas: list[dict], each dict contains 'sample_idx',
                       'scene_token', 'filename', 'can_bus', etc.
        """
        if not TSA_DEBUG or self._done:
            return
        if self._scene_meta_count >= _MAX_FRAMES:
            return

        fi = self._scene_meta_count
        meta_dir = self._ensure_dir('scene_meta')

        meta = img_metas[0]
        scene_info = {
            'frame_idx': fi,
            'sample_idx': meta.get('sample_idx', None),
            'scene_token': meta.get('scene_token', None),
            'filename': meta.get('filename', []),
            'can_bus': meta.get('can_bus', None),
        }
        # can_bus may be a numpy array
        if hasattr(scene_info['can_bus'], 'tolist'):
            scene_info['can_bus'] = scene_info['can_bus'].tolist()

        with open(os.path.join(meta_dir, f'frame{fi}.json'), 'w') as f:
            json.dump(scene_info, f, indent=2, ensure_ascii=False)

        # Save camera calibration as separate .pt (numpy arrays not JSON-friendly)
        lidar2img = meta.get('lidar2img', None)
        img_shape = meta.get('img_shape', None)
        if lidar2img is not None:
            calib = {
                'lidar2img': np.array(lidar2img),
                'img_shape': img_shape,
            }
            torch.save(calib, os.path.join(meta_dir, f'frame{fi}_calib.pt'))

        print(f'[Collector] Scene meta: frame={fi}, '
              f'sample_idx={scene_info["sample_idx"]}')
        self._scene_meta_count += 1

    # ========================
    # Meta
    # ========================
    def _save_meta(self):
        self._meta['collection_completed'] = datetime.now().isoformat()
        self._meta['total_tsa_sample_calls'] = self._tsa_sample_count
        self._meta['total_tsa_fusion_calls'] = self._tsa_fusion_count
        self._meta['total_encoder_calls'] = self._encoder_count
        self._meta['total_head_calls'] = self._head_count
        self._meta['total_scene_meta_calls'] = self._scene_meta_count
        self._meta['total_sca_sample_calls'] = self._sca_sample_count
        self._meta['experiments_collected'] = []
        for exp in ['exp1_sampling_locations', 'exp2_attention_weights',
                    'exp3_reference_points', 'exp4_fusion_output',
                    'exp5_bev_pos', 'exp6_sca', 'scene_meta']:
            d = os.path.join(_OUTPUT_ROOT, exp)
            if os.path.isdir(d):
                self._meta['experiments_collected'].append(exp)

        meta_path = os.path.join(_OUTPUT_ROOT, 'meta.json')
        os.makedirs(_OUTPUT_ROOT, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump(self._meta, f, indent=2, ensure_ascii=False)
        print(f'[Collector] Meta saved to {meta_path}')


# Module-level singleton
collector = _Collector()
