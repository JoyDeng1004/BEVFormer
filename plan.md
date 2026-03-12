# BEV Query 全链路可视化方案

## 现状分析

### 已有数据（debug_collector 已收集）
- **exp1**: TSA sampling_locations, reference_points, sampling_offsets (per frame×layer)
- **exp2**: TSA attention_weights (bs, 40000, 8, 2, 4)
- **exp3**: ref_2d, shift_ref_2d, shift (encoder 级别)
- **exp4**: TSA fusion output (before/after mean)
- **exp5**: bev_pos (positional encoding)
- **scene_meta**: sample_idx, filename, can_bus

### 缺失数据（新可视化需要）
1. **SCA 侧**: `reference_points_cam`, `bev_mask`, SCA 内部的 `sampling_locations`/`sampling_offsets`/`attention_weights`
2. **scene_meta 缺失**: `lidar2img`, `img_shape`, `lidar2cam`, `cam_intrinsic`
3. **SCA 前后 query 特征**: 用于特征变化热力图
4. **prev_bev 特征**: 用于 TSA 时序对照

### 已有可视化工具（可复用）
- `visualize_exp2.py` 中的: `build_lidar2img`, `get_3d_box_corners`, `project_boxes_to_image`, `draw_bev_boxes`, `load_cam_images`, `_style_bev_ax` 等

---

## 实现计划

### Phase 1: 扩展 debug_collector — 补充 SCA 数据收集

**文件**: `projects/mmdet3d_plugin/bevformer/debug_collector.py`

新增方法:
- `save_sca_sampling(sampling_locations, reference_points_cam, bev_mask, sampling_offsets, attention_weights, query_before, query_after)` — 在 SCA forward 中调用
- `save_scene_meta` 扩展: 额外保存 `lidar2img`, `img_shape`, `lidar2cam`, `cam_intrinsic`

**文件**: `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`

在 `SpatialCrossAttention.forward()` 和 `MSDeformableAttention3D.forward()` 中插入 collector hook:
- `MSDeformableAttention3D.forward()`: 在 sampling_locations 计算后保存 sampling_locations, sampling_offsets, attention_weights
- `SpatialCrossAttention.forward()`: 保存 query before/after SCA, reference_points_cam, bev_mask, indexes

**文件**: `projects/mmdet3d_plugin/bevformer/modules/encoder.py`
s c a
在 `BEVFormerEncoder.forward()` 中扩展 `save_scene_meta` 调用，传入完整 img_metas（含 lidar2img 等）

保存格式:
```
debug_outputs/exp6_sca/frame{fi}_layer{li}.pt
  - sampling_locations: (bs*num_cams, max_len, num_heads, num_levels, num_all_points, 2)
  - sampling_offsets: same shape
  - attention_weights: (bs*num_cams, max_len, num_heads, num_levels, num_all_points)
  - reference_points_cam: (num_cams, bs, num_query, D, 2)
  - bev_mask: (num_cams, bs, num_query, D)
  - indexes: list of per-cam query indices
  - query_before_sca: (bs, num_query, embed_dims)
  - query_after_sca: (bs, num_query, embed_dims)
```

### Phase 2: 提取公共工具模块

**新文件**: `tools/debug_visualize/vis_utils.py`

从 `visualize_exp2.py` 提取公共函数:
- GT 加载: `load_scene_meta`, `load_nuscenes_info`, `load_cam_images`
- BEV 坐标转换: `lidar_to_bev`, `bev_to_lidar`
- 3D box 绘制: `build_lidar2img`, `get_3d_box_corners`, `project_boxes_to_image`, `draw_bev_boxes`, `get_bev_box_corners`
- BEV 轴样式: `_style_bev_ax`, `_add_stats_text`
- 相机面板: `_draw_cameras`, `cam_name_from_path`
- 常量: `PC_RANGE`, `BEV_H`, `BEV_W`, `CAM_LAYOUT`, `BOX_EDGES`, `HEAD_COLORS`

新增:
- `bev_idx_to_physical(row, col)` → (x_m, y_m) 物理坐标
- `physical_to_bev_idx(x_m, y_m)` → (row, col)
- 代表性 query 定义（物理坐标版）:
  - 正前方 10m: x=10, y=0
  - 左前方 30m: x=21.2, y=21.2 (约 45°)
  - 右前方 20m: x=17.3, y=-10
  - 正后方 15m: x=-15, y=0
  - 自车位置: x=0, y=0

### Phase 3: 可视化 1 — SCA 采样过程

**新文件**: `tools/debug_visualize/visualize_sca.py`

功能:
1. **BEV 侧面板**: 在 200×200 BEV 网格上标出选定 query 位置，标注物理坐标(m)，叠加 GT boxes
2. **相机侧面板**: 对每个可见相机:
   - 加载原始图像，叠加 GT 3D bbox 投影（绿色线框）
   - 用圆圈标出 `reference_points_cam`（3D 参考点的 2D 投影，按 Z anchor 分色）
   - 用三角标出 `sampling_locations`（加 offset 后的实际采样位置）
   - 用箭头连接 reference_point → sampling_point，展示偏移方向
3. **特征变化面板**: query_after - query_before 的 L2 norm 热力图 (200×200)
4. **Warp 对比**: 传入 warp 参数时，并排生成 warp 前后两组图

### Phase 4: 可视化 2 — TSA 时序对齐

**新文件**: `tools/debug_visualize/visualize_tsa_alignment.py`

功能:
1. **对齐过程面板**: BEV 平面上同时标出:
   - `ref_2d` (当前帧参考点，蓝色)
   - `shift_ref_2d` (ego motion 补偿后，红色)
   - TSA `sampling_locations` (各 head 不同标记)
   - 箭头: ref_2d → shift_ref_2d 展示 ego motion shift
2. **时序对照面板**: 当前帧 BEV 特征 PCA 和前一帧 BEV 特征 PCA 并排，用连线标出 TSA 对应关系
3. **融合权重面板**:
   - 由于 TSA 是 per-queue softmax + hard mean，展示每个 queue 的 attention concentration
   - Bar chart: 对选定 query，展示 8 heads × 2 queues 的权重分配
4. **Warp 对比**: warp 前后的 shift_ref_2d 差异

### Phase 5: 可视化 3 — 端到端总览图

**新文件**: `tools/debug_visualize/visualize_overview.py`

布局 (一张大图):
```
┌─────────────────┬──────────────────┬──────────────────┐
│  左: 6 相机图像  │  中: 当前帧 BEV   │  右: 前一帧 BEV   │
│  (2×3 网格)      │                  │                  │
│  + GT 3D bbox    │  + GT boxes      │  + TSA 采样位置   │
│  + SCA 采样点    │  + query 位置    │  + 时序对应连线   │
│                  │  + TSA 轨迹      │                  │
└─────────────────┴──────────────────┴──────────────────┘
```

- 选定 query 使用一致颜色编码（跨三个面板）
- 每个 query 标注物理坐标
- 支持 warp 参数，生成 warp/no-warp 对比

### Phase 6: Warp 支持

所有三个可视化脚本共享 warp 接口:

```python
# CLI 参数
--warp_dx 0.5    # x 方向平移 (m)
--warp_dy 0.3    # y 方向平移 (m)
--warp_dtheta 2  # 旋转角度 (度)
--compare         # 生成 warp 前后对比图
```

Warp 实现: SE(2) 变换应用于 BEV 坐标系
- 对 ref_2d / shift_ref_2d 施加扰动
- 对 reference_points_cam 通过修改 lidar2img 间接影响
- 对比图: 左半 = original, 右半 = warped, 差异用颜色编码

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `debug_collector.py` | 新增 `save_sca_sampling`, 扩展 `save_scene_meta` |
| 修改 | `spatial_cross_attention.py` | 插入 SCA collector hook |
| 修改 | `encoder.py` | 扩展 scene_meta 保存内容 |
| 新建 | `tools/debug_visualize/vis_utils.py` | 公共工具模块 |
| 新建 | `tools/debug_visualize/visualize_sca.py` | 可视化 1: SCA 采样 |
| 新建 | `tools/debug_visualize/visualize_tsa_alignment.py` | 可视化 2: TSA 时序对齐 |
| 新建 | `tools/debug_visualize/visualize_overview.py` | 可视化 3: 端到端总览 |

## 执行顺序

1. Phase 1 + Phase 2 先行（数据收集 + 工具提取）
2. 重新运行 `TSA_DEBUG=1` 收集包含 SCA 数据的新 debug_outputs
3. Phase 3-5 可视化脚本（依赖新数据）
4. Phase 6 warp 支持贯穿所有脚本
