# TSA (Temporal Self-Attention) 可视化实验提案

## 背景

通过 pdb 调试，我们已经验证了 TSA 的核心机制。本提案设计一系列可视化实验，
从直觉层面深入理解 TSA 的运作方式。

---

## 项目信息

- 代码路径: `/gs/bs/tga-RLA/qdeng/BEVFormer`
- Checkpoint: `ckpts/bevformer_r101_dcn_24ep.pth`
- Config: `projects/configs/bevformer/bevformer_base.py`
- 数据集: nuScenes, 位于 `/gs/bs/tga-RLA/qdeng/data/nuscenes`
- Debug branch: `debug/tsa-analysis`
- 调试脚本: `tools/debug_tsa.sh` (单GPU eval模式)
- BEV尺寸: 200x200, embed_dims=256, num_heads=8, num_points=4, num_bev_queue=2

## 关键文件

| 文件 | 作用 |
|------|------|
| `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` | TSA 核心实现 (hook: exp1+exp2+exp4) |
| `projects/mmdet3d_plugin/bevformer/modules/encoder.py` | BEVFormerEncoder, prev_bev stacking, ref_2d 生成 (hook: exp3) |
| `projects/mmdet3d_plugin/bevformer/modules/transformer.py` | ego motion 补偿 (rotation + shift) |
| `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` | BEV query/PE 生成 (hook: exp5) |
| `projects/mmdet3d_plugin/bevformer/debug_collector.py` | 统一数据收集器 (TSA_DEBUG=1 启用) |

---

## 调试中已验证的关键发现

| 内容 | 第一帧 (无 history) | 第二帧 (有 history) |
|------|---------------------|---------------------|
| value | None → self-copy `stack([query,query])` | 真实的 `(bs*2, 40000, 256)` |
| history vs current 相似度 | cosine_sim = 1.0 | cosine_sim = -0.13 |
| reference_points shift | 无偏移 (history = current) | dy≈0.0416 (约8像素 ego motion) |
| attention weights 集中度 | 分散 (max≈0.86) | 集中,接近 one-hot (max≈0.999) |

---

## 可视化实验设计

### 实验 1: Sampling Points 在 BEV 平面上的分布

**目的**: 直观看到每个 head 在 history/current BEV 上"去哪里看"

**数据来源**: `temporal_self_attention.py` forward 中的 `sampling_locations`
- shape: `(bs*2, 40000, num_heads, 1, num_points, 2)`, 值域 [0,1]
- 前 bs 个 = history BEV 上的采样位置, 后 bs 个 = current BEV 上的采样位置

**可视化方案**:
- 选取若干代表性 query 位置: BEV中心(idx=20100), 左上角(idx=0), 右下角(idx=39999), 以及若干中间位置
- 画 200x200 的 BEV grid
- 对每个选定的 query, 用红色 X 标记其 reference point 位置
- 用不同颜色标记 8 个 head 的 4 个 sampling point (共 32 个点)
- 左右两幅子图: 左=history BEV, 右=current BEV

**对比实验**:
- (a) 第一帧 (value=None, self-copy) vs 第二帧 (真实 history)
- (b) BEV 中心 query vs BEV 边缘 query
- (c) Encoder 第 1 层 vs 第 6 层 (看深层是否学到不同的采样模式)

**提取代码**: 由统一收集器 `debug_collector.py` 的 `save_tsa_sampling()` 自动处理

**保存位置**: `debug_outputs/exp1_sampling_locations/frame{i}_layer{j}.pt`

**保存内容**:
```python
{
    'sampling_locations': ...,  # (bs*2, 40000, 8, 1, 4, 2)
    'reference_points': ...,   # (bs*2, 40000, 1, 2)
    'sampling_offsets': ...,   # (bs*2, 40000, 8, 1, 4, 2)
    'bev_h': 200, 'bev_w': 200,
    'layer_idx': int, 'frame_idx': int,
    'has_history': bool,
}
```

---

### 实验 2: Attention Weights 热力图

**目的**: 看不同 head 分别更关注 history 还是 current, 以及关注哪些采样点

**数据来源**: `temporal_self_attention.py` forward 中的 `attention_weights`
- softmax 之后, reshape 之前: shape `(bs, 40000, 8, 2, 4)`
- dim=3 的 2 = [history, current], dim=4 的 4 = 4个采样点

**可视化方案 2a: 单 query 的 head×point 热力图**
- 选取 BEV 中心 query
- 画一个 (8 heads) × (2×4=8 points) 的热力图
- 行 = 8 个 head, 列 = [hist_p0, hist_p1, hist_p2, hist_p3, curr_p0, curr_p1, curr_p2, curr_p3]
- 颜色深浅表示权重大小
- 对比第一帧 vs 第二帧

**可视化方案 2b: History vs Current 权重的空间分布图**
- 对每个 query, 计算 history_total_weight = sum(所有 head 的 history 权重) / total
- 得到 40000 个值, reshape 成 200x200
- 画热力图: 红色=偏好 history, 蓝色=偏好 current
- 这能揭示 BEV 不同区域对时序信息的依赖程度

**提取代码**: 由统一收集器 `debug_collector.py` 的 `save_tsa_sampling()` 自动处理

**保存位置**: `debug_outputs/exp2_attention_weights/frame{i}_layer{j}.pt`

**保存内容**:
```python
{
    'attention_weights': ...,  # (bs, 40000, 8, 2, 4) — softmax后, reshape前
    'layer_idx': int, 'frame_idx': int,
    'has_history': bool,
}
```

---

### 实验 3: Reference Points 的 Ego Motion Shift 可视化

**目的**: 直观看到 ego motion 补偿如何改变 history 的参考点位置

**数据来源**: `encoder.py` forward 中的 `ref_2d`, `shift_ref_2d`, `hybird_ref_2d`
- ref_2d: 原始 BEV grid, shape (bs, 40000, 1, 2), 值域 [0,1]
- shift_ref_2d: 加上 ego shift 后的 ref_2d
- hybird_ref_2d: stack([shift_ref_2d, ref_2d]), shape (bs*2, 40000, 1, 2)

**可视化方案**:
- 画 200x200 grid
- 蓝色箭头: 从 current ref_point 指向 history ref_point (shift_ref_2d)
- 箭头方向和大小 = ego motion 的方向和幅度
- 可以每隔 10 个 query 画一个箭头 (避免太密)

**提取代码**: 由统一收集器 `debug_collector.py` 的 `save_encoder()` 自动处理

**保存位置**: `debug_outputs/exp3_reference_points/frame{i}.pt`

**保存内容**:
```python
{
    'ref_2d': ...,        # (bs, 40000, 1, 2)
    'shift_ref_2d': ...,  # (bs, 40000, 1, 2)
    'shift': ...,         # (bs, 2)
    'has_prev_bev': bool,
    'frame_idx': int,
}
```

---

### 实验 4: History-Current 融合前后的 BEV Feature Map

**目的**: 看 TSA 融合的效果, history 和 current 各贡献了什么

**数据来源**: `temporal_self_attention.py` forward 中的 output
- line 261-262: `output.view(Q, C, bs, 2)` → `.mean(-1)`
- 在 mean 之前保存, dim=-1 的 2 = [history分支输出, current分支输出]

**可视化方案 4a: Feature Map 对比**
- 将 history 分支输出 reshape 成 (200, 200, 256), 取前3个channel 做伪彩色图
- 同样处理 current 分支输出 和 融合后输出
- 三张图并排: history_output | current_output | fused_output

**可视化方案 4b: 逐像素 cosine similarity**
- 计算 history_output 和 current_output 在每个 BEV 位置的 cosine similarity
- reshape 成 200x200 热力图
- 高相似度区域 = 静态背景 (两帧一致), 低相似度区域 = 动态物体/遮挡

**提取代码**: 由统一收集器 `debug_collector.py` 的 `save_tsa_fusion()` 自动处理

**保存位置**: `debug_outputs/exp4_fusion_output/frame{i}_layer{j}.pt`

**保存内容**:
```python
{
    'output_before_fusion': ...,  # (num_query, embed_dims, bs, 2)
    'output_after_fusion': ...,   # (num_query, embed_dims, bs)
    'layer_idx': int, 'frame_idx': int,
    'has_history': bool,
}
```

---

### 实验 5: BEV PE (Positional Encoding) 可视化

**目的**: 理解 Learned PE 编码了什么空间信息

**数据来源**: `bevformer_head.py` 中的 `bev_pos`
- 由 LearnedPositionalEncoding 生成, shape (bs, 256, 200, 200)
- 本质是 row_embed(200,128) 和 col_embed(200,128) 的组合

**可视化方案 5a: PE channel 可视化**
- 取若干 channel (如 channel 0, 64, 128, 192), reshape 成 200x200, 画热力图
- 前 128 channel 来自 col_embed (应该只沿 x 方向变化)
- 后 128 channel 来自 row_embed (应该只沿 y 方向变化)

**可视化方案 5b: PE 的 PCA 降维**
- 将 (200, 200, 256) 的 PE 做 PCA 降到 3 维
- 用 RGB 颜色表示, 画 200x200 的图
- 空间上相近的位置颜色应该相近

**提取代码**: 由统一收集器 `debug_collector.py` 的 `save_head()` 自动处理

**保存位置**: `debug_outputs/exp5_bev_pos/bev_pos.pt`

**保存内容**:
```python
{
    'bev_pos': ...,  # (bs, 256, 200, 200)
}
```

---

### 实验 6: 逐 Encoder Layer 的演变

**目的**: 观察 6 层 encoder 中 TSA 行为的逐层变化

**数据来源**: 每层 TSA 的 sampling_locations 和 attention_weights

**可视化方案**:
- 固定选取 BEV 中心 query
- 6 张子图, 每张对应一层 encoder
- 每张图画该层 TSA 在 history BEV 上的 32 个采样点 (8 heads × 4 points)
- 观察: 浅层 → 深层, 采样范围是否扩大? 注意力是否更集中?

**实现方式**: 收集器已自动按 `frame{i}_layer{j}.pt` 保存每层数据，无需额外修改代码。
实验 1 和 2 的数据天然包含所有 6 层，直接用 `visualize_exp1.py` 中的 layer comparison 功能即可。

---

## 实验管理

### 目录结构

按实验编号分目录，`.pt` 原始数据和可视化图片共存但分开，可随时用不同参数重新画图而不必重跑模型：

```
debug_outputs/
├── meta.json                         ← 全局元数据 (git hash, checkpoint, 时间戳)
├── exp1_sampling_locations/
│   ├── frame0_layer0.pt
│   ├── frame0_layer5.pt
│   ├── frame1_layer0.pt
│   ├── ...
│   └── figures/
│       ├── frame0_layer0_center.png
│       ├── frame1_layer5_overview.png
│       └── layer_compare_frame1_center.png
├── exp2_attention_weights/
│   ├── frame0_layer0.pt
│   ├── ...
│   └── figures/
├── exp3_reference_points/
│   ├── frame0.pt
│   ├── frame1.pt
│   └── figures/
├── exp4_fusion_output/
│   ├── frame0_layer0.pt
│   ├── ...
│   └── figures/
├── exp5_bev_pos/
│   ├── bev_pos.pt
│   └── figures/
└── summary/                          ← 最终挑选的对比图 (用于汇报/论文)
    ├── first_vs_second_frame.png
    └── shallow_vs_deep_layer.png
```

### 元数据 (meta.json)

每次数据收集自动生成，记录完整的实验上下文，确保事后可追溯：

```json
{
  "timestamp": "2026-03-10T14:30:00",
  "git_hash": "66b65f3...",
  "git_branch": "debug-tsa-analysis",
  "checkpoint": "ckpts/bevformer_r101_dcn_24ep.pth",
  "config": "projects/configs/bevformer/bevformer_base.py",
  "bev_h": 200, "bev_w": 200,
  "max_frames": 2, "num_layers": 6,
  "experiments_collected": ["exp1_sampling_locations", "exp2_attention_weights", ...]
}
```

### 统一数据收集器

所有实验的数据在**同一次推理**中一次性收集，避免反复跑模型：

- 收集器: `projects/mmdet3d_plugin/bevformer/debug_collector.py`
- 通过环境变量 `TSA_DEBUG=1` 开关控制，不设置时所有 hook 为 no-op，零开销
- TSA forward 中 2 个 hook point → exp1 + exp2 + exp4
- Encoder forward 中 1 个 hook point → exp3
- BEVFormerHead forward 中 1 个 hook point → exp5

### 脚本分离

| 脚本 | 用途 | 需要 GPU |
|------|------|----------|
| `tools/debug_tsa.sh` | 数据收集 (跑模型, 存 `.pt`) | 是 |
| `tools/visualize_exp1.py` | 实验1 可视化 | 否 |
| `tools/visualize_exp2.py` | 实验2 可视化 | 否 |
| ... | ... | 否 |

可视化脚本只读取 `.pt` 文件画图，可在本地笔记本上运行（scp `.pt` 下来即可），
可视化参数（配色、选哪些 query、子图布局）会反复调整，不应跟模型推理耦合。

---

## 实现计划

### 阶段 1: 数据收集 (一次推理, 全部实验)
1. ~~修改 TSA / encoder / head 的 forward, 在关键位置插入数据收集 hook~~ ✅ 已完成
2. ~~创建统一收集器 `debug_collector.py`~~ ✅ 已完成
3. 运行 `TSA_DEBUG=1 bash tools/debug_tsa.sh` (看到 "complete" 后 Ctrl+C)
4. 验证 `debug_outputs/meta.json` 和各实验子目录的 `.pt` 文件

### 阶段 2: 可视化脚本
1. ~~`tools/visualize_exp1.py`~~ ✅ 已完成
2. `tools/visualize_exp2.py` — attention weights 热力图
3. `tools/visualize_exp3.py` — ego motion shift 箭头图
4. ~~`tools/visualize_exp4.py` — 融合前后 feature map~~ ✅ 已完成
5. `tools/visualize_exp5.py` — BEV PE 可视化

### 阶段 3: 对比分析
1. 第一帧 vs 第二帧
2. BEV 中心 vs 边缘
3. 浅层 vs 深层
4. 静态区域 vs 动态物体区域

---

## 优先级建议

| 优先级 | 实验 | 理由 |
|--------|------|------|
| P0 (最先做) | 实验 1: Sampling Points 分布 | 最直观, 能直接看到 TSA "在看哪里" |
| P0 | 实验 2b: History/Current 权重空间分布 | 能揭示 BEV 不同区域对时序信息的依赖 |
| P1 | 实验 4b: 融合前 cosine similarity | 能识别动态 vs 静态区域 |
| P1 | 实验 3: Ego Motion Shift | 验证运动补偿是否正确 |
| P2 | 实验 5: PE 可视化 | 辅助理解, 非 TSA 核心 |
| P2 | 实验 6: 逐层演变 | 需要改动较多代码, 但 insight 很深 |
