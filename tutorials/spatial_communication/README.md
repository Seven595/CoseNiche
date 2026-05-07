# 空间通讯分析教程

本模块展示如何利用 CoseNiche 模型的空间注意力权重分析细胞-细胞通讯和配体-受体相互作用。

## 📋 概述

CoseNiche 模型在学习空间上下文时，通过空间交叉注意力机制捕获了相邻 spots 之间的信息流动。这些注意力权重可以揭示：

- 空间相邻的 spots 之间的基因交互
- 组织边界处的细胞通讯模式
- 配体-受体对的空间共定位
- 疾病相关的通讯异常

## 🎯 主要特点

- **Spot-to-Spot 注意力**: 分析空间位点间的注意力流向
- **边界检测**: 自动识别组织边界的 spots
- **配体-受体分析**: 结合 CellChat 等数据库识别通讯对
- **极坐标可视化**: 直观展示空间方向性通讯
- **Domain-aware 分析**: 考虑组织区域的通讯模式

## 📂 文件说明

```
spatial_communication/
├── README.md                           # 本文件
├── config.yaml                         # 配置文件
├── 0_check_data.py                     # 数据完整性检查
├── 1_export_spatial_data.py            # 导出空间注意力数据
├── 2_expression_analysis.py            # 基因表达与注意力关联分析
├── 3_prepare_polar.py                  # 准备极坐标图数据
├── 3.5_prepare_polar_domain_aware.py   # Domain-aware 极坐标数据
├── 4_boundary_visualizer.py            # 边界分析和可视化
├── 5_plot_polar.py                     # 极坐标图绘制
└── utils.py                            # 工具函数
```

## 🚀 快速开始

### 前置条件

确保已运行模型推理并保存了空间注意力权重：

```bash
python ../../scripts/extract_embeddings.py \
    --model_path /path/to/model.safetensors \
    --h5ad_path /path/to/data.h5ad \
    --output_dir ./embeddings_output \
    --save_attention \
    --save_spatial_attention \
    --device cuda:0
```

### 步骤 0: 数据完整性检查

```bash
python 0_check_data.py --data_dir ./embeddings_output
```

**检查内容**:
- 注意力文件是否存在
- 空间坐标是否完整
- Cluster/Domain 标注是否存在
- Marker 基因是否可用

### 步骤 1: 导出空间注意力数据

```bash
python 1_export_spatial_data.py \
    --dataset PDAC \
    --base_dir ./embeddings_output \
    --layer 5 \
    --batch_size 50
```

**输出**:
- `attention_exports/attention_layer{N}.csv`: 注意力详细数据
- `attention_exports/spatial_coords.csv`: 空间坐标
- `attention_exports/optimized_structures.pkl`: 优化的数据结构

**attention_layer{N}.csv 格式**:
| center_idx | neigh_idx | distance | q_gene | kv_gene | attn_score | q_expr | kv_expr |
|------------|-----------|----------|--------|---------|------------|--------|---------|
| 0 | 1 | 45.2 | KRAS | EGFR | 0.234 | 3.45 | 2.89 |

### 步骤 2: 基因表达与注意力关联分析（可选）

```bash
python 2_expression_analysis.py \
    --data_dir ./attention_exports \
    --layer 5
```

分析注意力权重与基因表达的相关性。

### 步骤 3: 准备极坐标图数据

```bash
python 3_prepare_polar.py \
    --out_dir ./attention_exports \
    --layer 5 \
    --topk 10
```

**输出**:
- `aggregated_spot_level_topk10.csv`: Spot 级别 top genes
- `aggregated_domain_neighbors_topk10.csv`: Domain 邻居级别
- `aggregated_domain_global_topk10.csv`: Domain 全局级别

或使用 domain-aware 版本（考虑组织边界）：

```bash
python 3.5_prepare_polar_domain_aware.py \
    --out_dir ./attention_exports \
    --layer 5 \
    --topk 10 \
    --domain_weight 0.6
```

### 步骤 4: 边界分析和可视化

```bash
python 4_boundary_visualizer.py \
    --data_dir ./attention_exports \
    --output_dir ./boundary_analysis \
    --threshold 0.3
```

**输出**:
- `boundary_spots.csv`: 边界 spots 列表
- `cross_cluster_ratio.csv`: 跨 cluster 通讯比例
- `gene_pairs.csv`: 边界处的基因对
- `network_plot.png`: 空间交互网络图
- `lr_interactions.png`: 配体-受体点图

### 步骤 5: 极坐标图绘制

```bash
python 5_plot_polar.py \
    --data_dir ./attention_exports \
    --output_dir ./polar_plots \
    --layer 5 \
    --view kv
```

**输出**:
- `polar_plots/polar_{domain}_{gene}.png`: 每个 domain 每个基因的极坐标图
- `polar_plots/summary_grid.png`: 汇总网格图

## 📊 输出结果详解

### 1. 注意力导出数据 (Step 1)

**attention_layer{N}.csv**:

详细的注意力信息，每行表示一个 query gene 对一个 neighbor spot 中的 kv gene 的注意力：

```
center_idx: 中心 spot 索引
neigh_idx: 邻居 spot 索引
distance: 空间距离
q_gene: Query 基因（中心 spot）
kv_gene: Key/Value 基因（邻居 spot）
attn_score: 注意力得分
q_expr: Query 基因表达量
kv_expr: Key/Value 基因表达量
```

### 2. 极坐标数据 (Step 3)

**Spot 级别** (`aggregated_spot_level_topk10.csv`):
| center_idx | center_domain | kv_gene | total_score | hit_neighbors | avg_score | rank |
|------------|---------------|---------|-------------|---------------|-----------|------|
| 0 | Tumor | EGFR | 2.345 | 5 | 0.469 | 1 |

**Domain 邻居级别** (`aggregated_domain_neighbors_topk10.csv`):
| center_domain | neigh_domain | kv_gene | total_score | hit_pairs | avg_score | rank |
|---------------|--------------|---------|-------------|-----------|-----------|------|
| Tumor | Stroma | FN1 | 15.67 | 89 | 0.176 | 1 |

### 3. 边界分析 (Step 4)

**boundary_spots.csv**:
| spot_idx | cluster | cross_cluster_ratio | in_cluster_attn | out_cluster_attn |
|----------|---------|---------------------|-----------------|------------------|
| 45 | Tumor | 0.67 | 1.23 | 2.45 |

边界 spots 的判定标准：`cross_cluster_ratio > threshold`

**gene_pairs.csv**:
| center_idx | neigh_idx | center_cluster | neigh_cluster | gene_A | gene_B | attn_score | is_lr_pair |
|------------|-----------|----------------|---------------|--------|--------|------------|------------|
| 45 | 78 | Tumor | Stroma | TGFB1 | TGFBR2 | 0.234 | True |

### 4. 极坐标图 (Step 5)

**图形解释**:
- **角度**: 表示邻居 spot 相对于中心 spot 的空间方向
- **半径**: 表示注意力强度
- **颜色**: 可以表示不同的基因或 domains
- **扇形宽度**: 可以表示该方向的 spot 数量

**两种视角**:
- `--view kv`: 显示邻居 spot 的 KV genes（默认）
- `--view q`: 显示中心 spot 的 Query genes

## 🔬 算法原理

### 1. 空间注意力提取

从 CoseNiche 的 spatial cross-attention 层提取权重：

```
Attn(Q_center, K_neighbors, V_neighbors) = softmax(Q·K^T / √d_k)·V
```

提取 `softmax(Q·K^T / √d_k)` 作为注意力权重。

### 2. Spot-to-Spot 注意力聚合

将基因级别的注意力聚合到 spot 级别：

```
A_spot(i, j) = Σ_g Σ_h attn(g_i, h_j)
```

其中：
- `g_i`: spot i 的基因 g
- `h_j`: spot j 的基因 h

### 3. 边界识别

计算每个 spot 的跨 cluster 注意力比例：

```
cross_cluster_ratio = Σ_{j∈out-cluster} A(i,j) / Σ_j A(i,j)
```

当 `cross_cluster_ratio > threshold` 时，spot i 被认为是边界 spot。

### 4. 极坐标转换

将空间坐标转换为极坐标：

```
angle = arctan2(y_neigh - y_center, x_neigh - x_center)
radius = attention_score
```

### 5. Domain-aware 聚合

考虑 domain 边界的聚合方式：

```
score_final = α × score_within_domain + (1-α) × score_cross_domain
```

其中 `α` 是 domain weight（默认 0.6）。

## 📈 应用案例

### 案例 1: 肿瘤-基质界面分析

识别肿瘤与基质边界的通讯基因：

```bash
python 4_boundary_visualizer.py --threshold 0.4

# 查看边界基因对
grep "Tumor.*Stroma" boundary_analysis/gene_pairs.csv | head -20
```

**发现**:
- TGF-β 信号通路在肿瘤-基质界面高度激活
- 纤维化相关基因（FN1, COL1A1）显著富集
- 免疫检查点分子（PD-L1, PD-1）共定位

### 案例 2: 配体-受体对发现

结合 CellChat 数据库识别有效的 L-R 对：

```bash
python 4_boundary_visualizer.py \
    --data_dir ./attention_exports \
    --lr_database CellChatDB.human.csv

# 筛选显著的 L-R 对
awk '$9=="True" && $8>0.1' boundary_analysis/gene_pairs.csv
```

### 案例 3: 方向性通讯可视化

使用极坐标图展示通讯的空间方向性：

```bash
python 5_plot_polar.py \
    --data_dir ./attention_exports \
    --focus_gene CXCL12

# 输出 CXCL12 在不同 domains 的极坐标图
```

**解释**:
- 如果极坐标图在某个方向有明显的峰，说明该基因在该方向的通讯更强
- 可以发现趋化因子的梯度方向

### 案例 4: 时空演化分析

对于有多个时间点的数据，比较通讯模式的变化：

```bash
for timepoint in T0 T1 T2; do
    python 4_boundary_visualizer.py \
        --data_dir ./embeddings_${timepoint} \
        --output_dir ./boundary_${timepoint}
done

# 比较不同时间点的边界基因
```

## 🔧 高级用法

### 自定义边界阈值

根据数据分布调整边界识别阈值：

```bash
# 计算 cross_cluster_ratio 的分布
python -c "
import pandas as pd
df = pd.read_csv('spot_to_spot.csv')
print(df['cross_cluster_ratio'].describe())
"

# 使用 75 分位数作为阈值
python 4_boundary_visualizer.py --threshold 0.35
```

### 筛选特定基因

只分析特定的基因列表：

```bash
python 5_plot_polar.py \
    --data_dir ./attention_exports \
    --gene_list genes_of_interest.txt  # 每行一个基因
```

### 批量处理多个数据集

```bash
for dataset in PDAC HBRC MouseBrain; do
    echo "Processing $dataset..."
    python 1_export_spatial_data.py --dataset $dataset
    python 3_prepare_polar.py --out_dir ./attention_exports_$dataset
    python 5_plot_polar.py --data_dir ./attention_exports_$dataset
done
```

### 集成外部配体-受体数据库

使用自定义的 L-R 数据库：

```python
# 准备 L-R 数据库 CSV 文件
# 列名: ligand, receptor, interaction_name, evidence_score

python 4_boundary_visualizer.py \
    --lr_database my_custom_lr_db.csv \
    --lr_score_threshold 0.5
```

## 💡 常见问题

### Q1: 极坐标图为空或没有数据？

**可能原因**:
- Top-K 参数太小
- 该 domain 的 spots 数量太少
- 空间邻居数设置不当

**解决方案**:
```bash
# 增加 top-K
python 3_prepare_polar.py --topk 20

# 检查 domain 的 spots 数量
python -c "
import scanpy as sc
adata = sc.read_h5ad('data.h5ad')
print(adata.obs['domain'].value_counts())
"
```

### Q2: 边界 spots 太多或太少？

**调整阈值**:
```bash
# 查看 cross_cluster_ratio 分布
python -c "
import pandas as pd
df = pd.read_csv('spot_to_spot.csv')
print('Percentiles:')
for p in [25, 50, 75, 90, 95]:
    print(f'{p}%: {df[\"cross_cluster_ratio\"].quantile(p/100):.3f}')
"

# 根据分布调整阈值
python 4_boundary_visualizer.py --threshold 0.4  # 例如使用 75 分位数
```

### Q3: 内存不足？

**优化**:
```bash
# 减小批处理大小
python 1_export_spatial_data.py --batch_size 20

# 使用流式读取
python 3_prepare_polar.py --chunksize 1000000

# 只处理特定 layer
python 1_export_spatial_data.py --layer 5  # 不要处理所有层
```

### Q4: L-R 对识别效果不好？

**改进**:
- 更新 L-R 数据库（使用最新的 CellChat 或 CellPhoneDB）
- 调整表达阈值
- 考虑基因的空间共定位（而不仅仅是表达）

## 📚 相关教程

- [反卷积分析](../deconvolution/README.md): 推断细胞类型组成
- [注意力分析](../attention_analysis/README.md): 分析基因交互网络

## 🔗 参考资料

- [CellChat](https://github.com/sqjin/CellChat): 细胞通讯分析
- [CellPhoneDB](https://www.cellphonedb.org/): L-R 数据库
- [Giotto](https://github.com/drieslab/Giotto): 空间转录组分析工具
- [COMMOT](https://github.com/zcang/COMMOT): 空间通讯推断

---

**更新日期**: 2024-01
