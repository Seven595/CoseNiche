# 注意力分析教程

本模块展示如何深入分析 CoseNiche 模型的自注意力机制，揭示基因-基因交互网络和功能富集模式。

## 📋 概述

CoseNiche 模型在编码过程中学习基因之间的注意力权重，这些权重反映了基因间的功能关联和交互模式。通过分析这些注意力模式，我们可以：

- 发现疾病相关的基因交互网络
- 识别关键信号通路和生物学过程
- 比较不同组织区域的基因功能差异
- 分析单个基因在不同环境中的功能可塑性

## 🎯 主要特点

- **Domain/Cluster 级别分析**: 按组织区域聚合注意力模式
- **Top Partner 识别**: 找出每个基因的主要交互伙伴
- **富集分析**: GO/KEGG/Reactome 多数据库功能富集
- **多样化可视化**: 气泡图、桑基图、点图等
- **单基因追踪**: 分析特定基因在不同区域的功能变化

## 📂 文件说明

```
attention_analysis/
├── README.md                           # 本文件
├── config.yaml                         # 配置文件
├── 1_export_attention.py               # 导出注意力数据
├── 2_enrichment_analysis.py            # 功能富集分析
├── 3_bubble_plot.py                    # 气泡图可视化
├── 4_prepare_sankey.py                 # 桑基图数据准备
├── 5_single_gene_analysis.py           # 单基因可塑性分析
└── utils.py                            # 工具函数
```

## 🚀 快速开始

### 前置条件

确保已运行模型推理并保存了注意力权重：

```bash
python ../../scripts/extract_embeddings.py \
    --model_path /path/to/model.safetensors \
    --h5ad_path /path/to/data.h5ad \
    --output_dir ./embeddings_output \
    --save_attention \
    --save_context_genes \
    --device cuda:0
```

这将生成：
- `context_attention_scores.pkl`: 注意力矩阵
- `context_genes.pkl`: 每个 spot 的基因列表
- `updated_embeddings.npy`: 嵌入向量

### 步骤 1: 导出注意力数据

```bash
python 1_export_attention.py --dataset PDAC --layer 5
```

**输出**:
- `result_output/hbrc_layer{N}_clusters/*.csv`: 每个 cluster 的长表格
- `result_output/enrichment_prepared/domain_tables/*.csv`: 每个 domain 的聚合表
- `result_output/enrichment_prepared/all_domains_top_partners.csv`: Top partners 汇总

### 步骤 2: 功能富集分析

```bash
python 2_enrichment_analysis.py \
    --dataset PDAC \
    --enrichment_dir ./enrichment_results \
    --top_n 100 \
    --libraries GO_Biological_Process_2023 KEGG_2021_Human Reactome_2022
```

**输出**:
- `enrichment_results/{domain}_enrichment.csv`: 每个 domain 的富集结果
- `enrichment_results/plots/*.png`: 富集分析可视化

### 步骤 3: 气泡图可视化

```bash
python 3_bubble_plot.py \
    --input ./result_output/enrichment_prepared/all_domains_top_partners.csv \
    --output ./bubble_plots \
    --top_n 3
```

**输出**:
- `bubble_plots/bubble_{domain}.png`: 每个 domain 的气泡图
- 展示 top partner 基因及其强度

### 步骤 4: 桑基图数据准备

```bash
python 4_prepare_sankey.py --dataset PDAC
```

**输出**:
- `sankey_data/{domain}_sankey.csv`: 每个 domain 的桑基图数据
- 包含 domain → library → term → genes 的层级关系

### 步骤 5: 单基因可塑性分析

```bash
python 5_single_gene_analysis.py \
    --dataset PDAC \
    --gene KRAS \
    --min_domains 2
```

**输出**:
- `single_gene_analysis/KRAS/`: KRAS 基因的完整分析
  - `domain_comparison.csv`: 不同 domain 的 pathway 比较
  - `enrichment_*.csv`: 各 domain 的富集结果
  - `plasticity_heatmap.png`: 可塑性热图

## 📊 输出结果详解

### 1. 长表格 (Step 1)

每个 cluster/domain 的详细注意力数据：

| spot_idx | spot_name | cluster | gene_symbol | rank | partner_symbol | score |
|----------|-----------|---------|-------------|------|----------------|-------|
| 0 | SPOT_1 | Tumor | KRAS | 1 | EGFR | 0.1234 |
| 0 | SPOT_1 | Tumor | KRAS | 2 | MYC | 0.0987 |

**列说明**:
- `spot_idx`: spot 索引
- `gene_symbol`: 查询基因
- `rank`: 该 partner 在该基因的排名
- `partner_symbol`: 交互伙伴基因
- `score`: 注意力得分

### 2. 聚合表 (Step 1)

每个 domain 的聚合统计：

| partner_symbol | hit_spots | sum_strength | avg_strength |
|----------------|-----------|--------------|--------------|
| EGFR | 45 | 12.34 | 0.274 |
| MYC | 38 | 9.87 | 0.260 |

**列说明**:
- `hit_spots`: 该基因出现在多少个 spots
- `sum_strength`: 所有 spots 的注意力总和
- `avg_strength`: 平均注意力强度

### 3. 富集分析结果 (Step 2)

GO/KEGG/Reactome 富集结果：

| Term | Library | Overlap | P-value | Adjusted P-value | Genes |
|------|---------|---------|---------|------------------|-------|
| Cell proliferation | GO_BP_2023 | 15/200 | 1.2e-08 | 3.4e-06 | KRAS;EGFR;... |

### 4. 气泡图 (Step 3)

可视化展示：
- X 轴: Top partner 基因
- Y 轴: Domain
- 气泡大小: Hit spots 数量
- 气泡颜色: 平均注意力强度

### 5. 单基因分析 (Step 5)

对比单个基因在不同 domain 的功能：

**Domain 比较表**:
| Pathway | Domain_Tumor | Domain_Normal | Enrichment_Diff |
|---------|--------------|---------------|-----------------|
| EGFR signaling | 0.0001 | 0.234 | 0.234 |
| Cell cycle | 0.002 | 0.456 | 0.454 |

## 🔬 算法原理

### 1. 注意力提取

从 CoseNiche 模型的 context encoder 提取自注意力矩阵：

```
A_layer = Attention(Q, K, V) / √d_k
```

其中 layer 通常选择第 5 层（中间层），平衡局部和全局模式。

### 2. 基因过滤

按照严格的质量控制流程过滤基因：

**步骤 1: 白名单**
- 保留 HLA- 和 MIR 开头的基因

**步骤 2: 点号检查**
- 移除含点号的基因（低质量注释）

**步骤 3: 正则模式**
- 排除假基因、反义 RNA 等（如 `-AS1`, `RP11-`）

**步骤 4: 前缀黑名单**
- 排除 AC, AL, LINC, RP, SNOR, SCARNA 开头的基因

### 3. 对称化

将注意力矩阵对称化以获得无向交互：

```
A_sym = 0.5 * (A + A^T)
```

### 4. Top-K 选择

为每个基因选择注意力最高的 K 个伙伴（默认 K=20）

### 5. Domain 聚合

在 domain 内聚合注意力模式：

**Level 1 (Spot 内)**:
```
score_sum_by_spot = Σ score(gene, partner)
```

**Level 2 (Domain 内)**:
```
avg_strength = Σ score_sum_by_spot / hit_spots
```

## 📈 应用案例

### 案例 1: 肿瘤微环境分析

识别肿瘤区域的关键通路：

```bash
python 1_export_attention.py --dataset PDAC
python 2_enrichment_analysis.py --dataset PDAC

# 查看肿瘤区域的富集结果
ls enrichment_results/Tumor_enrichment.csv
```

**发现**:
- 细胞周期通路高度激活
- DNA 修复通路富集
- 免疫逃逸相关基因上调

### 案例 2: 基因功能可塑性

分析 KRAS 在不同区域的功能：

```bash
python 5_single_gene_analysis.py --dataset PDAC --gene KRAS
```

**发现**:
- 在肿瘤区域: 与增殖相关基因高度交互
- 在正常区域: 与代谢基因交互
- 在免疫区域: 与炎症信号通路相关

### 案例 3: 跨区域比较

比较不同病理区域的通路差异：

```bash
python 3_bubble_plot.py --input all_domains_top_partners.csv
```

通过气泡图可视化不同区域的 top genes，发现区域特异性的基因网络。

## 🔧 高级用法

### 自定义基因过滤规则

在配置文件中修改过滤参数：

```yaml
gene_filtering:
  drop_with_dot: true
  extra_exclude_prefixes:
    - "AC"
    - "AL"
    - "LINC"
    - "CUSTOM_PREFIX"  # 添加自定义前缀
  
  whitelist:
    - "HLA-"
    - "MIR"
    - "MY_GENE"  # 添加自定义白名单
```

### 选择不同的注意力层

不同层捕获不同粒度的模式：

```bash
# 浅层 (局部模式)
python 1_export_attention.py --layer 2

# 中层 (平衡)
python 1_export_attention.py --layer 5

# 深层 (全局模式)
python 1_export_attention.py --layer 11
```

### 调整 Top-K 参数

```bash
python 1_export_attention.py --top_k 50  # 增加到 50 个 partners
python 2_enrichment_analysis.py --top_n 200  # 用前 200 个基因做富集
```

### 自定义富集数据库

```bash
python 2_enrichment_analysis.py \
    --libraries \
        GO_Biological_Process_2023 \
        GO_Molecular_Function_2023 \
        KEGG_2021_Human \
        Reactome_2022 \
        WikiPathways_2023_Human \
        BioCarta_2016 \
        MSigDB_Hallmark_2020
```

## 💡 常见问题

### Q1: 富集分析没有显著结果？

**可能原因**:
- Top genes 数量太少
- 基因过滤太严格
- Domain 内的 spots 数量不足

**解决方案**:
```bash
# 增加 top genes 数量
python 2_enrichment_analysis.py --top_n 200

# 放宽 p 值阈值
python 2_enrichment_analysis.py --cutoff 0.1

# 合并相似的 domains
```

### Q2: 注意力矩阵维度不匹配？

**检查**:
- 确保 `context_genes.pkl` 和 `context_attention_scores.pkl` 来自同一次推理
- 检查 vocab.json 是否正确加载
- 验证基因 ID 到 symbol 的映射

### Q3: 单基因分析失败？

**常见原因**:
- 该基因在某些 domains 中不存在
- 表达量太低被过滤

**调整参数**:
```bash
python 5_single_gene_analysis.py \
    --gene YOUR_GENE \
    --min_domains 1 \           # 降低最小 domain 要求
    --min_spots_per_domain 1    # 降低最小 spots 要求
```

### Q4: 气泡图太密集？

**优化**:
```bash
python 3_bubble_plot.py \
    --top_n 5 \                 # 减少每个 domain 的基因数
    --figsize 20 10 \           # 增大图形尺寸
    --min_size 20 \             # 增大最小气泡
    --max_size 500              # 增大最大气泡
```

## 📚 相关教程

- [反卷积分析](../deconvolution/README.md): 推断细胞类型组成
- [空间通讯](../spatial_communication/README.md): 研究细胞间通讯

## 🔗 参考资料

- [Enrichr](https://maayanlab.cloud/Enrichr/): 在线富集分析工具
- [GSEApy](https://gseapy.readthedocs.io/): Python 富集分析库
- [Gene Ontology](http://geneontology.org/): GO 数据库
- [KEGG](https://www.genome.jp/kegg/): 通路数据库

---

**更新日期**: 2024-01
