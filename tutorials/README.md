# CoseNiche 下游分析教程

本目录包含基于 CoseNiche 模型的下游分析教程和示例代码。这些分析展示了如何利用模型提取的嵌入和注意力权重进行深入的空间转录组数据分析。

## 📚 教程模块

### 1. 反卷积分析 (Deconvolution)
**目录**: `deconvolution/`

使用 CoseNiche 嵌入进行空间反卷积，推断每个空间位点的细胞类型组成。

**主要功能**:
- 基于嵌入的细胞类型反卷积
- 细胞类型比例可视化
- 空间分布分析
- 反卷积质量评估

**使用场景**:
- 理解组织微环境的细胞组成
- 识别疾病相关的细胞类型变化
- 分析细胞类型的空间分布模式

**快速开始**:
```bash
cd deconvolution
python 1_deconvolution.py --config config_pdac.yaml
python 2_plot_composition.py --config config_pdac.yaml
```

---

### 2. 注意力分析 (Attention Analysis)
**目录**: `attention_analysis/`

深入分析模型的自注意力机制，揭示基因-基因交互和功能富集模式。

**主要功能**:
- 导出每个 cluster 的基因注意力网络
- Top partner 基因识别
- 功能富集分析 (GO/KEGG/Reactome)
- 气泡图和桑基图可视化
- 单基因可塑性分析

**使用场景**:
- 发现疾病相关的基因交互网络
- 识别关键信号通路
- 比较不同组织区域的基因功能

**快速开始**:
```bash
cd attention_analysis
# 步骤1: 导出注意力数据
python 1_export_attention.py --dataset PDAC

# 步骤2: 运行富集分析
python 2_enrichment_analysis.py --dataset PDAC

# 步骤3: 可视化
python 3_bubble_plot.py --dataset PDAC
python 4_prepare_sankey.py --dataset PDAC

# 步骤4: 单基因分析
python 5_single_gene_analysis.py --dataset PDAC --gene KRAS
```

---

### 3. 空间通讯分析 (Spatial Communication)
**目录**: `spatial_communication/`

利用空间注意力权重分析细胞-细胞通讯和配体-受体相互作用。

**主要功能**:
- 空间注意力数据导出
- 边界 spot 识别
- 配体-受体相互作用分析
- 注意力流向可视化
- 极坐标图展示空间通讯模式

**使用场景**:
- 识别组织边界的细胞通讯
- 发现空间特异性的配体-受体对
- 分析疾病进展中的通讯变化

**快速开始**:
```bash
cd spatial_communication
# 步骤1: 导出空间数据
python 1_export_spatial_data.py --dataset PDAC

# 步骤2: 准备极坐标数据
python 3_prepare_polar.py --dataset PDAC

# 步骤3: 可视化
python 5_plot_polar.py --dataset PDAC
```

---

## 🚀 完整分析流程

### 前置步骤：模型推理

在运行下游分析之前，需要先使用 CoseNiche 模型提取嵌入和注意力权重：

```bash
# 1. 预处理数据
python ../scripts/preprocess.py \
    --h5ad_file /path/to/your/data.h5ad \
    --cache_dir ./cache \
    --max_neighbors 6

# 2. 提取嵌入和注意力（保存注意力权重）
python ../scripts/extract_embeddings.py \
    --model_path /path/to/model.safetensors \
    --h5ad_path /path/to/your/data.h5ad \
    --cache_dir ./cache \
    --output_dir ./embeddings_output \
    --save_attention \
    --device cuda:0
```

### 典型分析流程

以 PDAC（胰腺癌）数据集为例：

```bash
# ===== 1. 反卷积分析 =====
cd deconvolution
python 1_deconvolution.py --config config_pdac.yaml
python 2_plot_composition.py --config config_pdac.yaml

# ===== 2. 注意力分析 =====
cd ../attention_analysis
python 1_export_attention.py --dataset PDAC
python 2_enrichment_analysis.py --dataset PDAC
python 3_bubble_plot.py --dataset PDAC

# ===== 3. 空间通讯分析 =====
cd ../spatial_communication
python 1_export_spatial_data.py --dataset PDAC
python 3_prepare_polar.py --dataset PDAC
python 5_plot_polar.py --dataset PDAC
```

---

## 📊 输出结果

### 反卷积分析
- `deconvolution_out.h5ad`: 包含细胞类型比例的 AnnData 对象
- `cell_type_proportions/`: 细胞类型比例可视化
- `spatial_pie/`: 空间饼图展示细胞组成
- `reconstruction_quality.csv`: 重建质量评估

### 注意力分析
- `enrichment_results/`: 富集分析结果（GO/KEGG/Reactome）
- `bubble_plots/`: 气泡图展示 top partner 基因
- `sankey_data/`: 桑基图数据
- `single_gene_analysis/`: 单基因可塑性分析

### 空间通讯分析
- `attention_exports/`: 导出的注意力矩阵
- `boundary_analysis/`: 边界 spot 分析
- `polar_plots/`: 极坐标图展示空间通讯
- `lr_interactions/`: 配体-受体相互作用

---

## 🛠️ 环境配置

所有教程共享主项目的环境：

```bash
# 使用主环境
conda activate coseniche

# 如需额外依赖（用于富集分析）
pip install gseapy
```

---

## 📖 数据集

教程使用的示例数据集：

| 数据集 | 组织 | 平台 | Spots | 描述 |
|--------|------|------|-------|------|
| PDAC | 胰腺癌 | Visium | 428 | 包含肿瘤、正常组织区域 |
| HBRC | 乳腺癌 | Visium | 3,813 | 多个组织切片，包含多种病理区域 |

---

## 🤝 贡献

欢迎贡献新的分析教程！请遵循以下格式：

1. 在对应模块下创建脚本
2. 添加配置文件示例
3. 更新模块的 README
4. 提供示例输出和可视化

---

## 📞 支持

- **Issues**: [GitHub Issues](https://github.com/yourusername/CoseNiche/issues)
- **文档**: [完整文档](https://coseniche.readthedocs.io/)
- **论文**: [bioRxiv预印本](https://biorxiv.org/)

---

## 📄 许可证

本教程代码遵循 MIT 许可证，与主项目保持一致。
