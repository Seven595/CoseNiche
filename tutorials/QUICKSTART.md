# 快速开始指南

本指南将引导你完成从模型推理到下游分析的完整流程。

## 🎯 前置条件

1. **安装 CoseNiche**:
```bash
cd CoseNiche
pip install -e .
```

2. **准备数据**:
- 空间转录组数据（AnnData 格式 `.h5ad`）
- 单细胞参考数据（用于反卷积，可选）

3. **准备模型**:
- 下载预训练的 CoseNiche 模型
- 或使用你自己训练的模型

## 📋 完整流程

### 第一步：预处理数据

```bash
python scripts/preprocess.py \
    --h5ad_file /path/to/your/pdac.h5ad \
    --cache_dir ./cache/pdac \
    --max_neighbors 6 \
    --platform visium \
    --organ pancreas
```

**输出**:
- `./cache/pdac/`: 预处理后的数据缓存

### 第二步：模型推理并提取嵌入

```bash
python scripts/extract_embeddings.py \
    --model_path /path/to/model.safetensors \
    --h5ad_path /path/to/your/pdac.h5ad \
    --cache_dir ./cache/pdac \
    --output_dir ./embeddings_pdac \
    --save_attention \
    --save_context_genes \
    --save_spatial_attention \
    --device cuda:0
```

**输出**:
- `embeddings_pdac/updated_embeddings.npy`: Spot 嵌入向量
- `embeddings_pdac/reconstructed_expr.npy`: 重建的基因表达
- `embeddings_pdac/context_attention_scores.pkl`: 自注意力权重
- `embeddings_pdac/context_genes.pkl`: 每个 spot 的基因列表
- `embeddings_pdac/spatial_attention.pkl`: 空间注意力权重（可选）

### 第三步：下游分析

#### 3.1 反卷积分析

```bash
cd tutorials/deconvolution

# 准备单细胞参考数据的嵌入
python ../../scripts/extract_embeddings.py \
    --model_path /path/to/model.safetensors \
    --h5ad_path /path/to/pdac_sc_reference.h5ad \
    --cache_dir ./cache/pdac_sc \
    --output_dir ../../embeddings_pdac_sc \
    --device cuda:0

# 运行反卷积
python 1_deconvolution.py

# 可视化结果
python 2_plot_composition.py
```

**输出位置**: `./results/PDAC/`

#### 3.2 注意力分析

```bash
cd ../attention_analysis

# 步骤1: 导出注意力数据
python 1_export_attention.py --dataset PDAC --layer 5

# 步骤2: 功能富集分析
python 2_enrichment_analysis.py --dataset PDAC

# 步骤3: 气泡图可视化
python 3_bubble_plot.py --dataset PDAC

# 步骤4: 准备桑基图数据
python 4_prepare_sankey.py --dataset PDAC

# 步骤5: 单基因分析（示例）
python 5_single_gene_analysis.py --dataset PDAC --gene KRAS
```

**输出位置**: `./PDAC/result_output/`

#### 3.3 空间通讯分析

```bash
cd ../spatial_communication

# 步骤0: 检查数据
python 0_check_data.py --data_dir ../../embeddings_pdac

# 步骤1: 导出空间数据
python 1_export_spatial_data.py --dataset PDAC --layer 5

# 步骤2: 准备极坐标数据
python 3_prepare_polar.py --out_dir ./attention_exports --layer 5

# 步骤3: 边界分析
python 4_boundary_visualizer.py \
    --data_dir ./attention_exports \
    --output_dir ./boundary_analysis

# 步骤4: 极坐标图
python 5_plot_polar.py \
    --data_dir ./attention_exports \
    --output_dir ./polar_plots
```

**输出位置**: `./attention_exports/`, `./boundary_analysis/`, `./polar_plots/`

## 🗂️ 目录结构示例

完成所有步骤后，你的目录结构应该类似：

```
CoseNiche/
├── cache/
│   ├── pdac/                    # 预处理缓存
│   └── pdac_sc/                 # 单细胞预处理缓存
├── embeddings_pdac/             # 空间数据嵌入
│   ├── updated_embeddings.npy
│   ├── context_attention_scores.pkl
│   └── context_genes.pkl
├── embeddings_pdac_sc/          # 单细胞数据嵌入
│   └── updated_embeddings.npy
└── tutorials/
    ├── deconvolution/
    │   └── results/PDAC/       # 反卷积结果
    ├── attention_analysis/
    │   └── PDAC/               # 注意力分析结果
    └── spatial_communication/
        ├── attention_exports/   # 导出的注意力数据
        ├── boundary_analysis/   # 边界分析结果
        └── polar_plots/        # 极坐标图
```

## 💡 常用参数说明

### 预处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_neighbors` | 6 | 空间邻居数量 |
| `--platform` | None | 测序平台（visium/slideseq/...） |
| `--organ` | None | 组织器官类型 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_attention` | False | 是否保存自注意力权重 |
| `--save_context_genes` | False | 是否保存上下文基因列表 |
| `--save_spatial_attention` | False | 是否保存空间注意力权重 |
| `--batch_size` | 8 | 批处理大小 |
| `--device` | cuda | 计算设备（cuda/cpu） |

### 反卷积参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 2000 | 训练轮数 |
| `--lr` | 0.005 | 学习率 |
| `--lam_recon` | 100.0 | 重建损失权重 |
| `--lam_contrast` | 1.0 | 对比损失权重 |

### 注意力分析参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--layer` | 5 | 提取注意力的层数 |
| `--top_k` | 20 | Top-K partner 基因数 |
| `--top_n` | 100 | 用于富集分析的基因数 |

### 空间通讯参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--layer` | 5 | 分析的空间注意力层 |
| `--threshold` | 0.3 | 边界识别阈值 |
| `--topk` | 10 | 极坐标图的 top-K |

## 🐛 故障排除

### 问题 1: CUDA 内存不足

**解决方案**:
```bash
# 减小批处理大小
python scripts/extract_embeddings.py ... --batch_size 4

# 或使用 CPU
python scripts/extract_embeddings.py ... --device cpu
```

### 问题 2: 找不到某些基因

**原因**: 基因名称可能不匹配或被过滤

**解决方案**:
```bash
# 检查基因名称
python -c "
import scanpy as sc
adata = sc.read_h5ad('data.h5ad')
print('Total genes:', adata.n_vars)
print('Sample genes:', adata.var_names[:10].tolist())
"

# 检查 vocab.json
python -c "
import json
with open('/path/to/vocab.json') as f:
    vocab = json.load(f)
print('Vocab size:', len(vocab))
print('Sample:', list(vocab.items())[:5])
"
```

### 问题 3: 注意力文件损坏或格式不对

**解决方案**:
```bash
# 重新运行推理，确保保存注意力
python scripts/extract_embeddings.py \
    ... \
    --save_attention \
    --save_context_genes

# 检查文件
python -c "
import pickle
with open('embeddings_pdac/context_attention_scores.pkl', 'rb') as f:
    data = pickle.load(f)
print('Type:', type(data))
print('Length:', len(data) if isinstance(data, list) else 'N/A')
"
```

### 问题 4: 可视化图片空白或没有数据

**常见原因**:
- Domain/cluster 标注缺失
- 数据过滤太严格
- 参数设置不当

**检查步骤**:
```bash
# 检查 domain 标注
python -c "
import scanpy as sc
adata = sc.read_h5ad('data.h5ad')
if 'ground_truth' in adata.obs:
    print(adata.obs['ground_truth'].value_counts())
else:
    print('No ground_truth column!')
"

# 检查数据量
python 0_check_data.py --data_dir ./embeddings_pdac
```

## 📚 下一步

- 阅读各模块的详细文档:
  - [反卷积分析](deconvolution/README.md)
  - [注意力分析](attention_analysis/README.md)
  - [空间通讯分析](spatial_communication/README.md)

- 尝试自定义参数和配置
- 应用到你自己的数据集
- 组合多个分析模块获得全面的洞察

## 🤝 获取帮助

- **Issues**: [GitHub Issues](https://github.com/yourusername/CoseNiche/issues)
- **文档**: [完整文档](https://coseniche.readthedocs.io/)
- **示例**: `examples/` 目录下的示例脚本

---

**更新日期**: 2024-01
