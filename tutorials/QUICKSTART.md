# Quick Start Guide

This guide covers the workflow from model inference to downstream analysis.

## Prerequisites

1. Install CoseNiche:

```bash
cd CoseNiche
pip install -e .
```

2. Prepare spatial transcriptomics data in AnnData `.h5ad` format.
3. Prepare a pretrained CoseNiche model or a model trained on your own data.

## Step 1: Preprocess Data

```bash
python scripts/preprocess.py \
    --h5ad_file /path/to/your/pdac.h5ad \
    --cache_dir ./cache/pdac \
    --max_neighbors 6 \
    --platform visium \
    --organ pancreas
```

## Step 2: Run Inference

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

Main outputs:

- `updated_embeddings.npy`: spot embeddings
- `reconstructed_expr.npy`: reconstructed expression
- `context_attention_scores.pkl`: self-attention weights
- `context_genes.pkl`: context gene lists
- `spatial_attention.pkl`: spatial attention weights

## Step 3: Run Downstream Analyses

### Deconvolution

```bash
cd tutorials/deconvolution
python 1_deconvolution.py
python 2_plot_composition.py
```

### Attention Analysis

```bash
cd tutorials/attention_analysis
python 1_export_attention.py --dataset PDAC --layer 5
python 2_enrichment_analysis.py --dataset PDAC
python 3_bubble_plot.py --dataset PDAC
python 4_prepare_sankey.py --dataset PDAC
python 5_single_gene_analysis.py --dataset PDAC --gene KRAS
```

### Spatial Communication

```bash
cd tutorials/spatial_communication
python 0_check_data.py --data_dir ../../embeddings_pdac
python 1_export_spatial_data.py --dataset PDAC --layer 5
python 3_prepare_polar.py --out_dir ./attention_exports --layer 5
python 4_boundary_visualizer.py --data_dir ./attention_exports --output_dir ./boundary_analysis
python 5_plot_polar.py --data_dir ./attention_exports --output_dir ./polar_plots
```

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| `--max_neighbors` | Number of spatial neighbors |
| `--save_attention` | Save self-attention weights |
| `--save_context_genes` | Save context gene lists |
| `--save_spatial_attention` | Save spatial attention weights |
| `--batch_size` | Inference batch size |
| `--device` | Compute device, such as `cuda` or `cpu` |
| `--layer` | Attention layer to analyze |
| `--top_k` | Number of top partner genes |
| `--threshold` | Boundary detection threshold |

## Troubleshooting

If CUDA memory is insufficient, reduce `--batch_size` or use `--device cpu`.

If genes cannot be found, check `adata.var_names` and make sure `vocab.json` matches the model and data.

If attention files cannot be loaded, rerun inference with `--save_attention` and `--save_context_genes`.

If plots are blank, check that domain or cluster annotations exist and that filtering thresholds are not too strict.
