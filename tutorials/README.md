# CoseNiche Downstream Analysis Tutorials

This directory contains downstream analysis tutorials and example scripts for CoseNiche. The tutorials show how to use model embeddings and attention weights for spatial transcriptomics analysis.

## Tutorial Modules

### 1. Deconvolution

Directory: `deconvolution/`

Use CoseNiche embeddings to infer cell-type composition for each spatial spot.

Quick start:

```bash
cd deconvolution
python 1_deconvolution.py --config config_pdac.yaml
python 2_plot_composition.py --config config_pdac.yaml
```

### 2. Attention Analysis

Directory: `attention_analysis/`

Analyze self-attention to identify gene-gene interactions, top partner genes, and enriched biological pathways.

Quick start:

```bash
cd attention_analysis
python 1_export_attention.py --dataset PDAC
python 2_enrichment_analysis.py --dataset PDAC
python 3_bubble_plot.py --dataset PDAC
python 4_prepare_sankey.py --dataset PDAC
python 5_single_gene_analysis.py --dataset PDAC --gene KRAS
```

### 3. Spatial Communication

Directory: `spatial_communication/`

Use spatial attention weights to analyze cell-cell communication, boundary spots, and ligand-receptor interactions.

Quick start:

```bash
cd spatial_communication
python 1_export_spatial_data.py --dataset PDAC
python 3_prepare_polar.py --dataset PDAC
python 5_plot_polar.py --dataset PDAC
```

## Full Workflow

Run model inference before downstream analysis:

```bash
python ../scripts/preprocess.py \
    --h5ad_file /path/to/your/data.h5ad \
    --cache_dir ./cache \
    --max_neighbors 6

python ../scripts/extract_embeddings.py \
    --model_path /path/to/model.safetensors \
    --h5ad_path /path/to/your/data.h5ad \
    --cache_dir ./cache \
    --output_dir ./embeddings_output \
    --save_attention \
    --device cuda:0
```

## Outputs

- Deconvolution: cell-type proportions, spatial pie charts, reconstruction quality metrics.
- Attention analysis: enrichment results, bubble plots, Sankey data, single-gene analyses.
- Spatial communication: attention exports, boundary analysis, polar plots, ligand-receptor interactions.

## Environment

```bash
conda activate coseniche
pip install gseapy
```

## Datasets

| Dataset | Tissue | Platform | Spots | Description |
|---------|--------|----------|-------|-------------|
| PDAC | Pancreatic cancer | Visium | 428 | Tumor and normal tissue regions |
| HBRC | Breast cancer | Visium | 3,813 | Multiple sections with diverse pathological regions |

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/CoseNiche/issues)
- Documentation: [Full documentation](https://coseniche.readthedocs.io/)
- Paper: [bioRxiv preprint](https://biorxiv.org/)
