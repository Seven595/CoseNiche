# Spatial Deconvolution with CoseNiche

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Infer cell type compositions in spatial transcriptomics using CoseNiche foundation model embeddings**

This module performs cell type deconvolution on spatial transcriptomics data by leveraging high-quality embeddings from the CoseNiche foundation model. It learns an optimal mapping between spatial spots and single-cell reference data to accurately estimate cell type proportions at each tissue location.

## Overview

### Key Features

- **Foundation Model Embeddings**: Utilizes pre-trained CoseNiche embeddings that capture rich biological context
- **GraphST-Inspired Architecture**: Efficient mapping matrix learning with spatial awareness
- **Multiple Loss Functions**: Reconstruction (MSE/cosine/correlation) + contrastive loss for spatial coherence
- **Flexible Configuration**: YAML-based configuration with sensible defaults
- **Publication-Quality Visualization**: Nature journal-style plots with 300+ DPI output
- **Comprehensive Metrics**: Cell type proportions, reconstruction quality, spatial coherence scores

### Algorithm

The deconvolution pipeline learns a probabilistic mapping matrix **M** ∈ ℝ<sup>n_spots × n_cells</sup> that:

1. **Spatial Graph Construction**: Builds k-nearest neighbor graph from tissue coordinates
2. **Embedding Reconstruction**: Minimizes ||M ⊙ H<sub>cells</sub> - H<sub>spots</sub>||²
3. **Spatial Coherence**: Preserves neighborhood relationships via contrastive loss
4. **Cell Type Inference**: Aggregates mapping probabilities by cell type

**Mathematical Formulation**:
```
L_total = λ_recon · L_recon(M ⊙ H_c, H_s) + λ_contrast · L_contrast(M, G)

where:
- H_c: Single-cell embeddings (n_cells × d)
- H_s: Spatial embeddings (n_spots × d)  
- G: Spatial adjacency graph
- M: Softmax-normalized mapping matrix
```

## Quick Start

### Prerequisites

**Required Data**:
1. Spatial transcriptomics data (AnnData `.h5ad` format)
   - Must contain `adata.obsm['spatial']` with (x, y) coordinates
   - Optionally: `adata.obs['ground_truth']` for evaluation
2. Single-cell reference data (AnnData `.h5ad` format)
   - Must contain `adata.obs['cell_type']` with cell type annotations
3. Pre-computed CoseNiche embeddings (NumPy `.npy` format)
   - Spatial embeddings: shape (n_spots, embedding_dim)
   - Single-cell embeddings: shape (n_cells, embedding_dim)

**Installation**:
```bash
# Install required packages
pip install -r requirements.txt

# Or install individually
pip install scanpy numpy pandas torch matplotlib scipy pyyaml scikit-learn
```

### Step-by-Step Tutorial

#### 1. Extract Embeddings

First, extract embeddings using the CoseNiche foundation model:

```bash
# Extract spatial embeddings
python ../../scripts/extract_embeddings.py \
    --model-path /path/to/coseniche_model.safetensors \
    --h5ad-path data/PDAC/pdac_spatial.h5ad \
    --output-dir embeddings/PDAC_st \
    --device cuda:0

# Extract single-cell embeddings
python ../../scripts/extract_embeddings.py \
    --model-path /path/to/coseniche_model.safetensors \
    --h5ad-path data/PDAC/pdac_sc_reference.h5ad \
    --output-dir embeddings/PDAC_sc \
    --device cuda:0
```

#### 2. Run Deconvolution

**Method A: Using configuration file (recommended)**
```bash
# Copy template and modify for your data
cp config_template.yaml config_mydata.yaml
nano config_mydata.yaml  # Edit paths and parameters

# Run deconvolution
python 1_deconvolution.py --config config_mydata.yaml
```

**Method B: Direct command-line parameters**
```bash
python 1_deconvolution.py \
    --st-data data/PDAC/pdac_spatial.h5ad \
    --sc-data data/PDAC/pdac_sc_reference.h5ad \
    --st-embeddings embeddings/PDAC_st/updated_embeddings.npy \
    --sc-embeddings embeddings/PDAC_sc/updated_embeddings.npy \
    --output-dir results/PDAC \
    --epochs 2000 \
    --lr 0.005 \
    --loss-type combined \
    --device cuda
```

#### 3. Visualize Results

Generate publication-quality plots:

```bash
# Using configuration file
python 2_plot_composition.py --config config_mydata.yaml

# Or specify parameters directly
python 2_plot_composition.py \
    --deconv-file results/PDAC/deconvolution_result.h5ad \
    --sc-file results/PDAC/sc_reference.h5ad \
    --output-dir results/PDAC/plots \
    --plot-types bar pie spatial \
    --dpi 300
```

### Expected Runtime

| Dataset Size | Deconvolution | Visualization |
|--------------|---------------|---------------|
| ~500 spots   | 5-10 min      | 2-5 min       |
| ~2000 spots  | 15-30 min     | 5-10 min      |
| ~5000 spots  | 30-60 min     | 10-20 min     |

*Times assume GPU (NVIDIA V100/A100) for deconvolution*

## Configuration

### Creating a Configuration File

Copy the template and modify for your dataset:

```bash
cp config_template.yaml config_mydata.yaml
# Edit config_mydata.yaml with your paths and parameters
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbors` | 6 | Number of spatial neighbors for graph |
| `epochs` | 2000 | Training epochs |
| `lr` | 0.005 | Learning rate |
| `lam_recon` | 100.0 | Reconstruction loss weight |
| `lam_contrast` | 1.0 | Contrastive loss weight |
| `loss_type` | mse | Loss type (mse/cosine/correlation/combined) |
| `tau` | 1.0 | Temperature for contrastive loss |

### Loss Types

- **mse**: Mean squared error (fast, stable)
- **cosine**: Cosine similarity (emphasizes direction)
- **correlation**: Pearson correlation (robust to scale)
- **combined**: Weighted combination of all three (best quality)

## Output Files

The pipeline generates the following outputs in `output_dir`:

```
results/PDAC/
├── deconvolution_result.h5ad      # Spatial data with deconvolution results
├── sc_reference.h5ad               # Single-cell reference data
├── cell_type_proportions.csv      # Cell type proportions matrix
├── deconvolution.log               # Training log
└── plots/                          # Visualization outputs (from step 2)
    ├── component3_stacked_bar_*.png
    ├── component3_pie_*.png
    └── component4_spatial_pie_*.png
```

### AnnData Structure

The output `deconvolution_result.h5ad` contains:

```python
adata.obsm['map_matrix']              # (n_spots, n_cells) mapping matrix
adata.obsm['pred_embeddings']         # (n_spots, d) predicted embeddings
adata.obsm['cell_type_proportions']   # (n_spots, n_celltypes) proportions
```

## Workflow

### Step 1: Prepare Data

Ensure your data has the required structure:

**Spatial data (`adata`):**
- `adata.X`: Gene expression matrix
- `adata.obsm['spatial']`: Spatial coordinates
- `adata.obs['ground_truth']`: (Optional) True labels for evaluation

**Single-cell reference (`adata_sc`):**
- `adata_sc.X`: Gene expression matrix
- `adata_sc.obs['cell_type']`: Cell type annotations

### Step 2: Extract Embeddings

Use CoseNiche model to extract embeddings:

```bash
# For spatial data
python ../../scripts/extract_embeddings.py \
    --model_path /path/to/model.safetensors \
    --h5ad_path ../../Data/PDAC/pdac.h5ad \
    --cache_dir ./cache/pdac_st \
    --output_dir ../../embeddings/PDAC_st \
    --device cuda:0

# For single-cell reference
python ../../scripts/extract_embeddings.py \
    --model_path /path/to/model.safetensors \
    --h5ad_path ../../h5ad/PDAC_sc/GSE111672_scRNA_combined.h5ad \
    --cache_dir ./cache/pdac_sc \
    --output_dir ../../embeddings/PDAC_sc \
    --device cuda:0
```

### Step 3: Run Deconvolution

```bash
python 1_deconvolution.py --config config_pdac.yaml
```

### Step 4: Visualize Results

```bash
python 2_plot_composition.py --config config_pdac.yaml
```

## Advanced Usage

### Custom Loss Function

Modify the `loss_type` parameter or implement custom loss in `1_deconvolution.py`:

```python
# In graphst_deconvolution function
if loss_type == "custom":
    # Your custom loss implementation
    loss_recon = custom_loss(pred_sp_norm, Hs_norm)
```

### Fine-tuning Parameters

For different datasets, you may need to adjust:

1. **Learning rate (`lr`)**: 
   - Increase for faster convergence
   - Decrease if training is unstable

2. **Loss weights (`lam_recon`, `lam_contrast`)**:
   - Higher `lam_recon`: Better reconstruction
   - Higher `lam_contrast`: Better spatial coherence

3. **Number of neighbors (`n_neighbors`)**:
   - Increase for smoother spatial patterns
   - Decrease for more local information

### Handling Large Datasets

For large datasets (>10,000 spots):

```yaml
# In config file
advanced:
  use_amp: true  # Mixed precision training
  grad_clip_norm: 1.0  # Gradient clipping
  
  checkpoint:
    enabled: true
    save_every: 500
    keep_last_n: 3
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Option 1: Use CPU (slower but works for any dataset)
python 1_deconvolution.py --config config.yaml --device cpu

# Option 2: Reduce batch size (if using batched processing)
# Edit config: batch_size: 512

# Option 3: Use smaller embeddings (if possible)
# Re-extract with lower embedding dimension
```

**Prevention**: Monitor GPU memory with `nvidia-smi` during training.

---

#### Issue 2: Training Unstable (NaN/Inf Loss)

**Symptoms**: Loss suddenly becomes `nan` or `inf` during training

**Root Causes**:
- Learning rate too high
- Numerical instability in loss computation
- Poor embedding quality (not normalized)

**Solutions**:
```yaml
# In config file:
lr: 0.001              # Reduce from default 0.005
loss_type: "cosine"    # More stable than MSE
tau: 0.5               # Lower temperature for contrastive loss

# Advanced: Enable gradient clipping (already default)
grad_clip_norm: 1.0    # Prevents exploding gradients
```

**Debugging**:
```python
# Check if embeddings are normalized
import numpy as np
st_emb = np.load("embeddings/st.npy")
norms = np.linalg.norm(st_emb, axis=1)
print(f"Embedding norms: min={norms.min():.3f}, max={norms.max():.3f}, mean={norms.mean():.3f}")
# Should be close to 1.0 if normalized
```

---

#### Issue 3: Poor Reconstruction Quality

**Symptoms**: 
- Low cosine similarity between predicted and true embeddings (<0.7)
- Cell type proportions don't match expected biology

**Solutions**:
```yaml
# Increase training duration
epochs: 5000  # Up from default 2000

# Increase reconstruction weight
lam_recon: 200.0  # Up from default 100.0

# Use best loss function
loss_type: "combined"  # Weighted combination of MSE + cosine + correlation

# Increase spatial neighbors for more context
n_neighbors: 10  # Up from default 6
```

**Validation**:
- Check training loss curves (should converge smoothly)
- Visualize predicted embeddings with UMAP
- Compare with ground truth if available

---

#### Issue 4: Cell Type Names with Numeric Suffixes

**Symptoms**: Cell types like `"T cell.151"`, `"T cell.152"` treated as different types

**Solution**: Enable automatic normalization
```yaml
# In config file:
normalize_celltype: true  # Removes .XXX suffixes
```

Or manually normalize:
```python
import re
adata_sc.obs['cell_type'] = adata_sc.obs['cell_type'].str.replace(r'\.\d+$', '', regex=True)
```

---

#### Issue 5: Spatial Coordinates Not Found

**Symptoms**: `ValueError: Missing adata.obsm['spatial']`

**Solution**: Ensure spatial data has coordinate information
```python
import scanpy as sc

adata = sc.read_h5ad("spatial.h5ad")

# Check if spatial coordinates exist
if 'spatial' not in adata.obsm:
    # For 10x Visium data
    adata.obsm['spatial'] = adata.obs[['array_row', 'array_col']].values
    
    # Or from separate file
    import pandas as pd
    coords = pd.read_csv("coordinates.csv", index_col=0)
    adata.obsm['spatial'] = coords.loc[adata.obs_names, ['x', 'y']].values
```

---

#### Issue 6: Slow Training on CPU

**Symptoms**: Training takes hours even for small datasets

**Solutions**:
```bash
# Option 1: Use GPU if available
python 1_deconvolution.py --config config.yaml --device cuda:0

# Option 2: Reduce epochs for quick testing
python 1_deconvolution.py --config config.yaml --epochs 500

# Option 3: Use faster loss function
# Edit config: loss_type: "mse"  # Faster than "combined"
```

---

#### Issue 7: Import Errors for Visualization

**Symptoms**: `ModuleNotFoundError: No module named 'utils_plot'`

**Solution**: Install missing dependencies or disable projection
```bash
# Install all dependencies
pip install -r requirements.txt

# Or skip cell projection (use existing proportions)
# Visualization will use adata.obsm['cell_type_proportions'] if available
```

## Performance Benchmarks

### Typical Performance Metrics

**PDAC Dataset** (428 spots, 22 cell types, ~20,000 genes):

| Metric | Value | Hardware |
|--------|-------|----------|
| Training time | ~5 minutes | NVIDIA V100 (16GB) |
| Training time | ~25 minutes | CPU (Intel Xeon, 32 cores) |
| Peak memory | ~2 GB | GPU VRAM |
| Peak memory | ~8 GB | System RAM (CPU) |
| Reconstruction quality | Cosine sim > 0.85 | - |
| Cell type accuracy | > 80% | With ground truth |

**Large Dataset** (~5000 spots, 30 cell types):

| Metric | Value | Hardware |
|--------|-------|----------|
| Training time | ~30-45 minutes | NVIDIA A100 (40GB) |
| Peak memory | ~6 GB | GPU VRAM |
| Reconstruction quality | Cosine sim > 0.82 | - |

### Performance Optimization Tips

#### Speed Optimization

1. **Use GPU**: 5-10x faster than CPU for typical datasets
   ```bash
   python 1_deconvolution.py --config config.yaml --device cuda:0
   ```

2. **Choose Fast Loss Function**: For prototyping
   ```yaml
   loss_type: "mse"  # Fastest option
   ```

3. **Reduce Epochs for Testing**: Quick iteration
   ```yaml
   epochs: 500  # Fast test run
   ```

4. **Optimize Spatial Graph**: Fewer neighbors = faster
   ```yaml
   n_neighbors: 3  # Minimum for reasonable results
   ```

#### Memory Optimization

1. **For Large Datasets (>10,000 spots)**:
   ```yaml
   # Use gradient accumulation (if supported)
   gradient_accumulation_steps: 4
   
   # Reduce embedding dimension during extraction
   embedding_dim: 256  # Instead of 512
   ```

2. **CPU Training for Limited GPU Memory**:
   ```bash
   python 1_deconvolution.py --config config.yaml --device cpu
   ```

3. **Mixed Precision Training** (if supported):
   ```yaml
   use_amp: true  # Automatic Mixed Precision
   ```

#### Quality Optimization

1. **For Best Results** (slower but higher quality):
   ```yaml
   epochs: 5000
   loss_type: "combined"  # Best reconstruction
   lam_recon: 200.0       # Higher weight
   lr: 0.003              # Slightly lower for stability
   ```

2. **Hyperparameter Tuning Order**:
   - First: Tune `lr` (try 0.001, 0.005, 0.01)
   - Second: Tune `lam_recon` (try 50, 100, 200)
   - Third: Tune `lam_contrast` (try 0.5, 1.0, 2.0)
   - Fourth: Tune `n_neighbors` (try 3, 6, 10)

3. **Validation Strategy**:
   ```python
   # Split data for validation
   from sklearn.model_selection import train_test_split
   
   # Hold out 20% of spots for validation
   train_idx, val_idx = train_test_split(
       range(adata.n_obs), 
       test_size=0.2, 
       random_state=42
   )
   ```

## Citation

If you use this deconvolution module, please cite:

```bibtex
@article{coseniche2024,
  title={CoseNiche: A Foundation Model for Spatial Transcriptomics},
  author={Your Name et al.},
  journal={bioRxiv},
  year={2024}
}
```

## Related Modules

- **Attention Analysis**: `../attention_analysis/`
- **Spatial Communication**: `../spatial_communication/`

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/CoseNiche/issues)
- **Documentation**: [Full Documentation](https://coseniche.readthedocs.io/)
- **Examples**: See `../../examples/` for more use cases

## License

MIT License - see project root for details.
