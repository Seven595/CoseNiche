# Spatial Communication Tutorial

This module uses CoseNiche spatial attention weights to analyze communication between neighboring spots, tissue boundary behavior, and ligand-receptor interactions.

## Files

```
spatial_communication/
├── 0_check_data.py
├── 1_export_spatial_data.py
├── 2_expression_analysis.py
├── 3_prepare_polar.py
├── 3.5_prepare_polar_domain_aware.py
├── 4_boundary_visualizer.py
├── 5_plot_polar.py
├── config.yaml
└── README.md
```

## Workflow

```bash
python 0_check_data.py --data_dir ./attention_exports
python 1_export_spatial_data.py --dataset PDAC --layer 5
python 3_prepare_polar.py --out_dir ./attention_exports --layer 5
python 3.5_prepare_polar_domain_aware.py
python 4_boundary_visualizer.py --data_dir ./attention_exports --output_dir ./boundary_analysis
python 5_plot_polar.py --data_dir ./attention_exports --output_dir ./polar_plots
```

## Main Outputs

- `attention_exports/`: exported spatial attention tables and optimized data structures.
- `spatial_coords.csv`: spot coordinates and optional cluster labels.
- `boundary_analysis/`: boundary spots, cross-cluster ratios, and gene-pair summaries.
- `polar_plots/`: polar plots showing directional communication patterns.

## Analysis Ideas

- Identify high-attention interactions across tissue domains.
- Detect boundary spots with strong cross-domain attention.
- Compare ligand-receptor pairs across neighboring regions.
- Visualize directional communication using polar coordinates.

## Required Inputs

- AnnData `.h5ad` file.
- Context gene lists and attention weights from CoseNiche inference.
- Spatial coordinates.
- Optional ligand-receptor database.
