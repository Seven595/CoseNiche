# Attention Analysis Tutorial

This module analyzes CoseNiche context self-attention to identify gene-gene interaction patterns, top partner genes, and enriched pathways across tissue domains.

## Files

```
attention_analysis/
├── 1_export_attention.py
├── 2_enrichment_analysis.py
├── 3_bubble_plot.py
├── 4_prepare_sankey.py
├── 5_single_gene_analysis.py
├── config.yaml
└── README.md
```

## Workflow

Run inference first and save attention outputs:

```bash
python ../../scripts/extract_embeddings.py \
    --model_path /path/to/model.safetensors \
    --h5ad_path /path/to/data.h5ad \
    --cache_dir ./cache \
    --output_dir ./embeddings \
    --save_attention \
    --save_context_genes
```

Then run the analysis:

```bash
python 1_export_attention.py --dataset PDAC --layer 5
python 2_enrichment_analysis.py --dataset PDAC
python 3_bubble_plot.py --dataset PDAC
python 4_prepare_sankey.py --dataset PDAC
python 5_single_gene_analysis.py --dataset PDAC --gene KRAS
```

## Outputs

- Cluster-level long tables with spot, query gene, partner gene, and attention score.
- Domain-level aggregation tables for enrichment preparation.
- Enrichment results for GO, KEGG, and Reactome.
- Bubble plots and Sankey plot input tables.
- Single-gene plasticity reports across domains.

## Notes

Gene filtering removes low-quality symbols, pseudogene-like symbols, and selected blacklist prefixes while preserving useful whitelist prefixes such as `HLA-` and `MIR`.
