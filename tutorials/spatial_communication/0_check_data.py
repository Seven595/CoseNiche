"""Data and gene check utility.

Run this before the full analysis to check required files, marker gene coverage,
attention data statistics, and cluster information.
"""

import os
import glob
import json
import pandas as pd
import scanpy as sc
import numpy as np

# HBRC marker genes.
HBRC_MARKER_GENES = {
    "B cell": ["MS4A1", "CD79A", "CD79B", "BANK1"],
    "NK cell": ["NKG7", "PRF1", "GNLY", "KLRD1"],
    "T cell": ["CD3D", "CD3E", "TRAC", "CD2"],
    "fibroblast": ["COL1A1", "DCN", "COL1A2", "LUM"],
    "luminal cell": ["KRT8", "EPCAM", "KRT18", "MUC1"],
    "luminal progenitor": ["KRT23", "KIT", "KRT8", "KRT18"],
    "lymphatic endothelial cell": ["PROX1", "PDPN", "LYVE1", "FLT4"],
    "macrophage/DC/monocyte": ["LYZ", "LST1", "MRC1", "FCGR3A"],
    "muscle cell": ["ACTA2", "TAGLN", "MYH11", "MYL9"],
    "myoepithelial cell": ["KRT14", "KRT5", "ACTA2", "TP63"],
    "pDC": ["GZMB", "IRF7", "TCF4", "CLEC4C"],
    "plasma cell": ["MZB1", "IGHG1", "SDC1", "IGKC"],
    "vascular endothelial cell": ["PECAM1", "VWF", "KDR", "ENG"],
}

def find_latest_data_dir():
    """Find the latest data directory"""
    base_dir = "/home/junning/projectnvme/ST/project-20-contrast-organ/Analysis/spatial_communications"
    data_dirs = glob.glob(os.path.join(base_dir, "HBRC1/whole_slice_data_*"))
    
    if data_dirs:
        latest = sorted(data_dirs)[-1]
        return latest
    return None

def check_files(data_dir):
    """Check whether required files exist"""
    print("=" * 70)
    print("1. Check files")
    print("=" * 70)
    
    required_files = {
        "Configuration file": "export_config.json",
        "spatial coordinates": "spatial_coordinates.csv",
        "expression data": "adata_with_metadata.h5ad",
        "Attention": ["whole_slice_attention_*.csv", "whole_slice_attention_*.parquet"]
    }
    
    all_exist = True
    
    for name, patterns in required_files.items():
        if isinstance(patterns, list):
            # Check multiple possible file patterns.
            found = False
            for pattern in patterns:
                files = glob.glob(os.path.join(data_dir, pattern))
                if files:
                    print(f"✓ {name}: {os.path.basename(files[0])}")
                    found = True
                    break
            if not found:
                print(f"✗ {name}: not found")
                all_exist = False
        else:
            # Check single file
            filepath = os.path.join(data_dir, patterns)
            if os.path.exists(filepath):
                print(f"✓ {name}: {patterns}")
            else:
                print(f"✗ {name}: {patterns} (not found)")
                all_exist = False
    
    print()
    return all_exist

def check_marker_genes(data_dir):
    """Check marker gene coverage."""
    print("=" * 70)
    print("2. Check HBRC marker gene coverage")
    print("=" * 70)
    
    # Load AnnData.
    adata_path = os.path.join(data_dir, "adata_with_metadata.h5ad")
    if not os.path.exists(adata_path):
        print("✗ Failed to load expression data")
        return
    
    print(f"Loading expression data: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  - Spot count: {adata.n_obs}")
    print(f"  - number of genes: {adata.n_vars}")
    print()
    
    # Check marker coverage.
    total_markers = 0
    found_markers = 0
    
    results = []
    
    for cell_type, genes in HBRC_MARKER_GENES.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        total_markers += len(genes)
        found_markers += len(valid_genes)
        
        coverage = len(valid_genes) / len(genes) * 100 if genes else 0
        
        results.append({
            "cell_type": cell_type,
            "total marker count": len(genes),
            "found count": len(valid_genes),
            "coverage": f"{coverage:.0f}%",
            "missing genes": [g for g in genes if g not in adata.var_names]
        })
        
        status = "✓" if len(valid_genes) > 0 else "✗"
        print(f"{status} {cell_type:30s} : {len(valid_genes):2d}/{len(genes):2d} ({coverage:5.1f}%)")
        if len(valid_genes) < len(genes) and len(genes) - len(valid_genes) <= 2:
            missing = [g for g in genes if g not in adata.var_names]
            print(f"    Missing: {', '.join(missing)}")
    
    print()
    print(f"Total: {found_markers}/{total_markers} marker genes found ({found_markers/total_markers*100:.1f}%)")
    print()
    
    return results, adata

def check_attention_data(data_dir):
    """Check attention data statistics."""
    print("=" * 70)
    print("3. Attention data statistics")
    print("=" * 70)
    
    # attention file
    attn_files = glob.glob(os.path.join(data_dir, "whole_slice_attention_*.csv"))
    if not attn_files:
        attn_files = glob.glob(os.path.join(data_dir, "whole_slice_attention_*.parquet"))
    
    if not attn_files:
        print("✗ Attention data file was not found")
        return
    
    attn_path = attn_files[0]
    print(f"Loading attention data: {os.path.basename(attn_path)}")
    
    try:
        if attn_path.endswith('.parquet'):
            attn_df = pd.read_parquet(attn_path)
        else:
            attn_df = pd.read_csv(attn_path)
        
        print(f"  - total records: {len(attn_df):,}")
        print(f"  - center spot count: {attn_df['center_global_idx'].nunique():,}")
        print(f"  - neighbor spot count: {attn_df['neighbor_global_idx'].nunique():,}")
        
        if 'kv_gene_symbol' in attn_df.columns:
            # filter
            kv_genes = attn_df[attn_df['kv_gene_symbol'].str.len() > 0]['kv_gene_symbol']
            print(f"  - KV gene count: {kv_genes.nunique():,}")
            
            # Top KV genes.
            top_kv = attn_df.groupby('kv_gene_symbol')['attn_score'].sum().sort_values(ascending=False).head(10)
            print("\n  Top 10 KV genes (by total attention):")
            for gene, score in top_kv.items():
                if gene and len(str(gene)) > 0:
                    print(f"    {gene:15s} : {score:.4f}")
        
        if 'q_gene_symbol' in attn_df.columns:
            q_genes = attn_df[attn_df['q_gene_symbol'].str.len() > 0]['q_gene_symbol']
            print(f"\n  - Query gene count: {q_genes.nunique():,}")
        
        print()
        
    except Exception as e:
        print(f"✗ Error while loading attention data: {e}")
        return

def check_cluster_info(data_dir):
    """Check cluster information."""
    print("=" * 70)
    print("4. cluster information")
    print("=" * 70)
    
    coords_path = os.path.join(data_dir, "spatial_coordinates.csv")
    if not os.path.exists(coords_path):
        print("✗ Spatial coordinate file was not found")
        return
    
    coords_df = pd.read_csv(coords_path)
    
    if 'cluster' in coords_df.columns:
        n_clusters = coords_df['cluster'].nunique()
        print(f"number of clusters: {n_clusters}")
        print("\nCluster spot counts:")
        
        cluster_counts = coords_df['cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = count / len(coords_df) * 100
            print(f"  Cluster {cluster}: {count:5d} spots ({percentage:5.1f}%)")
    else:
        print("Warning: No cluster information found in the spatial coordinate file")
    
    print()

def main():
    """Run all data checks."""
    print("\n")
    print("=" * 70)
    print("Data and gene check utility")
    print("=" * 70)
    print()
    
    # Data directory
    data_dir = find_latest_data_dir()
    
    if data_dir is None:
        print("✗ Data directory was not found")
        print("\nPlease run 1_enhanced_spatial_data_exporter.py to generate data")
        return
    
    print(f"Data directory: {data_dir}")
    print()
    
    # Run checks.
    files_ok = check_files(data_dir)
    
    if not files_ok:
        print("\n⚠ Some required files are missing; please check the data directory")
        return
    
    # marker_results, adata = check_marker_genes(data_dir)
    check_attention_data(data_dir)
    check_cluster_info(data_dir)
    
    # Summary.
    print("=" * 70)
    print("Check completed")
    print("=" * 70)
    print()
    print("If all checks pass, you can run the full analysis:")
    print("  bash run_kv_analysis.sh")
    print(" or ")
    print("  python 2_kv_gene_attention_expression_analysis.py")
    print()

if __name__ == "__main__":
    main()
