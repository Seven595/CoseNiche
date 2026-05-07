#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Sankey Diagram Data Format Converter

Unified script for all datasets - just change the dataset configuration!

Convert enrichment analysis results to Sankey diagram format

Output format (one CSV per domain):
Description,GeneRatio,pvalue,geneID,Count
Circadian rhythm,0.014084507,0.010497623,RORA/RORB,2
NOD-like receptor signaling pathway,0.028169014,0.05303255,CASP8/TRIP6/MAPK8/CASP1,4

Usage:
    python 4_prepare_sankey_data.py --dataset PDAC
    python 4_prepare_sankey_data.py --dataset HBRC
    python 4_prepare_sankey_data.py --dataset OvaryCancer
"""

import os
import glob
import argparse
from typing import Optional

import numpy as np
import pandas as pd

# Import common utilities
from common_utils import safe_filename
from dataset_config import get_config


def load_enrichment_result(csv_path: str) -> Optional[pd.DataFrame]:
    """Load single enrichment result file"""
    if not os.path.exists(csv_path):
        print(f"[WARN] File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        required_cols = ['term', 'adj_pval', 'genes']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[WARN] Missing columns {missing_cols} in {csv_path}")
            return None
        
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {csv_path}: {e}")
        return None


def convert_to_sankey_format(
    df: pd.DataFrame,
    top_n: int = 50,
    adj_pval_cutoff: float = 0.05,
    min_gene_count: int = 2
) -> pd.DataFrame:
    """
    Convert enrichment results to Sankey diagram format
    
    Output columns:
    - Description: Pathway name (using term_clean)
    - GeneRatio: Gene ratio (numeric)
    - pvalue: Adjusted p-value
    - geneID: Gene list (/ separated)
    - Count: Gene count
    """
    # Filter significant results
    df_sig = df[df['adj_pval'] <= adj_pval_cutoff].copy()
    
    if df_sig.empty:
        print(f"[WARN] No significant results with p < {adj_pval_cutoff}")
        return pd.DataFrame(columns=['Description', 'GeneRatio', 'pvalue', 'geneID', 'Count'])
    
    # Process gene lists
    if 'genes' in df_sig.columns:
        df_sig['gene_list'] = df_sig['genes'].apply(
            lambda x: [g.strip() for g in str(x).split(';')] if pd.notna(x) and str(x) != '' else []
        )
        df_sig['gene_count_actual'] = df_sig['gene_list'].apply(len)
        df_sig['geneID'] = df_sig['gene_list'].apply(lambda x: '/'.join(x) if x else '')
    else:
        df_sig['gene_count_actual'] = 0
        df_sig['geneID'] = ''
    
    # Filter by gene count
    df_sig = df_sig[df_sig['gene_count_actual'] >= min_gene_count]
    
    if df_sig.empty:
        print(f"[WARN] No terms with >= {min_gene_count} genes")
        return pd.DataFrame(columns=['Description', 'GeneRatio', 'pvalue', 'geneID', 'Count'])
    
    # Calculate neg_log10_pval for sorting
    if 'neg_log10_pval' not in df_sig.columns:
        df_sig['neg_log10_pval'] = -np.log10(df_sig['adj_pval'].clip(lower=1e-300))
    
    # Sort by significance
    df_sig = df_sig.sort_values('neg_log10_pval', ascending=False)
    
    # Take top N
    df_sig = df_sig.head(top_n)
    
    # Prepare output DataFrame
    output_df = pd.DataFrame()
    
    # Description: use term_clean if exists, otherwise clean term
    if 'term_clean' in df_sig.columns:
        output_df['Description'] = df_sig['term_clean']
    elif 'term' in df_sig.columns:
        output_df['Description'] = df_sig['term'].astype(str).str.replace(
            r'\s*\([A-Z]{2}:\d+\)$|\s*\(WP\d+\)$', '', regex=True
        )
    else:
        output_df['Description'] = 'Unknown'
    
    # GeneRatio: numeric gene ratio
    if 'gene_ratio' in df_sig.columns:
        output_df['GeneRatio'] = df_sig['gene_ratio']
    else:
        output_df['GeneRatio'] = 0.0
    
    # pvalue: adjusted p-value
    output_df['pvalue'] = df_sig['adj_pval']
    
    # geneID: gene list (/ separated)
    output_df['geneID'] = df_sig['geneID']
    
    # Count: gene count
    output_df['Count'] = df_sig['gene_count_actual']
    
    output_df = output_df.reset_index(drop=True)
    
    return output_df


def process_all_enrichment_files(
    enrichment_dir: str,
    output_dir: str,
    top_n: int = 50,
    adj_pval_cutoff: float = 0.05,
    min_gene_count: int = 2
):
    """Batch process all enrichment result files"""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(enrichment_dir, "*_enrichment.csv"))
    
    if not csv_files:
        print(f"[ERROR] No enrichment files found in {enrichment_dir}")
        return
    
    print(f"\n[INFO] Found {len(csv_files)} enrichment result files")
    print("=" * 60)
    
    success_count = 0
    failed_domains = []
    
    for csv_path in csv_files:
        domain = os.path.basename(csv_path).replace("_enrichment.csv", "")
        print(f"\n[Processing] Domain: {domain}")
        
        try:
            # Load data
            df = load_enrichment_result(csv_path)
            
            if df is None or df.empty:
                print(f"[SKIP] {domain}: Failed to load or empty file")
                failed_domains.append(domain)
                continue
            
            print(f"  Total terms in file: {len(df)}")
            
            # Convert format
            sankey_df = convert_to_sankey_format(
                df=df,
                top_n=top_n,
                adj_pval_cutoff=adj_pval_cutoff,
                min_gene_count=min_gene_count
            )
            
            if sankey_df.empty:
                print(f"[SKIP] {domain}: No terms passed filtering criteria")
                failed_domains.append(domain)
                continue
            
            print(f"  Terms after filtering: {len(sankey_df)}")
            
            # Save
            safe_name = safe_filename(domain)
            output_path = os.path.join(output_dir, f"{safe_name}_sankey.csv")
            
            sankey_df.to_csv(output_path, sep=',', index=False, encoding='utf-8-sig')
            
            print(f"[SAVE] {output_path}")
            
            # Preview
            print(f"\n  Preview (first 3 rows):")
            preview = sankey_df.head(3)
            for idx, row in preview.iterrows():
                genes_preview = row['geneID'][:50] + '...' if len(str(row['geneID'])) > 50 else row['geneID']
                print(f"    {row['Description'][:40]:40s} | GeneRatio: {row['GeneRatio']:.4f} | "
                      f"p-val: {row['pvalue']:.2e} | Count: {row['Count']} | Genes: {genes_preview}")
            
            success_count += 1
            print(f"[SUCCESS] {domain}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {domain}: {e}")
            failed_domains.append(domain)
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("Data Conversion Complete!")
    print("=" * 60)
    print(f"Total domains: {len(csv_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_domains)}")
    
    if failed_domains:
        print(f"\nFailed domains: {', '.join(failed_domains)}")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles are saved in CSV format (comma-separated, UTF-8 with BOM for Excel compatibility)")
    print(f"Columns: Description, GeneRatio, pvalue, geneID, Count")


def main(dataset_name: str):
    """Main pipeline"""
    config = get_config(dataset_name)
    
    print("=" * 60)
    print(f"Sankey Data Preparation - {config.name}")
    print("=" * 60)
    
    if not os.path.exists(config.enrichment_results_dir):
        print(f"[ERROR] Enrichment results directory not found: {config.enrichment_results_dir}")
        print("Please run enrichment analysis first (script 2)")
        return
    
    print(f"\n[Config]")
    print(f"  Input directory: {config.enrichment_results_dir}")
    print(f"  Output directory: {config.sankey_prepare_dir}")
    print(f"  Top terms per domain: {config.sankey_top_terms}")
    print(f"  P-value cutoff: {config.adj_pval_cutoff}")
    print(f"  Min gene count: {config.sankey_min_gene_count}")
    
    process_all_enrichment_files(
        enrichment_dir=config.enrichment_results_dir,
        output_dir=config.sankey_prepare_dir,
        top_n=config.sankey_top_terms,
        adj_pval_cutoff=config.adj_pval_cutoff,
        min_gene_count=config.sankey_min_gene_count
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Sankey diagram data from enrichment results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (PDAC, HBRC, OvaryCancer)"
    )
    
    args = parser.parse_args()
    main(args.dataset)
