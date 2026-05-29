# -*- coding: utf-8 -*-
""" Domain-aware of TopK select - Consider domain information when selecting TopK genes - Neighbors in the same domain tend to select similar genes - domaingene,filter """

import os
import gc
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import anndata as ad

# ---------------- Domainload ----------------

def load_domains(adata_path: str) -> pd.Series:
    """Load domain information"""
    adata = ad.read_h5ad(adata_path)
    if "ground_truth" not in adata.obs:
        raise ValueError("adata.obs  in not found ground_truth column")
    s = adata.obs["ground_truth"].astype(str).copy()
    s.index = s.index.astype(str)
    return s

def get_neighbor_domains(df: pd.DataFrame, spot_domains: pd.Series) -> dict:
    """Get the domain for each neighbor"""
    neighbor_domains = {}
    for neighbor in df['neighbor_name'].unique():
        # neighbor for
        try:
            neighbor_idx = int(neighbor)
            neighbor_name = str(neighbor_idx)
        except:
            neighbor_name = str(neighbor)
        
        if neighbor_name in spot_domains.index:
            neighbor_domains[neighbor] = spot_domains[neighbor_name]
        else:
            neighbor_domains[neighbor] = "Unknown"
    
    return neighbor_domains

# ---------------- Domain-aware of TopK select  ----------------

def topk_domain_aware(df: pd.DataFrame, 
                      neighbor_domains: dict,
                      k: int = 10,
                      domain_weight: float = 0.6,
                      metric: str = "mean") -> pd.DataFrame:
    """ Domain-aware of TopK select Parameters: ----------- df : pd.DataFrame contains center_name, neighbor_name, gene, mean/sum neighbor_domains : dict neighbor-to-domain mapping k : int number of genes selected for each neighbor domain_weight : float (0-1) weight for domain-level ranking - 1.0: fully use unified domain ranking (all domaingene) - 0.0: fully use independent neighbor ranking (ignore domain) - 0.5-0.7: recommended value,balance domain consistency and neighbor specificity metric : str ranking metric (mean or sum) Returns: -------- pd.DataFrame Domain-awareTopK data after selection """
    print(f"\n=== Domain-awareTopK select  (k={k}, domain_weight={domain_weight}) ===")
    
    # 1. computedomainlevel of gene
    print("\n1️⃣ Computing domain-level gene ranking...")
    domain_gene_scores = {}
    unique_domains = set(neighbor_domains.values())
    
    for domain in unique_domains:
        # domain of all
        domain_neighbors = [n for n, d in neighbor_domains.items() if d == domain]
        domain_df = df[df['neighbor_name'].isin(domain_neighbors)]
        
        if len(domain_df) == 0:
            continue
        
        # computedomaingenes of attention
        gene_scores = domain_df.groupby('gene')[metric].mean().sort_values(ascending=False)
        domain_gene_scores[domain] = gene_scores
        
        print(f"  {domain}: {len(domain_neighbors)} neighbors, {len(gene_scores)} genes")
        print(f"    Top5gene: {list(gene_scores.head(5).index)}")
    
    # 2. Selecting TopK
    print(f"\n2️⃣ Selecting Top{k}gene...")
    result_dfs = []
    
    for neighbor in df['neighbor_name'].unique():
        neighbor_df = df[df['neighbor_name'] == neighbor].copy()
        domain = neighbor_domains.get(neighbor, "Unknown")
        
        if domain not in domain_gene_scores:
            # if domain not, to
            topk_df = neighbor_df.nlargest(k, metric)
            result_dfs.append(topk_df)
            continue
        
        # compute
        # ()
        neighbor_df['local_rank'] = neighbor_df[metric].rank(method='first', ascending=False)
        
        # domainlevel ()
        domain_scores = domain_gene_scores[domain]
        neighbor_df['domain_rank'] = neighbor_df['gene'].map(
            lambda g: domain_scores.index.get_loc(g) + 1 if g in domain_scores.index else len(domain_scores) + 1
        )
        
        # ()
        neighbor_df['mixed_rank'] = (
            domain_weight * neighbor_df['domain_rank'] + 
            (1 - domain_weight) * neighbor_df['local_rank']
        )
        
        #  select TopK
        topk_df = neighbor_df.nsmallest(k, 'mixed_rank')
        topk_df = topk_df.drop(columns=['local_rank', 'domain_rank', 'mixed_rank'])
        
        result_dfs.append(topk_df)
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # 3. domaingene
    print(f"\n3️⃣ Within-domain gene recurrence statistics:")
    for domain in unique_domains:
        domain_neighbors = [n for n, d in neighbor_domains.items() if d == domain]
        if len(domain_neighbors) < 2:
            continue
        
        domain_result = result[result['neighbor_name'].isin(domain_neighbors)]
        
        # genes in neighbors in
        gene_freq = domain_result.groupby('gene')['neighbor_name'].nunique()
        
        # : in 2 of gene
        repeated_genes = gene_freq[gene_freq >= 2]
        repeat_rate = len(repeated_genes) / len(gene_freq) if len(gene_freq) > 0 else 0
        
        print(f"  {domain}:")
        print(f"    neighbor count: {len(domain_neighbors)}")
        print(f"    unique gene count: {len(gene_freq)}")
        print(f"    repeated gene count: {len(repeated_genes)} ({repeat_rate*100:.1f}%)")
        print(f" high-frequency genes(>=3): {list(gene_freq[gene_freq >= 3].head(5).index)}")
    
    print(f"\n=== Completed ===\n")
    return result

# ---------------- ----------------

def prepare_domain_aware_topk(
    out_dir: str,
    layer: int = 5,
    value_col: str = "attn_score",
    topk: int = 10,
    domain_weight: float = 0.6,
    gene_view: str = "kv"
):
    """ Generate domain-aware TopK CSV files Parameters: ----------- out_dir : str Output directory layer : int layer number to use value_col : str attention column name topk : int number of genes selected for each neighbor domain_weight : float (0-1) domain - : 0.5-0.7 - higher: stronger domain consistency - lower: more neighbor specificity gene_view : str kv or q """
    print("="*80)
    print(f"Domain-awareTopK")
    print(f"  Output directory: {out_dir}")
    print(f"  Layer: {layer}")
    print(f"  TopK: {topk}")
    print(f" Domain: {domain_weight}")
    print("="*80)
    
    # 1. Load domain information
    adata_path = os.path.join(out_dir, "adata_with_metadata.h5ad")
    spot_domains = load_domains(adata_path)
    print(f"\n✅ Loaded {len(spot_domains)} spot domain entries")
    print(f"   Domains: {spot_domains.unique()}")
    
    # 2. Readingspot-level
    agg_dir = os.path.join(out_dir, "agg_csv")
    full_csv = os.path.join(agg_dir, f"spot_level_{gene_view}.csv")
    
    if not os.path.exists(full_csv):
        raise FileNotFoundError(f"Full data was not found: {full_csv}")
    
    print(f"\n✅ Reading full data: {full_csv}")
    df = pd.read_csv(full_csv)
    print(f"   record count: {len(df)}")
    print(f"   center spots: {df['center_name'].nunique()}")
    
    # 3. for each centerprocessing
    all_results = []
    centers = df['center_name'].unique()
    
    print(f"\nStart processing {len(centers)}  center spots...")
    
    for i, center in enumerate(centers):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing progress: {i+1}/{len(centers)}")
        
        center_df = df[df['center_name'] == center].copy()
        
        # center of domains
        neighbor_domains = get_neighbor_domains(center_df, spot_domains)
        
        # Domain-awareTopK select 
        topk_df = topk_domain_aware(
            center_df,
            neighbor_domains,
            k=topk,
            domain_weight=domain_weight,
            metric="mean"
        )
        
        all_results.append(topk_df)
    
    # 4. save
    result = pd.concat(all_results, ignore_index=True)
    
    output_path = os.path.join(agg_dir, f"spot_level_{gene_view}_top{topk}_domain_aware_w{int(domain_weight*10)}.csv")
    result.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Saved domain-aware TopK file:")
    print(f"   {output_path}")
    print(f"   record count: {len(result)}")
    print(f"={'='*80}\n")
    
    return output_path

# ---------------- ----------------

if __name__ == "__main__":
    out_dir = "./PDAC/whole_slice_data_20251028_173836"
    
    # Test different of domain
    for domain_weight in [0.5, 0.6, 0.7]:
        print(f"\n{'#'*80}")
        print(f"# Test domain_weight = {domain_weight}")
        print(f"{'#'*80}\n")
        
        prepare_domain_aware_topk(
            out_dir=out_dir,
            layer=5,
            topk=5,
            domain_weight=domain_weight,
            gene_view="kv"
        )

