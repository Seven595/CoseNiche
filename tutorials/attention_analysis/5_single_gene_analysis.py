#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" gene function can analysis:gene in different spatial domains of : 1. of gene (tumor, immunegene) 2. in different domain in random containsgene of spots 3. gene in each spot of Top partner genes 4. different domain in genepartner of pathway gene in different spatial microenvironments in of can Usage: python 5_single_gene_in_spots_analysis.py --dataset PDAC python 5_single_gene_in_spots_analysis.py --dataset HBRC python 5_single_gene_in_spots_analysis.py --dataset Ovary Cancer """

import os
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
import re

# Import dataset configuration
from dataset_config import get_config

# =========================
# Default Analysis Parameters
# =========================
# genefilterParameters
MIN_DOMAINS_WITH_GENE = 2  # gene in domain in table
MIN_SPOTS_PER_DOMAIN = 3   # gene in each domain in table of spot count

# parameters
SPOTS_PER_DOMAIN = 5  # each domain of spot count
TOP_K_PARTNERS = 100  # Top-K partnergene
MIN_EXPRESSION = 0.1  # genetable (optional)

# of gene (optional, if is emptyfilter)
MANUAL_QUERY_GENES = [
    # 'KRAS', # pancreatic cancergene
    # 'TP53', # tumorgene
    # 'CD8A', # T
    # 'COL1A1', # (stroma)
    # 'KRT19', #
    # 'PDGFRA', #
    # 'MKI67', #
    # 'CD68', #
]

# Analysis Parameters - from configReading
ADJ_PVAL_CUTOFF = 0.05

# =========================
# Helper Functions
# =========================
def load_data(config):
    """load all required"""
    print("[INFO] Loading data...")
    
    # loadh5ad
    adata = sc.read_h5ad(config.h5ad_path)
    
    # loadground truth (if)
    if config.truth_path is not None:
        df_meta = pd.read_csv(config.truth_path)[config.truth_column]
        adata.obs['domain'] = df_meta.values
    else:
        # of column
        if config.obs_column in adata.obs.columns:
            adata.obs['domain'] = adata.obs[config.obs_column]
        else:
            raise ValueError(f"Column '{config.obs_column}' not found in adata.obs")
    
    # loadvocab
    with open(config.vocab_path, 'r') as f:
        vocab = json.load(f)
    # build symbol -> id and id -> symbol
    sym2id = {str(k).strip(): int(v) for k, v in vocab.items()}
    id2sym = {int(v): str(k).strip() for k, v in vocab.items()}
    
    # loadcontext genes
    with open(config.ctx_genes_pkl_path, "rb") as f:
        ctx_genes = pickle.load(f)
    
    # loadattention scores
    with open(config.attn_pkl_path, "rb") as f:
        attn_packs = pickle.load(f)
    
    # attention to
    layer_key = f"context_encoder_layer_{config.use_layer}"
    A_list = []
    for pack in attn_packs:
        A_raw = np.array(pack[layer_key]["center_self_attention"])
        if A_raw.ndim == 3:
            for i in range(A_raw.shape[0]):
                A_list.append(A_raw[i])
        else:
            A_list.append(A_raw)
    
    print(f"  Loaded {len(A_list)} attention matrices")
    print(f" Loaded {len(ctx_genes)} context gene lists")
    print(f"  Total spots: {adata.n_obs}")
    print(f"  Domains: {adata.obs['domain'].unique()}")
    
    return adata, sym2id, id2sym, ctx_genes, A_list


def filter_genes_by_domain_coverage(adata, ctx_genes: List, id2sym: Dict,
                                    min_domains: int = 2,
                                    min_spots_per_domain: int = 3) -> List[str]:
    """ filter in multiple domain in table of gene Args: adata: Ann Data,containsdomain ctx_genes: each spot of gene list id2sym: gene ID to symbol of min_domains: gene in domain in table min_spots_per_domain: gene in each domain in table of spot count Returns: of gene list """
    print("\n[INFO] Filtering genes by domain coverage...")
    
    domains = adata.obs['domain'].unique()
    print(f"  Domains: {domains}")
    
    # genes in each domain in of table
    gene_domain_counts = defaultdict(lambda: defaultdict(int))
    
    for spot_idx in range(len(ctx_genes)):
        domain = adata.obs['domain'].iloc[spot_idx]
        genes_in_spot = ctx_genes[spot_idx]
        
        for gene_id in genes_in_spot:
            gene_symbol = id2sym.get(int(gene_id), str(gene_id))
            gene_domain_counts[gene_symbol][domain] += 1
    
    # filtergene
    qualified_genes = []
    
    for gene_symbol, domain_counts in gene_domain_counts.items():
        # gene in domain in spot count
        domains_with_sufficient_spots = sum(
            1 for domain, count in domain_counts.items()
            if count >= min_spots_per_domain
        )
        
        if domains_with_sufficient_spots >= min_domains:
            qualified_genes.append(gene_symbol)
    
    print(f"\n  Total genes found: {len(gene_domain_counts)}")
    print(f"  Genes expressed in >= {min_domains} domains: {len(qualified_genes)}")
    
    # Note.
    if qualified_genes:
        print(f"\n  Top 20 qualified genes (by total spots):")
        gene_total_spots = {
            gene: sum(gene_domain_counts[gene].values())
            for gene in qualified_genes
        }
        top_genes = sorted(gene_total_spots.items(), key=lambda x: x[1], reverse=True)[:20]
        for gene, count in top_genes:
            domains_info = {d: gene_domain_counts[gene][d] for d in domains if d in gene_domain_counts[gene]}
            print(f"    {gene:15s}: {count:4d} spots across {len(domains_info)} domains - {domains_info}")
    
    return qualified_genes


def find_gene_in_spot(spot_idx: int, gene_symbol: str, ctx_genes: List, id2sym: Dict) -> Tuple[bool, int]:
    """Check spotcontainsgene, (contains, gene in spot in of)"""
    genes_in_spot = ctx_genes[spot_idx]
    symbols_in_spot = [id2sym.get(int(g), str(g)) for g in genes_in_spot]
    
    try:
        gene_idx = symbols_in_spot.index(gene_symbol)
        return True, gene_idx
    except ValueError:
        return False, -1


def get_top_partners_for_gene(spot_idx: int, gene_idx: int, 
                               ctx_genes: List, A_list: List, id2sym: Dict,
                               top_k: int = 100) -> List[Tuple[str, float]]:
    """gene in spot of Top-K partner genes"""
    genes_in_spot = ctx_genes[spot_idx]
    A = A_list[spot_idx]
    
    # Note.
    A_sym = 0.5 * (A + A.T) if A.ndim == 2 else A
    
    # gene of rows
    if gene_idx >= A_sym.shape[0]:
        return []
    
    attention_row = A_sym[gene_idx, :]
    
    # align:
    min_len = min(len(genes_in_spot), len(attention_row))
    if min_len == 0:
        return []
    
    # of
    valid_attention = attention_row[:min_len]
    
    # ()
    sorted_indices = np.argsort(-valid_attention)
    partners = []
    
    for idx in sorted_indices:
        # check
        if idx >= min_len:
            continue
        if idx == gene_idx:  # Skipping
            continue
        if len(partners) >= top_k:
            break
        
        partner_id = genes_in_spot[idx]
        partner_symbol = id2sym.get(int(partner_id), str(partner_id))
        score = float(valid_attention[idx])
        partners.append((partner_symbol, score))
    
    return partners


def sample_spots_with_gene(adata, gene_symbol: str, domain: str, 
                           ctx_genes: List, id2sym: Dict,
                           n_samples: int = 5, min_expr: float = 0.1) -> List[int]:
    """ in domain in random containsgene of spots"""
    # domain of all spots
    domain_mask = adata.obs['domain'] == domain
    domain_indices = np.where(domain_mask)[0].tolist()
    
    # filtercontainsgeneexpression level of spots
    valid_spots = []
    for spot_idx in domain_indices:
        has_gene, _ = find_gene_in_spot(spot_idx, gene_symbol, ctx_genes, id2sym)
        if has_gene:
            # optional:Checkexpression level
            # expr_val = adata.X[spot_idx, adata.var_names == gene_symbol]
            # if expr_val > min_expr:
            valid_spots.append(spot_idx)
    
    # random
    if len(valid_spots) == 0:
        return []
    
    n_samples = min(n_samples, len(valid_spots))
    sampled = np.random.choice(valid_spots, size=n_samples, replace=False)
    
    return sampled.tolist()


def run_enrichment_for_gene_list(gene_list: List[str], 
                                  config,
                                  cutoff: float = None) -> pd.DataFrame:
    """gene listperformanalysis"""
    if len(gene_list) == 0:
        return pd.DataFrame()
    
    if cutoff is None:
        cutoff = config.adj_pval_cutoff
    
    all_results = []
    for library in config.enrich_libraries:
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=library,
                organism="Human",
                outdir=None,
                cutoff=cutoff,
                no_plot=True,
            )
            
            df = enr.results
            if df is not None and not df.empty:
                df["Library"] = library
                df = df.rename(columns={
                    "Term": "term",
                    "Adjusted P-value": "adj_pval",
                    "P-value": "pval",
                })
                df["term_clean"] = df["term"].str.replace(
                    r'\s*\([A-Z]{2}:\d+\)$|\s*\(WP\d+\)$', '', regex=True
                )
                all_results.append(df)
        except Exception as e:
            print(f"    [WARN] Enrichment failed for {library}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def compare_pathway_enrichment(df1: pd.DataFrame, df2: pd.DataFrame, 
                                domain1: str, domain2: str) -> pd.DataFrame:
    """ domain of,pathway"""
    if df1.empty or df2.empty:
        return pd.DataFrame()
    
    # domain of pathway
    terms1 = set(df1[df1['adj_pval'] < ADJ_PVAL_CUTOFF]['term_clean'].unique())
    terms2 = set(df2[df2['adj_pval'] < ADJ_PVAL_CUTOFF]['term_clean'].unique())
    
    # pathway
    unique_to_domain1 = terms1 - terms2
    unique_to_domain2 = terms2 - terms1
    shared = terms1 & terms2
    
    comparison = []
    
    # Domain1pathway
    for term in unique_to_domain1:
        row = df1[df1['term_clean'] == term].iloc[0]
        comparison.append({
            'term': term,
            'category': f'Unique to {domain1}',
            'domain1_pval': row['adj_pval'],
            'domain2_pval': np.nan,
            'library': row['Library']
        })
    
    # Domain2pathway
    for term in unique_to_domain2:
        row = df2[df2['term_clean'] == term].iloc[0]
        comparison.append({
            'term': term,
            'category': f'Unique to {domain2}',
            'domain1_pval': np.nan,
            'domain2_pval': row['adj_pval'],
            'library': row['Library']
        })
    
    # of pathway
    for term in shared:
        row1 = df1[df1['term_clean'] == term].iloc[0]
        row2 = df2[df2['term_clean'] == term].iloc[0]
        
        # computep
        pval_ratio = row1['adj_pval'] / row2['adj_pval']
        if pval_ratio > 2 or pval_ratio < 0.5:  # 2
            comparison.append({
                'term': term,
                'category': 'Shared (differential significance)',
                'domain1_pval': row1['adj_pval'],
                'domain2_pval': row2['adj_pval'],
                'pval_ratio': pval_ratio,
                'library': row1['Library']
            })
    
    return pd.DataFrame(comparison)


# =========================
# Main Analysis Pipeline
# =========================
def analyze_gene_plasticity(gene_symbol: str, adata, sym2id, id2sym, 
                            ctx_genes, A_list, config):
    """analysisgenes in different domain of can """
    print(f"\n{'='*60}")
    print(f"Analyzing gene: {gene_symbol}")
    print(f"{'='*60}")
    
    # Checkgene in vocab in
    if gene_symbol not in sym2id:
        print(f"[SKIP] Gene {gene_symbol} not in vocabulary")
        return
    
    domains = adata.obs['domain'].unique()
    print(f"  Domains: {domains}")
    
    results = {}
    
    # config in of Parameters
    spots_per_domain = config.spots_per_domain_sample
    top_k_partners = config.top_k_partners_plasticity
    
    # each domainspotspartners
    for domain in domains:
        print(f"\n[Domain: {domain}]")
        
        # spots
        sampled_spots = sample_spots_with_gene(
            adata, gene_symbol, domain, ctx_genes, id2sym, 
            n_samples=spots_per_domain
        )
        
        if len(sampled_spots) == 0:
            print(f"  No spots found containing {gene_symbol}")
            continue
        
        print(f"  Sampled {len(sampled_spots)} spots")
        
        # all spots in gene of partners
        all_partners = []
        for spot_idx in sampled_spots:
            has_gene, gene_idx = find_gene_in_spot(spot_idx, gene_symbol, ctx_genes, id2sym)
            if has_gene:
                partners = get_top_partners_for_gene(
                    spot_idx, gene_idx, ctx_genes, A_list, id2sym, 
                    top_k=top_k_partners
                )
                all_partners.extend([p[0] for p in partners])
        
        # Note.
        partner_counts = pd.Series(all_partners).value_counts()
        top_partners = partner_counts.head(top_k_partners).index.tolist()
        
        print(f"  Found {len(partner_counts)} unique partners")
        print(f"  Top 5 partners: {', '.join(top_partners[:5])}")
        
        # analysis
        print(f"  Running enrichment analysis...")
        enrichment_df = run_enrichment_for_gene_list(
            top_partners, config
        )
        
        if not enrichment_df.empty:
            sig_terms = enrichment_df[enrichment_df['adj_pval'] < config.adj_pval_cutoff]
            print(f"  Found {len(sig_terms)} significant pathways")
        else:
            print(f"  No significant pathways found")
        
        results[domain] = {
            'spots': sampled_spots,
            'partners': top_partners,
            'enrichment': enrichment_df
        }
    
    # domains
    print(f"\n[Pairwise Domain Comparison]")
    domain_list = list(results.keys())
    
    comparisons = []
    for i in range(len(domain_list)):
        for j in range(i+1, len(domain_list)):
            d1, d2 = domain_list[i], domain_list[j]
            print(f"\n  Comparing {d1} vs {d2}")
            
            comparison_df = compare_pathway_enrichment(
                results[d1]['enrichment'],
                results[d2]['enrichment'],
                d1, d2
            )
            
            if not comparison_df.empty:
                print(f"    Found {len(comparison_df)} differential pathways")
                comparison_df['gene'] = gene_symbol
                comparison_df['domain_pair'] = f"{d1}_vs_{d2}"
                comparisons.append(comparison_df)
    
    # save
    gene_output_dir = os.path.join(config.gene_plasticity_dir, safe_filename(gene_symbol))
    os.makedirs(gene_output_dir, exist_ok=True)
    
    # Save each domain of
    for domain, data in results.items():
        # 1. save
        domain_file = os.path.join(gene_output_dir, f"{safe_filename(domain)}_enrichment.csv")
        if not data['enrichment'].empty:
            data['enrichment'].to_csv(domain_file, index=False)
            print(f"[SAVE] {domain_file}")
        
        # 2. Savetop partnergene list
        partners_file = os.path.join(gene_output_dir, f"{safe_filename(domain)}_top_partners.txt")
        with open(partners_file, 'w') as f:
            f.write(f"# Top partner genes for {gene_symbol} in {domain}\n")
            f.write(f"# Total unique partners: {len(data['partners'])}\n")
            f.write(f"# Sampled from {len(data['spots'])} spots\n")
            f.write("#" + "="*50 + "\n")
            for i, partner in enumerate(data['partners'], 1):
                f.write(f"{i}\t{partner}\n")
        print(f"[SAVE] {partners_file}")
        
        # 3. save of partners (contains)
        partners_stats_file = os.path.join(gene_output_dir, f"{safe_filename(domain)}_partners_stats.csv")
        # computepartner (from)
        all_partners_with_scores = []
        for spot_idx in data['spots']:
            has_gene, gene_idx = find_gene_in_spot(spot_idx, gene_symbol, ctx_genes, id2sym)
            if has_gene:
                partners = get_top_partners_for_gene(
                    spot_idx, gene_idx, ctx_genes, A_list, id2sym, 
                    top_k=config.top_k_partners_plasticity
                )
                all_partners_with_scores.extend(partners)
        
        # each partner of and attention score
        partner_stats = defaultdict(lambda: {'count': 0, 'total_score': 0.0, 'scores': []})
        for partner, score in all_partners_with_scores:
            partner_stats[partner]['count'] += 1
            partner_stats[partner]['total_score'] += score
            partner_stats[partner]['scores'].append(score)
        
        # build DataFrame
        stats_rows = []
        for partner in data['partners']:  # by of top partners
            if partner in partner_stats:
                stats = partner_stats[partner]
                stats_rows.append({
                    'partner_gene': partner,
                    'frequency': stats['count'],
                    'avg_attention_score': stats['total_score'] / stats['count'],
                    'max_attention_score': max(stats['scores']),
                    'min_attention_score': min(stats['scores'])
                })
        
        if stats_rows:
            stats_df = pd.DataFrame(stats_rows)
            stats_df.to_csv(partners_stats_file, index=False)
            print(f"[SAVE] {partners_stats_file}")
    
    # save
    if comparisons:
        comparison_file = os.path.join(gene_output_dir, "domain_comparisons.csv")
        pd.concat(comparisons, ignore_index=True).to_csv(comparison_file, index=False)
        print(f"[SAVE] {comparison_file}")
    
    # savetotal
    summary_file = os.path.join(gene_output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Gene Plasticity Analysis Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Query Gene: {gene_symbol}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  - Spots sampled per domain: {config.spots_per_domain_sample}\n")
        f.write(f"  - Top K partners extracted: {config.top_k_partners_plasticity}\n")
        f.write(f"  - Enrichment libraries: {', '.join(config.enrich_libraries)}\n")
        f.write(f"  - Adj p-value cutoff: {config.adj_pval_cutoff}\n\n")
        
        f.write(f"Results by Domain:\n")
        f.write(f"{'-'*60}\n")
        for domain, data in results.items():
            f.write(f"\n{domain}:\n")
            f.write(f"  - Spots sampled: {len(data['spots'])}\n")
            f.write(f"  - Unique partner genes: {len(data['partners'])}\n")
            f.write(f"  - Top 10 partners: {', '.join(data['partners'][:10])}\n")
            if not data['enrichment'].empty:
                sig_count = len(data['enrichment'][data['enrichment']['adj_pval'] < ADJ_PVAL_CUTOFF])
                f.write(f"  - Significant pathways: {sig_count}\n")
            else:
                f.write(f"  - Significant pathways: 0\n")
        
        f.write(f"\n{'-'*60}\n")
        f.write(f"Domain Comparisons:\n")
        if comparisons:
            total_diff = sum(len(c) for c in comparisons)
            f.write(f"  - Total differential pathways: {total_diff}\n")
            for comp_df in comparisons:
                if not comp_df.empty:
                    domain_pair = comp_df['domain_pair'].iloc[0]
                    f.write(f"  - {domain_pair}: {len(comp_df)} pathways\n")
        else:
            f.write(f"  - No differential pathways found\n")
    
    print(f"[SAVE] {summary_file}")
    
    return results, comparisons


def safe_filename(s: str) -> str:
    """ for file"""
    return re.sub(r'[^\w\-]', '_', str(s))


def main(dataset_name: str):
    """"""
    # load
    config = get_config(dataset_name)
    
    # createOutput directory
    os.makedirs(config.gene_plasticity_dir, exist_ok=True)
    
    print("="*60)
    print(f"Gene Functional Plasticity Analysis - {config.name}")
    print("="*60)
    print(f"\n[Config]")
    print(f"  Dataset: {config.name}")
    print(f"  Output directory: {config.gene_plasticity_dir}")
    print(f"  Min domains with gene: {config.min_domains_with_gene}")
    print(f"  Min spots per domain: {config.min_spots_per_domain}")
    print(f"  Spots sampled per domain: {config.spots_per_domain_sample}")
    print(f"  Top K partners: {config.top_k_partners_plasticity}")
    
    # load
    adata, sym2id, id2sym, ctx_genes, A_list = load_data(config)
    
    # :filter in multiple domain in table of gene
    if MANUAL_QUERY_GENES:
        query_genes = MANUAL_QUERY_GENES
        print(f"\n[INFO] Using manually specified genes: {len(query_genes)} genes")
    else:
        qualified_genes = filter_genes_by_domain_coverage(
            adata, ctx_genes, id2sym,
            min_domains=config.min_domains_with_gene,
            min_spots_per_domain=config.min_spots_per_domain
        )
        
        if not qualified_genes:
            print("\n[ERROR] No genes found meeting the domain coverage criteria!")
            print(f"  Try lowering min_domains_with_gene (current: {config.min_domains_with_gene})")
            print(f"  or min_spots_per_domain (current: {config.min_spots_per_domain})")
            return
        
        # select Topgeneperformanalysis (by totaltablespots)
        gene_total_spots = {}
        for gene in qualified_genes:
            total_spots = 0
            for spot_idx in range(len(ctx_genes)):
                has_gene, _ = find_gene_in_spot(spot_idx, gene, ctx_genes, id2sym)
                if has_gene:
                    total_spots += 1
            gene_total_spots[gene] = total_spots
        
        # select Top 20genes (can)
        top_n = min(20, len(qualified_genes))
        query_genes = sorted(gene_total_spots.items(), key=lambda x: x[1], reverse=True)[:top_n]
        query_genes = [g[0] for g in query_genes]
        
        print(f"\n[INFO] Selected top {len(query_genes)} genes for analysis:")
        for gene in query_genes:
            print(f"  - {gene} ({gene_total_spots[gene]} spots)")
    
    # Savefilter of gene list
    genes_file = os.path.join(config.gene_plasticity_dir, "analyzed_genes.txt")
    with open(genes_file, 'w') as f:
        f.write('\n'.join(query_genes))
    print(f"\n[SAVE] Gene list saved to: {genes_file}")
    
    # :analysis each gene
    all_results = {}
    for gene in query_genes:
        try:
            results, comparisons = analyze_gene_plasticity(
                gene, adata, sym2id, id2sym, ctx_genes, A_list, config
            )
            all_results[gene] = {'results': results, 'comparisons': comparisons}
        except Exception as e:
            print(f"[ERROR] Failed to analyze {gene}: {e}")
            import traceback
            traceback.print_exc()
    
    # total
    print("\n" + "="*60)
    print("Summary Report")
    print("="*60)
    
    for gene, data in all_results.items():
        print(f"\n{gene}:")
        if data['comparisons']:
            total_differential = sum(len(c) for c in data['comparisons'])
            print(f"  Total differential pathways: {total_differential}")
        else:
            print(f"  No differential pathways found")
    
    print(f"\nResults saved to: {config.gene_plasticity_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gene Functional Plasticity Analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (PDAC, HBRC, OvaryCancer)"
    )
    
    args = parser.parse_args()
    
    np.random.seed(42)  # can
    main(args.dataset)