#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基因功能可塑性分析：比较同一基因在不同空间域的交互网络

功能：
1. 对预定义的关键基因（如肿瘤标志物、免疫基因等）
2. 在不同domain中随机采样包含该基因的spots
3. 提取该基因在每个spot的Top partner genes
4. 对比不同domain中该基因partner的富集通路差异

这揭示了基因在不同空间微环境中的功能可塑性

Usage:
    python 5_single_gene_in_spots_analysis.py --dataset PDAC
    python 5_single_gene_in_spots_analysis.py --dataset HBRC
    python 5_single_gene_in_spots_analysis.py --dataset OvaryCancer
"""

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
# 基因筛选参数
MIN_DOMAINS_WITH_GENE = 2  # 基因至少在多少个domain中表达
MIN_SPOTS_PER_DOMAIN = 3   # 基因在每个domain中至少表达的spot数量

# 采样参数
SPOTS_PER_DOMAIN = 5  # 每个domain采样的spot数量
TOP_K_PARTNERS = 100  # 提取Top-K个partner基因
MIN_EXPRESSION = 0.1  # 基因最低表达阈值（可选）

# 手动指定感兴趣的基因（可选，如果为空则自动筛选）
MANUAL_QUERY_GENES = [
    # 'KRAS',      # 胰腺癌驱动基因
    # 'TP53',      # 肿瘤抑制基因
    # 'CD8A',      # T细胞标志物
    # 'COL1A1',    # 胶原蛋白（基质）
    # 'KRT19',     # 上皮细胞标志物
    # 'PDGFRA',    # 成纤维细胞标志物
    # 'MKI67',     # 增殖标志物
    # 'CD68',      # 巨噬细胞标志物
]

# 富集分析参数 - 将从config读取
ADJ_PVAL_CUTOFF = 0.05

# =========================
# Helper Functions
# =========================
def load_data(config):
    """加载所有必需数据"""
    print("[INFO] Loading data...")
    
    # 加载h5ad
    adata = sc.read_h5ad(config.h5ad_path)
    
    # 加载ground truth（如果需要）
    if config.truth_path is not None:
        df_meta = pd.read_csv(config.truth_path)[config.truth_column]
        adata.obs['domain'] = df_meta.values
    else:
        # 使用已有的列
        if config.obs_column in adata.obs.columns:
            adata.obs['domain'] = adata.obs[config.obs_column]
        else:
            raise ValueError(f"Column '{config.obs_column}' not found in adata.obs")
    
    # 加载vocab
    with open(config.vocab_path, 'r') as f:
        vocab = json.load(f)
    # 构建 symbol -> id 和 id -> symbol 映射
    sym2id = {str(k).strip(): int(v) for k, v in vocab.items()}
    id2sym = {int(v): str(k).strip() for k, v in vocab.items()}
    
    # 加载context genes
    with open(config.ctx_genes_pkl_path, "rb") as f:
        ctx_genes = pickle.load(f)
    
    # 加载attention scores
    with open(config.attn_pkl_path, "rb") as f:
        attn_packs = pickle.load(f)
    
    # 展平attention到指定层
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
    print(f"  Loaded {len(ctx_genes)} context gene lists")
    print(f"  Total spots: {adata.n_obs}")
    print(f"  Domains: {adata.obs['domain'].unique()}")
    
    return adata, sym2id, id2sym, ctx_genes, A_list


def filter_genes_by_domain_coverage(adata, ctx_genes: List, id2sym: Dict,
                                    min_domains: int = 2,
                                    min_spots_per_domain: int = 3) -> List[str]:
    """
    筛选在多个domain中都有表达的基因
    
    Args:
        adata: AnnData对象，包含domain信息
        ctx_genes: 每个spot的基因列表
        id2sym: 基因ID到symbol的映射
        min_domains: 基因至少在多少个domain中表达
        min_spots_per_domain: 基因在每个domain中至少表达的spot数量
    
    Returns:
        符合条件的基因列表
    """
    print("\n[INFO] Filtering genes by domain coverage...")
    
    domains = adata.obs['domain'].unique()
    print(f"  Domains: {domains}")
    
    # 统计每个基因在每个domain中的表达情况
    gene_domain_counts = defaultdict(lambda: defaultdict(int))
    
    for spot_idx in range(len(ctx_genes)):
        domain = adata.obs['domain'].iloc[spot_idx]
        genes_in_spot = ctx_genes[spot_idx]
        
        for gene_id in genes_in_spot:
            gene_symbol = id2sym.get(int(gene_id), str(gene_id))
            gene_domain_counts[gene_symbol][domain] += 1
    
    # 筛选基因
    qualified_genes = []
    
    for gene_symbol, domain_counts in gene_domain_counts.items():
        # 统计该基因在多少个domain中满足最小spot数量要求
        domains_with_sufficient_spots = sum(
            1 for domain, count in domain_counts.items()
            if count >= min_spots_per_domain
        )
        
        if domains_with_sufficient_spots >= min_domains:
            qualified_genes.append(gene_symbol)
    
    print(f"\n  Total genes found: {len(gene_domain_counts)}")
    print(f"  Genes expressed in >= {min_domains} domains: {len(qualified_genes)}")
    
    # 显示统计信息
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
    """检查某个spot是否包含指定基因，返回 (是否包含, 基因在spot中的索引)"""
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
    """获取指定基因在指定spot的Top-K partner genes"""
    genes_in_spot = ctx_genes[spot_idx]
    A = A_list[spot_idx]
    
    # 对称化
    A_sym = 0.5 * (A + A.T) if A.ndim == 2 else A
    
    # 提取该基因的注意力行
    if gene_idx >= A_sym.shape[0]:
        return []
    
    attention_row = A_sym[gene_idx, :]
    
    # 确保尺寸对齐：取两者最小长度
    min_len = min(len(genes_in_spot), len(attention_row))
    if min_len == 0:
        return []
    
    # 只使用有效范围的索引
    valid_attention = attention_row[:min_len]
    
    # 排序（排除自身）
    sorted_indices = np.argsort(-valid_attention)
    partners = []
    
    for idx in sorted_indices:
        # 边界检查
        if idx >= min_len:
            continue
        if idx == gene_idx:  # 跳过自身
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
    """在指定domain中随机采样包含目标基因的spots"""
    # 获取该domain的所有spots
    domain_mask = adata.obs['domain'] == domain
    domain_indices = np.where(domain_mask)[0].tolist()
    
    # 筛选包含目标基因且表达量足够的spots
    valid_spots = []
    for spot_idx in domain_indices:
        has_gene, _ = find_gene_in_spot(spot_idx, gene_symbol, ctx_genes, id2sym)
        if has_gene:
            # 可选：检查表达量
            # expr_val = adata.X[spot_idx, adata.var_names == gene_symbol]
            # if expr_val > min_expr:
            valid_spots.append(spot_idx)
    
    # 随机采样
    if len(valid_spots) == 0:
        return []
    
    n_samples = min(n_samples, len(valid_spots))
    sampled = np.random.choice(valid_spots, size=n_samples, replace=False)
    
    return sampled.tolist()


def run_enrichment_for_gene_list(gene_list: List[str], 
                                  config,
                                  cutoff: float = None) -> pd.DataFrame:
    """对基因列表进行富集分析"""
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
    """比较两个domain的富集结果，识别差异性通路"""
    if df1.empty or df2.empty:
        return pd.DataFrame()
    
    # 获取两个domain的显著通路
    terms1 = set(df1[df1['adj_pval'] < ADJ_PVAL_CUTOFF]['term_clean'].unique())
    terms2 = set(df2[df2['adj_pval'] < ADJ_PVAL_CUTOFF]['term_clean'].unique())
    
    # 分类通路
    unique_to_domain1 = terms1 - terms2
    unique_to_domain2 = terms2 - terms1
    shared = terms1 & terms2
    
    comparison = []
    
    # Domain1特异性通路
    for term in unique_to_domain1:
        row = df1[df1['term_clean'] == term].iloc[0]
        comparison.append({
            'term': term,
            'category': f'Unique to {domain1}',
            'domain1_pval': row['adj_pval'],
            'domain2_pval': np.nan,
            'library': row['Library']
        })
    
    # Domain2特异性通路
    for term in unique_to_domain2:
        row = df2[df2['term_clean'] == term].iloc[0]
        comparison.append({
            'term': term,
            'category': f'Unique to {domain2}',
            'domain1_pval': np.nan,
            'domain2_pval': row['adj_pval'],
            'library': row['Library']
        })
    
    # 共享但显著性差异的通路
    for term in shared:
        row1 = df1[df1['term_clean'] == term].iloc[0]
        row2 = df2[df2['term_clean'] == term].iloc[0]
        
        # 计算p值差异
        pval_ratio = row1['adj_pval'] / row2['adj_pval']
        if pval_ratio > 2 or pval_ratio < 0.5:  # 2倍差异
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
    """分析单个基因在不同domain的功能可塑性"""
    print(f"\n{'='*60}")
    print(f"Analyzing gene: {gene_symbol}")
    print(f"{'='*60}")
    
    # 检查基因是否在vocab中
    if gene_symbol not in sym2id:
        print(f"[SKIP] Gene {gene_symbol} not in vocabulary")
        return
    
    domains = adata.obs['domain'].unique()
    print(f"  Domains: {domains}")
    
    results = {}
    
    # 使用config中的参数
    spots_per_domain = config.spots_per_domain_sample
    top_k_partners = config.top_k_partners_plasticity
    
    # 对每个domain采样spots并提取partners
    for domain in domains:
        print(f"\n[Domain: {domain}]")
        
        # 采样spots
        sampled_spots = sample_spots_with_gene(
            adata, gene_symbol, domain, ctx_genes, id2sym, 
            n_samples=spots_per_domain
        )
        
        if len(sampled_spots) == 0:
            print(f"  No spots found containing {gene_symbol}")
            continue
        
        print(f"  Sampled {len(sampled_spots)} spots")
        
        # 收集所有采样spots中该基因的partners
        all_partners = []
        for spot_idx in sampled_spots:
            has_gene, gene_idx = find_gene_in_spot(spot_idx, gene_symbol, ctx_genes, id2sym)
            if has_gene:
                partners = get_top_partners_for_gene(
                    spot_idx, gene_idx, ctx_genes, A_list, id2sym, 
                    top_k=top_k_partners
                )
                all_partners.extend([p[0] for p in partners])
        
        # 去重并统计频率
        partner_counts = pd.Series(all_partners).value_counts()
        top_partners = partner_counts.head(top_k_partners).index.tolist()
        
        print(f"  Found {len(partner_counts)} unique partners")
        print(f"  Top 5 partners: {', '.join(top_partners[:5])}")
        
        # 富集分析
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
    
    # 两两比较domains
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
    
    # 保存结果
    gene_output_dir = os.path.join(config.gene_plasticity_dir, safe_filename(gene_symbol))
    os.makedirs(gene_output_dir, exist_ok=True)
    
    # 保存每个domain的结果
    for domain, data in results.items():
        # 1. 保存富集结果
        domain_file = os.path.join(gene_output_dir, f"{safe_filename(domain)}_enrichment.csv")
        if not data['enrichment'].empty:
            data['enrichment'].to_csv(domain_file, index=False)
            print(f"[SAVE] {domain_file}")
        
        # 2. 保存top partner基因列表
        partners_file = os.path.join(gene_output_dir, f"{safe_filename(domain)}_top_partners.txt")
        with open(partners_file, 'w') as f:
            f.write(f"# Top partner genes for {gene_symbol} in {domain}\n")
            f.write(f"# Total unique partners: {len(data['partners'])}\n")
            f.write(f"# Sampled from {len(data['spots'])} spots\n")
            f.write("#" + "="*50 + "\n")
            for i, partner in enumerate(data['partners'], 1):
                f.write(f"{i}\t{partner}\n")
        print(f"[SAVE] {partners_file}")
        
        # 3. 保存详细的partners统计信息（包含频率）
        partners_stats_file = os.path.join(gene_output_dir, f"{safe_filename(domain)}_partners_stats.csv")
        # 重新计算partner频率（从原始数据）
        all_partners_with_scores = []
        for spot_idx in data['spots']:
            has_gene, gene_idx = find_gene_in_spot(spot_idx, gene_symbol, ctx_genes, id2sym)
            if has_gene:
                partners = get_top_partners_for_gene(
                    spot_idx, gene_idx, ctx_genes, A_list, id2sym, 
                    top_k=config.top_k_partners_plasticity
                )
                all_partners_with_scores.extend(partners)
        
        # 统计每个partner的出现频率和平均attention score
        partner_stats = defaultdict(lambda: {'count': 0, 'total_score': 0.0, 'scores': []})
        for partner, score in all_partners_with_scores:
            partner_stats[partner]['count'] += 1
            partner_stats[partner]['total_score'] += score
            partner_stats[partner]['scores'].append(score)
        
        # 构建统计DataFrame
        stats_rows = []
        for partner in data['partners']:  # 按照已排序的top partners顺序
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
    
    # 保存比较结果
    if comparisons:
        comparison_file = os.path.join(gene_output_dir, "domain_comparisons.csv")
        pd.concat(comparisons, ignore_index=True).to_csv(comparison_file, index=False)
        print(f"[SAVE] {comparison_file}")
    
    # 保存汇总信息
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
    """转换为安全文件名"""
    return re.sub(r'[^\w\-]', '_', str(s))


def main(dataset_name: str):
    """主函数"""
    # 加载配置
    config = get_config(dataset_name)
    
    # 创建输出目录
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
    
    # 加载数据
    adata, sym2id, id2sym, ctx_genes, A_list = load_data(config)
    
    # 第一步：筛选在多个domain中都有表达的基因
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
        
        # 选择Top基因进行分析（按总表达spots数排序）
        gene_total_spots = {}
        for gene in qualified_genes:
            total_spots = 0
            for spot_idx in range(len(ctx_genes)):
                has_gene, _ = find_gene_in_spot(spot_idx, gene, ctx_genes, id2sym)
                if has_gene:
                    total_spots += 1
            gene_total_spots[gene] = total_spots
        
        # 选择Top 20个基因（可调整）
        top_n = min(20, len(qualified_genes))
        query_genes = sorted(gene_total_spots.items(), key=lambda x: x[1], reverse=True)[:top_n]
        query_genes = [g[0] for g in query_genes]
        
        print(f"\n[INFO] Selected top {len(query_genes)} genes for analysis:")
        for gene in query_genes:
            print(f"  - {gene} ({gene_total_spots[gene]} spots)")
    
    # 保存筛选的基因列表
    genes_file = os.path.join(config.gene_plasticity_dir, "analyzed_genes.txt")
    with open(genes_file, 'w') as f:
        f.write('\n'.join(query_genes))
    print(f"\n[SAVE] Gene list saved to: {genes_file}")
    
    # 第二步：分析每个查询基因
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
    
    # 生成汇总报告
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
    
    np.random.seed(42)  # 可重复性
    main(args.dataset)