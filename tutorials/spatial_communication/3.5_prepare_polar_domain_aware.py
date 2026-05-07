# -*- coding: utf-8 -*-
"""
Domain感知的TopK选择策略
- 在选择TopK基因时考虑domain信息
- 同domain内的邻居倾向于选择相似的基因
- 提高domain内基因重复率，增强过滤效果
"""

import os
import gc
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import anndata as ad

# ---------------- Domain信息加载 ----------------

def load_domains(adata_path: str) -> pd.Series:
    """加载domain信息"""
    adata = ad.read_h5ad(adata_path)
    if "ground_truth" not in adata.obs:
        raise ValueError("adata.obs 中未找到 ground_truth 列")
    s = adata.obs["ground_truth"].astype(str).copy()
    s.index = s.index.astype(str)
    return s

def get_neighbor_domains(df: pd.DataFrame, spot_domains: pd.Series) -> dict:
    """获取每个邻居的domain"""
    neighbor_domains = {}
    for neighbor in df['neighbor_name'].unique():
        # 尝试将neighbor转为索引
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

# ---------------- Domain感知的TopK选择 ----------------

def topk_domain_aware(df: pd.DataFrame, 
                      neighbor_domains: dict,
                      k: int = 10,
                      domain_weight: float = 0.6,
                      metric: str = "mean") -> pd.DataFrame:
    """
    Domain感知的TopK选择
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含 center_name, neighbor_name, gene, mean/sum
    neighbor_domains : dict
        邻居到domain的映射
    k : int
        每个邻居选择的基因数
    domain_weight : float (0-1)
        domain级别排名的权重
        - 1.0: 完全按domain统一排名（所有同domain邻居选相同基因）
        - 0.0: 完全按邻居独立排名（忽略domain）
        - 0.5-0.7: 推荐值，平衡domain一致性和邻居特异性
    metric : str
        排名指标（mean或sum）
    
    Returns:
    --------
    pd.DataFrame
        Domain感知选择后的TopK数据
    """
    print(f"\n=== Domain感知TopK选择 (k={k}, domain_weight={domain_weight}) ===")
    
    # 1. 计算domain级别的基因排名
    print("\n1️⃣ 计算domain级别基因排名...")
    domain_gene_scores = {}
    unique_domains = set(neighbor_domains.values())
    
    for domain in unique_domains:
        # 该domain的所有邻居
        domain_neighbors = [n for n, d in neighbor_domains.items() if d == domain]
        domain_df = df[df['neighbor_name'].isin(domain_neighbors)]
        
        if len(domain_df) == 0:
            continue
        
        # 计算该domain内每个基因的平均attention
        gene_scores = domain_df.groupby('gene')[metric].mean().sort_values(ascending=False)
        domain_gene_scores[domain] = gene_scores
        
        print(f"  {domain}: {len(domain_neighbors)} 个邻居, {len(gene_scores)} 个基因")
        print(f"    Top5基因: {list(gene_scores.head(5).index)}")
    
    # 2. 为每个邻居选择TopK
    print(f"\n2️⃣ 为每个邻居选择Top{k}基因...")
    result_dfs = []
    
    for neighbor in df['neighbor_name'].unique():
        neighbor_df = df[df['neighbor_name'] == neighbor].copy()
        domain = neighbor_domains.get(neighbor, "Unknown")
        
        if domain not in domain_gene_scores:
            # 如果domain未知，回退到独立排名
            topk_df = neighbor_df.nlargest(k, metric)
            result_dfs.append(topk_df)
            continue
        
        # 计算混合得分
        # 邻居本地排名（越小越好）
        neighbor_df['local_rank'] = neighbor_df[metric].rank(method='first', ascending=False)
        
        # domain级别排名（越小越好）
        domain_scores = domain_gene_scores[domain]
        neighbor_df['domain_rank'] = neighbor_df['gene'].map(
            lambda g: domain_scores.index.get_loc(g) + 1 if g in domain_scores.index else len(domain_scores) + 1
        )
        
        # 混合得分（越小越好）
        neighbor_df['mixed_rank'] = (
            domain_weight * neighbor_df['domain_rank'] + 
            (1 - domain_weight) * neighbor_df['local_rank']
        )
        
        # 选择TopK
        topk_df = neighbor_df.nsmallest(k, 'mixed_rank')
        topk_df = topk_df.drop(columns=['local_rank', 'domain_rank', 'mixed_rank'])
        
        result_dfs.append(topk_df)
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # 3. 统计domain内基因重复率
    print(f"\n3️⃣ Domain内基因重复率统计:")
    for domain in unique_domains:
        domain_neighbors = [n for n, d in neighbor_domains.items() if d == domain]
        if len(domain_neighbors) < 2:
            continue
        
        domain_result = result[result['neighbor_name'].isin(domain_neighbors)]
        
        # 统计每个基因在多少个邻居中出现
        gene_freq = domain_result.groupby('gene')['neighbor_name'].nunique()
        
        # 重复率：出现在2个以上邻居的基因比例
        repeated_genes = gene_freq[gene_freq >= 2]
        repeat_rate = len(repeated_genes) / len(gene_freq) if len(gene_freq) > 0 else 0
        
        print(f"  {domain}:")
        print(f"    邻居数: {len(domain_neighbors)}")
        print(f"    独特基因数: {len(gene_freq)}")
        print(f"    重复基因数: {len(repeated_genes)} ({repeat_rate*100:.1f}%)")
        print(f"    高频基因(>=3邻居): {list(gene_freq[gene_freq >= 3].head(5).index)}")
    
    print(f"\n=== 完成 ===\n")
    return result

# ---------------- 主函数 ----------------

def prepare_domain_aware_topk(
    out_dir: str,
    layer: int = 5,
    value_col: str = "attn_score",
    topk: int = 10,
    domain_weight: float = 0.6,
    gene_view: str = "kv"
):
    """
    生成domain感知的TopK CSV文件
    
    Parameters:
    -----------
    out_dir : str
        输出目录
    layer : int
        使用的层号
    value_col : str
        attention列名
    topk : int
        每个邻居选择的基因数
    domain_weight : float (0-1)
        domain权重
        - 推荐: 0.5-0.7
        - 更高: 更强的domain一致性
        - 更低: 更多的邻居特异性
    gene_view : str
        kv 或 q
    """
    print("="*80)
    print(f"Domain感知TopK生成")
    print(f"  输出目录: {out_dir}")
    print(f"  Layer: {layer}")
    print(f"  TopK: {topk}")
    print(f"  Domain权重: {domain_weight}")
    print("="*80)
    
    # 1. 加载domain信息
    adata_path = os.path.join(out_dir, "adata_with_metadata.h5ad")
    spot_domains = load_domains(adata_path)
    print(f"\n✅ 加载了 {len(spot_domains)} 个spots的domain信息")
    print(f"   Domains: {spot_domains.unique()}")
    
    # 2. 读取全量spot-level数据
    agg_dir = os.path.join(out_dir, "agg_csv")
    full_csv = os.path.join(agg_dir, f"spot_level_{gene_view}.csv")
    
    if not os.path.exists(full_csv):
        raise FileNotFoundError(f"未找到全量数据: {full_csv}")
    
    print(f"\n✅ 读取全量数据: {full_csv}")
    df = pd.read_csv(full_csv)
    print(f"   记录数: {len(df)}")
    print(f"   中心spots: {df['center_name'].nunique()}")
    
    # 3. 为每个center分别处理
    all_results = []
    centers = df['center_name'].unique()
    
    print(f"\n开始处理 {len(centers)} 个中心spots...")
    
    for i, center in enumerate(centers):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  处理进度: {i+1}/{len(centers)}")
        
        center_df = df[df['center_name'] == center].copy()
        
        # 获取该center的邻居domains
        neighbor_domains = get_neighbor_domains(center_df, spot_domains)
        
        # Domain感知TopK选择
        topk_df = topk_domain_aware(
            center_df,
            neighbor_domains,
            k=topk,
            domain_weight=domain_weight,
            metric="mean"
        )
        
        all_results.append(topk_df)
    
    # 4. 合并并保存
    result = pd.concat(all_results, ignore_index=True)
    
    output_path = os.path.join(agg_dir, f"spot_level_{gene_view}_top{topk}_domain_aware_w{int(domain_weight*10)}.csv")
    result.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ 已保存Domain感知TopK文件:")
    print(f"   {output_path}")
    print(f"   记录数: {len(result)}")
    print(f"={'='*80}\n")
    
    return output_path

# ---------------- 使用示例 ----------------

if __name__ == "__main__":
    out_dir = "./PDAC/whole_slice_data_20251028_173836"
    
    # 测试不同的domain权重
    for domain_weight in [0.5, 0.6, 0.7]:
        print(f"\n{'#'*80}")
        print(f"# 测试 domain_weight = {domain_weight}")
        print(f"{'#'*80}\n")
        
        prepare_domain_aware_topk(
            out_dir=out_dir,
            layer=5,
            topk=5,
            domain_weight=domain_weight,
            gene_view="kv"
        )

