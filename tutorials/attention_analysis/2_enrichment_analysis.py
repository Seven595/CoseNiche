#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrichment Analysis and Publication-Quality Dot Plot
Generate Nature-style enrichment analysis figures from domain gene tables
"""

import os
import glob
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import gseapy as gp

# =========================
# Configuration
# =========================
# Input directory with domain tables
# DOMAIN_TABLES_DIR = "/home/junning/projectnvme/ST/project-20-contrast-organ/Analysis/self_attention_analysis/HBRC1/result_output/enrichment_prepared/domain_tables"
# Output directory
# OUTPUT_DIR = "./HBRC1/result_output/enrichment_plots"


DOMAIN_TABLES_DIR = "/home/junning/projectnvme/ST/project-20-contrast-organ/Analysis/self_attention_analysis/PDAC/result_output/enrichment_prepared/domain_tables"
OUTPUT_DIR = "./PDAC/result_output/enrichment_plots"



# DOMAIN_TABLES_DIR = "/home/junning/projectnvme/ST/project-20-contrast-organ/Analysis/self_attention_analysis/HBRC/result_output/enrichment_prepared/domain_tables"
# OUTPUT_DIR = "./HBRC/result_output/enrichment_plots"


# Number of top genes per domain to use for enrichment
TOP_GENES = 100

# Enrichment parameters
ENRICH_LIBRARIES = [
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
]
ORGANISM = "Human"
PVALUE_CUTOFF = 0.05
ADJ_PVALUE_CUTOFF = 0.05

# Plotting parameters
TOP_TERMS_PER_DOMAIN = 10  # Top enriched terms to show per domain
MAX_TERMS_TOTAL = 30  # Maximum total terms to display
MIN_GENE_RATIO = 0.01  # Minimum gene ratio to display

# Nature journal color schemes
NATURE_COLORS = {
    'blue': '#4472C4',
    'orange': '#ED7D31',
    'green': '#70AD47',
    'yellow': '#FFC000',
    'light_blue': '#5B9BD5',
    'purple': '#7030A0',
    'red': '#C00000',
    'teal': '#00B0F0',
}

# Library-specific colors (for distinguishing different enrichment libraries)
LIBRARY_COLORS = {
    'GO_Biological_Process_2023': '#fbad91',  # Orange
    'GO_Cellular_Component_2023': '#CAC8E1',  # Green
    'GO_Molecular_Function_2023': '#b9e3b3',  # Yellow
    'KEGG_2021_Human': '#AFD436',  # Light blue
    'Reactome_2022': '#71A24F',  # Purple
    'WikiPathways_2019_Human': '#C00000',  # Red
}

# =========================
# Data Loading
# =========================
def load_domain_genes(domain_tables_dir: str, top_n: int = 100) -> Dict[str, List[str]]:
    """
    Load top N genes from each domain table.
    
    Returns:
        Dict mapping domain_name -> list of gene symbols
    """
    csv_files = sorted(glob.glob(os.path.join(domain_tables_dir, "*_partners_full.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {domain_tables_dir}")
    
    domain_genes = {}
    for csv_path in csv_files:
        # Extract domain name from filename
        basename = os.path.basename(csv_path)
        domain_name = basename.replace("_partners_full.csv", "")
        
        # Read domain data
        df = pd.read_csv(csv_path)
        
        # Check required columns
        if "partner_symbol" not in df.columns:
            print(f"[WARN] No partner_symbol column in {basename}, skipping")
            continue
        
        # Get top N genes (already sorted by avg_strength in the input files)
        top_genes = df["partner_symbol"].head(top_n).dropna().unique().tolist()
        
        if len(top_genes) > 0:
            domain_genes[domain_name] = top_genes
            print(f"[INFO] Loaded {len(top_genes)} genes from {domain_name}")
    
    return domain_genes

# =========================
# Check Existing Results
# =========================
def check_existing_enrichment_results(
    domain_genes: Dict[str, List[str]],
    outdir: str
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Check for existing enrichment results.
    
    Returns:
        Tuple of (existing_results_dict, missing_domains_list)
    """
    existing_results = {}
    missing_domains = []
    
    for domain in domain_genes.keys():
        csv_path = os.path.join(outdir, f"{domain}_enrichment.csv")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                # Check if file has required columns
                required_cols = ['term', 'adj_pval', 'gene_ratio', 'gene_count']
                if all(col in df.columns for col in required_cols):
                    # Calculate neg_log10_pval if not present
                    if 'neg_log10_pval' not in df.columns:
                        df['neg_log10_pval'] = -np.log10(df['adj_pval'].clip(lower=1e-300))
                    
                    # Ensure term_clean exists
                    if 'term_clean' not in df.columns:
                        if 'term' in df.columns:
                            df['term_clean'] = df['term'].astype(str).str.replace(
                                r'\s*\([A-Z]{2}:\d+\)$|\s*\(WP\d+\)$', '', regex=True
                            )
                        else:
                            df['term_clean'] = df['term']
                    
                    existing_results[domain] = df
                    print(f"  ✓ Found existing: {domain} ({len(df)} terms)")
                else:
                    print(f"  ✗ Invalid format: {domain} (missing columns)")
                    missing_domains.append(domain)
            except Exception as e:
                print(f"  ✗ Error loading: {domain} ({e})")
                missing_domains.append(domain)
        else:
            missing_domains.append(domain)
    
    return existing_results, missing_domains


# =========================
# Enrichment Analysis
# =========================
def check_enrichment_dir_status(outdir: str) -> Tuple[bool, List[str]]:
    """
    Check if enrichment directory exists and has valid CSV files.
    
    Returns:
        (has_valid_results, list_of_csv_files)
    """
    if not os.path.exists(outdir):
        return False, []
    
    csv_files = glob.glob(os.path.join(outdir, "*_enrichment.csv"))
    
    if not csv_files:
        return False, []
    
    # Check if at least one file has valid content
    valid_files = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            required_cols = ['term', 'adj_pval', 'gene_ratio', 'gene_count']
            if all(col in df.columns for col in required_cols) and len(df) > 0:
                valid_files.append(csv_path)
        except Exception:
            continue
    
    return len(valid_files) > 0, valid_files


def run_enrichment_analysis(
    domain_genes: Dict[str, List[str]],
    libraries: List[str],
    organism: str = "Human",
    outdir: str = "./enrichment_results",
    cutoff: float = 0.05,
    force_rerun: bool = False
    ) -> Dict[str, pd.DataFrame]:
    """
    Run enrichment analysis for each domain using multiple libraries.
    
    Args:
        domain_genes: Dictionary of domain names to gene lists
        libraries: List of enrichment libraries to use
        organism: Organism name
        outdir: Output directory for results
        cutoff: P-value cutoff
        force_rerun: If True, rerun even if results exist
    
    Returns:
        Dict mapping domain_name -> combined enrichment results DataFrame
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Check if enrichment directory has valid results
    has_results, csv_files = check_enrichment_dir_status(outdir)
    
    if has_results and not force_rerun:
        print("\n[INFO] Found existing enrichment results directory with valid data!")
        print(f"  Directory: {outdir}")
        print(f"  Files found: {len(csv_files)}")
        print("  Skipping enrichment analysis, loading existing results...")
        print("  (Use --force-rerun to regenerate all results)")
        
        # Load existing results
        enrichment_results = {}
        for csv_path in csv_files:
            domain = os.path.basename(csv_path).replace("_enrichment.csv", "")
            try:
                df = pd.read_csv(csv_path)
                # Add missing columns if needed
                if 'neg_log10_pval' not in df.columns and 'adj_pval' in df.columns:
                    df['neg_log10_pval'] = -np.log10(df['adj_pval'].clip(lower=1e-300))
                if 'term_clean' not in df.columns and 'term' in df.columns:
                    df['term_clean'] = df['term'].astype(str).str.replace(
                        r'\s*\([A-Z]{2}:\d+\)$|\s*\(WP\d+\)$', '', regex=True
                    )
                enrichment_results[domain] = df
                print(f"  ✓ Loaded {domain}: {len(df)} terms")
            except Exception as e:
                print(f"  ✗ Error loading {domain}: {e}")
        
        return enrichment_results
    
    # Check for existing results (old behavior for partial completion)
    if not force_rerun:
        print("\n[INFO] Checking for existing enrichment results...")
        existing_results, missing_domains = check_existing_enrichment_results(
            domain_genes, outdir
        )
        
        if existing_results and not missing_domains:
            print(f"\n✓ All {len(existing_results)} domains have existing results!")
            print("  Skipping enrichment analysis, will use existing data for plotting.")
            print("  (Use --force-rerun to regenerate all results)")
            return existing_results
        elif existing_results:
            print(f"\n[INFO] Found {len(existing_results)} existing results")
            print(f"[INFO] Will analyze {len(missing_domains)} missing domains:")
            for d in missing_domains:
                print(f"  - {d}")
            # Filter domain_genes to only missing domains
            domain_genes = {k: v for k, v in domain_genes.items() if k in missing_domains}
        else:
            print("[INFO] No existing results found, will run full analysis")
    else:
        print("\n[INFO] Force rerun enabled, will regenerate all results")
        existing_results = {}
    
    enrichment_results = existing_results.copy()
    
    for domain, genes in domain_genes.items():
        print(f"\n[INFO] Running enrichment for {domain} ({len(genes)} genes)")
        
        combined_results = []
        
        for library in libraries:
            print(f"  - Library: {library}")
            try:
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=library,
                    organism=organism,
                    outdir=None,  # Don't save intermediate files
                    cutoff=cutoff,
                    no_plot=True,
                )
                
                df = enr.results
                if df is not None and not df.empty:
                    # Add library information
                    df["Library"] = library
                    df["Domain"] = domain
                    
                    # Standardize column names
                    rename_map = {
                        "Term": "term",
                        "Adjusted P-value": "adj_pval",
                        "P-value": "pval",
                        "Combined Score": "combined_score",
                        "Odds Ratio": "odds_ratio",
                        "Overlap": "overlap",
                        "Genes": "genes"
                    }
                    df = df.rename(columns=rename_map)
                    
                    # Clean term names (remove GO:XXX, WP:XXX prefixes)
                    df["term_clean"] = df["term"].astype(str).str.replace(
                        r'\s*\([A-Z]{2}:\d+\)$|\s*\(WP\d+\)$', '', regex=True
                    )
                    
                    combined_results.append(df)
                    print(f"    Found {len(df)} significant terms (p < {cutoff})")
                
            except Exception as e:
                print(f"    [ERROR] Failed: {e}")
                continue
        
        if combined_results:
            # Combine all library results for this domain
            domain_df = pd.concat(combined_results, ignore_index=True)
            
            # Calculate gene ratio
            domain_df["gene_count"] = domain_df["overlap"].apply(
                lambda x: int(x.split("/")[0]) if isinstance(x, str) and "/" in x else 0
            )
            domain_df["gene_set_size"] = domain_df["overlap"].apply(
                lambda x: int(x.split("/")[1]) if isinstance(x, str) and "/" in x else 1
            )
            domain_df["gene_ratio"] = domain_df["gene_count"] / len(genes)
            
            # Sort by adjusted p-value and combined score
            domain_df = domain_df.sort_values(
                ["adj_pval", "combined_score"], 
                ascending=[True, False]
            ).reset_index(drop=True)
            
            enrichment_results[domain] = domain_df
            
            # Save individual domain results
            out_csv = os.path.join(outdir, f"{domain}_enrichment.csv")
            domain_df.to_csv(out_csv, index=False)
            print(f"[SAVE] {out_csv}")
        else:
            print(f"[WARN] No enrichment results for {domain}")
    
    return enrichment_results

# =========================
# Data Preparation for Plotting
# =========================
def prepare_plot_data(
    enrichment_results: Dict[str, pd.DataFrame],
    top_terms_per_domain: int = 10,
    max_terms_total: int = 30,
    adj_pval_cutoff: float = 0.05,
    min_gene_ratio: float = 0.01
    ) -> pd.DataFrame:
    """
    Prepare data for dot plot by selecting top terms from each domain.
    """
    plot_data_list = []
    
    for domain, df in enrichment_results.items():
        # Filter by adjusted p-value
        df_sig = df[df["adj_pval"] <= adj_pval_cutoff].copy()
        
        if df_sig.empty:
            print(f"[INFO] No significant terms for {domain}")
            continue
        
        # Filter by minimum gene ratio
        df_sig = df_sig[df_sig["gene_ratio"] >= min_gene_ratio]
        
        # Select top terms
        df_top = df_sig.head(top_terms_per_domain).copy()
        plot_data_list.append(df_top)
    
    if not plot_data_list:
        raise ValueError("No significant enrichment results to plot")
    
    # Combine all domains
    plot_df = pd.concat(plot_data_list, ignore_index=True)
    
    # Remove duplicate terms (keep the one with lowest p-value)
    plot_df = plot_df.sort_values("adj_pval").drop_duplicates(
        subset=["term_clean"], keep="first"
    )
    
    # Limit total number of terms
    plot_df = plot_df.head(max_terms_total)
    
    # Calculate -log10(adjusted p-value) for color scale
    plot_df["neg_log10_pval"] = -np.log10(plot_df["adj_pval"].clip(lower=1e-300))
    
    return plot_df

# =========================
# Publication-Quality Dot Plot
# =========================
def create_enrichment_dotplot(
    plot_df: pd.DataFrame,
    output_path: str,
    figsize: Tuple[float, float] = (10, 12),
    dpi: int = 300,
    title: str = "Enrichment Analysis"
):
    """
    Create a publication-quality enrichment dot plot similar to Nature journals.
    """
    # Sort terms by combined score or adjusted p-value
    plot_df = plot_df.sort_values("combined_score", ascending=True).reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Define color map (white -> orange -> red for p-values)
    colors_pval = ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', 
                   '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
    cmap = LinearSegmentedColormap.from_list('custom_reds', colors_pval)
    
    # Normalize color scale
    norm = Normalize(
        vmin=plot_df["neg_log10_pval"].min(),
        vmax=plot_df["neg_log10_pval"].max()
    )
    
    # Create scatter plot
    scatter = ax.scatter(
        plot_df["gene_ratio"],
        plot_df.index,
        c=plot_df["neg_log10_pval"],
        s=plot_df["gene_count"] * 20,  # Size by number of genes
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        edgecolors='black',
        linewidths=0.5,
        zorder=3
    )
    
    # Set y-axis labels (term names)
    ax.set_yticks(plot_df.index)
    ax.set_yticklabels(plot_df["term_clean"], fontsize=9)
    
    # Set x-axis
    ax.set_xlabel("Gene Ratio", fontsize=12, fontweight='bold')
    ax.set_xlim(left=0, right=plot_df["gene_ratio"].max() * 1.1)
    ax.tick_params(axis='x', labelsize=10)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar for p-values
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('-log$_{10}$(Adjusted P-value)', 
                   fontsize=11, fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.5)
    
    # Add size legend for gene count
    gene_counts = [5, 10, 20]
    legend_elements = []
    for gc in gene_counts:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='gray', markersize=np.sqrt(gc * 20 / np.pi),
                      label=f'{gc}', markeredgecolor='black', markeredgewidth=0.5)
        )
    
    legend = ax.legend(
        handles=legend_elements,
        title='Gene Count',
        loc='lower right',
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9,
        edgecolor='black',
        fancybox=False
    )
    legend.get_frame().set_linewidth(1.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"[SAVE] {output_path}")
    
    plt.close()


# =========================
# 单个Domain的条形图（按Library颜色区分）
# =========================
def create_domain_enrichment_barplot(
    domain: str,
    df: pd.DataFrame,
    output_path: str,
    top_terms_per_library: int = 10,
    adj_pval_cutoff: float = 0.05,
    figsize: Tuple[float, float] = (8, 14),
    dpi: int = 300
):
    """
    Create enrichment bar plot for a single domain with library-specific colors.
    - Each library's results are grouped together
    - Within each library, bars are sorted by -log10(adj_pval) in descending order
    - X-axis uses -log10(Adjusted P-value)
    
    Args:
        domain: Domain name
        df: Enrichment results DataFrame with 'Library' column
        output_path: Path to save figure
        top_terms_per_library: Number of top terms to show per library
        adj_pval_cutoff: Adjusted p-value cutoff
        figsize: Figure size
        dpi: Resolution
    """
    # 按p值筛选
    df_sig = df[df["adj_pval"] <= adj_pval_cutoff].copy()
    
    if df_sig.empty:
        print(f"[WARN] No significant terms for {domain}")
        return
    
    # 计算-log10 p-value（如果不存在）
    if 'neg_log10_pval' not in df_sig.columns:
        df_sig['neg_log10_pval'] = -np.log10(df_sig['adj_pval'].clip(lower=1e-300))
    
    # 获取唯一的libraries，按ENRICH_LIBRARIES顺序排序
    available_libraries = df_sig['Library'].unique()
    # 按ENRICH_LIBRARIES顺序排列libraries
    library_order = [lib for lib in ENRICH_LIBRARIES if lib in available_libraries]
    # 将不在ENRICH_LIBRARIES中的library添加到末尾
    library_order.extend([lib for lib in available_libraries if lib not in library_order])
    
    # 准备绘图数据：按library分组，每个library内部按neg_log10_pval排序
    plot_rows = []
    y_positions = []
    y_labels = []
    y_ticks = []
    current_y = 0
    
    for library in library_order:
        library_data = df_sig[df_sig['Library'] == library].copy()
        
        # 按neg_log10_pval降序排序（最显著的在前）
        library_data = library_data.sort_values('neg_log10_pval', ascending=False)
        
        # 取该library的前N个terms
        library_data = library_data.head(top_terms_per_library)
        
        if len(library_data) == 0:
            continue
        
        # 添加该library的所有entries
        for idx, row in library_data.iterrows():
            plot_rows.append({
                'y_pos': current_y,
                'term_clean': row['term_clean'],
                'library': row['Library'],
                'neg_log10_pval': row['neg_log10_pval'],
                'adj_pval': row['adj_pval'],
                'gene_ratio': row['gene_ratio']
            })
            
            # 所有terms都显示标签
            y_labels.append(row['term_clean'])
            y_ticks.append(current_y)
            current_y += 1
    
    if not plot_rows:
        print(f"[WARN] No data to plot for {domain}")
        return
    
    plot_df = pd.DataFrame(plot_rows)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 根据library获取颜色
    bar_colors = [LIBRARY_COLORS.get(lib, '#808080') for lib in plot_df['library']]
    
    # 创建水平条形图（完全无间隙）
    # 关键：y_pos是连续整数(0,1,2,...)，设置height=1.0确保bars完全填充，无任何间隙
    # 注意：bar_height参数保留但不使用，因为要确保无间隙必须使用height=1.0
    bar_height = 0.6  # 此参数保留用于文档说明，但实际使用1.0确保无间隙
    
    # 逐个绘制bar，使用height=1.0确保完全无间隙
    for idx, row in plot_df.iterrows():
        ax.barh(
            row['y_pos'],
            row['neg_log10_pval'],
            height=1.0,  # 固定为1.0以确保bars完全填充y轴单位空间，无任何间隙
            left=0,
            color=bar_colors[idx],
            edgecolor='white',
            linewidth=0.5,
            alpha=0.85,
            align='center'
        )
    
    # 设置Y轴标签
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    
    # 设置X轴
    ax.set_xlabel('-log$_{10}$(Adjusted P-value)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlim(left=0, right=plot_df['neg_log10_pval'].max() * 1.1)
 
    # 设置标题
    ax.set_title(f"Enrichment Analysis - {domain}", fontsize=14, fontweight='bold', pad=20)
    
    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 创建libraries图例
    unique_libraries = plot_df['library'].unique()
    legend_elements = []
    for lib in library_order:
        if lib in unique_libraries:
            color = LIBRARY_COLORS.get(lib, '#808080')
            # 简化library名称用于图例
            lib_name = lib.replace('_2023', '').replace('_2021', '').replace('_2022', '').replace('_Human', '').replace('_', ' ')
            legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=lib_name, linewidth=0.5))
    
    # 智能选择legend位置：检查右下角是否有足够空间
    # 计算右侧bars的最大值，如果太短则将legend放在右上角或外部
    max_pval = plot_df['neg_log10_pval'].max()
    # 如果bars普遍较短（最大值小于阈值），将legend放在外部或上方
    if max_pval < 5:  # 如果最大-log10(p-value) < 5，说明bars较短
        legend_loc = 'upper right'
        bbox_to_anchor = None
    else:
        # bars较长时，放在右下角内部
        legend_loc = 'lower right'
        bbox_to_anchor = None
    
    ax.legend(handles=legend_elements, loc=legend_loc, bbox_to_anchor=bbox_to_anchor,
             fontsize=9, frameon=True, framealpha=0.95, edgecolor='black', 
             title='Enrichment Library', title_fontsize=10)
    
    # 设置Y轴范围以适配所有bars
    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    ax.invert_yaxis()
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    plt.close()


# =========================
# 单个Domain的点图
# =========================
# def create_domain_enrichment_dotplot(
#     domain: str,
#     df: pd.DataFrame,
#     output_path: str,
#     top_terms_per_library: int = 10,
#     adj_pval_cutoff: float = 0.05,
#     figsize: Tuple[float, float] = (10, 12),
#     dpi: int = 300
#     ):
#     """
#     Create enrichment dot plot for a single domain.
#     - Each library's results are grouped together
#     - Different libraries have different background colors
#     - Within each library, terms are sorted by -log10(adj_pval) descending
    
#     Args:
#         domain: Domain name
#         df: Enrichment results DataFrame
#         output_path: Path to save figure
#         top_terms_per_library: Number of top terms to show per library
#         adj_pval_cutoff: Adjusted p-value cutoff
#         figsize: Figure size
#         dpi: Resolution
#     """
#     # 按p值筛选
#     df_sig = df[df["adj_pval"] <= adj_pval_cutoff].copy()
    
#     if df_sig.empty:
#         print(f"[WARN] No significant terms for {domain}")
#         return
    
#     # 计算-log10 p-value（如果不存在）
#     if 'neg_log10_pval' not in df_sig.columns:
#         df_sig['neg_log10_pval'] = -np.log10(df_sig['adj_pval'].clip(lower=1e-300))
    
#     # 获取唯一的libraries，按ENRICH_LIBRARIES顺序排序
#     available_libraries = df_sig['Library'].unique()
#     library_order = [lib for lib in ENRICH_LIBRARIES if lib in available_libraries]
#     library_order.extend([lib for lib in available_libraries if lib not in library_order])
    
#     # 准备绘图数据：按library分组，每个library内部排序
#     plot_rows = []
#     y_pos = 0
#     library_ranges = []  # 存储(start, end, library)用于背景颜色
    
#     for library in library_order:
#         library_data = df_sig[df_sig['Library'] == library].copy()
        
#         # 按neg_log10_pval降序排序（最显著的在前）
#         library_data = library_data.sort_values('neg_log10_pval', ascending=False)
        
#         # 取该library的前N个terms
#         library_data = library_data.head(top_terms_per_library)
        
#         if len(library_data) == 0:
#             continue
        
#         start_pos = y_pos
        
#         # 添加该library的所有entries
#         for idx, row in library_data.iterrows():
#             plot_rows.append({
#                 'y_pos': y_pos,
#                 'term_clean': row['term_clean'],
#                 'library': row['Library'],
#                 'neg_log10_pval': row['neg_log10_pval'],
#                 'gene_ratio': row['gene_ratio'],
#                 'gene_count': row['gene_count']
#             })
#             y_pos += 1
        
#         # 存储该library的范围
#         library_ranges.append((start_pos, y_pos - 1, library))
    
#     if not plot_rows:
#         print(f"[WARN] No data to plot for {domain}")
#         return
    
#     df_plot = pd.DataFrame(plot_rows)
#     df_plot = df_plot.reset_index(drop=True)
    
#     # 创建图形
#     fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
#     # 为每个library组添加背景颜色
#     library_bg_colors = {
#         'GO_Biological_Process_2023': '#D9D9D9',
#         'GO_Cellular_Component_2023': '#FFFFFF',
#         'GO_Molecular_Function_2023': '#D9D9D9',
#         'KEGG_2021_Human': '#FFFFFF',
#         'Reactome_2022': '#D9D9D9',
#         'WikiPathways_2019_Human': '#D9D9D9',
#     }
    
#     for start, end, library in library_ranges:
#         bg_color = library_bg_colors.get(library, '#F8F8F8')
#         ax.axhspan(start - 0.5, end + 0.5, facecolor=bg_color, alpha=0.3, zorder=0)
    
#     # 定义颜色映射（白色 -> 橙色 -> 红色用于p值）
#     colors_pval = ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', 
#                    '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
   
#     colors_deep_purple = ['#F4F0F7', '#E8DDF0', '#D4BFE1', '#BF9DD1',
#                       '#AA7BC0', '#9459B0', '#7E3BA0', '#682D8F', '#52237E']
#     cmap = LinearSegmentedColormap.from_list('deep_purple', colors_deep_purple)
    
#     # 标准化颜色范围
#     norm = Normalize(
#         vmin=df_plot["neg_log10_pval"].min(),
#         vmax=df_plot["neg_log10_pval"].max()
#     )
    
#     # 创建散点图
#     scatter = ax.scatter(
#         df_plot["gene_ratio"],
#         df_plot["y_pos"],
#         c=df_plot["neg_log10_pval"],
#         s=df_plot["gene_count"] * 20,
#         cmap=cmap,
#         norm=norm,
#         alpha=0.85,
#         edgecolors='black',
#         linewidths=0.5,
#         zorder=3
#     )
    
#     # 设置Y轴标签
#     ax.set_yticks(df_plot["y_pos"])
#     ax.set_yticklabels(df_plot["term_clean"], fontsize=9)
    
#     # 设置X轴
#     ax.set_xlabel("Gene Ratio", fontsize=12, fontweight='bold')
#     ax.set_xlim(left=0, right=df_plot["gene_ratio"].max() * 1.1)
#     ax.tick_params(axis='x', labelsize=10)
    
#     # 添加网格
#     ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
#     ax.set_axisbelow(True)
    
#     # 移除边框
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_linewidth(1.5)
#     ax.spines['bottom'].set_linewidth(1.5)
    
#     # 添加标题
#     ax.set_title(f"Enrichment Analysis - {domain}", fontsize=14, fontweight='bold', pad=20)
    
#     # 添加颜色条
#     cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
#     cbar.set_label('-log$_{10}$(Adjusted P-value)', 
#                    fontsize=11, fontweight='bold', rotation=270, labelpad=20)
#     cbar.ax.tick_params(labelsize=9)
#     cbar.outline.set_linewidth(1.5)
    
#     # 添加基因数量图例
#     gene_counts = [5, 10, 20]
#     legend_elements = []
#     for gc in gene_counts:
#         legend_elements.append(
#             plt.Line2D([0], [0], marker='o', color='w', 
#                       markerfacecolor='gray', markersize=np.sqrt(gc * 20 / np.pi),
#                       label=f'{gc}', markeredgecolor='black', markeredgewidth=0.5)
#         )
    
#     legend = ax.legend(
#         handles=legend_elements,
#         title='Gene Count',
#         loc='lower right',
#         frameon=True,
#         fontsize=9,
#         title_fontsize=10,
#         framealpha=0.9,
#         edgecolor='black',
#         fancybox=False
#     )
#     legend.get_frame().set_linewidth(1.5)
    
#     # 紧凑布局
#     plt.tight_layout()
    
#     # 保存图片
#     plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
#     plt.close()



def create_domain_enrichment_dotplot(
    domain: str,
    df: pd.DataFrame,
    output_path: str,
    top_terms_per_library: int = 10,
    adj_pval_cutoff: float = 0.05,
    figsize: Tuple[float, float] = (10, 100),
    dpi: int = 300
):
    """
    Create enrichment dot plot for a single domain.
    - Each library's results are grouped together
    - Different libraries have different background colors
    - Within each library, terms are sorted by -log10(adj_pval) descending
    
    Args:
        domain: Domain name
        df: Enrichment results DataFrame
        output_path: Path to save figure
        top_terms_per_library: Number of top terms to show per library
        adj_pval_cutoff: Adjusted p-value cutoff
        figsize: Figure size
        dpi: Resolution
    """
    # 按p值筛选
    df_sig = df[df["adj_pval"] <= adj_pval_cutoff].copy()
    
    if df_sig.empty:
        print(f"[WARN] No significant terms for {domain}")
        return
    
    # 计算-log10 p-value（如果不存在）
    if 'neg_log10_pval' not in df_sig.columns:
        df_sig['neg_log10_pval'] = -np.log10(df_sig['adj_pval'].clip(lower=1e-300))
    
    # 获取唯一的libraries，按ENRICH_LIBRARIES顺序排序
    available_libraries = df_sig['Library'].unique()
    library_order = [lib for lib in ENRICH_LIBRARIES if lib in available_libraries]
    library_order.extend([lib for lib in available_libraries if lib not in library_order])
    
    # 准备绘图数据：按library分组，每个library内部排序
    # 准备绘图数据：按library分组，每个library内部排序
    plot_rows = []
    y_pos = 0
    library_ranges = []  # 存储(start, end, library)用于背景颜色
    gap_size = 0  # 设置library之间的间隔行数

    for i, library in enumerate(library_order):
        library_data = df_sig[df_sig['Library'] == library].copy()
        
        # 按neg_log10_pval降序排序（最显著的在前）
        library_data = library_data.sort_values('neg_log10_pval', ascending=False)
        
        # 取该library的前N个terms
        # library_data = library_data.head(top_terms_per_library)
        library_data = library_data.head(8)
        
        if len(library_data) == 0:
            continue
        
        # 如果不是第一个library，添加间隔
        if i > 0 and len(plot_rows) > 0:
            y_pos += gap_size
        
        start_pos = y_pos
        
        # 添加该library的所有entries
        for idx, row in library_data.iterrows():
            plot_rows.append({
                'y_pos': y_pos,
                'term_clean': row['term_clean'],
                'library': row['Library'],
                'neg_log10_pval': row['neg_log10_pval'],
                'gene_ratio': row['gene_ratio'],
                'gene_count': row['gene_count']
            })
            y_pos += 1
        
        # 存储该library的范围
        library_ranges.append((start_pos, y_pos - 1, library))
    
    if not plot_rows:
        print(f"[WARN] No data to plot for {domain}")
        return
    
    df_plot = pd.DataFrame(plot_rows)
    df_plot = df_plot.reset_index(drop=True)
    
    # 创建图形 - 使用GridSpec来添加左侧色块
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(10,22), dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[0.03, 1], wspace=0.02, figure=fig)
    
    # 左侧色块axis
    ax_colors = fig.add_subplot(gs[0])
    # 主图axis
    ax = fig.add_subplot(gs[1])
    
    # 定义每个library的颜色（类似参考图）
    library_colors = {
        'GO_Biological_Process_2023': '#fbad91',  # Orange
        'GO_Cellular_Component_2023': '#CAC8E1',  # Green
        'GO_Molecular_Function_2023': '#b9e3b3',  # Yellow
        'KEGG_2021_Human': '#AFD436',  # Light blue
        'Reactome_2022': '#71A24F',  # Purple
        'WikiPathways_2019_Human': '#C00000', 
    }
    
    # 在左侧ax_colors上绘制色块
    # for start, end, library in library_ranges:
    #     color = library_colors.get(library, '#CCCCCC')
    #     ax_colors.barh(
    #         y=(start + end) / 2,
    #         width=1,
    #         height=(end - start + 1),
    #         color=color,
    #         alpha=1,
    #         edgecolor='none'
    #     )

    for start, end, library in library_ranges:
        color = library_colors.get(library, '#CCCCCC')
        ax_colors.axhspan(
            ymin=start - 0.5,      # 起始位置
            ymax=end + 0.5,        # 结束位置
            xmin=0,
            xmax=1,
            facecolor=color,
            edgecolor='none',
            alpha=1
        )   
    
    # 设置左侧色块axis
    ax_colors.set_xlim(0, 10)
    ax_colors.set_ylim(-0.5, len(df_plot) - 0.5)
    ax_colors.axis('off')
    

    
    # 定义颜色映射
    colors_deep_purple = ['#FFFFFF', '#E8DDF0', '#D4BFE1', '#BF9DD1',
                          '#AA7BC0', '#9459B0', '#7E3BA0', '#682D8F', '#52237E']
    cmap = LinearSegmentedColormap.from_list('deep_purple', colors_deep_purple)
    
    # 标准化颜色范围
    norm = Normalize(
        vmin=df_plot["neg_log10_pval"].min(),
        vmax=df_plot["neg_log10_pval"].max()
    )
    
    # 创建散点图
    scatter = ax.scatter(
        df_plot["gene_ratio"],
        df_plot["y_pos"],
        c=df_plot["neg_log10_pval"],
        s=df_plot["gene_count"] * 50,
        cmap=cmap,
        norm=norm,
        alpha=0.95,
        edgecolors='black',
        linewidths=0.5,
        zorder=3
    )
    
    # 设置Y轴标签
    ax.set_yticks(df_plot["y_pos"])
    ax.set_yticklabels(df_plot["term_clean"], fontsize=28)
    ax.tick_params(axis='y', which='major', pad=25,  length=0)  # 增加padding
    
    # 设置X轴
    # ax.set_xlabel("Gene Count  ", fontsize=18)
    ax.set_xlim(left=0, right=df_plot["gene_ratio"].max() * 1.1)
    ax.tick_params(axis='x', labelsize=28)
    
    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 添加标题
    ax.set_title(f"{domain}", fontsize=40,  pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('-log$_{10}$(Adjusted P-value)', 
                   fontsize=28,  rotation=270, labelpad=40)
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_linewidth(1.5)
    
    # 添加基因数量图例
    gene_counts = [10, 30, 50]
    legend_elements = []
    for gc in gene_counts:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='gray', markersize=np.sqrt(gc * 30 / np.pi),  # 从20改为50，与散点图保持一致
                    label=f'{gc}', markeredgecolor='black', markeredgewidth=0.5, alpha=0.6)
        )
    
    legend = ax.legend(
        handles=legend_elements,
        title='Gene Count(%)',
        loc='lower right',
        frameon=True,
        fontsize=20,
        title_fontsize=20,
        framealpha=0.6,
        edgecolor='black',
        fancybox=False
    )
    legend.get_frame().set_linewidth(1.5)
    
    # 添加library颜色图例
    library_legend_elements = []
    # 只为实际使用的library创建图例
    used_libraries = [lib for _, _, lib in library_ranges]
    for library in dict.fromkeys(used_libraries):  # 保持顺序并去重
        color = library_colors.get(library, '#CCCCCC')
        # 清理library名称，使其更易读
        display_name = library.replace('_', ' ').replace('2023', '').replace('2021', '').replace('2022', '').replace('2019', '').strip()
        library_legend_elements.append(
            plt.Line2D([0], [0], marker='s', color='w',
                    markerfacecolor=color, markersize=20,
                    label=display_name, markeredgecolor='none')
        )

    # 创建第二个图例（library图例）
    library_legend = ax.legend(
        handles=library_legend_elements,
        title='Library',
        loc='upper right',  # 放在右上角
        frameon=True,
        fontsize=20,
        title_fontsize=20,
        framealpha=0.6,
        edgecolor='black',
        fancybox=False
    )
    library_legend.get_frame().set_linewidth(1.5)

    # 重要：添加第一个图例回图中（因为第二个图例会覆盖第一个）
    ax.add_artist(legend)

    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    plt.close()


# =========================
# Load Existing Results and Plot
# =========================
def load_and_plot_only(enrichment_dir: str, output_dir: str):
    """
    Load existing enrichment results and create plots without running analysis.
    
    Args:
        enrichment_dir: Directory with existing enrichment CSV files
        output_dir: Output directory for plots
    """
    print("=" * 60)
    print("Loading Existing Results and Plotting")
    print("=" * 60)
    
    # Find all enrichment CSV files
    csv_files = glob.glob(os.path.join(enrichment_dir, "*_enrichment.csv"))
    
    if not csv_files:
        print(f"[ERROR] No enrichment CSV files found in {enrichment_dir}")
        print("Please run enrichment analysis first.")
        return
    
    print(f"\n[INFO] Found {len(csv_files)} enrichment result files")
    
    # Load all results
    enrichment_results = {}
    for csv_path in csv_files:
        domain = os.path.basename(csv_path).replace("_enrichment.csv", "")
        try:
            df = pd.read_csv(csv_path)
            
            # Ensure required columns exist
            if 'neg_log10_pval' not in df.columns and 'adj_pval' in df.columns:
                df['neg_log10_pval'] = -np.log10(df['adj_pval'].clip(lower=1e-300))
            
            if 'term_clean' not in df.columns and 'term' in df.columns:
                df['term_clean'] = df['term'].astype(str).str.replace(
                    r'\s*\([A-Z]{2}:\d+\)$|\s*\(WP\d+\)$', '', regex=True
                )
            
            enrichment_results[domain] = df
            print(f"  ✓ Loaded {domain}: {len(df)} terms")
        except Exception as e:
            print(f"  ✗ Error loading {domain}: {e}")
    
    if not enrichment_results:
        print("[ERROR] Failed to load any enrichment results")
        return
    
    print(f"\n[SUCCESS] Loaded {len(enrichment_results)} domains")
    
    # Create visualizations
    print("\n[INFO] Creating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Combined dot plot
    try:
        print("\n  Creating combined dot plot...")
        plot_df = prepare_plot_data(
            enrichment_results=enrichment_results,
            top_terms_per_domain=TOP_TERMS_PER_DOMAIN,
            max_terms_total=MAX_TERMS_TOTAL,
            adj_pval_cutoff=ADJ_PVALUE_CUTOFF,
            min_gene_ratio=MIN_GENE_RATIO
        )
        
        create_enrichment_dotplot(
            plot_df=plot_df,
            output_path=os.path.join(output_dir, "enrichment_dotplot_combined.png"),
            figsize=(10, 12),
            dpi=300,
            title="Enrichment Analysis - Top Pathways"
        )
    except Exception as e:
        print(f"[ERROR] Failed to create combined dot plot: {e}")
    
  
    
 
    
    # Individual domain plots
    print("\n[INFO] Creating individual domain plots...")
    domain_plots_dir = os.path.join(output_dir, "domain_plots")
    os.makedirs(domain_plots_dir, exist_ok=True)
    
    for domain, df in enrichment_results.items():
        try:
            safe_name = re.sub(r'[^\w\-]', '_', domain)
            
            # Barplot with library colors and grouped by library
            barplot_path = os.path.join(domain_plots_dir, f"{safe_name}_barplot.png")
            create_domain_enrichment_barplot(
                domain=domain,
                df=df,
                output_path=barplot_path,
                top_terms_per_library=10,  # Top 10 terms per library
                adj_pval_cutoff=ADJ_PVALUE_CUTOFF,
                figsize=(8, 14),
                dpi=300
            )
            print(f"  [SAVE] Barplot: {safe_name}_barplot.png")
            
            # Dotplot with library backgrounds
            dotplot_path = os.path.join(domain_plots_dir, f"{safe_name}_dotplot.png")
            create_domain_enrichment_dotplot(
                domain=domain,
                df=df,
                output_path=dotplot_path,
                top_terms_per_library=10,  # Top 10 terms per library
                adj_pval_cutoff=ADJ_PVALUE_CUTOFF,
                figsize=(10, 12),
                dpi=300
            )
            print(f"  [SAVE] Dotplot: {safe_name}_dotplot.png")
            
        except Exception as e:
            print(f"  [ERROR] Failed to create plots for {domain}: {e}")
    
    print("\n" + "=" * 60)
    print("Plotting Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


# =========================
# Main Execution
# =========================
def main():
    """Main execution function."""
    import sys
    
    # Check for plot-only mode
    plot_only = '--plot-only' in sys.argv or '-p' in sys.argv
    force_rerun = '--force-rerun' in sys.argv or '-f' in sys.argv
    enrichment_dir = os.path.join(OUTPUT_DIR, "enrichment_results")
    
    if plot_only:
        # Only plot from existing results
        load_and_plot_only(enrichment_dir, OUTPUT_DIR)
        return
    
    if not force_rerun:
        has_valid_results, _ = check_enrichment_dir_status(enrichment_dir)
        if has_valid_results:
            print("=" * 60)
            print("Existing enrichment results detected")
            print("=" * 60)
            load_and_plot_only(enrichment_dir, OUTPUT_DIR)
            return
    
    print("=" * 60)
    print("Enrichment Analysis and Visualization")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load domain genes
    print("\n[STEP 1] Loading domain gene tables...")
    domain_genes = load_domain_genes(DOMAIN_TABLES_DIR, top_n=TOP_GENES)
    print(f"Loaded {len(domain_genes)} domains")
    
    # 2. Run enrichment analysis (with auto-detection of existing results)
    print("\n[STEP 2] Running enrichment analysis...")
    enrichment_results = run_enrichment_analysis(
        domain_genes=domain_genes,
        libraries=ENRICH_LIBRARIES,
        organism=ORGANISM,
        outdir=enrichment_dir,
        cutoff=PVALUE_CUTOFF,
        force_rerun=force_rerun  # Use command line flag
    )
    
    if not enrichment_results:
        print("[ERROR] No enrichment results generated. Exiting.")
        return
    
    print(f"\n[INFO] Enrichment completed for {len(enrichment_results)} domains")
    
    # 3. Create visualizations
    print("\n[STEP 3] Creating visualizations...")
    
    # 3a. Combined dot plot (all domains merged)
    try:
        print("\n  Creating combined dot plot...")
        plot_df = prepare_plot_data(
            enrichment_results=enrichment_results,
            top_terms_per_domain=TOP_TERMS_PER_DOMAIN,
            max_terms_total=MAX_TERMS_TOTAL,
            adj_pval_cutoff=ADJ_PVALUE_CUTOFF,
            min_gene_ratio=MIN_GENE_RATIO
        )
        
        create_enrichment_dotplot(
            plot_df=plot_df,
            output_path=os.path.join(OUTPUT_DIR, "enrichment_dotplot_combined.png"),
            figsize=(10, 12),
            dpi=300,
            title="Enrichment Analysis - Top Pathways"
        )
    except Exception as e:
        print(f"[ERROR] Failed to create combined dot plot: {e}")
    

    # 4. Create individual domain plots
    print("\n[STEP 4] Creating individual domain plots...")
    domain_plots_dir = os.path.join(OUTPUT_DIR, "domain_plots")
    os.makedirs(domain_plots_dir, exist_ok=True)
    
    for domain, df in enrichment_results.items():
        try:
            safe_name = re.sub(r'[^\w\-]', '_', domain)
            
            # 4a. Barplot with library colors and grouped by library
            barplot_path = os.path.join(domain_plots_dir, f"{safe_name}_barplot.png")
            create_domain_enrichment_barplot(
                domain=domain,
                df=df,
                output_path=barplot_path,
                top_terms_per_library=10,  # Top 10 terms per library
                adj_pval_cutoff=ADJ_PVALUE_CUTOFF,
                figsize=(8, 14),
                dpi=300
            )
            print(f"  [SAVE] Barplot: {barplot_path}")
            
            # 4b. Dotplot with library backgrounds
            dotplot_path = os.path.join(domain_plots_dir, f"{safe_name}_dotplot.png")
            create_domain_enrichment_dotplot(
                domain=domain,
                df=df,
                output_path=dotplot_path,
                top_terms_per_library=5,  # Top 10 terms per library
                adj_pval_cutoff=ADJ_PVALUE_CUTOFF,
                figsize=(10, 130),
                dpi=300
            )
            print(f"  [SAVE] Dotplot: {dotplot_path}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to create plots for {domain}: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()

