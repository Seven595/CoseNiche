#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble Plot for Top Partner Genes per Domain
Generate Nature-style bubble plot visualization
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as mpatches
import random

# Nature journal style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

def _safe_name(s: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", str(s)).strip("_") or "NA"

def load_all_top_table(all_top_csv: str) -> pd.DataFrame:
    """
    Load top partners table with required columns.
    
    Args:
        all_top_csv: Path to CSV file with top partners
        
    Returns:
        DataFrame with domain, partner_symbol, hit_spots, avg_strength columns
    """
    if not os.path.isfile(all_top_csv):
        raise FileNotFoundError(f"Not found: {all_top_csv}")
    df = pd.read_csv(all_top_csv)
    need = {"domain", "partner_symbol", "hit_spots", "avg_strength"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{all_top_csv} missing columns: {missing}")
    return df

def pick_global_x_set_from_domain_top3(df_all: pd.DataFrame, top_n: int = 3) -> tuple:
    """
    Pick top N genes per domain and create global gene set.
    
    Args:
        df_all: Full dataset with all domains
        top_n: Number of top genes to select per domain
    
    Returns:
        tuple: (per_domain_filtered_genes_dict, global_ordered_genes_list)
    """
    per_dom_top = {}
    x_set = []
    
    # 第一步：获取每个domain的top_n基因
    for dom, sub in df_all.groupby("domain", sort=False):
        sub_sorted = sub.sort_values(["avg_strength", "hit_spots"], ascending=[False, False])
        genes = sub_sorted["partner_symbol"].astype(str).head(top_n).tolist()
        per_dom_top[dom] = genes
        x_set.extend(genes)
    
    # 第二步：去重形成全局并集
    seen = set()
    x_order = []
    for g in x_set:
        if g not in seen:
            seen.add(g)
            x_order.append(g)
    
    # 第三步：随机打乱顺序
    global_ordered = x_order.copy()
    random.shuffle(global_ordered)
    
    # 第四步：为每个domain筛选出既在top_n中又在全局并集中的基因
    per_dom_filtered = {}
    for dom, top_genes in per_dom_top.items():
        # 取交集：该domain的top基因 ∩ 全局并集
        filtered_genes = [g for g in top_genes if g in global_ordered]
        per_dom_filtered[dom] = filtered_genes
    
    return per_dom_filtered, global_ordered

def build_plot_matrix_per_domain(df_all: pd.DataFrame, per_domain_genes: dict, global_ordered_genes: list) -> pd.DataFrame:
    """
    为每个domain构建只包含其top基因的绘图矩阵。
    每个domain只显示自己的top基因，但x轴位置按全局基因顺序排列。
    
    Args:
        df_all: 原始数据
        per_domain_genes: 每个domain的top基因字典 {domain: [genes]}
        global_ordered_genes: 全局排序的基因列表
    
    Returns:
        DataFrame: columns = [domain, partner_symbol, avg_strength, hit_spots, x_position]
    """
    plot_data = []
    
    # 创建全局基因到x位置的映射
    gene_to_x = {gene: i for i, gene in enumerate(global_ordered_genes)}
    
    for domain, genes in per_domain_genes.items():
        # 获取该domain的数据
        domain_data = df_all[df_all["domain"] == domain].copy()
        
        # 只保留该domain的top基因
        domain_top = domain_data[domain_data["partner_symbol"].isin(genes)].copy()
        
        # 聚合（防重）
        domain_top = (domain_top.groupby(["domain", "partner_symbol"], as_index=False)
                      .agg(avg_strength=("avg_strength", "max"),
                           hit_spots=("hit_spots", "max")))
        
        # 按全局基因顺序排列
        domain_top["partner_symbol"] = pd.Categorical(domain_top["partner_symbol"], 
                                                    categories=global_ordered_genes, ordered=True)
        domain_top = domain_top.sort_values("partner_symbol")
        
        # 添加x轴位置信息（按全局顺序）
        domain_top["x_position"] = domain_top["partner_symbol"].map(gene_to_x)
        
        plot_data.append(domain_top)
    
    return pd.concat(plot_data, ignore_index=True)

def build_plot_matrix(df_all: pd.DataFrame, x_genes: list) -> pd.DataFrame:
    """
    行：domain；列：x_genes；值：两列 avg_strength 和 hit_spots。
    返回一个长表，便于绘图：columns = [domain, partner_symbol, avg_strength, hit_spots]
    对于缺失项填充为 NaN，再在绘图时处理为 0 大小和透明。
    """
    # 只保留 x_genes
    df = df_all[df_all["partner_symbol"].isin(x_genes)].copy()
    # 聚合（防重），选择 avg_strength 最大的记录（理论上同一 domain-partner 应唯一）
    df = (df.groupby(["domain", "partner_symbol"], as_index=False)
            .agg(avg_strength=("avg_strength", "max"),
                 hit_spots=("hit_spots", "max")))
    # 确保域×基因的全组合
    domains = df["domain"].unique().tolist()
    idx = pd.MultiIndex.from_product([domains, x_genes], names=["domain", "partner_symbol"])
    df_full = df.set_index(["domain", "partner_symbol"]).reindex(idx).reset_index()
    return df_full



def plot_bubble_per_domain(df_long: pd.DataFrame,
                          global_ordered_genes: list,
                          out_png: str,
                          figsize=(14, 9),
                          dpi=300,
                          cmap_name="Reds",
                          min_size=10,
                          max_size=300,
                          size_floor_zero=True,
                          fixed_size=False,
                          x_spacing=1.0,
                          title="Top Partner Genes per Domain"):
    """
    Create Nature-style bubble plot showing top genes per domain.
    
    Args:
        df_long: DataFrame with columns [domain, partner_symbol, avg_strength, hit_spots, x_position]
        global_ordered_genes: List of all genes in display order
        out_png: Output path for PNG file
        figsize: Figure size (width, height)
        dpi: Resolution
        cmap_name: Colormap name (default: Reds for Nature style)
        min_size: Minimum bubble size
        max_size: Maximum bubble size
        size_floor_zero: If True, zero values have size 0
        fixed_size: If True, use fixed figure size
        x_spacing: Spacing between x-axis positions
        title: Plot title
        
    Notes:
        - Bubble size represents hit_spots (number of observations)
        - Bubble color represents avg_strength (average signal strength)
    """
    # 排序domain：按domain名称
    doms = sorted(df_long["domain"].dropna().unique().tolist())
    
    # 尺度映射
    hs = df_long["hit_spots"].astype(float)
    av = df_long["avg_strength"].astype(float)

    # 大小映射（线性）
    hs_valid = hs.fillna(0.0)
    if hs_valid.max() == hs_valid.min():
        sizes = np.where(hs_valid > 0, (min_size + max_size) / 2.0, 0 if size_floor_zero else min_size)
    else:
        s_norm = (hs_valid - hs_valid.min()) / (hs_valid.max() - hs_valid.min())
        sizes = s_norm * (max_size - min_size) + min_size
        if size_floor_zero:
            sizes = np.where(hs_valid > 0, sizes, 0)

    # Nature期刊风格颜色映射（使用红色系）
    if cmap_name == "Reds" or cmap_name == "custom_reds":
        # 自定义红色渐变（白色->橙色->红色->深红色）
        colors_pval = ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', 
                       '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
        cmap = LinearSegmentedColormap.from_list('custom_reds', colors_pval)
    else:
        cmap = get_cmap(cmap_name)
    
    av_valid = av.copy()
    vmin = np.nanmin(av_valid) if np.isfinite(av_valid).any() else 0.0
    vmax = np.nanmax(av_valid) if np.isfinite(av_valid).any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-12
    norm = Normalize(vmin=vmin, vmax=vmax)
    colors = [cmap(norm(v)) if np.isfinite(v) else (0,0,0,0) for v in av_valid]

    # 坐标网格 - 使用全局基因顺序
    dom_to_y = {d: i for i, d in enumerate(doms)}
    ys = df_long["domain"].map(dom_to_y).astype(float).values
    
    # x坐标直接使用全局位置
    xs = df_long["x_position"].astype(float).values * x_spacing

    # 图尺寸：根据domain数动态扩展高度（除非固定尺寸）
    if not fixed_size and isinstance(figsize, tuple) and len(figsize) == 2 and figsize[1] == 0.6:
        height = max(3.0, 0.6 * max(1, len(doms)))
        figsize = (figsize[0], height)

    # 创建图形
    # fig, ax = plt.subplots(figsize=(16, 7), dpi=dpi) # PDAC
    fig, ax = plt.subplots(figsize=(20, 7), dpi=dpi)    #HBRC
    # 绘制散点图（使用黑色边缘以增强视觉效果）
    sc = ax.scatter(xs, ys, s=sizes, c=colors, marker="o", 
                   edgecolors="black", linewidths=0.7, alpha=0.9, zorder=3)

    # Y轴设置
    ax.set_yticks(range(len(doms)))
    ax.set_yticklabels(doms, fontsize=16, fontweight='normal')
    ax.set_ylabel("", fontsize=18, fontweight='bold')
    
    # X轴设置 - 使用全局基因顺序
    ax.set_xticks([i * x_spacing for i in range(len(global_ordered_genes))])
    ax.set_xticklabels(global_ordered_genes, rotation=45, ha="right", fontsize=12, 
                      fontstyle='italic')  # 基因名用斜体（Nature风格）
    ax.set_xlabel(" ", fontsize=14, fontweight='bold')

    # 标题
    ax.set_title(title, fontsize=26, fontweight='bold', pad=20)
    
    # 添加网格（淡色背景网格）
    # ax.grid(True, axis='x', linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # 边框设置（Nature风格：只保留左侧和底部）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # 设置轴范围
    ax.set_xlim(-0.5 * x_spacing, (len(global_ordered_genes) - 0.5) * x_spacing)
    ax.set_ylim(-0.5, len(doms) - 0.5)

    # 颜色条（放置在右侧上半部分）
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # 缩小色条的纵向范围，让它只占上半部分
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # 创建色条，占据右侧上半部分（纵向 0.5-1.0）
    cbar_ax = inset_axes(ax,
                        width="2%",  # 色条宽度
                        height="48%",  # 色条高度（上半部分）
                        loc='upper right',
                        bbox_to_anchor=(0.04, -0.1, 1, 1),  # 相对于ax的位置
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Average Strength", fontsize=16, fontweight='bold', 
                   rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.5)

    # Hit Spots 图例（放置在右侧下半部分）
    if np.nanmax(hs_valid) > 0:
        handles = []
        labels = []
        # 选择3-4个代表性大小
        ticks = np.linspace(hs_valid[hs_valid > 0].min(), hs_valid.max(), num=4)
        for t in ticks:
            s_t = (t - hs_valid.min()) / (hs_valid.max() - hs_valid.min() + 1e-12)
            s_t = s_t * (max_size - min_size) + min_size
            handles.append(plt.scatter([], [], s=s_t, color="gray", 
                                      edgecolors='black', linewidths=0.5))
            labels.append(f"{int(t)}")
        
        # 图例放在右侧下半部分（纵向 0-0.5），居中对齐
        leg = ax.legend(handles, labels, title="Hit Spots", 
                       scatterpoints=1,
                       loc='center',  # 图例框居中
                       bbox_to_anchor=(1.04, 0.18),  # x: 右侧，y: 下半部分中心（0.25）
                       bbox_transform=ax.transAxes,  # 使用轴坐标系
                       frameon=True, 
                       framealpha=0.9, 
                       edgecolor='black',
                       fontsize=11, 
                       title_fontsize=12,
                       labelspacing=1,
                       handletextpad=1.0,
                       borderpad=1.0)
        leg.get_frame().set_linewidth(1.5)
    
    # 紧凑布局
    plt.tight_layout()

    # 保存图片（PNG和PDF）
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor='white')
    pdf_path = out_png.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {pdf_path}")



def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate Nature-style bubble plot for top partner genes per domain"
    )

    # HBRC
    parser.add_argument("--prepared_dir", type=str, 
                       default="./HBRC/result_output/enrichment_prepared",
                       help="Directory containing all_domains_top_partners.csv")
    parser.add_argument("--out_png", type=str, 
                       default="./HBRC/result_output/bubble_top10.png",
                       help="Output path for bubble plot PNG")
    
    
    
    # # PDAC
    # parser.add_argument("--prepared_dir", type=str, 
    #                    default="./PDAC/result_output/enrichment_prepared",
    #                    help="Directory containing all_domains_top_partners.csv")

    # parser.add_argument("--out_png", type=str, 
    #                    default="./PDAC/result_output/bubble_top10.png",
    #                    help="Output path for bubble plot PNG")
                       
    parser.add_argument("--top_csv_name", type=str, 
                       default="all_domains_top_partners.csv",
                       help="Name of the top partners CSV file")
    parser.add_argument("--top_n", type=int, default=10, 
                       help="Number of top genes to select per domain")
   
    parser.add_argument("--cmap", type=str, default="Reds",
                       help="Colormap name (default: Reds for Nature style)")
    parser.add_argument("--min_size", type=float, default=20,
                       help="Minimum bubble size")
    parser.add_argument("--max_size", type=float, default=300,
                       help="Maximum bubble size")
    parser.add_argument("--fixed_size", default=True, type=bool, 
                       help="Use fixed figure size (not dynamically adjusted)")
    parser.add_argument("--x_spacing", type=float, default=0.5, 
                       help="Spacing multiplier between x-axis genes")
    args = parser.parse_args()

    print("=" * 60)
    print("Nature-Style Bubble Plot Generation")
    print("=" * 60)
    
    # 加载数据
    all_top_csv = os.path.join(args.prepared_dir, args.top_csv_name)
    print(f"\n[INFO] Loading data from: {all_top_csv}")
    df_all = load_all_top_table(all_top_csv)
    print(f"[INFO] Loaded {len(df_all)} total records")

    # 选取每个domain的top基因
    print(f"\n[INFO] Selecting top {args.top_n} genes per domain...")
    per_domain_genes, global_ordered_genes = pick_global_x_set_from_domain_top3(
        df_all, top_n=args.top_n
    )
    if not per_domain_genes:
        raise ValueError("Failed to select any top genes from input table")
    
    print(f"[INFO] Selected {len(global_ordered_genes)} unique genes across {len(per_domain_genes)} domains")

    # 构造绘图长表（每个domain只显示自己的top基因，但按全局顺序排列）
    print("\n[INFO] Building plot matrix...")
    df_plot = build_plot_matrix_per_domain(df_all, per_domain_genes, global_ordered_genes)
    print(f"[INFO] Plot matrix shape: {df_plot.shape}")

    # 绘制Nature风格bubble图
    print("\n[INFO] Creating Nature-style bubble plot...")
    plot_bubble_per_domain(
        df_long=df_plot,
        global_ordered_genes=global_ordered_genes,
        out_png=args.out_png,
        figsize=(16, 6),
        dpi=300,
        cmap_name=args.cmap,
        min_size=args.min_size,
        max_size=args.max_size,
        size_floor_zero=True,
        fixed_size=True,
        x_spacing=args.x_spacing,
        title=f"Top {args.top_n} Partner Genes per Domain"
    )
    
    print("\n" + "=" * 60)
    print("Bubble Plot Generation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()