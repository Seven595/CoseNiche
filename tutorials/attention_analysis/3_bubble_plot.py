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
    
    # : each domain of top_ngene
    for dom, sub in df_all.groupby("domain", sort=False):
        sub_sorted = sub.sort_values(["avg_strength", "hit_spots"], ascending=[False, False])
        genes = sub_sorted["partner_symbol"].astype(str).head(top_n).tolist()
        per_dom_top[dom] = genes
        x_set.extend(genes)
    
    # :
    seen = set()
    x_order = []
    for g in x_set:
        if g not in seen:
            seen.add(g)
            x_order.append(g)
    
    # : random
    global_ordered = x_order.copy()
    random.shuffle(global_ordered)
    
    # : for each domainfilter in top_n in in in of gene
    per_dom_filtered = {}
    for dom, top_genes in per_dom_top.items():
        # :domain of topgene ∩
        filtered_genes = [g for g in top_genes if g in global_ordered]
        per_dom_filtered[dom] = filtered_genes
    
    return per_dom_filtered, global_ordered

def build_plot_matrix_per_domain(df_all: pd.DataFrame, per_domain_genes: dict, global_ordered_genes: list) -> pd.DataFrame:
    """ for each domainbuildcontainstop gene of plot. each domain of top gene,x by genecolumn. Args: df_all: per_domain_genes: each domain of top gene {domain: [genes]} global_ordered_genes: of gene list Returns: Data Frame: columns = [domain, partner_symbol, avg_strength, hit_spots, x_position] """
    plot_data = []
    
    # creategene to x of
    gene_to_x = {gene: i for i, gene in enumerate(global_ordered_genes)}
    
    for domain, genes in per_domain_genes.items():
        # domain of
        domain_data = df_all[df_all["domain"] == domain].copy()
        
        # domain of topgene
        domain_top = domain_data[domain_data["partner_symbol"].isin(genes)].copy()
        
        # aggregation ()
        domain_top = (domain_top.groupby(["domain", "partner_symbol"], as_index=False)
                      .agg(avg_strength=("avg_strength", "max"),
                           hit_spots=("hit_spots", "max")))
        
        # by genecolumn
        domain_top["partner_symbol"] = pd.Categorical(domain_top["partner_symbol"], 
                                                    categories=global_ordered_genes, ordered=True)
        domain_top = domain_top.sort_values("partner_symbol")
        
        # x (by)
        domain_top["x_position"] = domain_top["partner_symbol"].map(gene_to_x)
        
        plot_data.append(domain_top)
    
    return pd.concat(plot_data, ignore_index=True)

def build_plot_matrix(df_all: pd.DataFrame, x_genes: list) -> pd.DataFrame:
    """ rows:domain;column:x_genes;:column avg_strength and hit_spots. table,plot:columns = [domain, partner_symbol, avg_strength, hit_spots] Missing for Na N, in plotprocessing for 0 size and. """
    # x_genes
    df = df_all[df_all["partner_symbol"].isin(x_genes)].copy()
    # aggregation (), select avg_strength of (domain-partner)
    df = (df.groupby(["domain", "partner_symbol"], as_index=False)
            .agg(avg_strength=("avg_strength", "max"),
                 hit_spots=("hit_spots", "max")))
    # ×gene of
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
    # domain: by domain
    doms = sorted(df_long["domain"].dropna().unique().tolist())
    
    # Note.
    hs = df_long["hit_spots"].astype(float)
    av = df_long["avg_strength"].astype(float)

    # size ()
    hs_valid = hs.fillna(0.0)
    if hs_valid.max() == hs_valid.min():
        sizes = np.where(hs_valid > 0, (min_size + max_size) / 2.0, 0 if size_floor_zero else min_size)
    else:
        s_norm = (hs_valid - hs_valid.min()) / (hs_valid.max() - hs_valid.min())
        sizes = s_norm * (max_size - min_size) + min_size
        if size_floor_zero:
            sizes = np.where(hs_valid > 0, sizes, 0)

    # Nature ()
    if cmap_name == "Reds" or cmap_name == "custom_reds":
        # (->->->)
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

    # - gene
    dom_to_y = {d: i for i, d in enumerate(doms)}
    ys = df_long["domain"].map(dom_to_y).astype(float).values
    
    # x
    xs = df_long["x_position"].astype(float).values * x_spacing

    # plot:domain ()
    if not fixed_size and isinstance(figsize, tuple) and len(figsize) == 2 and figsize[1] == 0.6:
        height = max(3.0, 0.6 * max(1, len(doms)))
        figsize = (figsize[0], height)

    # createplot
    # fig, ax = plt.subplots(figsize=(16, 7), dpi=dpi) # PDAC
    fig, ax = plt.subplots(figsize=(20, 7), dpi=dpi)    #HBRC
    # plotdot plot ()
    sc = ax.scatter(xs, ys, s=sizes, c=colors, marker="o", 
                   edgecolors="black", linewidths=0.7, alpha=0.9, zorder=3)

    # Y
    ax.set_yticks(range(len(doms)))
    ax.set_yticklabels(doms, fontsize=16, fontweight='normal')
    ax.set_ylabel("", fontsize=18, fontweight='bold')
    
    # X - gene
    ax.set_xticks([i * x_spacing for i in range(len(global_ordered_genes))])
    ax.set_xticklabels(global_ordered_genes, rotation=45, ha="right", fontsize=12, 
                      fontstyle='italic')  # gene (Nature)
    ax.set_xlabel(" ", fontsize=14, fontweight='bold')

    # Note.
    ax.set_title(title, fontsize=26, fontweight='bold', pad=20)
    
    # ()
    # ax.grid(True, axis='x', linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # (Nature: and)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Note.
    ax.set_xlim(-0.5 * x_spacing, (len(global_ordered_genes) - 0.5) * x_spacing)
    ax.set_ylim(-0.5, len(doms) - 0.5)

    # (in)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # of,
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # create, (0.5-1.0)
    cbar_ax = inset_axes(ax,
                        width="2%",  # Note.
                        height="48%",  # ()
                        loc='upper right',
                        bbox_to_anchor=(0.04, -0.1, 1, 1),  # ax of
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Average Strength", fontsize=16, fontweight='bold', 
                   rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(1.5)

    # Hit Spots plot (in)
    if np.nanmax(hs_valid) > 0:
        handles = []
        labels = []
        # select 3-4 tablesize
        ticks = np.linspace(hs_valid[hs_valid > 0].min(), hs_valid.max(), num=4)
        for t in ticks:
            s_t = (t - hs_valid.min()) / (hs_valid.max() - hs_valid.min() + 1e-12)
            s_t = s_t * (max_size - min_size) + min_size
            handles.append(plt.scatter([], [], s=s_t, color="gray", 
                                      edgecolors='black', linewidths=0.5))
            labels.append(f"{int(t)}")
        
        # plot in (0-0.5), in align
        leg = ax.legend(handles, labels, title="Hit Spots", 
                       scatterpoints=1,
                       loc='center',  # plot in
                       bbox_to_anchor=(1.04, 0.18),  # x:,y: in (0.25)
                       bbox_transform=ax.transAxes,  # Note.
                       frameon=True, 
                       framealpha=0.9, 
                       edgecolor='black',
                       fontsize=11, 
                       title_fontsize=12,
                       labelspacing=1,
                       handletextpad=1.0,
                       borderpad=1.0)
        leg.get_frame().set_linewidth(1.5)
    
    # Note.
    plt.tight_layout()

    # Saveplot (PNG and PDF)
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
    
    # load
    all_top_csv = os.path.join(args.prepared_dir, args.top_csv_name)
    print(f"\n[INFO] Loading data from: {all_top_csv}")
    df_all = load_all_top_table(all_top_csv)
    print(f"[INFO] Loaded {len(df_all)} total records")

    # each domain of topgene
    print(f"\n[INFO] Selecting top {args.top_n} genes per domain...")
    per_domain_genes, global_ordered_genes = pick_global_x_set_from_domain_top3(
        df_all, top_n=args.top_n
    )
    if not per_domain_genes:
        raise ValueError("Failed to select any top genes from input table")
    
    print(f"[INFO] Selected {len(global_ordered_genes)} unique genes across {len(per_domain_genes)} domains")

    # plottable (each domain of topgene, by column)
    print("\n[INFO] Building plot matrix...")
    df_plot = build_plot_matrix_per_domain(df_all, per_domain_genes, global_ordered_genes)
    print(f"[INFO] Plot matrix shape: {df_plot.shape}")

    # plotNaturebubbleplot
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