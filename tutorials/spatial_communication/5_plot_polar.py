# # -*- coding: utf-8 -*-
# """
# plot (Nature):
# - Nature and
# -, of can
# - of and
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from matplotlib import colormaps as cmaps
# import json
# import scanpy as sc

# # Nature
# plt.rcParams.update({
#     'font.family': 'sans-serif',
#     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
# 'font.size': 10, # :8 -> 10
#     'axes.linewidth': 0.5,
#     'xtick.major.width': 0.5,
#     'ytick.major.width': 0.5,
#     'xtick.major.size': 2.5,
#     'ytick.major.size': 2.5,
#     'legend.frameon': False,
# 'legend.fontsize': 9, # plot:7 -> 9
#     'figure.dpi': 300,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.1,
# 'pdf.fonttype': 42, # True Type,Nature
#     'ps.fonttype': 42,
# })

# # Nature of - (domain)
# NATURE_COLORS = {
# # : to 12 x 2 = 24,12 domain
# # each of (sequential of)
# # and, in of,
#     'primary': [
# # === :6 ===
# # (Domain 0)-
# '#5D4A8E', # ()
# '#8E6FC9', # in (and)
        
# # (Domain 1)-
# '#D4A870', # (and)
# '#EAD7B3', # ()
        
# # (Domain 2)-
# '#3C5C95', # ()
# '#5BA3C7', # (and)
        
# # (Domain 3)-
# '#B85054', # ()
# '#E87B8A', # (and)
        
# # (Domain 4)-
# '#71A24F', # ()
# '#6BB881', # (and)
        
# # (Domain 5)-
# '#D9935C', # (and)
# '#A66C3D', # ()
        
# # === :6 ===
# # (Domain 6)-
# '#2E4A6B', #
# '#4A6FA5', #
        
# # (Domain 7)-
# '#C84D7B', #
# '#E8739C', #
        
# # (Domain 8)-
# '#3B9B8F', #
# '#5EC4B6', #
        
# # (Domain 9)-
# '#A84032', #
# '#D16B5E', #
        
# # (Domain 10)-
# '#7B904B', #
# '#A5B875', #
        
# # (Domain 11)- in
# '#5A6F7D', #
# '#7B93A3', #
#     ],
# # (gene expression)-
#     'sequential': ['#4A3A6B', '#6B5491', '#9A72DD', '#B794E8',
#                    '#D1AEFA', '#E5D4FC', '#F3EBFD', '#FAF7FE'],
# # (for)-
#     'diverging': ['#4A3A6B', '#6F5099', '#9A72DD', '#BEA3ED', '#E5D4FC',
#                   '#F9EDD8', '#F5E0BD', '#EDD0A3', '#DCBD8A', '#CAAA75'],
# # region (, for /)
#     'backgrounds': {
# 'same_domain': '#FAF7FE', # -
# 'diff_domain': '#FEFBF7', # -
#     }
# }

# # and - Nature
# GRID_COLOR = '#E5E5E5' # of
# BG_ALPHA = 0.25 # (from 0.12 to 0.25)

# # Domain
# # each domain of
# # in all plot in, domain of
# GLOBAL_DOMAIN_COLOR_MAP = {
# 'Duct Epithelium': 0, # (0: #5D4A8E, #8E6FC9)
# 'Stroma': 1, # (2: #D4A870, #EAD7B3)
# 'Tumor': 2, # (4: #3C5C95, #5BA3C7)
# 'Acinar': 3, # (6: #B85054, #E87B8A)
# 'Islet': 4, # (8: #71A24F, #6BB881)
# 'Immune': 5, # (10: #D9935C, #A66C3D)
# # can actual of domain...
# }


# def get_domain_color(domain_name: str, nature_palette: list) -> str:
#     """
# domain of
    
#     Args:
# domain_name: domain of
# nature_palette: columntable
    
#     Returns:
# domain of ()
#     """
# # if domain in in, of
#     if domain_name in GLOBAL_DOMAIN_COLOR_MAP:
#         color_idx = GLOBAL_DOMAIN_COLOR_MAP[domain_name]
# # each domain of (, * 2)
#         palette_idx = (color_idx * 2) % len(nature_palette)
#         return nature_palette[palette_idx]
    
# # if not of domain,hash of
# # of domain, in different plot in of
# # of hash each domain different of
#     hash_value = hash(domain_name)
# # ():0, 2, 4, 6, 8, 10
#     color_idx = abs(hash_value) % (len(nature_palette) // 2)
#     palette_idx = (color_idx * 2) % len(nature_palette)
#     return nature_palette[palette_idx]


# # ---------------- ----------------

# def load_boundary_spots(data_dir: str) -> list:
#     boundary_spots_path = os.path.join(data_dir, 'boundary_spots.json')  # JSON path
#     print(f"boundary spotsfile path: {boundary_spots_path}")
#     if os.path.exists(boundary_spots_path):
#         with open(boundary_spots_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
# return data.get('boundary_spots', []) # columntable
#     else:
#         print(f"Warning: not foundboundary spotsfile {boundary_spots_path}")
#         return []

# def load_adata_and_coords(data_dir: str) -> tuple:
# h5ad_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')] # h5ad
#     if h5ad_files:
# adata_path = os.path.join(data_dir, h5ad_files[0]) #
#         adata = sc.read_h5ad(adata_path)  # Reading
#         print(f"loadAnnData: {adata_path}")
#     else:
#         print("Warning: not foundAnnDatafile")
#         adata = None
    
# coord_files = [f for f in os.listdir(data_dir) if 'coordinates' in f and f.endswith('.csv')] # CSV
#     if coord_files:
#         coords_path = os.path.join(data_dir, coord_files[0])
# coords_df = pd.read_csv(coords_path) # Readingtable
# print(f"load: {coords_path}")
#     else:
# print("Warning: not foundfile")
#         coords_df = None
    
#     return adata, coords_df

# def get_spot_name_from_index(spot_idx: int, adata=None, coords_df=None) -> str:
# if adata is not None and spot_idx < len(adata.obs): # obs.index
#         return adata.obs.index[spot_idx]
# elif coords_df is not None and 'spot_name' in coords_df.columns: # CSV
#         spot_row = coords_df[coords_df['spot_idx'] == spot_idx]
#         if not spot_row.empty:
#             return spot_row['spot_name'].iloc[0]
#     return str(spot_idx)

# def get_domain_info(spot_name: str, adata=None, coords_df=None, spot_idx=None) -> str:
# """spot of domain/cluster
    
# :
# 1. if spot_idx, from coords_df (can)
# 2. from adata.obsspot_name
# 3. from coords_dfspot_name
#     """
# # 1: spot_idx (can)
#     if spot_idx is not None and coords_df is not None:
#         if 'spot_idx' in coords_df.columns:
#             spot_row = coords_df[coords_df['spot_idx'] == spot_idx]
#             if not spot_row.empty:
#                 if 'cluster' in coords_df.columns:
#                     return str(spot_row['cluster'].iloc[0])
#                 elif 'domain' in coords_df.columns:
#                     return str(spot_row['domain'].iloc[0])
    
# # 2: spot_name from adata
#     if adata is not None:
#         if spot_name in adata.obs.index:
#             if 'ground_truth' in adata.obs.columns:
#                 return str(adata.obs.loc[spot_name, 'ground_truth'])
#             elif 'domain' in adata.obs.columns:
#                 return str(adata.obs.loc[spot_name, 'domain'])
#             elif 'cluster' in adata.obs.columns:
#                 return str(adata.obs.loc[spot_name, 'cluster'])
    
# # 3: spot_name from coords_df
#     if coords_df is not None and 'spot_name' in coords_df.columns:
#         spot_row = coords_df[coords_df['spot_name'] == spot_name]
#         if not spot_row.empty:
#             if 'cluster' in coords_df.columns:
#                 return str(spot_row['cluster'].iloc[0])
#             elif 'domain' in coords_df.columns:
#                 return str(spot_row['domain'].iloc[0])
    
#     return "Unknown"



# def normalize_array(x):
# """ (, for)"""
#     x = np.asarray(x, float)
#     vmax = x.max() if x.size and np.isfinite(x).any() else 1.0
#     return x / (vmax if vmax > 0 else 1.0)

# def filter_cross_domain_genes(df: pd.DataFrame, 
#                               neighbor_col: str = "neighbor_name",
#                               gene_col: str = "gene",
#                               adata=None,
#                               coords_df=None) -> pd.DataFrame:
#     """
# filterdomain of gene,domain or domain of gene
    
# :
# - genes, in domain in
# - if gene in multiple different domain in -> filter (domain)
# - if gene in single domain in -> (domain or domain)
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
# contains and gene of
#     neighbor_col : str
# column
#     gene_col : str
# genecolumn
#     adata : AnnData, optional
# for domain
#     coords_df : pd.DataFrame, optional
# for domain
    
#     Returns:
#     --------
#     pd.DataFrame
# after filtering of (domaingene)
#     """
# print("\n=== Startfilterdomaingene ===")
    
# # for neighborsdomain
#     neighbor_domains = {}
#     for neighbor in df[neighbor_col].unique():
# # neighbor for spot_idx
#         neighbor_idx = None
#         try:
#             neighbor_idx = int(neighbor)
#         except (ValueError, TypeError):
# # if, from coords_df
#             if coords_df is not None and 'spot_name' in coords_df.columns:
#                 neighbor_row = coords_df[coords_df['spot_name'] == str(neighbor)]
#                 if not neighbor_row.empty and 'spot_idx' in coords_df.columns:
#                     neighbor_idx = neighbor_row['spot_idx'].iloc[0]
        
# # spot_idxdomain
#         domain = get_domain_info(str(neighbor), adata=adata, coords_df=coords_df, spot_idx=neighbor_idx)
#         neighbor_domains[neighbor] = domain
    
# # of domain
# print(f"\n -domain:")
#     for neighbor, domain in sorted(neighbor_domains.items(), key=lambda x: str(x[0])):
# print(f" {neighbor} -> {domain}")
    
# print(f"\n of domain:")
#     domain_count = {}
#     for neighbor, domain in neighbor_domains.items():
#         domain_count[domain] = domain_count.get(domain, 0) + 1
#     for domain, count in sorted(domain_count.items()):
#         print(f"    {domain}: {count} neighbors")
    
# # for domaincolumn
#     df_with_domain = df.copy()
#     df_with_domain['neighbor_domain'] = df_with_domain[neighbor_col].map(neighbor_domains)
    
# # genes in domain in
#     gene_domain_count = df_with_domain.groupby(gene_col)['neighbor_domain'].nunique()
    
# # :genes in of domain in
#     gene_domains_detail = df_with_domain.groupby(gene_col)['neighbor_domain'].apply(
#         lambda x: sorted(x.unique())
#     ).to_dict()
    
# # in single domain in of gene
#     genes_to_keep = gene_domain_count[gene_domain_count == 1].index.tolist()
    
# Note.
#     total_genes = df[gene_col].nunique()
#     cross_domain_genes = gene_domain_count[gene_domain_count > 1].index.tolist()
    
# print(f"\n gene count: {total_genes}")
# print(f" domaingene count: {len(cross_domain_genes)}")
# print(f" of domain/gene count: {len(genes_to_keep)}")
    
#     if len(cross_domain_genes) > 0:
# print(f"\n domaingene (before 10):")
#         for gene in cross_domain_genes[:10]:
#             domains = gene_domains_detail.get(gene, [])
# print(f" {gene:15s}: in {domains}")
    
# # filter
#     df_filtered = df[df[gene_col].isin(genes_to_keep)].copy()
    
#     print(f"\n  filter before record count: {len(df)}")
#     print(f"   after filtering record count: {len(df_filtered)}")
#     print("=== filterCompleted ===\n")
    
#     return df_filtered


# def read_csv_or_topk(csv_path: str,
#                      required_cols,
#                      topk_in_memory: int = None,
#                      by_sum_or_mean: str = "sum",
#                      entity_cols=None) -> pd.DataFrame:
# df = pd.read_csv(csv_path) # CSV
# miss = [c for c in required_cols if c not in df.columns] # columnCheck
#     if miss:
# raise ValueError(f"file {csv_path} column: {miss}")
# if topk_in_memory is None or entity_cols is None: # TopK
#         return df
# k = int(topk_in_memory) # TopK
#     metric = by_sum_or_mean
#     if metric not in ("sum","mean"):
# raise ValueError("by_sum_or_mean for 'sum' or 'mean'")
#     df = df.copy()
# df["__rank__"] = df.groupby(entity_cols)[metric].rank(method="first", ascending=False) #
# out = df[df["__rank__"] <= k].drop(columns="__rank__") # TopK
#     return out

# # ---------------- plot () ----------------

# def build_domain_colormap(domains, palette="primary", alpha=None):
#     """
# Nature for domain (RGBA,)
    
#     Parameters:
#     -----------
#     domains : list
#         domaincolumntable
#     palette : str
# :'primary' (), 'pastel' (and), 'accent' ()
#     alpha : float
# ,default BG_ALPHA
#     """
#     if alpha is None:
#         alpha = BG_ALPHA
        
#     color_list = NATURE_COLORS.get(palette, NATURE_COLORS['primary'])
#     unique_domains = list(pd.Index(domains).unique())
#     dom2rgba = {}
    
#     for d in unique_domains:
# # get_domain_color() and plot
#         hex_color = get_domain_color(d, color_list)
#         rgba = list(colors.to_rgba(hex_color))
#         rgba[3] = float(alpha)
#         dom2rgba[d] = tuple(rgba)
    
#     return dom2rgba

# def fill_ring_sector(ax, th0, th1, r0, r1, color, n=128):
#     """
# in in,.
# th0, th1: ()
# r0, r1:
# color: RGBA
# n:
#     """
# thetas_outer = np.linspace(th0, th1, n) #
# thetas_inner = np.linspace(th1, th0, n) #
# r_outer = np.full_like(thetas_outer, r1) #
# r_inner = np.full_like(thetas_inner, r0) #
# th = np.concatenate([thetas_outer, thetas_inner]) # column
# rr = np.concatenate([r_outer, r_inner]) #
# ax.fill(th, rr, color=color, edgecolor=None, linewidth=0) # in


# def draw_sector_middle_band(ax, theta_edges, r_inner, r_outer,
#                             sector_labels, sector2domain, domain2color,
#                             band_frac=(0.0, 1.0)):
#     """
# in each of.
# band_frac=(0.0,1.0) means from to.
# :sector_labels of sector2domain of, in domain2color.
#     """
# r_span = (r_outer - r_inner) #
# r0 = r_inner + r_span * float(band_frac[0]) #
# r1 = r_inner + r_span * float(band_frac[1]) # End
    
# Note.
#     print("Debug - Domain colors mapping:")
#     if domain2color:
#         for dom, col in domain2color.items():
#             print(f"  Domain '{dom}' -> Color {col}")
    
# for i in range(len(sector_labels)): #
# th0 = theta_edges[i] #
# th1 = theta_edges[i+1] # End
# # of domain; not in
#         dom = sector2domain.get(sector_labels[i], "Unknown") if sector2domain is not None else "Unknown"
#         col = domain2color.get(dom, (0.7,0.7,0.7,0.18)) if domain2color is not None else (0.7,0.7,0.7,0.18)
#         print(f"  Sector {i} ({sector_labels[i]}) -> Domain '{dom}' -> Color {col}")
# fill_ring_sector(ax, th0, th1, r0, r1, col, n=256) #


# def polar_bar_chart_v2(ax, sectors, bars_per_sector, heights,
#                        sector_labels=None, bar_labels=None,
#                        sector_colors=None, bar_colors=None,
#                        r_inner=0.2, r_outer=1.0,
#                        bar_width_frac=0.92,
#                        grid_rings=5, grid_color=None,
#                        sector_label_pad=0.02,
# label_fontsize=9, # :8 -> 10
# gene_label_fontsize=9, # gene:7 -> 9
#                        gene_label_pad=0.02,
# # : and in
#                        sector2domain=None, domain2color=None,
#                        middle_band_frac=(0.0, 1.0),
# center_=None, center__fontsize=10, # in :9 -> 11
# # : for of columntable (if and sector_labels different)
#                        sector_keys=None,
# # : (for domaincompute)
#                        sector_angles_list=None):
#     import numpy as np

#     if grid_color is None:
#         grid_color = GRID_COLOR

# ax.set_theta_direction(-1) #
# ax.set_theta_offset(np.pi/2.0) # 0 in
# ax.set_axis_off() #

# # bars_per_sector
#     if isinstance(bars_per_sector, int):
#         bps = [bars_per_sector] * sectors
#     else:
#         bps = list(bars_per_sector)
#     assert len(bps) == sectors
#     T = sum(bps)
#     heights = np.clip(np.asarray(heights, float), 0, 1)
#     assert len(heights) == T
#     if bar_labels is None:
#         bar_labels = [None] * T

# # - Nature:
#     for i in range(grid_rings+1):
#         r = r_inner + (r_outer - r_inner) * i / grid_rings
#         ax.plot(np.linspace(0, 2*np.pi, 400), np.full(400, r), 
#                 color=grid_color, lw=0.4, alpha=0.6, zorder=0)

# # by
#     total_bars = sum(bps)
#     full_angle = 2*np.pi
#     sector_angles = [full_angle * (b / total_bars) if total_bars>0 else full_angle/sectors for b in bps]

# # compute (for)
#     theta = 0.0
#     theta_edges = [0.0]
#     for s in range(sectors):
#         theta += sector_angles[s]
#         theta_edges.append(theta)

# # : each
#     if sector_labels is None:
#         sector_labels = [f"S{s+1}" for s in range(sectors)]
    
# # if of sector_keys, for
#     keys_for_mapping = sector_keys if sector_keys is not None else sector_labels
    
#     draw_sector_middle_band(ax=ax, theta_edges=theta_edges,
#                             r_inner=r_inner, r_outer=r_outer,
#                             sector_labels=keys_for_mapping,
#                             sector2domain=sector2domain,
#                             domain2color=domain2color,
#                             band_frac=middle_band_frac)

# # Startplot
#     theta = 0.0
#     idx = 0
#     sector_centers = []
#     for s in range(sectors):
#         ang = sector_angles[s]
#         bars_s = bps[s]
# gap = ang * (1 - bar_width_frac) #
#         usable = ang - gap
#         bar_ang = usable / bars_s if bars_s>0 else usable
#         t0 = theta + gap/2

#         scolor = None
#         if sector_colors is not None:
#             if isinstance(sector_colors, dict):
#                 scolor = sector_colors.get(sector_labels[s], "#888")
#             else:
#                 scolor = sector_colors[s]

#         for b in range(bars_s):
#             h = float(heights[idx])
#             r0 = r_inner
#             r1 = r_inner + (r_outer - r_inner) * h
#             th0 = t0 + b * bar_ang
#             th1 = th0 + bar_ang * 0.95
#             th_mid = (th0 + th1) / 2.0

#             bcolor = bar_colors[idx] if bar_colors is not None else scolor or NATURE_COLORS['primary'][0]
# Note.
#             ax.bar(x=th_mid, height=r1 - r0, width=(th1 - th0),
#                    bottom=r0, color=bcolor, edgecolor="black", linewidth=0.3, align="center")

# # gene: in ()
#             if bar_labels[idx]:
# # : after of hcompute of actual
# actual_height = r_inner + (r_outer - r_inner) * h # of actual
# pad_r = (r_outer - r_inner) * gene_label_pad * 0.5 #
                
#                 ang_deg = np.degrees(th_mid)
#                 rot = 90 - ang_deg
                
# # :
#                 if -90 <= rot <= 90:
# # : ()
#                     rt = actual_height + pad_r
#                     ha = "left"; va = "center"
#                 else:
# # :
# # 180, in
# rot = rot + 180 #
# rt = actual_height + pad_r #
#                     ha = "right"; va = "center"
                    
# ax.(th_mid, rt, str(bar_labels[idx]),
#                         rotation=rot, rotation_mode="anchor",
#                         ha=ha, va=va, fontsize=gene_label_fontsize, 
#                         color="#2C3E50", weight='normal', alpha=0.9)
#             idx += 1

# # - Nature:
#         ax.plot([theta, theta], [r_inner, r_outer], color=grid_color, lw=0.4, alpha=0.6, zorder=0)

#         sector_centers.append(theta + ang/2)
#         theta += ang
#     ax.plot([theta, theta], [r_inner, r_outer], color=grid_color, lw=0.4, alpha=0.6, zorder=0)

# #, in plot in
# # (,)
    
# # Domainplot, in plot in
# # (description)

# # in - Nature (size and)
# if center_:
# ax.(0.0, 0.0, center_,
#                 ha="center", va="center",
# fontsize=center__fontsize, color="#2C3E50", fontweight="bold",
#                 linespacing=1.3)


# def plot_spot_level_from_csv(base_dir: str,
#                              center_name: str,
#                              gene_view: str = "kv",
#                              metric: str = "mean",
#                              topk: int = 5,
#                              use_topk_file: bool = True,
#                              figsize=(8,8),
#                              r_inner=0.2, r_outer=0.8,
# gene_label_fontsize=9, # default:8 -> 9
#                              save_png: bool = False,
#                              out_png: str = None,
#                              adata=None,
#                              coords_df=None,
#                              center_idx=None,
# show_background: bool = True, # :
# show_legend: bool = True, # :plot
# filter_cross_domain: bool = False, # :domain_aware
# use_domain_aware: bool = False, # :domainTopKfile
# domain_weight: float = 0.6): # :domain (for select file)
#     agg_dir = os.path.join(base_dir, "agg_csv")
#     if gene_view not in ("kv","q"):
# raise ValueError("gene_view for 'kv' or 'q'")
#     if metric not in ("sum","mean"):
# raise ValueError("metric for 'sum' or 'mean'")

# # CSV - :domain_aware > TopK >
#     if use_domain_aware:
# # domainTopKfile
#         weight_suffix = f"w{int(domain_weight*10)}"
#         csv_path = os.path.join(agg_dir, f"spot_level_{gene_view}_top{topk}_domain_aware_{weight_suffix}.csv")
        
#         if not os.path.exists(csv_path):
# print(f"⚠️ not founddomainTopKfile: {csv_path}")
#             print(f"   Please run: python 3.5_prepare_polar_domain_aware.py")
# print(f" to TopKfile...")
#             use_domain_aware = False
#             use_topk_file = True
#         else:
# print(f"✅ domainTopKfile (domain_weight={domain_weight})")
# use_topk_file = True # domain_awarefilecompute of TopK
    
#     if use_topk_file and not use_domain_aware:
# # TopKfile
#         csv_path = os.path.join(agg_dir, f"spot_level_{gene_view}_top{topk}_by_{metric}.csv")
#         if not os.path.exists(csv_path):
#             csv_path = os.path.join(agg_dir, f"spot_level_{gene_view}.csv")
#             use_topk_file = False
# print(f"⚠️ not found TopKfile,")
#     elif not use_topk_file:
# Note.
#         csv_path = os.path.join(agg_dir, f"spot_level_{gene_view}.csv")
# print(f"📊 ")

#     req = ["center_name","neighbor_name","gene","sum","mean"]
#     df = read_csv_or_topk(csv_path, required_cols=req,
#                           topk_in_memory=None if use_topk_file else topk,
#                           by_sum_or_mean=metric,
#                           entity_cols=["center_name","neighbor_name"])

#     dfc = df[df["center_name"].astype(str) == str(center_name)].copy()
#     if dfc.empty:
# raise ValueError(f" in {csv_path} in not found center_name={center_name} of ")

# # if not usedTopKfile, select TopK
#     if use_topk_file is False:
#         dfc["__rank__"] = dfc.groupby(["neighbor_name"])[metric].rank(method="first", ascending=False)
#         dfc = dfc[dfc["__rank__"] <= topk].drop(columns="__rank__")

# # domaingene filtering (and domain_aware)
#     if filter_cross_domain:
# print(f"\ndomaingene filtering (in spot: {center_name})")
# print(f" filter before : {len(dfc)},{dfc['gene'].nunique()} gene")
        
#         dfc = filter_cross_domain_genes(
#             dfc, 
#             neighbor_col="neighbor_name",
#             gene_col="gene",
#             adata=adata,
#             coords_df=coords_df
#         )
        
#         if dfc.empty:
# print(f"Warning: after filtering no gene, all TopKgenedomain")
# raise ValueError(f"filterdomaingene after no ")
        
# print(f" after filtering : {len(dfc)},{dfc['gene'].nunique()} gene")
    
# # description
#     if use_domain_aware:
# print(f"\n💡 Domain-aware:")
# print(f" - domain select similar of gene")
# print(f" - domaingene")
#         if filter_cross_domain:
# print(f" ✅ domainfilter,domaingene")
#         else:
# print(f" ⚠️ filter_cross_domain=True filterdomaingene")

# # in domain
#     center_domain = get_domain_info(center_name, adata, coords_df, spot_idx=center_idx)
# print(f" in spot {center_name} (idx={center_idx}) domain: {center_domain}")

# # (by domain, by - domain of in)
#     neighbor_strength = dfc.groupby("neighbor_name")[metric].sum()
    
# # Get the domain for each neighbor (for)
#     temp_neighbor_domains = {}
#     for neighbor in neighbor_strength.index:
#         neighbor_idx = None
#         try:
#             neighbor_idx = int(neighbor)
#         except (ValueError, TypeError):
#             if coords_df is not None and 'spot_name' in coords_df.columns:
#                 neighbor_row = coords_df[coords_df['spot_name'] == neighbor]
#                 if not neighbor_row.empty and 'spot_idx' in coords_df.columns:
#                     neighbor_idx = neighbor_row['spot_idx'].iloc[0]
#         temp_neighbor_domains[neighbor] = get_domain_info(neighbor, adata, coords_df, spot_idx=neighbor_idx)
    
# # by domain,domain by
#     neighbor_df = pd.DataFrame({
#         'neighbor': neighbor_strength.index,
#         'strength': neighbor_strength.values,
#         'domain': [temp_neighbor_domains[n] for n in neighbor_strength.index]
#     })
# # by domain, in domain by
#     neighbor_df = neighbor_df.sort_values(['domain', 'strength'], ascending=[True, False])
#     sectors = neighbor_df['neighbor'].tolist()
    
# # -> domain
# # from neighbor_namespot_idx (if neighbor_name or can)
#     neighbor_domains = {}
#     for neighbor in sectors:
# # neighbor of spot_idx
#         neighbor_idx = None
#         try:
# # 1: if neighbor_name,
#             neighbor_idx = int(neighbor)
#         except (ValueError, TypeError):
# # 2: from coords_df
#             if coords_df is not None and 'spot_name' in coords_df.columns:
#                 neighbor_row = coords_df[coords_df['spot_name'] == neighbor]
#                 if not neighbor_row.empty and 'spot_idx' in coords_df.columns:
#                     neighbor_idx = neighbor_row['spot_idx'].iloc[0]
        
#         neighbor_domain = get_domain_info(neighbor, adata, coords_df, spot_idx=neighbor_idx)
#         neighbor_domains[neighbor] = neighbor_domain
    
# print(f" in spot {center_name} of domains: {neighbor_domains}")

# # Optimization: different domain different,domain of
#     import matplotlib.colors as mcolors
#     nature_palette = NATURE_COLORS['primary']
    
# # for each unique domain of ()
#     unique_domains = list(dict.fromkeys([neighbor_domains[n] for n in sectors]))
# # domain in all plot in of
#     domain_base_colors = {dom: get_domain_color(dom, nature_palette) 
#                           for dom in unique_domains}
    
# # for neighbors (domain)
# # domain of different of ()
#     sector_base_colors = {}
# domain_neighbor_count = {} # each domain of neighbor count
# domain_neighbor_index = {} # each domain before processing to neighbors
    
#     for n in sectors:
#         dom = neighbor_domains[n]
#         if dom not in domain_neighbor_count:
#             domain_neighbor_count[dom] = sum(1 for s in sectors if neighbor_domains[s] == dom)
#             domain_neighbor_index[dom] = 0
    
#     for n in sectors:
#         dom = neighbor_domains[n]
#         base_color = domain_base_colors[dom]
        
# # if domainneighbors,
#         if domain_neighbor_count[dom] == 1:
#             sector_base_colors[n] = base_color
#         else:
# # domain of different : (from of 75% to 115%)
#             index = domain_neighbor_index[dom]
#             total = domain_neighbor_count[dom]
# # :0.75 to 1.15
#             brightness_factor = 0.75 + (0.4 * index / max(1, total - 1))
            
# # for RGB
#             rgb = mcolors.to_rgb(base_color)
#             adjusted_rgb = tuple(min(1.0, c * brightness_factor) for c in rgb)
#             sector_base_colors[n] = mcolors.to_hex(adjusted_rgb)
            
#             domain_neighbor_index[dom] += 1

#     bars_per_sector, heights, bar_colors, bar_labels = [], [], [], []
# sector_total_attention = [] # each of totalattention

# # (: + domain)
# sector_labels = [] # for (=)
# sector_labels_with_domain = [] # for ()
    
# # build sector2domain,
#     sector2domain = {}
    
#     for n in sectors:
#         neighbor_domain = neighbor_domains[n]
#         is_same_domain = neighbor_domain == center_domain
# sector_labels.append(n) # :, for sector2domain of
        
# # of all gene, after filtertop k (in)
#         sub = dfc[dfc["neighbor_name"] == n].sort_values(metric, ascending=False)
#         vals = sub[metric].to_numpy()
        
# # compute of totalattention
#         total_attn = vals.sum()
#         sector_total_attention.append(total_attn)
        
# # : (domain in)
#         sector_labels_with_domain.append(f"N{n}")
        
# # to domain
#         sector2domain[n] = neighbor_domain
        
#         bars_per_sector.append(len(vals))
#         heights.extend(vals.tolist())
#         glist = sub["gene"].tolist()
#         bar_labels.extend(glist)
        
# # () of all gene
# # in :domain of different
#         bar_colors.extend([sector_base_colors[n]] * len(vals))

# # : all (,)
#     heights = normalize_array(heights)
    
# print(f": (all,)")
# print(f"totalattention: {sector_total_attention}")

# # : for each different of domain different
# # domain -> color (of domain)
#     unique_neighbor_domains = list(pd.Index([neighbor_domains[n] for n in sectors]).unique())
    
# Note.
#     print(f"Debug - Unique neighbor domains: {unique_neighbor_domains}")
#     print(f"Debug - Neighbor domains mapping:")
#     for n, d in neighbor_domains.items():
#         print(f"  Neighbor {n} -> Domain {d}")
    
# # :parameters
#     if show_background:
# # Nature for different domain (of alpha, and domain-level)
#         domain2rgba = build_domain_colormap(unique_neighbor_domains, palette='primary', alpha=0.15)
# # (and domain-level): in in 30%-70%
#         band_frac = (0.30, 0.60)
#     else:
#         domain2rgba = None
#         sector2domain = None
#         band_frac = (0.0, 0.0)

# # in (,)
# center_ = f"Center Spot\n{center_name}\n Domain {center_domain}"

#     fig = plt.figure(figsize=figsize, facecolor='white')
#     ax = plt.subplot(111, projection="polar")
#     polar_bar_chart_v2(ax,
#                        sectors=len(sectors),
#                        bars_per_sector=bars_per_sector,
#                        heights=heights,
#                        sector_labels=sector_labels_with_domain,
#                        bar_labels=bar_labels,
#                        bar_colors=bar_colors,
#                        r_inner=r_inner, r_outer=r_outer,
# bar_width_frac=0.92, # and domain-level
#                        grid_rings=5,
# sector_label_pad=0.02, # and domain-level
# label_fontsize=9, #
#                        gene_label_fontsize=gene_label_fontsize,
# gene_label_pad=0.02, # and domain-level
#                        sector2domain=sector2domain if show_background else None,
#                        domain2color=domain2rgba,
# middle_band_frac=band_frac, #
# center_=center_,
# center__fontsize=10, # in
#                        sector_keys=sector_labels)

# # plot:optional (parameters)
#     if show_legend:
#         from matplotlib.patches import Patch
        
#         handles = []
#         legend_labels = []
        
# # all unique domains ()
#         unique_domains_ordered = list(dict.fromkeys([neighbor_domains[n] for n in sectors]))
        
# # :Domains
#         legend_labels.append("═══ Domains ═══")
# handles.append(plt.Line2D([0],[0], color='none')) #
        
#         for dom in unique_domains_ordered:
#             dom_color = domain_base_colors[dom]
#             handles.append(Patch(facecolor=dom_color, edgecolor='white', linewidth=1.5, alpha=0.9))
#             legend_labels.append(f"Domain {dom}")
        
# Note.
#         legend_labels.append("")
#         handles.append(plt.Line2D([0],[0], color='none'))
        
# # :Neighbors (by domain, in)
#         legend_labels.append("═══ Neighbors ═══")
#         handles.append(plt.Line2D([0],[0], color='none'))
        
#         for dom in unique_domains_ordered:
#             domain_neighbors = [n for n in sectors if neighbor_domains[n] == dom]
#             for n in domain_neighbors:
#                 handles.append(plt.Line2D([0],[0], color=sector_base_colors[n], lw=5, solid_capstyle='butt'))
#                 legend_labels.append(f"N{n} ({dom})")
        
#         legend = plt.legend(handles, legend_labels, bbox_to_anchor=(0.95, 1.0), loc="upper left",
# title=None, # "Legend"
# fontsize=9, frameon=True, # plot:9 -> 10
#                             fancybox=False, shadow=False, edgecolor='#DDDDDD',
# handlelength=1.5, handletextpad=0.6, labelspacing=0.8,
#                             borderpad=0.4, columnspacing=1.0)
# legend.get_frame().set_linewidth(0.8) #
#     plt.tight_layout()

#     if save_png:
#         if out_png is None:
#             out_png = f"spot_level_{gene_view}_top{topk}_by_{metric}_{center_name}.png"
#         plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
#         print(f"Savedimage to : {out_png}")
#     plt.close()

# # ---------------- Domain-level (global, kv, top K=5) ----------------

# def plot_domain_level_kv_global_topk_from_csv(base_dir: str,
#                                               metric: str = "mean",
#                                               topk: int = 5,
#                                               figsize=(8,8),
#                                               r_inner=0.2, r_outer=1.0,
# gene_label_fontsize=9, # default:8 -> 9
#                                               save_png: bool = False,
#                                               out_png: str = None,
#                                               out_dir: str = None,
# show_background: bool = False): # :
#     agg_dir = os.path.join(base_dir, "agg_csv")
#     if metric not in ("sum","mean"):
# raise ValueError("metric for 'sum' or 'mean'")

#     csv_topk = os.path.join(agg_dir, f"domain_level_kv_global_top{topk}_by_{metric}.csv")
#     if os.path.exists(csv_topk):
#         df = pd.read_csv(csv_topk)
#         req = ["domain","gene","sum","mean"]
#         miss = [c for c in req if c not in df.columns]
#         if miss:
# raise ValueError(f"{csv_topk} column: {miss}")
#         dfg = df.copy()
#     else:
#         csv_full = os.path.join(agg_dir, "domain_level_kv_global.csv")
#         req = ["domain","gene","sum","mean"]
#         dfall = read_csv_or_topk(csv_full, required_cols=req)
#         dfall["__rank__"] = dfall.groupby("domain")[metric].rank(method="first", ascending=False)
#         dfg = dfall[dfall["__rank__"] <= topk].drop(columns="__rank__").copy()

# Note.
#     sector_order = dfg.groupby("domain")[metric].sum().sort_values(ascending=False).index.tolist()
#     sectors = sector_order

# # of : each domain of
# # for each domain different of, by (:0, 2, 4, 6, 8, 10)
#     nature_palette = NATURE_COLORS['primary']
    
# # createdomain to of, each domain
#     domain_base_colors = {}
# available_colors = [nature_palette[i] for i in range(0, len(nature_palette), 2)] #
    
#     for idx, dom in enumerate(sectors):
#         if dom in GLOBAL_DOMAIN_COLOR_MAP:
# # of
#             color_idx = GLOBAL_DOMAIN_COLOR_MAP[dom]
#             domain_base_colors[dom] = available_colors[color_idx % len(available_colors)]
#         else:
# # by,
#             domain_base_colors[dom] = available_colors[idx % len(available_colors)]
    
#     import matplotlib.colors as mcolors
    
#     bars_per_sector = []
#     heights = []
#     bar_colors = []
#     bar_labels = []

#     for dom in sectors:
#         sub = dfg[dfg["domain"] == dom].sort_values(metric, ascending=False)
#         vals = sub[metric].to_numpy()
#         bars_per_sector.append(len(vals))
#         heights.extend(vals.tolist())
#         glist = sub["gene"].tolist()
#         bar_labels.extend(glist)
        
# # domain of all gene
#         base_color = domain_base_colors[dom]
#         n_genes = len(glist)
        
# # for domain of all gene (not used)
#         for i in range(n_genes):
#             bar_colors.append(base_color)

#     heights = normalize_array(heights)

# # - Domain-leveldefault
#     if show_background:
#         domain2rgba = build_domain_colormap(sectors, palette='primary', alpha=0.08)
#         sector2domain = {dom: dom for dom in sectors}
#     else:
#         domain2rgba = None
#         sector2domain = None

# center_ = "Domains\n Top genes"

#     fig = plt.figure(figsize=figsize, facecolor='white')
#     ax = plt.subplot(111, projection="polar")
#     polar_bar_chart_v2(ax,
#                        sectors=len(sectors),
#                        bars_per_sector=bars_per_sector,
#                        heights=heights,
#                        sector_labels=sectors,
#                        bar_labels=bar_labels,
#                        bar_colors=bar_colors,
#                        r_inner=r_inner, r_outer=r_outer,
#                        bar_width_frac=0.92,
#                        grid_rings=5,
#                        sector_label_pad=0.02,
# label_fontsize=9, #
#                        gene_label_fontsize=gene_label_fontsize,
#                        gene_label_pad=0.02,
#                        sector2domain=sector2domain,
#                        domain2color=domain2rgba,
#                        middle_band_frac=(0.30, 0.70),
# center_=center_,
# center__fontsize=10) # in

# # plot: each Domain of
#     from matplotlib.patches import Patch
    
#     handles = []
#     legend_labels = []
    
# # for each domain createplot
#     for dom in sectors:
#         dom_color = domain_base_colors[dom]
#         handles.append(Patch(facecolor=dom_color, edgecolor='white', linewidth=1.5, alpha=0.9))
#         legend_labels.append(f"{dom}")
    
#     legend = plt.legend(handles, legend_labels, bbox_to_anchor=(0.92, 0.9), loc="upper left",
#                         title="",
# title_fontsize=9, # plot
# fontsize=14, frameon=True, # plot:9 -> 10
#                         fancybox=False, shadow=False, edgecolor='#DDDDDD',
# handlelength=1.5, handletextpad=0.8, labelspacing=0.8,
#                         borderpad=0.4, columnspacing=1.0)
#     legend.get_frame().set_linewidth(0.8)
    

#     plt.tight_layout()

#     if save_png:
# # if Output directory, in
#         if out_dir is not None:
#             os.makedirs(out_dir, exist_ok=True)
            
#         if out_png is None:
#             filename = f"domain_level_kv_global_top{topk}_by_{metric}.png"
# # if Output directory,file and directory
#             if out_dir is not None:
#                 out_png = os.path.join(out_dir, filename)
#             else:
#                 out_png = filename
                
#         plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
#         print(f"Savedimage to : {out_png}")
#     plt.close()

# def plot_domain_level_kv_domain_aware(base_dir: str,
#                                       metric: str = "mean",
#                                       topk: int = 10,
#                                       domain_weight: float = 0.6,
#                                       figsize=(10,10),
#                                       r_inner=0.2, r_outer=1.0,
#                                       gene_label_fontsize=9,
#                                       save_png: bool = False,
#                                       out_png: str = None,
#                                       out_dir: str = None,
#                                       show_background: bool = False,
#                                       adata=None,
#                                       coords_df=None):
#     """
# domain aware of spot level,aggregationdomain levelplot
    
# :
# 1. Readingdomain aware of spot level TopK
# 2. by domainaggregation,compute each gene in domain in of total and /
#     3.  for  each domain select TopKgene
# 4. plotdomain levelplot
    
#     Parameters:
#         base_dir: base directory
#         metric: aggregationmetrics ("mean"  or  "sum")
# topk: each domain of TopKgene count
# domain_weight: domain (for select of domain awarefile)
# parameters plot_domain_level_kv_global_topk_from_csv
#     """
#     agg_dir = os.path.join(base_dir, "agg_csv")
#     if metric not in ("sum","mean"):
# raise ValueError("metric for 'sum' or 'mean'")
    
# # Readingdomain aware of spot level
#     weight_suffix = f"w{int(domain_weight*10)}"
#     spot_csv = os.path.join(agg_dir, f"spot_level_kv_top{topk}_domain_aware_{weight_suffix}.csv")
    
#     if not os.path.exists(spot_csv):
#         raise FileNotFoundError(f"not founddomain awarefile: {spot_csv}\nPlease run 3.5_prepare_polar_domain_aware.py")
    
#     print(f"\n{'='*80}")
# print(f"📊 Domain Level aggregation (Domain Aware)")
#     print(f"{'='*80}")
#     print(f"Input: {spot_csv}")
# print(f"Domain: {domain_weight}")
#     print(f"aggregationmetrics: {metric}")
#     print(f" each domain TopK: {topk}")
    
# # Readingspot level
#     df_spot = pd.read_csv(spot_csv)
# print(f"\nrecord count: {len(df_spot)}")
# print(f"center spots: {df_spot['center_name'].nunique()}")
# print(f"spots: {df_spot['neighbor_name'].nunique()}")
    
# # each neighbor of domain
#     if adata is None or coords_df is None:
#         adata, coords_df = load_metadata(base_dir)
    
# # for each neighbordomain
# print("\n of domain...")
#     neighbor_domains = {}
#     for neighbor in df_spot['neighbor_name'].unique():
#         neighbor_idx = None
#         try:
#             neighbor_idx = int(neighbor)
#         except (ValueError, TypeError):
#             if coords_df is not None and 'spot_name' in coords_df.columns:
#                 neighbor_row = coords_df[coords_df['spot_name'] == str(neighbor)]
#                 if not neighbor_row.empty and 'spot_idx' in coords_df.columns:
#                     neighbor_idx = neighbor_row['spot_idx'].iloc[0]
        
#         domain = get_domain_info(str(neighbor), adata=adata, coords_df=coords_df, spot_idx=neighbor_idx)
#         neighbor_domains[neighbor] = domain
    
# # domaincolumn
#     df_spot['neighbor_domain'] = df_spot['neighbor_name'].map(neighbor_domains)
    
# # filterUnknown domain
#     df_spot = df_spot[df_spot['neighbor_domain'] != 'Unknown'].copy()
#     print(f"filterUnknown after record count: {len(df_spot)}")
    
#     domain_counts = df_spot['neighbor_domain'].value_counts()
# print(f"\n Domain:")
#     for domain, count in domain_counts.items():
# print(f" {domain}: {count} ")
    
#     #  by domain and geneaggregation
#     print(f"\n by domain and geneaggregation...")
#     if metric == "mean":
# # mean,compute
# # sumcolumnattention of total and,countcolumnrecord count
#         df_agg = df_spot.groupby(['neighbor_domain', 'gene']).agg({
#             'sum': 'sum',
#             'count': 'sum'
#         }).reset_index()
#         df_agg['mean'] = df_agg['sum'] / df_agg['count']
#     else:  # sum
#         df_agg = df_spot.groupby(['neighbor_domain', 'gene']).agg({
#             'sum': 'sum',
#             'count': 'sum'
#         }).reset_index()
#         df_agg['mean'] = df_agg['sum'] / df_agg['count']
    
#     df_agg.rename(columns={'neighbor_domain': 'domain'}, inplace=True)
    
#     print(f"aggregation after record count: {len(df_agg)}")
# print(f"domains: {df_agg['domain'].nunique()}")
# print(f"genes: {df_agg['gene'].nunique()}")
    
#     #  for  each domain select TopKgene
#     print(f"\n for  each domain select Top{topk}gene...")
#     df_agg["__rank__"] = df_agg.groupby("domain")[metric].rank(method="first", ascending=False)
#     dfg = df_agg[df_agg["__rank__"] <= topk].drop(columns="__rank__").copy()
    
#     print(f" select TopK after record count: {len(dfg)}")
    
#     #  each domain of TopKgene
# print(f"\ndomain of Top{min(5, topk)}gene:")
#     for domain in dfg['domain'].unique():
#         domain_genes = dfg[dfg['domain'] == domain].nlargest(min(5, topk), metric)['gene'].tolist()
#         print(f"  {domain}: {domain_genes}")
    
# Note.
#     sector_order = dfg.groupby("domain")[metric].sum().sort_values(ascending=False).index.tolist()
#     sectors = sector_order

# # of : each domain of
# # for each domain different of, by (:0, 2, 4, 6, 8, 10)
#     nature_palette = NATURE_COLORS['primary']
    
# # createdomain to of, each domain
#     domain_base_colors = {}
# available_colors = [nature_palette[i] for i in range(0, len(nature_palette), 2)] #
    
#     for idx, dom in enumerate(sectors):
#         if dom in GLOBAL_DOMAIN_COLOR_MAP:
# # of
#             color_idx = GLOBAL_DOMAIN_COLOR_MAP[dom]
#             domain_base_colors[dom] = available_colors[color_idx % len(available_colors)]
#         else:
# # by,
#             domain_base_colors[dom] = available_colors[idx % len(available_colors)]
    
#     import matplotlib.colors as mcolors
    
#     bars_per_sector = []
#     heights = []
#     bar_colors = []
#     bar_labels = []

#     for dom in sectors:
#         sub = dfg[dfg["domain"] == dom].sort_values(metric, ascending=False)
#         vals = sub[metric].to_numpy()
#         bars_per_sector.append(len(vals))
#         heights.extend(vals.tolist())
#         glist = sub["gene"].tolist()
#         bar_labels.extend(glist)
        
# # domain of all gene
#         base_color = domain_base_colors[dom]
#         n_genes = len(glist)
        
# # for domain of all gene (not used)
#         for i in range(n_genes):
#             bar_colors.append(base_color)

#     heights = normalize_array(heights)

# Note.
#     if show_background:
#         domain2rgba = build_domain_colormap(sectors, palette='primary', alpha=0.08)
#         sector2domain = {dom: dom for dom in sectors}
#     else:
#         domain2rgba = None
#         sector2domain = None

# center_ = "Domains\n(Domain Aware)\n Top genes"

#     fig = plt.figure(figsize=figsize, facecolor='white')
#     ax = plt.subplot(111, projection="polar")
#     polar_bar_chart_v2(ax,
#                        sectors=len(sectors),
#                        bars_per_sector=bars_per_sector,
#                        heights=heights,
#                        sector_labels=sectors,
#                        bar_labels=bar_labels,
#                        bar_colors=bar_colors,
#                        r_inner=r_inner, r_outer=r_outer,
#                        bar_width_frac=0.92,
#                        grid_rings=5,
#                        sector_label_pad=0.02,
#                        label_fontsize=9,
#                        gene_label_fontsize=gene_label_fontsize,
#                        gene_label_pad=0.02,
#                        sector2domain=sector2domain,
#                        domain2color=domain2rgba,
#                        middle_band_frac=(0.30, 0.70),
# center_=center_,
# center__fontsize=10)

# # plot
#     from matplotlib.patches import Patch
    
#     handles = []
#     legend_labels = []
    
#     for dom in sectors:
#         dom_color = domain_base_colors[dom]
#         handles.append(Patch(facecolor=dom_color, edgecolor='white', linewidth=1.5, alpha=0.9))
#         legend_labels.append(f"{dom}")
    
#     legend = plt.legend(handles, legend_labels, bbox_to_anchor=(0.92, 0.9), loc="upper left",
#                         title="",
#                         title_fontsize=9,
#                         fontsize=14, frameon=True,
#                         fancybox=True, shadow=False,
#                         edgecolor='gray', framealpha=0.9)
#     legend.get_frame().set_linewidth(1.5)

#     if save_png:
#         if out_png is None:
#             filename = f"domain_level_kv_domain_aware_w{int(domain_weight*10)}_top{topk}_by_{metric}.png"
#             if out_dir is not None:
#                 out_png = os.path.join(out_dir, filename)
#             else:
#                 out_png = filename
                
#         plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
#         print(f"\n✅ Savedimage to : {out_png}")
#         print(f"{'='*80}\n")
#     plt.close()

# # ---------------- ----------------
# if __name__ == "__main__":
#     base_dir = "./PDAC/whole_slice_data_20251028_173836"
#     boundary_dir = "./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_boundary"
#     adata, coords_df = load_adata_and_coords(base_dir)
    
#     boundary_spots_indices = load_boundary_spots(boundary_dir)
#     print(f" found  {len(boundary_spots_indices)}  boundary spots")
    
# # Saveboundary spots of and of
#     boundary_spots_names = []
#     boundary_spots_idx_map = {}  # spot_name -> spot_idx
#     for spot_idx in boundary_spots_indices:
#         spot_name = get_spot_name_from_index(spot_idx, adata, coords_df)
#         boundary_spots_names.append(spot_name)
#         boundary_spots_idx_map[spot_name] = spot_idx
    
# print(f"boundary spots: {boundary_spots_names[:5]}...")
# print(f"boundary spots: {list(boundary_spots_idx_map.items())[:5]}...")

#     # 1) Domain-level
# # createdomain-levelplot of file
#     domain_plots_dir = "./PDAC/domain_polar_plots"
#     os.makedirs(domain_plots_dir, exist_ok=True)
# print(f"domain-levelplot,save: {domain_plots_dir}")
    
#     plot_domain_level_kv_global_topk_from_csv(base_dir,
#                                               metric="mean",
#                                               topk=10,
#                                               figsize=(10,10),
#                                               r_inner=0.2, r_outer=1.0,
# gene_label_fontsize=10, # gene
#                                               save_png=True,
#                                               out_dir=domain_plots_dir,
# show_background=False) # Domain-level

# # 2) Spot-level - (all gene)
#     if boundary_spots_names:
#         center_spot = boundary_spots_names[0]
#         center_spot_idx = boundary_spots_idx_map.get(center_spot)
# print(f"spot for in : {center_spot} (idx={center_spot_idx})")
        
# print("\n=== : all topgene ===")
#         plot_spot_level_from_csv(base_dir,
#                                  center_name=center_spot,
#                                  gene_view="kv",
#                                  metric="mean",
#                                  topk=30,
#                                  use_topk_file=True,
#                                  figsize=(10,10),
#                                  r_inner=0.2, r_outer=1.0,
# gene_label_fontsize=9, # gene
#                                  save_png=False,
#                                  adata=adata,
#                                  coords_df=coords_df,
#                                  center_idx=center_spot_idx,
#                                  show_background=True,
#                                  show_legend=True,
# filter_cross_domain=False) # filter
        
# # 3) Spot-level - Domain-aware ()
# print("\n=== Domain-aware:domain select similar gene ===")
#         plot_spot_level_from_csv(base_dir,
#                                  center_name=center_spot,
#                                  gene_view="kv",
#                                  metric="mean",
#                                  topk=30,
#                                  use_topk_file=True,
#                                  figsize=(10,10),
#                                  r_inner=0.2, r_outer=1.0,
#                                  gene_label_fontsize=9,
#                                  save_png=False,
#                                  adata=adata,
#                                  coords_df=coords_df,
#                                  center_idx=center_spot_idx,
#                                  show_background=True,
#                                  show_legend=True,
# use_domain_aware=True, # domainTopK
# domain_weight=0.6) # domain (0.5-0.7)
        
#         if len(boundary_spots_names) > 1:
# # createspotplot of file
#             boundary_plots_dir = "./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_boundary/"
#             os.makedirs(boundary_plots_dir, exist_ok=True)
# print(f" for before 3 boundary spotsplot,save: {boundary_plots_dir}")
            
#             for i, spot_name in enumerate(boundary_spots_names[:]):
#                 spot_idx = boundary_spots_idx_map.get(spot_name)
# print(f"processingspot {i+1}/{min(10, len(boundary_spots_names))}: {spot_name} (idx={spot_idx})")
                
# # of get_domain_infodomain
#                 domain = get_domain_info(spot_name, adata, coords_df, spot_idx=spot_idx)
#                 print(f"Spot {spot_name} (idx={spot_idx})  of domain: {domain}")
                
# # buildOutputfile path (in file in)
#                 out_file = os.path.join(boundary_plots_dir, f"boundary_spot_{i+1}_{spot_idx}_polar.png")
                
#                 plot_spot_level_from_csv(base_dir,
#                                          center_name=spot_name,
#                                          gene_view="kv",
#                                          metric="mean",
#                                          topk=10,
#                                          use_topk_file=True,
#                                          figsize=(10,10),
#                                          r_inner=0.2, r_outer=1.0,
# gene_label_fontsize=8, # gene
#                                          save_png=True,
#                                          out_png=out_file,
#                                          adata=adata,
#                                          coords_df=coords_df,
#                                          center_idx=spot_idx,
#                                          show_background=True,
#                                          show_legend=True,
# filter_cross_domain=False) #
            
# # 4) Domain-aware+filterplot
# print(f"\n=== Domain-aware+filterplot ===")
#             domain_aware_plots_dir = "./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_domain_aware_filtered/"
#             os.makedirs(domain_aware_plots_dir, exist_ok=True)
# print(f"Domain-aware+filterplot,save: {domain_aware_plots_dir}")
# print(f": domainTopK select + domaingene filtering")
            
#             for i, spot_name in enumerate(boundary_spots_names[:]):  #  before 10 boundary spots
#                 spot_idx = boundary_spots_idx_map.get(spot_name)
# print(f"\nprocessingspot {i+1}/10: {spot_name} (idx={spot_idx})")
                
#                 domain = get_domain_info(spot_name, adata, coords_df, spot_idx=spot_idx)
#                 print(f"  Spot {spot_name} (idx={spot_idx})  of domain: {domain}")
                
#                 # buildOutputfile path
#                 out_file = os.path.join(domain_aware_plots_dir, f"domain_aware_filtered_spot_{i+1}_{spot_idx}_polar.png")
                
#                 try:
#                     plot_spot_level_from_csv(base_dir,
#                                              center_name=spot_name,
#                                              gene_view="kv",
#                                              metric="mean",
#                                              topk=10,
#                                              use_topk_file=True,
#                                              figsize=(10,10),
#                                              r_inner=0.2, r_outer=1.0,
#                                              gene_label_fontsize=8,
#                                              save_png=True,
#                                              out_png=out_file,
#                                              adata=adata,
#                                              coords_df=coords_df,
#                                              center_idx=spot_idx,
#                                              show_background=True,
#                                              show_legend=True,
# use_domain_aware=True, # domainTopK
# domain_weight=0.6, # domain
# filter_cross_domain=True) # filterdomaingene
#                     print(f"  ✓ Saved: {out_file}")
#                 except Exception as e:
#                     print(f"  ✗ Skippingspot {spot_name}: {str(e)}")
#                     continue
#     else:
# print("Warning: not foundboundary spots,defaultspot")
    
# # 5) Domain Levelplot (Domain Aware) - rows,boundary spots
#     print(f"\n{'='*80}")
# print(f"=== Domain Levelplot (Domain Aware) ===")
#     print(f"{'='*80}")
#     domain_plots_dir = "./PDAC/domain_polar_plots_domain_aware/"
#     os.makedirs(domain_plots_dir, exist_ok=True)
    
#     # Test different  of TopK and domain_weight
#     for topk_val in [5, 10]:
# for weight in [0.6]: # can Test multiple : [0.5, 0.6, 0.7]
#             print(f"\n{'#'*80}")
#             print(f"# TopK={topk_val}, Domain Weight={weight}")
#             print(f"{'#'*80}")
#             try:
#                 plot_domain_level_kv_domain_aware(
#                     base_dir=base_dir,
#                     metric="mean",
#                     topk=topk_val,
#                     domain_weight=weight,
#                     figsize=(12, 12),
#                     r_inner=0.2,
#                     r_outer=1.0,
#                     gene_label_fontsize=10,
#                     save_png=True,
#                     out_dir=domain_plots_dir,
#                     show_background=False,
#                     adata=adata,
#                     coords_df=coords_df
#                 )
#             except Exception as e:
# print(f" ✗ failed: {str(e)}")
#                 import traceback
#                 traceback.print_exc()
    
#     print(f"\n{'='*80}")
# print(f"✅ Domain Levelplotsaved: {domain_plots_dir}")
#     print(f"{'='*80}")
    
# # 6) from each domain in random sample 5 spotplotplot
#     print(f"\n{'='*80}")
# print(f"=== from each Domain random sample Spotsplotplot ===")
#     print(f"{'='*80}")
    
#     random_spots_dir = "./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_random_by_domain/"
#     os.makedirs(random_spots_dir, exist_ok=True)
    
# # all domain of spots
#     domain_to_spots = None
    
# # from coords_dfdomain
#     if coords_df is not None:
#         domain_col = None
#         if 'cluster' in coords_df.columns:
#             domain_col = 'cluster'
#         elif 'domain' in coords_df.columns:
#             domain_col = 'domain'
        
#         if domain_col is not None:
# print(f" from coords_df of '{domain_col}'columndomain")
#             domain_to_spots = {}
#             for idx, row in coords_df.iterrows():
#                 domain = str(row[domain_col])
#                 spot_idx = int(row['spot_idx']) if 'spot_idx' in row else idx
#                 spot_name = row['spot_name'] if 'spot_name' in row else str(spot_idx)
                
#                 if domain not in domain_to_spots:
#                     domain_to_spots[domain] = []
#                 domain_to_spots[domain].append((spot_name, spot_idx))
    
# # if coords_df no domain, from adata
#     if domain_to_spots is None and adata is not None and 'ground_truth' in adata.obs.columns:
# print(f" from adata.obs of 'ground_truth'columndomain")
#         domain_to_spots = {}
#         for spot_idx in range(len(adata.obs)):
#             spot_name = adata.obs.index[spot_idx]
#             domain = str(adata.obs.loc[spot_name, 'ground_truth'])
#             if domain not in domain_to_spots:
#                 domain_to_spots[domain] = []
#             domain_to_spots[domain].append((spot_name, spot_idx))
    
# # if no,
#     if domain_to_spots is None:
# print("✗ domain,Skipping Step")
# print(" coords_df'cluster' or 'domain'column, or adata.obs'ground_truth'column")
    
#     if domain_to_spots:
#         import random
# random.seed(42) # random can
        
#         print(f"\nFindings {len(domain_to_spots)}  domain:")
#         for domain, spots in domain_to_spots.items():
#             print(f"  - {domain}: {len(spots)} spots")
        
#         #  for  each domain random  sample 5 spot
#         for domain, spots in domain_to_spots.items():
#             print(f"\n{'#'*80}")
#             print(f"# Domain: {domain}")
#             print(f"{'#'*80}")
            
# # createdomain of file
#             domain_subdir = os.path.join(random_spots_dir, f"domain_{domain.replace(' ', '_')}")
#             os.makedirs(domain_subdir, exist_ok=True)
            
# # random sample 5 spot (if domain of spot count5,all)
#             n_samples = min(5, len(spots))
#             sampled_spots = random.sample(spots, n_samples)
            
#             print(f" from domain '{domain}'  in  random  sample  {n_samples}  spots:")
            
#             for i, (spot_name, spot_idx) in enumerate(sampled_spots):
#                 print(f"\n  [{i+1}/{n_samples}] Spot: {spot_name} (idx={spot_idx})")
                
#                 # buildOutputfile path
#                 out_file = os.path.join(domain_subdir, f"spot_{spot_name}_{spot_idx}_polar.png")
                
#                 try:
#                     plot_spot_level_from_csv(
#                         base_dir,
#                         center_name=spot_name,
#                         gene_view="kv",
#                         metric="mean",
#                         topk=10,
#                         use_topk_file=True,
#                         figsize=(10, 10),
#                         r_inner=0.2,
#                         r_outer=1.0,
#                         gene_label_fontsize=8,
#                         save_png=True,
#                         out_png=out_file,
#                         adata=adata,
#                         coords_df=coords_df,
#                         center_idx=spot_idx,
#                         show_background=True,
#                         show_legend=True,
# use_domain_aware=True, # domainTopK
# domain_weight=0.6, # domain
# filter_cross_domain=True # filterdomaingene
#                     )
#                     print(f"    ✓ Saved: {out_file}")
#                 except Exception as e:
#                     print(f"    ✗ Skippingspot {spot_name}: {str(e)}")
#                     continue
        
#         print(f"\n{'='*80}")
# print(f"✅ random spotplotsaved: {random_spots_dir}")
#         print(f"{'='*80}\n")
#     else:
# print("✗ domain-spot,Skipping random Step\n")





# (for of rows)
try:
    from .utils_5_plot_polar import *
except ImportError:
    # if failed, (when running the script directly)
    from utils_5_plot_polar import *




# ---------------- ----------------
if __name__ == "__main__":
    base_dir = "./PDAC/whole_slice_data_20251028_173836"
    boundary_dir = base_dir + "/spatial_attention_visualizer_boundary"
    # boundary_dir = "./HBRC/whole_slice_data_20251102_141320/spatial_attention_visualizer_boundary"
    adata, coords_df = load_adata_and_coords(base_dir)
    
    boundary_spots_indices = load_boundary_spots(boundary_dir)
    print(f" found  {len(boundary_spots_indices)}  boundary spots")
    
    # Saveboundary spots of and of
    boundary_spots_names = []
    boundary_spots_idx_map = {}  # spot_name -> spot_idx
    for spot_idx in boundary_spots_indices:
        spot_name = get_spot_name_from_index(spot_idx, adata, coords_df)
        boundary_spots_names.append(spot_name)
        boundary_spots_idx_map[spot_name] = spot_idx
    
    print(f"boundary spots: {boundary_spots_names[:5]}...")
    print(f"boundary spots: {list(boundary_spots_idx_map.items())[:5]}...")



    # 2) Spot-level - Domain-aware+filterplot
    if boundary_spots_names:

        if len(boundary_spots_names) > 1:
            # 4) Domain-aware+filterplot
            print(f"\n=== Domain-aware+filterplot ===")
            domain_aware_plots_dir = base_dir + "/spatial_attention_visualizer_domain_aware_filtered1/"
            # domain_aware_plots_dir = "./HBRC/whole_slice_data_20251102_141320/spatial_attention_visualizer_domain_aware_filtered1/"
            os.makedirs(domain_aware_plots_dir, exist_ok=True)
            print(f"Domain-aware+filterplot,save: {domain_aware_plots_dir}")
            print(f": domainTopK select + domaingene filtering")
            
            for i, spot_name in enumerate(boundary_spots_names[:]):  #  before 10 boundary spots
                spot_idx = boundary_spots_idx_map.get(spot_name)
                print(f"\nprocessingspot {i+1}/10: {spot_name} (idx={spot_idx})")
                
                domain = get_domain_info(spot_name, adata, coords_df, spot_idx=spot_idx)
                print(f"  Spot {spot_name} (idx={spot_idx})  of domain: {domain}")
                
                # buildOutputfile path
                out_file = os.path.join(domain_aware_plots_dir, f"domain_aware_filtered_spot_{i+1}_{spot_idx}_polar.png")
                
                try:
                    plot_spot_level_from_csv(base_dir,
                                             center_name=spot_name,
                                             gene_view="kv",
                                             metric="mean",
                                             topk=5,
                                             use_topk_file=True,
                                             figsize=(10,10),
                                             r_inner=0.3, r_outer=1.0,
                                             gene_label_fontsize=14,
                                             save_png=True,
                                             out_png=out_file,
                                             adata=adata,
                                             coords_df=coords_df,
                                             center_idx=spot_idx,
                                             show_background=True,
                                             show_legend=True,
                                             use_domain_aware=True,       # domainTopK
                                             domain_weight=0.6,           # domain
                                             filter_cross_domain=True)    # filterdomaingene
                    print(f"  ✓ Saved: {out_file}")
                except Exception as e:
                    print(f"  ✗ Skippingspot {spot_name}: {str(e)}")
                    continue
    else:
        print("Warning: not foundboundary spots,defaultspot")
    
    # # 5) Domain Levelplot (Domain Aware) - rows,boundary spots
    # print(f"\n{'='*80}")
    # print(f"=== Domain Levelplot (Domain Aware) ===")
    # print(f"{'='*80}")
    # domain_plots_dir = base_dir + "/domain_polar_plots_domain_aware1/"
    # os.makedirs(domain_plots_dir, exist_ok=True)
    
    # # Test different  of TopK and domain_weight
    # for topk_val in [5, 10]:
    #     try:
    #         plot_domain_level_kv_domain_aware(
    #             base_dir=base_dir,
    #             metric="sum",
    #             topk=topk_val,
    #             domain_weight=0.7,
    #             figsize=(12, 12),
    #             r_inner=0.3,
    #             r_outer=1.0,
    #             gene_label_fontsize=12,
    #             save_png=True,
    #             out_dir=domain_plots_dir,
    #             show_background=False,
    #             adata=adata,
    #             coords_df=coords_df
    #         )
    #     except Exception as e:
    # print(f" ✗ failed: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    
    # print(f"\n{'='*80}")
    # print(f"✅ Domain Levelplotsaved: {domain_plots_dir}")
    # print(f"{'='*80}")




