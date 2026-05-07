# # -*- coding: utf-8 -*-
# """
# 极坐标绘图（Nature风格）：
# - Nature期刊标准配色与审美
# - 高对比度、清晰的可视化
# - 专业的字体与布局
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from matplotlib import colormaps as cmaps
# import json
# import scanpy as sc

# # Nature风格配置
# plt.rcParams.update({
#     'font.family': 'sans-serif',
#     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
#     'font.size': 10,  # 增大基础字体：8 -> 10
#     'axes.linewidth': 0.5,
#     'xtick.major.width': 0.5,
#     'ytick.major.width': 0.5,
#     'xtick.major.size': 2.5,
#     'ytick.major.size': 2.5,
#     'legend.frameon': False,
#     'legend.fontsize': 9,  # 增大图例字体：7 -> 9
#     'figure.dpi': 300,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.1,
#     'pdf.fonttype': 42,  # TrueType字体，Nature要求
#     'ps.fonttype': 42,
# })

# # Nature推荐的配色方案 - 多色系实色配色（支持多domain）
# NATURE_COLORS = {
#     # 主色调：扩展到12大色系 x 2梯度 = 24色，支持12个domain
#     # 每个色系有明显的深浅对比（类似sequential的对比度）
#     # 使用饱和、实在的颜色，不透不淡
#     'primary': [
#         # === 第一组：原有6色系 ===
#         # 紫色系（Domain 0）- 优雅神秘
#         '#5D4A8E',  # 深紫（明显更深）
#         '#8E6FC9',  # 中紫（饱和实色）
        
#         # 金黄色系（Domain 1）- 温暖醇厚
#         '#D4A870',  # 金黄（饱和实色）
#         '#EAD7B3',  # 浅金（明显更浅）
        
#         # 青蓝色系（Domain 2）- 清新明快
#         '#3C5C95',  # 深青（明显更深）
#         '#5BA3C7',  # 青蓝（饱和实色）
        
#         # 珊瑚红系（Domain 3）- 温暖活力
#         '#B85054',  # 深珊瑚（明显更深）
#         '#E87B8A',  # 珊瑚红（饱和实色）
        
#         # 翠绿色系（Domain 4）- 生机自然
#         '#71A24F',  # 深绿（明显更深）
#         '#6BB881',  # 翠绿（饱和实色）
        
#         # 橙棕色系（Domain 5）- 沉稳大气
#         '#D9935C',  # 橙棕（饱和实色）
#         '#A66C3D',  # 深橙棕（明显更深）
        
#         # === 第二组：新增6色系 ===
#         # 深蓝色系（Domain 6）- 深邃稳重
#         '#2E4A6B',  # 深海蓝
#         '#4A6FA5',  # 宝蓝
        
#         # 洋红色系（Domain 7）- 鲜艳活泼
#         '#C84D7B',  # 洋红
#         '#E8739C',  # 粉红
        
#         # 青绿色系（Domain 8）- 清新自然
#         '#3B9B8F',  # 青绿
#         '#5EC4B6',  # 浅青绿
        
#         # 深红色系（Domain 9）- 热情醇厚
#         '#A84032',  # 深红
#         '#D16B5E',  # 朱红
        
#         # 橄榄绿系（Domain 10）- 沉稳自然
#         '#7B904B',  # 橄榄绿
#         '#A5B875',  # 浅橄榄
        
#         # 深灰蓝系（Domain 11）- 中性专业
#         '#5A6F7D',  # 深灰蓝
#         '#7B93A3',  # 灰蓝
#     ],
#     # 单色渐变（适合基因表达）- 紫色系
#     'sequential': ['#4A3A6B', '#6B5491', '#9A72DD', '#B794E8',
#                    '#D1AEFA', '#E5D4FC', '#F3EBFD', '#FAF7FE'],
#     # 对比色（用于强调差异）- 紫米渐变
#     'diverging': ['#4A3A6B', '#6F5099', '#9A72DD', '#BEA3ED', '#E5D4FC',
#                   '#F9EDD8', '#F5E0BD', '#EDD0A3', '#DCBD8A', '#CAAA75'],
#     # 区域背景色（淡色，用于同域/异域标记）
#     'backgrounds': {
#         'same_domain': '#FAF7FE',    # 极淡紫 - 同域
#         'diff_domain': '#FEFBF7',    # 极淡米 - 异域
#     }
# }

# # 网格和背景颜色 - Nature标准
# GRID_COLOR = '#E5E5E5'  # 更淡的灰色
# BG_ALPHA = 0.25  # 提高透明度以增强对比（从0.12提升到0.25）

# # 全局Domain颜色映射
# # 这个字典定义了每个domain名称对应的固定颜色索引
# # 确保在所有图中，同一个domain始终使用相同的颜色
# GLOBAL_DOMAIN_COLOR_MAP = {
#     'Duct Epithelium': 0,        # 紫色系 (索引0: #5D4A8E, #8E6FC9)
#     'Stroma': 1,                 # 金黄色系 (索引2: #D4A870, #EAD7B3)
#     'Tumor': 2,                  # 青蓝色系 (索引4: #3C5C95, #5BA3C7)
#     'Acinar': 3,                 # 珊瑚红系 (索引6: #B85054, #E87B8A)
#     'Islet': 4,                  # 翠绿色系 (索引8: #71A24F, #6BB881)
#     'Immune': 5,                 # 橙棕色系 (索引10: #D9935C, #A66C3D)
#     # 可以根据实际的domain名称继续添加...
# }


# def get_domain_color(domain_name: str, nature_palette: list) -> str:
#     """
#     根据domain名称获取其固定的颜色
    
#     Args:
#         domain_name: domain的名称
#         nature_palette: 颜色列表
    
#     Returns:
#         该domain对应的颜色（十六进制字符串）
#     """
#     # 如果domain在全局映射中，使用固定的颜色索引
#     if domain_name in GLOBAL_DOMAIN_COLOR_MAP:
#         color_idx = GLOBAL_DOMAIN_COLOR_MAP[domain_name]
#         # 每个domain使用色系的第一个颜色（深色，索引 * 2）
#         palette_idx = (color_idx * 2) % len(nature_palette)
#         return nature_palette[palette_idx]
    
#     # 如果是未知的domain，使用hash来分配一个一致的颜色
#     # 这样即使是新的domain，在不同图中也会使用相同的颜色
#     # 使用稳定的hash并确保每个domain获得不同的颜色
#     hash_value = hash(domain_name)
#     # 使用深色（偶数索引）：0, 2, 4, 6, 8, 10
#     color_idx = abs(hash_value) % (len(nature_palette) // 2)
#     palette_idx = (color_idx * 2) % len(nature_palette)
#     return nature_palette[palette_idx]


# # ---------------- 通用工具 ----------------

# def load_boundary_spots(data_dir: str) -> list:
#     boundary_spots_path = os.path.join(data_dir, 'boundary_spots.json')  # JSON 路径
#     print(f"边界spots文件路径: {boundary_spots_path}")
#     if os.path.exists(boundary_spots_path):
#         with open(boundary_spots_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         return data.get('boundary_spots', [])  # 返回索引列表
#     else:
#         print(f"警告: 未找到边界spots文件 {boundary_spots_path}")
#         return []

# def load_adata_and_coords(data_dir: str) -> tuple:
#     h5ad_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]  # 查找h5ad
#     if h5ad_files:
#         adata_path = os.path.join(data_dir, h5ad_files[0])  # 使用第一个
#         adata = sc.read_h5ad(adata_path)  # 读取
#         print(f"加载AnnData: {adata_path}")
#     else:
#         print("警告: 未找到AnnData文件")
#         adata = None
    
#     coord_files = [f for f in os.listdir(data_dir) if 'coordinates' in f and f.endswith('.csv')]  # 查找坐标CSV
#     if coord_files:
#         coords_path = os.path.join(data_dir, coord_files[0])
#         coords_df = pd.read_csv(coords_path)  # 读取坐标表
#         print(f"加载坐标数据: {coords_path}")
#     else:
#         print("警告: 未找到坐标文件")
#         coords_df = None
    
#     return adata, coords_df

# def get_spot_name_from_index(spot_idx: int, adata=None, coords_df=None) -> str:
#     if adata is not None and spot_idx < len(adata.obs):  # 优先 obs.index
#         return adata.obs.index[spot_idx]
#     elif coords_df is not None and 'spot_name' in coords_df.columns:  # 其次坐标CSV
#         spot_row = coords_df[coords_df['spot_idx'] == spot_idx]
#         if not spot_row.empty:
#             return spot_row['spot_name'].iloc[0]
#     return str(spot_idx)

# def get_domain_info(spot_name: str, adata=None, coords_df=None, spot_idx=None) -> str:
#     """获取spot的domain/cluster信息
    
#     优先顺序：
#     1. 如果提供了spot_idx，直接从coords_df查找（最可靠）
#     2. 从adata.obs通过spot_name查找
#     3. 从coords_df通过spot_name查找
#     """
#     # 方法1: 通过spot_idx直接查找（最可靠）
#     if spot_idx is not None and coords_df is not None:
#         if 'spot_idx' in coords_df.columns:
#             spot_row = coords_df[coords_df['spot_idx'] == spot_idx]
#             if not spot_row.empty:
#                 if 'cluster' in coords_df.columns:
#                     return str(spot_row['cluster'].iloc[0])
#                 elif 'domain' in coords_df.columns:
#                     return str(spot_row['domain'].iloc[0])
    
#     # 方法2: 通过spot_name从adata查找
#     if adata is not None:
#         if spot_name in adata.obs.index:
#             if 'ground_truth' in adata.obs.columns:
#                 return str(adata.obs.loc[spot_name, 'ground_truth'])
#             elif 'domain' in adata.obs.columns:
#                 return str(adata.obs.loc[spot_name, 'domain'])
#             elif 'cluster' in adata.obs.columns:
#                 return str(adata.obs.loc[spot_name, 'cluster'])
    
#     # 方法3: 通过spot_name从coords_df查找
#     if coords_df is not None and 'spot_name' in coords_df.columns:
#         spot_row = coords_df[coords_df['spot_name'] == spot_name]
#         if not spot_row.empty:
#             if 'cluster' in coords_df.columns:
#                 return str(spot_row['cluster'].iloc[0])
#             elif 'domain' in coords_df.columns:
#                 return str(spot_row['domain'].iloc[0])
    
#     return "Unknown"



# def normalize_array(x):
#     """全局归一化（已弃用，保留用于兼容）"""
#     x = np.asarray(x, float)
#     vmax = x.max() if x.size and np.isfinite(x).any() else 1.0
#     return x / (vmax if vmax > 0 else 1.0)

# def filter_cross_domain_genes(df: pd.DataFrame, 
#                               neighbor_col: str = "neighbor_name",
#                               gene_col: str = "gene",
#                               adata=None,
#                               coords_df=None) -> pd.DataFrame:
#     """
#     过滤跨domain共有的基因，只保留domain内特异或domain内共有的基因
    
#     策略：
#     - 对于每个基因，统计它出现在哪些domain中
#     - 如果基因在多个不同domain中都出现 -> 过滤掉（跨domain共有）
#     - 如果基因只在单个domain中出现 -> 保留（domain特异或domain内共有）
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         包含邻居和基因信息的数据
#     neighbor_col : str
#         邻居列名
#     gene_col : str
#         基因列名
#     adata : AnnData, optional
#         用于获取domain信息
#     coords_df : pd.DataFrame, optional
#         用于获取domain信息
    
#     Returns:
#     --------
#     pd.DataFrame
#         过滤后的数据（移除跨domain共有基因）
#     """
#     print("\n=== 开始过滤跨domain共有基因 ===")
    
#     # 为每个邻居获取其domain信息
#     neighbor_domains = {}
#     for neighbor in df[neighbor_col].unique():
#         # 尝试将neighbor转换为spot_idx
#         neighbor_idx = None
#         try:
#             neighbor_idx = int(neighbor)
#         except (ValueError, TypeError):
#             # 如果不是纯数字，尝试从coords_df查找
#             if coords_df is not None and 'spot_name' in coords_df.columns:
#                 neighbor_row = coords_df[coords_df['spot_name'] == str(neighbor)]
#                 if not neighbor_row.empty and 'spot_idx' in coords_df.columns:
#                     neighbor_idx = neighbor_row['spot_idx'].iloc[0]
        
#         # 使用spot_idx查询domain
#         domain = get_domain_info(str(neighbor), adata=adata, coords_df=coords_df, spot_idx=neighbor_idx)
#         neighbor_domains[neighbor] = domain
    
#     # 打印邻居的domain分布
#     print(f"\n  邻居-domain详细映射:")
#     for neighbor, domain in sorted(neighbor_domains.items(), key=lambda x: str(x[0])):
#         print(f"    邻居 {neighbor} -> {domain}")
    
#     print(f"\n  邻居的domain分布:")
#     domain_count = {}
#     for neighbor, domain in neighbor_domains.items():
#         domain_count[domain] = domain_count.get(domain, 0) + 1
#     for domain, count in sorted(domain_count.items()):
#         print(f"    {domain}: {count} 个邻居")
    
#     # 为数据添加domain列
#     df_with_domain = df.copy()
#     df_with_domain['neighbor_domain'] = df_with_domain[neighbor_col].map(neighbor_domains)
    
#     # 统计每个基因出现在哪些domain中
#     gene_domain_count = df_with_domain.groupby(gene_col)['neighbor_domain'].nunique()
    
#     # 详细统计：每个基因出现在哪些具体的domain中
#     gene_domains_detail = df_with_domain.groupby(gene_col)['neighbor_domain'].apply(
#         lambda x: sorted(x.unique())
#     ).to_dict()
    
#     # 只保留在单个domain中出现的基因
#     genes_to_keep = gene_domain_count[gene_domain_count == 1].index.tolist()
    
#     # 统计信息
#     total_genes = df[gene_col].nunique()
#     cross_domain_genes = gene_domain_count[gene_domain_count > 1].index.tolist()
    
#     print(f"\n  原始基因数: {total_genes}")
#     print(f"  跨domain共有基因数: {len(cross_domain_genes)}")
#     print(f"  保留的domain特异/内共有基因数: {len(genes_to_keep)}")
    
#     if len(cross_domain_genes) > 0:
#         print(f"\n  跨domain共有基因详情（前10个）:")
#         for gene in cross_domain_genes[:10]:
#             domains = gene_domains_detail.get(gene, [])
#             print(f"    {gene:15s}: 出现在 {domains}")
    
#     # 过滤数据
#     df_filtered = df[df[gene_col].isin(genes_to_keep)].copy()
    
#     print(f"\n  过滤前记录数: {len(df)}")
#     print(f"  过滤后记录数: {len(df_filtered)}")
#     print("=== 过滤完成 ===\n")
    
#     return df_filtered


# def read_csv_or_topk(csv_path: str,
#                      required_cols,
#                      topk_in_memory: int = None,
#                      by_sum_or_mean: str = "sum",
#                      entity_cols=None) -> pd.DataFrame:
#     df = pd.read_csv(csv_path)  # 读CSV
#     miss = [c for c in required_cols if c not in df.columns]  # 缺列检查
#     if miss:
#         raise ValueError(f"文件 {csv_path} 缺少必要列: {miss}")
#     if topk_in_memory is None or entity_cols is None:  # 不筛TopK则直接返回
#         return df
#     k = int(topk_in_memory)  # TopK 数
#     metric = by_sum_or_mean
#     if metric not in ("sum","mean"):
#         raise ValueError("by_sum_or_mean 必须为 'sum' 或 'mean'")
#     df = df.copy()
#     df["__rank__"] = df.groupby(entity_cols)[metric].rank(method="first", ascending=False)  # 分组排名
#     out = df[df["__rank__"] <= k].drop(columns="__rank__")  # 取TopK
#     return out

# # ---------------- 背景绘制工具（极坐标数据坐标下构造扇区多边形） ----------------

# def build_domain_colormap(domains, palette="primary", alpha=None):
#     """
#     使用Nature风格配色为domain生成颜色映射（RGBA，带透明度）
    
#     Parameters:
#     -----------
#     domains : list
#         domain列表
#     palette : str
#         配色方案：'primary'（主色）, 'pastel'（柔和）, 'accent'（强调）
#     alpha : float
#         透明度，默认使用BG_ALPHA
#     """
#     if alpha is None:
#         alpha = BG_ALPHA
        
#     color_list = NATURE_COLORS.get(palette, NATURE_COLORS['primary'])
#     unique_domains = list(pd.Index(domains).unique())
#     dom2rgba = {}
    
#     for d in unique_domains:
#         # 使用 get_domain_color() 确保与条形图颜色一致
#         hex_color = get_domain_color(d, color_list)
#         rgba = list(colors.to_rgba(hex_color))
#         rgba[3] = float(alpha)
#         dom2rgba[d] = tuple(rgba)
    
#     return dom2rgba

# def fill_ring_sector(ax, th0, th1, r0, r1, color, n=128):
#     """
#     在极坐标数据坐标中填充环形扇区，避免出现水滴形。
#     th0, th1: 扇区起止角（弧度）
#     r0, r1:  内外半径
#     color:   RGBA 颜色
#     n:       沿角方向采样点数
#     """
#     thetas_outer = np.linspace(th0, th1, n)            # 外弧角度采样
#     thetas_inner = np.linspace(th1, th0, n)            # 内弧反向采样
#     r_outer = np.full_like(thetas_outer, r1)           # 外弧半径
#     r_inner = np.full_like(thetas_inner, r0)           # 内弧半径
#     th = np.concatenate([thetas_outer, thetas_inner])  # 闭合角序列
#     rr = np.concatenate([r_outer, r_inner])            # 对应半径
#     ax.fill(th, rr, color=color, edgecolor=None, linewidth=0)  # 直接在极坐标数据坐标填充


# def draw_sector_middle_band(ax, theta_edges, r_inner, r_outer,
#                             sector_labels, sector2domain, domain2color,
#                             band_frac=(0.0, 1.0)):
#     """
#     在每个扇区的指定径向范围填充背景色。
#     band_frac=(0.0,1.0) 表示从内圆到外圆全覆盖。
#     关键点：sector_labels 的元素必须是 sector2domain 的键，才能命中 domain2color。
#     """
#     r_span = (r_outer - r_inner)                    # 半径跨度
#     r0 = r_inner + r_span * float(band_frac[0])     # 起始半径
#     r1 = r_inner + r_span * float(band_frac[1])     # 结束半径
    
#     # 调试信息
#     print("Debug - Domain colors mapping:")
#     if domain2color:
#         for dom, col in domain2color.items():
#             print(f"  Domain '{dom}' -> Color {col}")
    
#     for i in range(len(sector_labels)):             # 遍历扇区
#         th0 = theta_edges[i]                        # 起始角
#         th1 = theta_edges[i+1]                      # 结束角
#         # 取该扇区对应的 domain；若未命中则回退灰色
#         dom = sector2domain.get(sector_labels[i], "Unknown") if sector2domain is not None else "Unknown"
#         col = domain2color.get(dom, (0.7,0.7,0.7,0.18)) if domain2color is not None else (0.7,0.7,0.7,0.18)
#         print(f"  Sector {i} ({sector_labels[i]}) -> Domain '{dom}' -> Color {col}")
#         fill_ring_sector(ax, th0, th1, r0, r1, col, n=256)     # 真正填充背景扇形


# def polar_bar_chart_v2(ax, sectors, bars_per_sector, heights,
#                        sector_labels=None, bar_labels=None,
#                        sector_colors=None, bar_colors=None,
#                        r_inner=0.2, r_outer=1.0,
#                        bar_width_frac=0.92,
#                        grid_rings=5, grid_color=None,
#                        sector_label_pad=0.02,
#                        label_fontsize=9,  # 增大扇区标签：8 -> 10
#                        gene_label_fontsize=9,  # 增大基因标签：7 -> 9
#                        gene_label_pad=0.02,
#                        # 新增：背景与中心文本
#                        sector2domain=None, domain2color=None,
#                        middle_band_frac=(0.0, 1.0),
#                        center_text=None, center_text_fontsize=10,  # 增大中心文字：9 -> 11
#                        # 新增：用于背景匹配的键列表（如果与sector_labels不同）
#                        sector_keys=None,
#                        # 新增：扇区角度（用于domain标签计算）
#                        sector_angles_list=None):
#     import numpy as np

#     if grid_color is None:
#         grid_color = GRID_COLOR

#     ax.set_theta_direction(-1)       # 顺时针
#     ax.set_theta_offset(np.pi/2.0)   # 0度在正上
#     ax.set_axis_off()                # 隐藏极轴

#     # 统一 bars_per_sector 形态
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

#     # 底层同心网格环 - Nature风格：更细更淡
#     for i in range(grid_rings+1):
#         r = r_inner + (r_outer - r_inner) * i / grid_rings
#         ax.plot(np.linspace(0, 2*np.pi, 400), np.full(400, r), 
#                 color=grid_color, lw=0.4, alpha=0.6, zorder=0)

#     # 按条数分配扇区角宽
#     total_bars = sum(bps)
#     full_angle = 2*np.pi
#     sector_angles = [full_angle * (b / total_bars) if total_bars>0 else full_angle/sectors for b in bps]

#     # 先计算扇区边界角（用于背景）
#     theta = 0.0
#     theta_edges = [0.0]
#     for s in range(sectors):
#         theta += sector_angles[s]
#         theta_edges.append(theta)

#     # 背景扇形：严格覆盖每个扇区
#     if sector_labels is None:
#         sector_labels = [f"S{s+1}" for s in range(sectors)]
    
#     # 如果提供了专用的sector_keys，则使用它们作为映射键
#     keys_for_mapping = sector_keys if sector_keys is not None else sector_labels
    
#     draw_sector_middle_band(ax=ax, theta_edges=theta_edges,
#                             r_inner=r_inner, r_outer=r_outer,
#                             sector_labels=keys_for_mapping,
#                             sector2domain=sector2domain,
#                             domain2color=domain2color,
#                             band_frac=middle_band_frac)

#     # 开始绘制条形
#     theta = 0.0
#     idx = 0
#     sector_centers = []
#     for s in range(sectors):
#         ang = sector_angles[s]
#         bars_s = bps[s]
#         gap = ang * (1 - bar_width_frac)         # 扇区整体留白
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
#             # 添加黑色边框以增强对比度
#             ax.bar(x=th_mid, height=r1 - r0, width=(th1 - th0),
#                    bottom=r0, color=bcolor, edgecolor="black", linewidth=0.3, align="center")

#             # 基因标签：统一放在条形末端外侧（考虑左右方向差异）
#             if bar_labels[idx]:
#                 # 关键：使用归一化后的高度h来计算条形的实际顶端位置
#                 actual_height = r_inner + (r_outer - r_inner) * h  # 归一化高度对应的实际半径
#                 pad_r = (r_outer - r_inner) * gene_label_pad * 0.5 # 增加间距
                
#                 ang_deg = np.degrees(th_mid)
#                 rot = 90 - ang_deg
                
#                 # 关键修复：左右两侧方向相反
#                 if -90 <= rot <= 90:
#                     # 右侧：向外延伸（增加半径）
#                     rt = actual_height + pad_r
#                     ha = "left"; va = "center"
#                 else:
#                     # 左侧：旋转文字并反向放置
#                     # 文字旋转180度使其正向，并且位置在条形外侧
#                     rot = rot + 180  # 翻转文字使其正向
#                     rt = actual_height + pad_r  # 同样向外延伸
#                     ha = "right"; va = "center"
                    
#                 ax.text(th_mid, rt, str(bar_labels[idx]),
#                         rotation=rot, rotation_mode="anchor",
#                         ha=ha, va=va, fontsize=gene_label_fontsize, 
#                         color="#2C3E50", weight='normal', alpha=0.9)
#             idx += 1

#         # 扇区分隔线 - Nature风格：更细
#         ax.plot([theta, theta], [r_inner, r_outer], color=grid_color, lw=0.4, alpha=0.6, zorder=0)

#         sector_centers.append(theta + ang/2)
#         theta += ang
#     ax.plot([theta, theta], [r_inner, r_outer], color=grid_color, lw=0.4, alpha=0.6, zorder=0)

#     # 扇区外弧标签已移除，邻居信息仅在图例中显示
#     # （简化视觉，避免拥挤）
    
#     # Domain标签已移至图例，不在图中显示
#     # （保留此注释以说明设计决策）

#     # 内圆中心文字 - Nature风格（调整字体大小和位置）
#     if center_text:
#         ax.text(0.0, 0.0, center_text,
#                 ha="center", va="center",
#                 fontsize=center_text_fontsize, color="#2C3E50", fontweight="bold",
#                 linespacing=1.3)


# def plot_spot_level_from_csv(base_dir: str,
#                              center_name: str,
#                              gene_view: str = "kv",
#                              metric: str = "mean",
#                              topk: int = 5,
#                              use_topk_file: bool = True,
#                              figsize=(8,8),
#                              r_inner=0.2, r_outer=0.8,
#                              gene_label_fontsize=9,  # 增大默认字体：8 -> 9
#                              save_png: bool = False,
#                              out_png: str = None,
#                              adata=None,
#                              coords_df=None,
#                              center_idx=None,
#                              show_background: bool = True,  # 新增：是否显示背景
#                              show_legend: bool = True,      # 新增：是否显示图例
#                              filter_cross_domain: bool = False,  # 已弃用：建议使用domain_aware模式
#                              use_domain_aware: bool = False,     # 新增：使用domain感知TopK文件
#                              domain_weight: float = 0.6):        # 新增：domain权重（用于选择对应文件）
#     agg_dir = os.path.join(base_dir, "agg_csv")
#     if gene_view not in ("kv","q"):
#         raise ValueError("gene_view 必须为 'kv' 或 'q'")
#     if metric not in ("sum","mean"):
#         raise ValueError("metric 必须为 'sum' 或 'mean'")

#     # 选源CSV - 优先级：domain_aware > 标准TopK > 全量数据
#     if use_domain_aware:
#         # 使用domain感知TopK文件
#         weight_suffix = f"w{int(domain_weight*10)}"
#         csv_path = os.path.join(agg_dir, f"spot_level_{gene_view}_top{topk}_domain_aware_{weight_suffix}.csv")
        
#         if not os.path.exists(csv_path):
#             print(f"⚠️  未找到domain感知TopK文件: {csv_path}")
#             print(f"   请先运行: python 3.5_prepare_polar_domain_aware.py")
#             print(f"   回退到标准TopK文件...")
#             use_domain_aware = False
#             use_topk_file = True
#         else:
#             print(f"✅ 使用domain感知TopK文件 (domain_weight={domain_weight})")
#             use_topk_file = True  # domain_aware文件也是预计算的TopK
    
#     if use_topk_file and not use_domain_aware:
#         # 使用标准TopK文件
#         csv_path = os.path.join(agg_dir, f"spot_level_{gene_view}_top{topk}_by_{metric}.csv")
#         if not os.path.exists(csv_path):
#             csv_path = os.path.join(agg_dir, f"spot_level_{gene_view}.csv")
#             use_topk_file = False
#             print(f"⚠️  未找到标准TopK文件，使用全量数据")
#     elif not use_topk_file:
#         # 使用全量数据
#         csv_path = os.path.join(agg_dir, f"spot_level_{gene_view}.csv")
#         print(f"📊 使用全量数据")

#     req = ["center_name","neighbor_name","gene","sum","mean"]
#     df = read_csv_or_topk(csv_path, required_cols=req,
#                           topk_in_memory=None if use_topk_file else topk,
#                           by_sum_or_mean=metric,
#                           entity_cols=["center_name","neighbor_name"])

#     dfc = df[df["center_name"].astype(str) == str(center_name)].copy()
#     if dfc.empty:
#         raise ValueError(f"在 {csv_path} 中未找到 center_name={center_name} 的记录")

#     # 如果不使用TopK文件，手动选择TopK
#     if use_topk_file is False:
#         dfc["__rank__"] = dfc.groupby(["neighbor_name"])[metric].rank(method="first", ascending=False)
#         dfc = dfc[dfc["__rank__"] <= topk].drop(columns="__rank__")

#     # 跨domain基因过滤（建议与domain_aware配合使用）
#     if filter_cross_domain:
#         print(f"\n启用跨domain基因过滤模式（中心spot: {center_name}）")
#         print(f"  过滤前: {len(dfc)} 条记录，{dfc['gene'].nunique()} 个独特基因")
        
#         dfc = filter_cross_domain_genes(
#             dfc, 
#             neighbor_col="neighbor_name",
#             gene_col="gene",
#             adata=adata,
#             coords_df=coords_df
#         )
        
#         if dfc.empty:
#             print(f"警告: 过滤后没有剩余基因，所有TopK基因都是跨domain共有")
#             raise ValueError(f"过滤跨domain基因后没有剩余数据")
        
#         print(f"  过滤后: {len(dfc)} 条记录，{dfc['gene'].nunique()} 个独特基因")
    
#     # 模式说明
#     if use_domain_aware:
#         print(f"\n💡 Domain感知模式:")
#         print(f"   - 同domain邻居倾向于选择相似的基因")
#         print(f"   - domain内基因重复率提高")
#         if filter_cross_domain:
#             print(f"   ✅ 配合跨domain过滤，移除跨domain共有基因")
#         else:
#             print(f"   ⚠️  建议启用 filter_cross_domain=True 以过滤跨domain共有基因")

#     # 中心 domain
#     center_domain = get_domain_info(center_name, adata, coords_df, spot_idx=center_idx)
#     print(f"中心spot {center_name} (idx={center_idx}) 属于domain: {center_domain}")

#     # 扇区顺序（先按domain分组，再按强度排序 - 让同domain的邻居聚集在一起）
#     neighbor_strength = dfc.groupby("neighbor_name")[metric].sum()
    
#     # 临时获取每个邻居的domain（用于分组）
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
    
#     # 按domain分组，domain内按强度排序
#     neighbor_df = pd.DataFrame({
#         'neighbor': neighbor_strength.index,
#         'strength': neighbor_strength.values,
#         'domain': [temp_neighbor_domains[n] for n in neighbor_strength.index]
#     })
#     # 先按domain排序，再在domain内按强度降序排序
#     neighbor_df = neighbor_df.sort_values(['domain', 'strength'], ascending=[True, False])
#     sectors = neighbor_df['neighbor'].tolist()
    
#     # 邻居 -> domain 映射
#     # 首先尝试从neighbor_name解析出spot_idx（如果neighbor_name是纯数字或可以提取索引）
#     neighbor_domains = {}
#     for neighbor in sectors:
#         # 尝试多种方法获取neighbor的spot_idx
#         neighbor_idx = None
#         try:
#             # 方法1: 如果neighbor_name就是数字，直接使用
#             neighbor_idx = int(neighbor)
#         except (ValueError, TypeError):
#             # 方法2: 从coords_df查找
#             if coords_df is not None and 'spot_name' in coords_df.columns:
#                 neighbor_row = coords_df[coords_df['spot_name'] == neighbor]
#                 if not neighbor_row.empty and 'spot_idx' in coords_df.columns:
#                     neighbor_idx = neighbor_row['spot_idx'].iloc[0]
        
#         neighbor_domain = get_domain_info(neighbor, adata, coords_df, spot_idx=neighbor_idx)
#         neighbor_domains[neighbor] = neighbor_domain
    
#     print(f"中心spot {center_name} 的邻居domains: {neighbor_domains}")

#     # 优化配色：不同domain用不同色系，同domain内的邻居使用色系变体
#     import matplotlib.colors as mcolors
#     nature_palette = NATURE_COLORS['primary']
    
#     # 为每个unique domain分配固定的颜色（基于全局映射）
#     unique_domains = list(dict.fromkeys([neighbor_domains[n] for n in sectors]))
#     # 使用全局函数确保同一个domain在所有图中使用相同的颜色
#     domain_base_colors = {dom: get_domain_color(dom, nature_palette) 
#                           for dom in unique_domains}
    
#     # 为每个邻居分配基础色（基于其domain）
#     # 同一个domain的不同邻居使用同色系的渐变色（便于区分）
#     sector_base_colors = {}
#     domain_neighbor_count = {}  # 记录每个domain的邻居数量
#     domain_neighbor_index = {}  # 记录每个domain当前处理到第几个邻居
    
#     for n in sectors:
#         dom = neighbor_domains[n]
#         if dom not in domain_neighbor_count:
#             domain_neighbor_count[dom] = sum(1 for s in sectors if neighbor_domains[s] == dom)
#             domain_neighbor_index[dom] = 0
    
#     for n in sectors:
#         dom = neighbor_domains[n]
#         base_color = domain_base_colors[dom]
        
#         # 如果该domain只有一个邻居，直接使用基础色
#         if domain_neighbor_count[dom] == 1:
#             sector_base_colors[n] = base_color
#         else:
#             # 同一domain的不同邻居：生成色系变体（从基础色的75%亮度到115%亮度）
#             index = domain_neighbor_index[dom]
#             total = domain_neighbor_count[dom]
#             # 亮度变化范围：0.75 到 1.15
#             brightness_factor = 0.75 + (0.4 * index / max(1, total - 1))
            
#             # 转换为RGB并调整亮度
#             rgb = mcolors.to_rgb(base_color)
#             adjusted_rgb = tuple(min(1.0, c * brightness_factor) for c in rgb)
#             sector_base_colors[n] = mcolors.to_hex(adjusted_rgb)
            
#             domain_neighbor_index[dom] += 1

#     bars_per_sector, heights, bar_colors, bar_labels = [], [], [], []
#     sector_total_attention = []  # 记录每个扇区的总attention

#     # 外弧扇区标签（简洁版：邻居编号 + domain编号）
#     sector_labels = []                 # 用于背景匹配（键=邻居名称）
#     sector_labels_with_domain = []     # 用于显示（简洁版）
    
#     # 直接构建 sector2domain 映射，确保键是邻居名称
#     sector2domain = {}
    
#     for n in sectors:
#         neighbor_domain = neighbor_domains[n]
#         is_same_domain = neighbor_domain == center_domain
#         sector_labels.append(n)  # 注意：这里用纯邻居名，作为 sector2domain 的键
        
#         # 先获取该邻居的所有基因，然后筛选top k（但不在扇区内归一化）
#         sub = dfc[dfc["neighbor_name"] == n].sort_values(metric, ascending=False)
#         vals = sub[metric].to_numpy()
        
#         # 计算该扇区的总attention
#         total_attn = vals.sum()
#         sector_total_attention.append(total_attn)
        
#         # 简洁扇区标签：只显示邻居编号（domain信息在扇形内部统一显示）
#         sector_labels_with_domain.append(f"N{n}")
        
#         # 将邻居名称映射到其domain
#         sector2domain[n] = neighbor_domain
        
#         bars_per_sector.append(len(vals))
#         heights.extend(vals.tolist())
#         glist = sub["gene"].tolist()
#         bar_labels.extend(glist)
        
#         # 同一邻居（扇区）内的所有基因使用相同颜色
#         # 渐变体现在：同一domain的不同邻居之间
#         bar_colors.extend([sector_base_colors[n]] * len(vals))

#     # 全局归一化：所有邻居一起归一化（恢复原始方式，便于邻居间对比）
#     heights = normalize_array(heights)
    
#     print(f"归一化方法: 全局归一化（所有邻居一起，便于比较）")
#     print(f"扇区总attention: {sector_total_attention}")

#     # 多色着色：为每个不同的domain分配不同颜色
#     # 构造 domain -> color 映射（只考虑邻居出现过的 domain）
#     unique_neighbor_domains = list(pd.Index([neighbor_domains[n] for n in sectors]).unique())
    
#     # 打印调试信息
#     print(f"Debug - Unique neighbor domains: {unique_neighbor_domains}")
#     print(f"Debug - Neighbor domains mapping:")
#     for n, d in neighbor_domains.items():
#         print(f"  Neighbor {n} -> Domain {d}")
    
#     # 背景配置：根据参数决定是否显示
#     if show_background:
#         # 使用Nature配色为不同domain分配颜色（使用更淡的alpha，与domain-level一致）
#         domain2rgba = build_domain_colormap(unique_neighbor_domains, palette='primary', alpha=0.15)
#         # 使用环形背景（与domain-level一致）：只在中间30%-70%显示
#         band_frac = (0.30, 0.60)
#     else:
#         domain2rgba = None
#         sector2domain = None
#         band_frac = (0.0, 0.0)

#     # 中心文字（简洁清晰，无括号）
#     center_text = f"Center Spot\n{center_name}\nDomain {center_domain}"

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
#                        bar_width_frac=0.92,  # 与domain-level一致
#                        grid_rings=5,
#                        sector_label_pad=0.02,  # 与domain-level一致
#                        label_fontsize=9,  # 增大扇区标签
#                        gene_label_fontsize=gene_label_fontsize,
#                        gene_label_pad=0.02,  # 与domain-level一致
#                        sector2domain=sector2domain if show_background else None,
#                        domain2color=domain2rgba,
#                        middle_band_frac=band_frac,  # 使用环形背景
#                        center_text=center_text,
#                        center_text_fontsize=10,  # 增大中心文字
#                        sector_keys=sector_labels)

#     # 图例：可选显示（根据参数）
#     if show_legend:
#         from matplotlib.patches import Patch
        
#         handles = []
#         legend_labels = []
        
#         # 获取所有unique domains（保持原始顺序）
#         unique_domains_ordered = list(dict.fromkeys([neighbor_domains[n] for n in sectors]))
        
#         # 第一部分：Domains
#         legend_labels.append("═══ Domains ═══")
#         handles.append(plt.Line2D([0],[0], color='none'))  # 占位符
        
#         for dom in unique_domains_ordered:
#             dom_color = domain_base_colors[dom]
#             handles.append(Patch(facecolor=dom_color, edgecolor='white', linewidth=1.5, alpha=0.9))
#             legend_labels.append(f"Domain {dom}")
        
#         # 添加分隔
#         legend_labels.append("")
#         handles.append(plt.Line2D([0],[0], color='none'))
        
#         # 第二部分：Neighbors（按domain分组，但视觉上集中）
#         legend_labels.append("═══ Neighbors ═══")
#         handles.append(plt.Line2D([0],[0], color='none'))
        
#         for dom in unique_domains_ordered:
#             domain_neighbors = [n for n in sectors if neighbor_domains[n] == dom]
#             for n in domain_neighbors:
#                 handles.append(plt.Line2D([0],[0], color=sector_base_colors[n], lw=5, solid_capstyle='butt'))
#                 legend_labels.append(f"N{n} ({dom})")
        
#         legend = plt.legend(handles, legend_labels, bbox_to_anchor=(0.95, 1.0), loc="upper left",
#                             title=None,  # 去掉"Legend"标题
#                             fontsize=9, frameon=True,  # 增大图例字体：9 -> 10
#                             fancybox=False, shadow=False, edgecolor='#DDDDDD',
#                             handlelength=1.5, handletextpad=0.6, labelspacing=0.8,
#                             borderpad=0.4, columnspacing=1.0)
#         legend.get_frame().set_linewidth(0.8)  # 设置边框线
#     plt.tight_layout()

#     if save_png:
#         if out_png is None:
#             out_png = f"spot_level_{gene_view}_top{topk}_by_{metric}_{center_name}.png"
#         plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
#         print(f"已保存图像到: {out_png}")
#     plt.close()

# # ---------------- Domain-level（global, kv, 每扇区topK=5） ----------------

# def plot_domain_level_kv_global_topk_from_csv(base_dir: str,
#                                               metric: str = "mean",
#                                               topk: int = 5,
#                                               figsize=(8,8),
#                                               r_inner=0.2, r_outer=1.0,
#                                               gene_label_fontsize=9,  # 增大默认字体：8 -> 9
#                                               save_png: bool = False,
#                                               out_png: str = None,
#                                               out_dir: str = None,
#                                               show_background: bool = False):  # 新增：是否显示背景
#     agg_dir = os.path.join(base_dir, "agg_csv")
#     if metric not in ("sum","mean"):
#         raise ValueError("metric 必须为 'sum' 或 'mean'")

#     csv_topk = os.path.join(agg_dir, f"domain_level_kv_global_top{topk}_by_{metric}.csv")
#     if os.path.exists(csv_topk):
#         df = pd.read_csv(csv_topk)
#         req = ["domain","gene","sum","mean"]
#         miss = [c for c in req if c not in df.columns]
#         if miss:
#             raise ValueError(f"{csv_topk} 缺少必要列: {miss}")
#         dfg = df.copy()
#     else:
#         csv_full = os.path.join(agg_dir, "domain_level_kv_global.csv")
#         req = ["domain","gene","sum","mean"]
#         dfall = read_csv_or_topk(csv_full, required_cols=req)
#         dfall["__rank__"] = dfall.groupby("domain")[metric].rank(method="first", ascending=False)
#         dfg = dfall[dfall["__rank__"] <= topk].drop(columns="__rank__").copy()

#     # 扇区顺序
#     sector_order = dfg.groupby("domain")[metric].sum().sort_values(ascending=False).index.tolist()
#     sectors = sector_order

#     # 改进的配色策略：每个domain使用固定且唯一的颜色
#     # 为了确保每个domain都有不同的颜色，按顺序分配颜色（使用深色：索引0, 2, 4, 6, 8, 10）
#     nature_palette = NATURE_COLORS['primary']
    
#     # 创建domain到颜色的映射，确保每个domain有唯一颜色
#     domain_base_colors = {}
#     available_colors = [nature_palette[i] for i in range(0, len(nature_palette), 2)]  # 只使用深色
    
#     for idx, dom in enumerate(sectors):
#         if dom in GLOBAL_DOMAIN_COLOR_MAP:
#             # 使用预定义的颜色
#             color_idx = GLOBAL_DOMAIN_COLOR_MAP[dom]
#             domain_base_colors[dom] = available_colors[color_idx % len(available_colors)]
#         else:
#             # 按顺序分配颜色，确保不重复
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
        
#         # 同一个domain的所有基因使用相同颜色
#         base_color = domain_base_colors[dom]
#         n_genes = len(glist)
        
#         # 为该domain的所有基因使用相同颜色（不使用渐变）
#         for i in range(n_genes):
#             bar_colors.append(base_color)

#     heights = normalize_array(heights)

#     # 背景颜色映射 - Domain-level默认不显示背景
#     if show_background:
#         domain2rgba = build_domain_colormap(sectors, palette='primary', alpha=0.08)
#         sector2domain = {dom: dom for dom in sectors}
#     else:
#         domain2rgba = None
#         sector2domain = None

#     center_text = "Domains\nTop genes"

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
#                        label_fontsize=9,  # 增大扇区标签
#                        gene_label_fontsize=gene_label_fontsize,
#                        gene_label_pad=0.02,
#                        sector2domain=sector2domain,
#                        domain2color=domain2rgba,
#                        middle_band_frac=(0.30, 0.70),
#                        center_text=center_text,
#                        center_text_fontsize=10)  # 增大中心文字

#     # 添加图例：显示每个 Domain 的颜色
#     from matplotlib.patches import Patch
    
#     handles = []
#     legend_labels = []
    
#     # 为每个 domain 创建图例项
#     for dom in sectors:
#         dom_color = domain_base_colors[dom]
#         handles.append(Patch(facecolor=dom_color, edgecolor='white', linewidth=1.5, alpha=0.9))
#         legend_labels.append(f"{dom}")
    
#     legend = plt.legend(handles, legend_labels, bbox_to_anchor=(0.92, 0.9), loc="upper left",
#                         title="",
#                         title_fontsize=9,  # 增大图例标题字体
#                         fontsize=14, frameon=True,  # 增大图例字体：9 -> 10
#                         fancybox=False, shadow=False, edgecolor='#DDDDDD',
#                         handlelength=1.5, handletextpad=0.8, labelspacing=0.8,
#                         borderpad=0.4, columnspacing=1.0)
#     legend.get_frame().set_linewidth(0.8)
    

#     plt.tight_layout()

#     if save_png:
#         # 如果提供了输出目录，确保它存在
#         if out_dir is not None:
#             os.makedirs(out_dir, exist_ok=True)
            
#         if out_png is None:
#             filename = f"domain_level_kv_global_top{topk}_by_{metric}.png"
#             # 如果指定了输出目录，将文件名与目录合并
#             if out_dir is not None:
#                 out_png = os.path.join(out_dir, filename)
#             else:
#                 out_png = filename
                
#         plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
#         print(f"已保存图像到: {out_png}")
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
#     基于domain aware的spot level数据，聚合生成domain level极坐标图
    
#     策略：
#     1. 读取domain aware的spot level TopK数据
#     2. 按照domain聚合，计算每个gene在该domain中的总和/平均
#     3. 为每个domain选择TopK基因
#     4. 绘制domain level极坐标图
    
#     参数:
#         base_dir: 基础目录
#         metric: 聚合指标 ("mean" 或 "sum")
#         topk: 每个domain保留的TopK基因数
#         domain_weight: domain权重（用于选择对应的domain aware文件）
#         其他参数同 plot_domain_level_kv_global_topk_from_csv
#     """
#     agg_dir = os.path.join(base_dir, "agg_csv")
#     if metric not in ("sum","mean"):
#         raise ValueError("metric 必须为 'sum' 或 'mean'")
    
#     # 读取domain aware的spot level数据
#     weight_suffix = f"w{int(domain_weight*10)}"
#     spot_csv = os.path.join(agg_dir, f"spot_level_kv_top{topk}_domain_aware_{weight_suffix}.csv")
    
#     if not os.path.exists(spot_csv):
#         raise FileNotFoundError(f"未找到domain aware文件: {spot_csv}\n请先运行 3.5_prepare_polar_domain_aware.py")
    
#     print(f"\n{'='*80}")
#     print(f"📊 Domain Level 聚合 (基于Domain Aware数据)")
#     print(f"{'='*80}")
#     print(f"输入: {spot_csv}")
#     print(f"Domain权重: {domain_weight}")
#     print(f"聚合指标: {metric}")
#     print(f"每个domain TopK: {topk}")
    
#     # 读取spot level数据
#     df_spot = pd.read_csv(spot_csv)
#     print(f"\n原始记录数: {len(df_spot)}")
#     print(f"涉及中心spots: {df_spot['center_name'].nunique()}")
#     print(f"涉及邻居spots: {df_spot['neighbor_name'].nunique()}")
    
#     # 需要获取每个neighbor的domain信息
#     if adata is None or coords_df is None:
#         adata, coords_df = load_metadata(base_dir)
    
#     # 为每个neighbor获取domain
#     print("\n获取邻居的domain信息...")
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
    
#     # 添加domain列
#     df_spot['neighbor_domain'] = df_spot['neighbor_name'].map(neighbor_domains)
    
#     # 过滤掉Unknown domain
#     df_spot = df_spot[df_spot['neighbor_domain'] != 'Unknown'].copy()
#     print(f"过滤Unknown后记录数: {len(df_spot)}")
    
#     domain_counts = df_spot['neighbor_domain'].value_counts()
#     print(f"\nDomain分布:")
#     for domain, count in domain_counts.items():
#         print(f"  {domain}: {count} 条记录")
    
#     # 按domain和gene聚合
#     print(f"\n按domain和gene聚合...")
#     if metric == "mean":
#         # 对于mean，我们需要重新计算加权平均
#         # sum列是attention的总和，count列是记录数
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
    
#     print(f"聚合后记录数: {len(df_agg)}")
#     print(f"涉及domains: {df_agg['domain'].nunique()}")
#     print(f"涉及genes: {df_agg['gene'].nunique()}")
    
#     # 为每个domain选择TopK基因
#     print(f"\n为每个domain选择Top{topk}基因...")
#     df_agg["__rank__"] = df_agg.groupby("domain")[metric].rank(method="first", ascending=False)
#     dfg = df_agg[df_agg["__rank__"] <= topk].drop(columns="__rank__").copy()
    
#     print(f"选择TopK后记录数: {len(dfg)}")
    
#     # 每个domain的TopK基因
#     print(f"\n各domain的Top{min(5, topk)}基因:")
#     for domain in dfg['domain'].unique():
#         domain_genes = dfg[dfg['domain'] == domain].nlargest(min(5, topk), metric)['gene'].tolist()
#         print(f"  {domain}: {domain_genes}")
    
#     # 扇区顺序
#     sector_order = dfg.groupby("domain")[metric].sum().sort_values(ascending=False).index.tolist()
#     sectors = sector_order

#     # 改进的配色策略：每个domain使用固定且唯一的颜色
#     # 为了确保每个domain都有不同的颜色，按顺序分配颜色（使用深色：索引0, 2, 4, 6, 8, 10）
#     nature_palette = NATURE_COLORS['primary']
    
#     # 创建domain到颜色的映射，确保每个domain有唯一颜色
#     domain_base_colors = {}
#     available_colors = [nature_palette[i] for i in range(0, len(nature_palette), 2)]  # 只使用深色
    
#     for idx, dom in enumerate(sectors):
#         if dom in GLOBAL_DOMAIN_COLOR_MAP:
#             # 使用预定义的颜色
#             color_idx = GLOBAL_DOMAIN_COLOR_MAP[dom]
#             domain_base_colors[dom] = available_colors[color_idx % len(available_colors)]
#         else:
#             # 按顺序分配颜色，确保不重复
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
        
#         # 同一个domain的所有基因使用相同颜色
#         base_color = domain_base_colors[dom]
#         n_genes = len(glist)
        
#         # 为该domain的所有基因使用相同颜色（不使用渐变）
#         for i in range(n_genes):
#             bar_colors.append(base_color)

#     heights = normalize_array(heights)

#     # 背景颜色映射
#     if show_background:
#         domain2rgba = build_domain_colormap(sectors, palette='primary', alpha=0.08)
#         sector2domain = {dom: dom for dom in sectors}
#     else:
#         domain2rgba = None
#         sector2domain = None

#     center_text = "Domains\n(Domain Aware)\nTop genes"

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
#                        center_text=center_text,
#                        center_text_fontsize=10)

#     # 添加图例
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
#         print(f"\n✅ 已保存图像到: {out_png}")
#         print(f"{'='*80}\n")
#     plt.close()

# # ---------------- 示例 ----------------
# if __name__ == "__main__":
#     base_dir = "./PDAC/whole_slice_data_20251028_173836"
#     boundary_dir = "./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_boundary"
#     adata, coords_df = load_adata_and_coords(base_dir)
    
#     boundary_spots_indices = load_boundary_spots(boundary_dir)
#     print(f"找到 {len(boundary_spots_indices)} 个边界spots")
    
#     # 保存边界spots的名称和索引的映射
#     boundary_spots_names = []
#     boundary_spots_idx_map = {}  # spot_name -> spot_idx
#     for spot_idx in boundary_spots_indices:
#         spot_name = get_spot_name_from_index(spot_idx, adata, coords_df)
#         boundary_spots_names.append(spot_name)
#         boundary_spots_idx_map[spot_name] = spot_idx
    
#     print(f"边界spots名称: {boundary_spots_names[:5]}...")
#     print(f"边界spots索引映射: {list(boundary_spots_idx_map.items())[:5]}...")

#     # 1) Domain-level
#     # 创建存储domain-level极坐标图的文件夹
#     domain_plots_dir = "./PDAC/domain_polar_plots"
#     os.makedirs(domain_plots_dir, exist_ok=True)
#     print(f"生成domain-level极坐标图，保存至: {domain_plots_dir}")
    
#     plot_domain_level_kv_global_topk_from_csv(base_dir,
#                                               metric="mean",
#                                               topk=10,
#                                               figsize=(10,10),
#                                               r_inner=0.2, r_outer=1.0,
#                                               gene_label_fontsize=10,  # 增大基因标签字体
#                                               save_png=True,
#                                               out_dir=domain_plots_dir,
#                                               show_background=False)  # Domain-level不显示背景

#     # 2) Spot-level - 标准模式（显示所有基因）
#     if boundary_spots_names:
#         center_spot = boundary_spots_names[0]
#         center_spot_idx = boundary_spots_idx_map.get(center_spot)
#         print(f"使用边界spot作为中心: {center_spot} (idx={center_spot_idx})")
        
#         print("\n=== 标准模式：显示所有top基因 ===")
#         plot_spot_level_from_csv(base_dir,
#                                  center_name=center_spot,
#                                  gene_view="kv",
#                                  metric="mean",
#                                  topk=30,
#                                  use_topk_file=True,
#                                  figsize=(10,10),
#                                  r_inner=0.2, r_outer=1.0,
#                                  gene_label_fontsize=9,  # 增大基因标签字体
#                                  save_png=False,
#                                  adata=adata,
#                                  coords_df=coords_df,
#                                  center_idx=center_spot_idx,
#                                  show_background=True,
#                                  show_legend=True,
#                                  filter_cross_domain=False)  # 标准模式不过滤
        
#         # 3) Spot-level - Domain感知模式（推荐）
#         print("\n=== Domain感知模式：同domain邻居选择相似基因 ===")
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
#                                  use_domain_aware=True,      # 使用domain感知TopK
#                                  domain_weight=0.6)          # domain权重（0.5-0.7推荐）
        
#         if len(boundary_spots_names) > 1:
#             # 创建存储边界spot极坐标图的文件夹
#             boundary_plots_dir = "./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_boundary/"
#             os.makedirs(boundary_plots_dir, exist_ok=True)
#             print(f"为前3个边界spots生成极坐标图，保存至: {boundary_plots_dir}")
            
#             for i, spot_name in enumerate(boundary_spots_names[:]):
#                 spot_idx = boundary_spots_idx_map.get(spot_name)
#                 print(f"处理边界spot {i+1}/{min(10, len(boundary_spots_names))}: {spot_name} (idx={spot_idx})")
                
#                 # 使用改进的get_domain_info函数获取domain
#                 domain = get_domain_info(spot_name, adata, coords_df, spot_idx=spot_idx)
#                 print(f"Spot {spot_name} (idx={spot_idx}) 的domain: {domain}")
                
#                 # 构建输出文件路径（放在新文件夹中）
#                 out_file = os.path.join(boundary_plots_dir, f"boundary_spot_{i+1}_{spot_idx}_polar.png")
                
#                 plot_spot_level_from_csv(base_dir,
#                                          center_name=spot_name,
#                                          gene_view="kv",
#                                          metric="mean",
#                                          topk=10,
#                                          use_topk_file=True,
#                                          figsize=(10,10),
#                                          r_inner=0.2, r_outer=1.0,
#                                          gene_label_fontsize=8,  # 增大基因标签字体
#                                          save_png=True,
#                                          out_png=out_file,
#                                          adata=adata,
#                                          coords_df=coords_df,
#                                          center_idx=spot_idx,
#                                          show_background=True,
#                                          show_legend=True,
#                                          filter_cross_domain=False)  # 标准模式
            
#             # 4) 批量生成Domain感知+过滤极坐标图
#             print(f"\n=== 批量生成Domain感知+过滤极坐标图 ===")
#             domain_aware_plots_dir = "./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_domain_aware_filtered/"
#             os.makedirs(domain_aware_plots_dir, exist_ok=True)
#             print(f"生成Domain感知+过滤极坐标图，保存至: {domain_aware_plots_dir}")
#             print(f"策略: domain感知TopK选择 + 跨domain共有基因过滤")
            
#             for i, spot_name in enumerate(boundary_spots_names[:]):  # 前10个边界spots
#                 spot_idx = boundary_spots_idx_map.get(spot_name)
#                 print(f"\n处理边界spot {i+1}/10: {spot_name} (idx={spot_idx})")
                
#                 domain = get_domain_info(spot_name, adata, coords_df, spot_idx=spot_idx)
#                 print(f"  Spot {spot_name} (idx={spot_idx}) 的domain: {domain}")
                
#                 # 构建输出文件路径
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
#                                              use_domain_aware=True,       # 使用domain感知TopK
#                                              domain_weight=0.6,           # domain权重
#                                              filter_cross_domain=True)    # 同时过滤跨domain共有基因
#                     print(f"  ✓ 已保存: {out_file}")
#                 except Exception as e:
#                     print(f"  ✗ 跳过spot {spot_name}: {str(e)}")
#                     continue
#     else:
#         print("警告: 未找到边界spots，使用默认spot")
    
#     # 5) 生成Domain Level极坐标图（Domain Aware版本） - 独立执行，不依赖边界spots
#     print(f"\n{'='*80}")
#     print(f"=== 生成Domain Level极坐标图 (Domain Aware) ===")
#     print(f"{'='*80}")
#     domain_plots_dir = "./PDAC/domain_polar_plots_domain_aware/"
#     os.makedirs(domain_plots_dir, exist_ok=True)
    
#     # 测试不同的TopK和domain_weight
#     for topk_val in [5, 10]:
#         for weight in [0.6]:  # 可以测试多个权重: [0.5, 0.6, 0.7]
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
#                 print(f"  ✗ 生成失败: {str(e)}")
#                 import traceback
#                 traceback.print_exc()
    
#     print(f"\n{'='*80}")
#     print(f"✅ Domain Level极坐标图已保存至: {domain_plots_dir}")
#     print(f"{'='*80}")
    
#     # 6) 从每个domain中随机抽取5个spot绘制极坐标图
#     print(f"\n{'='*80}")
#     print(f"=== 从每个Domain随机抽取Spots绘制极坐标图 ===")
#     print(f"{'='*80}")
    
#     random_spots_dir = "./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_random_by_domain/"
#     os.makedirs(random_spots_dir, exist_ok=True)
    
#     # 获取所有domain及其对应的spots
#     domain_to_spots = None
    
#     # 优先从coords_df获取domain信息
#     if coords_df is not None:
#         domain_col = None
#         if 'cluster' in coords_df.columns:
#             domain_col = 'cluster'
#         elif 'domain' in coords_df.columns:
#             domain_col = 'domain'
        
#         if domain_col is not None:
#             print(f"从coords_df的'{domain_col}'列获取domain信息")
#             domain_to_spots = {}
#             for idx, row in coords_df.iterrows():
#                 domain = str(row[domain_col])
#                 spot_idx = int(row['spot_idx']) if 'spot_idx' in row else idx
#                 spot_name = row['spot_name'] if 'spot_name' in row else str(spot_idx)
                
#                 if domain not in domain_to_spots:
#                     domain_to_spots[domain] = []
#                 domain_to_spots[domain].append((spot_name, spot_idx))
    
#     # 如果coords_df没有domain信息，尝试从adata获取
#     if domain_to_spots is None and adata is not None and 'ground_truth' in adata.obs.columns:
#         print(f"从adata.obs的'ground_truth'列获取domain信息")
#         domain_to_spots = {}
#         for spot_idx in range(len(adata.obs)):
#             spot_name = adata.obs.index[spot_idx]
#             domain = str(adata.obs.loc[spot_name, 'ground_truth'])
#             if domain not in domain_to_spots:
#                 domain_to_spots[domain] = []
#             domain_to_spots[domain].append((spot_name, spot_idx))
    
#     # 如果都没有，报错
#     if domain_to_spots is None:
#         print("✗ 无法获取domain信息，跳过此步骤")
#         print("  请确保coords_df有'cluster'或'domain'列，或adata.obs有'ground_truth'列")
    
#     if domain_to_spots:
#         import random
#         random.seed(42)  # 设置随机种子以便结果可复现
        
#         print(f"\n发现 {len(domain_to_spots)} 个domain:")
#         for domain, spots in domain_to_spots.items():
#             print(f"  - {domain}: {len(spots)} spots")
        
#         # 为每个domain随机抽取5个spot
#         for domain, spots in domain_to_spots.items():
#             print(f"\n{'#'*80}")
#             print(f"# Domain: {domain}")
#             print(f"{'#'*80}")
            
#             # 创建该domain的子文件夹
#             domain_subdir = os.path.join(random_spots_dir, f"domain_{domain.replace(' ', '_')}")
#             os.makedirs(domain_subdir, exist_ok=True)
            
#             # 随机抽取5个spot（如果该domain的spot数量少于5，则全部使用）
#             n_samples = min(5, len(spots))
#             sampled_spots = random.sample(spots, n_samples)
            
#             print(f"从domain '{domain}' 中随机抽取 {n_samples} 个spots:")
            
#             for i, (spot_name, spot_idx) in enumerate(sampled_spots):
#                 print(f"\n  [{i+1}/{n_samples}] Spot: {spot_name} (idx={spot_idx})")
                
#                 # 构建输出文件路径
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
#                         use_domain_aware=True,       # 使用domain感知TopK
#                         domain_weight=0.6,           # domain权重
#                         filter_cross_domain=True     # 过滤跨domain共有基因
#                     )
#                     print(f"    ✓ 已保存: {out_file}")
#                 except Exception as e:
#                     print(f"    ✗ 跳过spot {spot_name}: {str(e)}")
#                     continue
        
#         print(f"\n{'='*80}")
#         print(f"✅ 随机spot极坐标图已保存至: {random_spots_dir}")
#         print(f"{'='*80}\n")
#     else:
#         print("✗ 无法获取domain-spot映射，跳过随机抽样步骤\n")





# 尝试相对导入（作为包的一部分运行时）
try:
    from .utils_5_plot_polar import *
except ImportError:
    # 如果相对导入失败，使用绝对导入（直接运行脚本时）
    from utils_5_plot_polar import *




# ---------------- 示例 ----------------
if __name__ == "__main__":
    base_dir = "./PDAC/whole_slice_data_20251028_173836"
    boundary_dir = base_dir + "/spatial_attention_visualizer_boundary"
    # boundary_dir = "./HBRC/whole_slice_data_20251102_141320/spatial_attention_visualizer_boundary"
    adata, coords_df = load_adata_and_coords(base_dir)
    
    boundary_spots_indices = load_boundary_spots(boundary_dir)
    print(f"找到 {len(boundary_spots_indices)} 个边界spots")
    
    # 保存边界spots的名称和索引的映射
    boundary_spots_names = []
    boundary_spots_idx_map = {}  # spot_name -> spot_idx
    for spot_idx in boundary_spots_indices:
        spot_name = get_spot_name_from_index(spot_idx, adata, coords_df)
        boundary_spots_names.append(spot_name)
        boundary_spots_idx_map[spot_name] = spot_idx
    
    print(f"边界spots名称: {boundary_spots_names[:5]}...")
    print(f"边界spots索引映射: {list(boundary_spots_idx_map.items())[:5]}...")



    # 2) Spot-level - 批量生成Domain感知+过滤极坐标图
    if boundary_spots_names:

        if len(boundary_spots_names) > 1:
            # 4) 批量生成Domain感知+过滤极坐标图
            print(f"\n=== 批量生成Domain感知+过滤极坐标图 ===")
            domain_aware_plots_dir = base_dir + "/spatial_attention_visualizer_domain_aware_filtered1/"
            # domain_aware_plots_dir = "./HBRC/whole_slice_data_20251102_141320/spatial_attention_visualizer_domain_aware_filtered1/"
            os.makedirs(domain_aware_plots_dir, exist_ok=True)
            print(f"生成Domain感知+过滤极坐标图，保存至: {domain_aware_plots_dir}")
            print(f"策略: domain感知TopK选择 + 跨domain共有基因过滤")
            
            for i, spot_name in enumerate(boundary_spots_names[:]):  # 前10个边界spots
                spot_idx = boundary_spots_idx_map.get(spot_name)
                print(f"\n处理边界spot {i+1}/10: {spot_name} (idx={spot_idx})")
                
                domain = get_domain_info(spot_name, adata, coords_df, spot_idx=spot_idx)
                print(f"  Spot {spot_name} (idx={spot_idx}) 的domain: {domain}")
                
                # 构建输出文件路径
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
                                             use_domain_aware=True,       # 使用domain感知TopK
                                             domain_weight=0.6,           # domain权重
                                             filter_cross_domain=True)    # 同时过滤跨domain共有基因
                    print(f"  ✓ 已保存: {out_file}")
                except Exception as e:
                    print(f"  ✗ 跳过spot {spot_name}: {str(e)}")
                    continue
    else:
        print("警告: 未找到边界spots，使用默认spot")
    
    # # 5) 生成Domain Level极坐标图（Domain Aware版本） - 独立执行，不依赖边界spots
    # print(f"\n{'='*80}")
    # print(f"=== 生成Domain Level极坐标图 (Domain Aware) ===")
    # print(f"{'='*80}")
    # domain_plots_dir = base_dir + "/domain_polar_plots_domain_aware1/"
    # os.makedirs(domain_plots_dir, exist_ok=True)
    
    # # 测试不同的TopK和domain_weight
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
    #         print(f"  ✗ 生成失败: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    
    # print(f"\n{'='*80}")
    # print(f"✅ Domain Level极坐标图已保存至: {domain_plots_dir}")
    # print(f"{'='*80}")




