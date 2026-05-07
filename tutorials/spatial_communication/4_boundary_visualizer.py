"""
空间注意力可视化与分析脚本
- 基于导出的空间注意力数据进行高级分析和可视化
- 实现三个主要可视化:
  1. Spot-level空间交互图
  2. 边界Spots分析与关键基因交互
  3. 全局注意力流动向量场
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.cluster import KMeans
import argparse
import json
from typing import List, Dict, Tuple, Optional, Union, Any
from plot_polar import *
# Nature风格配置
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#2C3E50',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.color': '#2C3E50',
    'ytick.color': '#2C3E50',
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '#DDDDDD',
    'legend.fontsize': 9,
    'grid.color': '#E5E5E5',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 禁用不必要的警告
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============= 辅助函数 =============
def load_data(data_dir: str) -> Dict[str, Any]:
    """
    加载所需数据文件
    """
    data = {}
    
    # 找到综合注意力CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'combined' in f]
    if not csv_files:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'comprehensive' in f]
    if not csv_files:
        # 尝试找到whole_slice_attention文件
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'whole_slice_attention' in f]
    
    if csv_files:
        attn_csv_path = os.path.join(data_dir, csv_files[0])
        print(f"使用注意力数据: {attn_csv_path}")
        data['attn_df'] = pd.read_csv(attn_csv_path)
    else:
        raise FileNotFoundError(f"在{data_dir}中未找到注意力CSV文件")
    
    # 加载空间坐标
    coord_files = [f for f in os.listdir(data_dir) if 'coordinates' in f and f.endswith('.csv')]
    if coord_files:
        coord_csv_path = os.path.join(data_dir, coord_files[0])
        print(f"使用空间坐标数据: {coord_csv_path}")
        data['coords_df'] = pd.read_csv(coord_csv_path)
    else:
        raise FileNotFoundError(f"在{data_dir}中未找到空间坐标CSV文件")
    
    # 加载AnnData (如果存在)
    h5ad_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
    if h5ad_files:
        h5ad_path = os.path.join(data_dir, h5ad_files[0])
        print(f"使用AnnData文件: {h5ad_path}")
        data['adata'] = sc.read_h5ad(h5ad_path)
    else:
        print("未找到AnnData文件，将仅使用CSV数据")
        data['adata'] = None
    
    # 加载配体-受体数据库 (如果存在)
    lr_files = [f for f in os.listdir(data_dir) if 'lr_database' in f and f.endswith('.csv')]
    if lr_files:
        lr_path = os.path.join(data_dir, lr_files[0])
        print(f"使用配体-受体数据库: {lr_path}")
        data['lr_df'] = pd.read_csv(lr_path)
    else:
        print("未找到配体-受体数据库，L-R分析将受限")
        data['lr_df'] = None
    
    return data

def prepare_spot_to_spot_attention(attn_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算spot到spot的注意力总和
    """
    print("计算Spot-to-Spot注意力...")
    
    # 确保数据中有必要的列
    required_cols = ['center_global_idx', 'neighbor_global_idx', 'attn_sum_norm']
    for col in required_cols:
        if col not in attn_df.columns:
            if col == 'attn_sum_norm' and 'attn_sum' in attn_df.columns:
                # 计算归一化的注意力总和
                attn_sum_by_center = attn_df.groupby('center_global_idx')['attn_sum'].sum()
                attn_df['attn_sum_norm'] = attn_df.apply(
                    lambda x: x['attn_sum'] / attn_sum_by_center[x['center_global_idx']] 
                    if x['center_global_idx'] in attn_sum_by_center and attn_sum_by_center[x['center_global_idx']] > 0 
                    else 0, 
                    axis=1
                )
            else:
                raise ValueError(f"注意力数据中缺少必要列: {col}")
    
    # 计算从一个spot到另一个spot的总注意力
    spot_to_spot = attn_df.groupby(['center_global_idx', 'neighbor_global_idx'])['attn_sum_norm'].sum().reset_index()
    
    # 重命名列以更清晰
    spot_to_spot.columns = ['source_idx', 'target_idx', 'attention_weight']
    
    # 去除NaN值
    spot_to_spot = spot_to_spot.dropna()
    
    print(f"共找到{len(spot_to_spot)}个spot-to-spot连接")
    return spot_to_spot

def identify_boundary_spots(coords_df: pd.DataFrame, spot_to_spot: pd.DataFrame, 
                           threshold: float = 0.3) -> List[int]:
    """
    识别边界spots (与不同聚类的spots有强交互的spots)
    """
    print("识别边界spots...")
    
    # 确保有聚类信息
    if 'cluster' not in coords_df.columns:
        print("警告: 未找到聚类信息，将使用KMeans进行简单聚类")
        # 进行简单的KMeans聚类
        kmeans = KMeans(n_clusters=7, random_state=42)
        coords_df['cluster'] = kmeans.fit_predict(coords_df[['x', 'y']])
    
    # 将字符串聚类标签转换为数值（如果还没有转换）
    if coords_df['cluster'].dtype == 'object':
        unique_clusters = coords_df['cluster'].unique()
        cluster_to_num = {cluster: i for i, cluster in enumerate(unique_clusters)}
        coords_df['cluster_num'] = coords_df['cluster'].map(cluster_to_num)
    else:
        coords_df['cluster_num'] = coords_df['cluster']
    
    # 创建spot_idx到cluster的映射（使用数值标签）
    idx_to_cluster = dict(zip(coords_df['spot_idx'], coords_df['cluster_num']))
    
    # 找到边界spots
    boundary_spots = []
    cross_cluster_ratio = {}
    
    # 计算每个spot对不同聚类的spots的注意力比例
    for spot_idx in coords_df['spot_idx'].unique():
        if spot_idx not in idx_to_cluster:
            continue
            
        # 获取该spot的聚类
        spot_cluster = idx_to_cluster[spot_idx]
        
        # 找到该spot关注的所有目标spots
        targets = spot_to_spot[spot_to_spot['source_idx'] == spot_idx]
        
        if targets.empty:
            continue
            
        # 分离同类和异类目标
        same_cluster_attn = 0.0
        diff_cluster_attn = 0.0
        
        for _, row in targets.iterrows():
            target_idx = row['target_idx']
            if target_idx not in idx_to_cluster:
                continue
                
            target_cluster = idx_to_cluster[target_idx]
            
            if target_cluster == spot_cluster:
                same_cluster_attn += row['attention_weight']
            else:
                diff_cluster_attn += row['attention_weight']
        
        total_attn = same_cluster_attn + diff_cluster_attn
        
        if total_attn > 0:
            cross_ratio = diff_cluster_attn / total_attn
            cross_cluster_ratio[spot_idx] = cross_ratio
            
            # 如果跨聚类注意力比例超过阈值，则视为边界spot
            if cross_ratio > threshold:
                boundary_spots.append(spot_idx)
    
    print(f"已识别出{len(boundary_spots)}个边界spots (跨聚类注意力比例 > {threshold})")
    return boundary_spots, cross_cluster_ratio

def find_attention_flow_vectors(spot_to_spot: pd.DataFrame, coords_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    计算每个spot的最大注意力流向量
    """
    print("计算注意力流向量...")
    
    # 创建spot_idx到坐标的映射
    idx_to_coords = {}
    for _, row in coords_df.iterrows():
        idx_to_coords[row['spot_idx']] = (row['x'], row['y'])
    
    # 找出每个spot注意力最强的目标
    flow_vectors = {}
    
    for spot_idx in coords_df['spot_idx'].unique():
        if spot_idx not in idx_to_coords:
            continue
            
        # 找到该spot关注的所有目标spots
        targets = spot_to_spot[spot_to_spot['source_idx'] == spot_idx]
        
        if targets.empty:
            continue
            
        # 找出注意力最强的目标
        max_target_row = targets.loc[targets['attention_weight'].idxmax()]
        max_target_idx = max_target_row['target_idx']
        
        if max_target_idx not in idx_to_coords:
            continue
            
        # 计算向量
        source_pos = idx_to_coords[spot_idx]
        target_pos = idx_to_coords[max_target_idx]
        
        # 向量: 从源点指向目标点
        direction = np.array([target_pos[0] - source_pos[0], target_pos[1] - source_pos[1]])
        
        # 计算向量长度
        magnitude = np.linalg.norm(direction)
        
        if magnitude > 0:
            # 单位向量 * 注意力权重
            normalized_direction = direction / magnitude * max_target_row['attention_weight']
            
            flow_vectors[spot_idx] = {
                'position': source_pos,
                'target_idx': max_target_idx,
                'direction': normalized_direction,
                'magnitude': max_target_row['attention_weight'],
                'target_position': target_pos
            }
    
    print(f"为{len(flow_vectors)}个spots计算了注意力流向量")
    return flow_vectors

def find_gene_pairs_for_boundary_spots(attn_df: pd.DataFrame, boundary_spots: List[int], 
                                      coords_df: pd.DataFrame, lr_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    查找边界spots中的高注意力基因对，并标记潜在的配体-受体对
    """
    print("分析边界spots的基因对...")
    
    # 创建spot_idx到cluster的映射
    if 'cluster' in coords_df.columns:
        # 确保有数值化的聚类标签
        if coords_df['cluster'].dtype == 'object' and 'cluster_num' not in coords_df.columns:
            unique_clusters = coords_df['cluster'].unique()
            cluster_to_num = {cluster: i for i, cluster in enumerate(unique_clusters)}
            coords_df['cluster_num'] = coords_df['cluster'].map(cluster_to_num)
        
        idx_to_cluster = dict(zip(coords_df['spot_idx'], coords_df['cluster_num']))
    else:
        print("警告: 未找到聚类信息，无法区分同类/异类基因对")
        idx_to_cluster = {idx: 0 for idx in coords_df['spot_idx']}
    
    # 准备一个配体-受体查询字典
    lr_pairs = {}
    if lr_df is not None:
        for _, row in lr_df.iterrows():
            lr_pairs[(row['ligand'], row['receptor'])] = True
    
    # 收集边界spots的基因对信息
    gene_pairs_data = []
    
    for spot_idx in boundary_spots:
        if spot_idx not in idx_to_cluster:
            continue
            
        # 获取该spot的聚类
        spot_cluster = idx_to_cluster[spot_idx]
        
        # 筛选该spot的数据
        spot_data = attn_df[attn_df['center_global_idx'] == spot_idx]
        
        if spot_data.empty:
            continue
        
        # 筛选同类和异类交互
        for neighbor_idx in spot_data['neighbor_global_idx'].unique():
            if pd.isna(neighbor_idx) or neighbor_idx not in idx_to_cluster:
                continue
                
            neighbor_cluster = idx_to_cluster[int(neighbor_idx)]
            interaction_type = 'same_cluster' if neighbor_cluster == spot_cluster else 'diff_cluster'
            
            # 提取该邻居的基因对
            neighbor_data = spot_data[spot_data['neighbor_global_idx'] == neighbor_idx]
            
            for _, row in neighbor_data.iterrows():
                if pd.isna(row['q_gene_symbol']) or pd.isna(row['kv_gene_symbol']) or not row['q_gene_symbol'] or not row['kv_gene_symbol']:
                    continue
                    
                # 检查是否为配体-受体对
                is_lr_pair = False
                lr_direction = ""
                
                if lr_df is not None:
                    q_gene = row['q_gene_symbol']
                    kv_gene = row['kv_gene_symbol']
                    
                    if (q_gene, kv_gene) in lr_pairs:
                        is_lr_pair = True
                        lr_direction = f"{q_gene}(L)->{kv_gene}(R)"
                    elif (kv_gene, q_gene) in lr_pairs:
                        is_lr_pair = True
                        lr_direction = f"{kv_gene}(L)->{q_gene}(R)"
                
                gene_pairs_data.append({
                    'boundary_spot_idx': spot_idx,
                    'boundary_cluster': spot_cluster,
                    'neighbor_idx': int(neighbor_idx),
                    'neighbor_cluster': neighbor_cluster,
                    'interaction_type': interaction_type,
                    'query_gene': row['q_gene_symbol'],
                    'key_gene': row['kv_gene_symbol'],
                    'attn_score': row['attn_score'],
                    'is_lr_pair': is_lr_pair,
                    'lr_direction': lr_direction
                })
    
    # 创建DataFrame
    gene_pairs_df = pd.DataFrame(gene_pairs_data)
    
    if not gene_pairs_df.empty:
        print(f"共找到{len(gene_pairs_df)}个边界基因对，其中{gene_pairs_df['is_lr_pair'].sum()}个是已知配体-受体对")
    else:
        print("未找到边界基因对")
    
    return gene_pairs_df

# ============= 可视化函数 =============
def plot_spatial_interaction_network(coords_df: pd.DataFrame, spot_to_spot: pd.DataFrame, 
                                    boundary_spots: List[int], cross_cluster_ratio: Dict[int, float],
                                    output_dir: str, percentile_threshold: int = 60):
    """
    绘制空间交互网络，突出显示区域边界
    """
    print("绘制空间交互网络...")
    
    plt.figure(figsize=(16, 14))
    
    # 绘制所有spots
    if 'cluster' in coords_df.columns:
        # 将字符串聚类标签转换为数值
        unique_clusters = coords_df['cluster'].unique()
        cluster_to_num = {cluster: i for i, cluster in enumerate(unique_clusters)}
        coords_df['cluster_num'] = coords_df['cluster'].map(cluster_to_num)
        
        scatter = plt.scatter(
            coords_df['x'], coords_df['y'],
            c=coords_df['cluster_num'], cmap='tab20', alpha=0.7, s=50, edgecolors='black',
            label='Regular Spots'
        )
    else:
        scatter = plt.scatter(
            coords_df['x'], coords_df['y'],
            c='lightgray', alpha=0.7, s=50, edgecolors='black',
            label='Regular Spots'
        )
    
    # 高亮显示边界spots
    if boundary_spots:
        boundary_df = coords_df[coords_df['spot_idx'].isin(boundary_spots)]
        plt.scatter(
            boundary_df['x'], boundary_df['y'],
            s=100, facecolors='none', edgecolors='red', linewidth=2,
            label='Boundary Spots'
        )
    
    # 创建spot_idx到坐标的映射
    idx_to_coords = {}
    for _, row in coords_df.iterrows():
        idx_to_coords[row['spot_idx']] = (row['x'], row['y'])
    
    # 为避免过多连线，仅显示注意力权重在前N%的连接
    threshold = np.percentile(spot_to_spot['attention_weight'], percentile_threshold)
    strong_connections = spot_to_spot[spot_to_spot['attention_weight'] >= threshold]
    
    print(f"显示注意力权重在前{100-percentile_threshold}%的连接 (阈值: {threshold:.4f})")
    
    # 绘制注意力连接
    for _, row in strong_connections.iterrows():
        source_idx = row['source_idx']
        target_idx = row['target_idx']
        
        if source_idx not in idx_to_coords or target_idx not in idx_to_coords:
            continue
        
        source_pos = idx_to_coords[source_idx]
        target_pos = idx_to_coords[target_idx]
        
        # 计算注意力归一化后的线宽和透明度
        weight_norm = row['attention_weight'] / strong_connections['attention_weight'].max()
        line_width = max(0.5, weight_norm * 4)
        alpha = max(0.2, min(0.9, weight_norm * 1.5))
        
        # 如果是边界spot的连接，使用特殊颜色
        if source_idx in boundary_spots:
            line_color = 'red'
        else:
            line_color = 'blue'
        
        # 绘制带箭头的连接
        plt.annotate(
            '', xy=target_pos, xytext=source_pos,
            arrowprops=dict(
                arrowstyle='-|>', 
                color=line_color,
                lw=line_width,
                alpha=alpha,
                shrinkA=10,
                shrinkB=10
            )
        )
    
    # 添加颜色条（如果有聚类信息）
    if 'cluster' in coords_df.columns:
        cbar = plt.colorbar(scatter, orientation='vertical', pad=0.01)
        cbar.set_label('Cluster ID', fontsize=12)
    
    # 添加图例
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2, marker=None),
        Line2D([0], [0], color='red', lw=2, marker=None),
        Line2D([0], [0], color='black', marker='o', markerfacecolor='none', markeredgecolor='red', markersize=10, lw=0)
    ]
    plt.legend(custom_lines, ['Regular Attention', 'Boundary Attention', 'Boundary Spots'], loc='upper right', fontsize=12)
    
    plt.title('Spatial Interaction Network', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, 'spatial_interaction_network.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"空间交互网络图已保存到: {output_path}")
    
    # 可选：绘制边界spots的跨聚类注意力比例图
    if cross_cluster_ratio and len(cross_cluster_ratio) > 0:
        plt.figure(figsize=(10, 6))
        
        # 转换为DataFrame便于绘制
        ratio_df = pd.DataFrame({
            'spot_idx': list(cross_cluster_ratio.keys()),
            'cross_cluster_ratio': list(cross_cluster_ratio.values())
        })
        
        # 添加边界标记
        ratio_df['is_boundary'] = ratio_df['spot_idx'].isin(boundary_spots)
        
        # 对比图
        sns.barplot(
            x='spot_idx', y='cross_cluster_ratio', hue='is_boundary',
            data=ratio_df.sort_values('cross_cluster_ratio', ascending=False).head(50),
            palette=['blue', 'red']
        )
        
        plt.title('Top 50 Spots by Cross-Cluster Attention Ratio', fontsize=16)
        plt.xlabel('Spot Index', fontsize=14)
        plt.ylabel('Cross-Cluster Attention Ratio', fontsize=14)
        plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.6, label='Threshold')
        plt.legend(title='Is Boundary')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        ratio_path = os.path.join(output_dir, 'boundary_spots_ratio.png')
        plt.savefig(ratio_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"边界spots比例图已保存到: {ratio_path}")
    
    return output_path

def plot_boundary_gene_interactions(gene_pairs_df: pd.DataFrame, coords_df: pd.DataFrame, output_dir: str):
    """
    分析并可视化边界spots的基因交互（Nature风格）
    """
    if gene_pairs_df.empty:
        print("警告: 无边界基因交互数据可用")
        return None
    
    print("分析边界spots的基因交互...")
    
    # Nature配色方案
    nature_colors = {
        'same_cluster': '#5BA3C7',  # 青蓝色
        'diff_cluster': '#E87B8A'   # 珊瑚红色
    }
    
    # 1. 分析同类vs异类交互
    interaction_counts = gene_pairs_df.groupby('interaction_type').size()
    interaction_avg_score = gene_pairs_df.groupby('interaction_type')['attn_score'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    
    # 使用Nature配色的条形图
    x_pos = np.arange(len(interaction_counts))
    colors = [nature_colors.get(idx, '#808080') for idx in interaction_counts.index]
    
    bars = ax.bar(x_pos, interaction_counts.values, color=colors, 
                  edgecolor='white', linewidth=1.5, alpha=0.9, width=0.6)
    
    # 添加数值标签（更优雅的样式）
    for i, (v, avg) in enumerate(zip(interaction_counts.values, interaction_avg_score.values)):
        ax.text(i, v + max(interaction_counts.values) * 0.02, 
                f"{v}", 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2C3E50')
        ax.text(i, v / 2, 
                f"avg: {avg:.3f}", 
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # 设置标签（更简洁）
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Same Cluster', 'Different Cluster'], fontsize=11, color='#2C3E50')
    ax.set_ylabel('Count', fontsize=12, color='#2C3E50', fontweight='normal')
    ax.set_title('Boundary Spots: Interaction Analysis', fontsize=13, fontweight='bold', 
                 color='#2C3E50', pad=15)
    
    # 优化网格和边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#E5E5E5')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    count_path = os.path.join(output_dir, 'boundary_interaction_counts.png')
    plt.savefig(count_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 2. 分析配体-受体对（Nature风格）
    if 'is_lr_pair' in gene_pairs_df.columns and gene_pairs_df['is_lr_pair'].any():
        # 找出前15个高分配体-受体对（减少以提高可读性）
        top_lr_pairs = (gene_pairs_df[gene_pairs_df['is_lr_pair']]
                        .groupby(['lr_direction', 'interaction_type'])['attn_score']
                        .mean()
                        .reset_index()
                        .sort_values('attn_score', ascending=False)
                        .head(15))
        
        fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
        
        # 准备数据
        lr_pairs = top_lr_pairs['lr_direction'].unique()
        x_pos = np.arange(len(lr_pairs))
        
        # 分组绘制
        same_data = []
        diff_data = []
        for lr in lr_pairs:
            same_val = top_lr_pairs[(top_lr_pairs['lr_direction'] == lr) & 
                                   (top_lr_pairs['interaction_type'] == 'same_cluster')]['attn_score']
            diff_val = top_lr_pairs[(top_lr_pairs['lr_direction'] == lr) & 
                                   (top_lr_pairs['interaction_type'] == 'diff_cluster')]['attn_score']
            same_data.append(same_val.values[0] if len(same_val) > 0 else 0)
            diff_data.append(diff_val.values[0] if len(diff_val) > 0 else 0)
        
        # 绘制分组条形图
        width = 0.35
        ax.bar(x_pos - width/2, same_data, width, label='Same Cluster',
               color=nature_colors['same_cluster'], edgecolor='white', linewidth=1, alpha=0.9)
        ax.bar(x_pos + width/2, diff_data, width, label='Different Cluster',
               color=nature_colors['diff_cluster'], edgecolor='white', linewidth=1, alpha=0.9)
        
        # 设置标签和标题
        ax.set_xlabel('Ligand-Receptor Pair', fontsize=12, color='#2C3E50', fontweight='normal')
        ax.set_ylabel('Average Attention Score', fontsize=12, color='#2C3E50', fontweight='normal')
        ax.set_title('Top Ligand-Receptor Pairs in Boundary Interactions', 
                    fontsize=13, fontweight='bold', color='#2C3E50', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lr_pairs, rotation=45, ha='right', fontsize=9)
        
        # 图例
        ax.legend(loc='upper right', frameon=True, edgecolor='#DDDDDD', 
                 fontsize=9, framealpha=1.0)
        
        # 优化边框和网格
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#DDDDDD')
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='#E5E5E5')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        lr_path = os.path.join(output_dir, 'top_lr_pairs.png')
        plt.savefig(lr_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"配体-受体对分析已保存到: {lr_path}")
    
    # 3. 热图：边界spot对每个基因对的注意力分布
    try:
        # 选择前15个最高注意力分数的基因对
        top_gene_pairs = (gene_pairs_df
                        .groupby(['query_gene', 'key_gene'])['attn_score']
                        .mean()
                        .reset_index()
                        .sort_values('attn_score', ascending=False)
                        .head(15))
        
        if len(top_gene_pairs) > 1:  # 需要至少2个基因对来绘制热图
            # 创建交叉表：spot x gene_pair
            pivot_data = []
            
            for _, row in top_gene_pairs.iterrows():
                q_gene = row['query_gene']
                k_gene = row['key_gene']
                gene_pair = f"{q_gene}-{k_gene}"
                
                # 找到所有包含该基因对的行
                pair_data = gene_pairs_df[
                    (gene_pairs_df['query_gene'] == q_gene) & 
                    (gene_pairs_df['key_gene'] == k_gene)
                ]
                
                for _, p_row in pair_data.iterrows():
                    pivot_data.append({
                        'boundary_spot_idx': p_row['boundary_spot_idx'],
                        'gene_pair': gene_pair,
                        'attn_score': p_row['attn_score'],
                        'is_lr_pair': p_row['is_lr_pair']
                    })
            
            pivot_df = pd.DataFrame(pivot_data)
            
            if not pivot_df.empty:
                # 透视为热图格式
                heatmap_data = pivot_df.pivot_table(
                    index='boundary_spot_idx', 
                    columns='gene_pair', 
                    values='attn_score', 
                    aggfunc='mean',
                    fill_value=0
                )
                
                # 绘制热图（Nature风格）
                fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
                
                # 使用Nature友好的配色方案
                im = ax.imshow(heatmap_data.values, cmap='viridis', aspect='auto', 
                              interpolation='nearest')
                
                # 设置刻度标签
                ax.set_xticks(np.arange(len(heatmap_data.columns)))
                ax.set_yticks(np.arange(len(heatmap_data.index)))
                ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=9)
                ax.set_yticklabels(heatmap_data.index, fontsize=9)
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Attention Score', fontsize=11, color='#2C3E50')
                cbar.ax.tick_params(labelsize=9, colors='#2C3E50')
                cbar.outline.set_edgecolor('#DDDDDD')
                cbar.outline.set_linewidth(0.8)
                
                # 设置标签和标题
                ax.set_xlabel('Gene Pairs', fontsize=12, color='#2C3E50', fontweight='normal')
                ax.set_ylabel('Boundary Spot Index', fontsize=12, color='#2C3E50', fontweight='normal')
                ax.set_title('Gene Pair Attention Scores Across Boundary Spots', 
                           fontsize=13, fontweight='bold', color='#2C3E50', pad=15)
                
                # 优化边框
                for spine in ax.spines.values():
                    spine.set_edgecolor('#DDDDDD')
                    spine.set_linewidth(0.8)
                
                plt.tight_layout()
                
                heatmap_path = os.path.join(output_dir, 'boundary_gene_pairs_heatmap.png')
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
                
                print(f"基因对热图已保存到: {heatmap_path}")
    except Exception as e:
        print(f"绘制基因对热图时出错: {e}")
    
    print(f"边界交互统计已保存到: {count_path}")
    return count_path

def create_lr_dot_plot(gene_pairs_df: pd.DataFrame, coords_df: pd.DataFrame, output_dir: str):
    """
    创建配体-受体点图：显示不同聚类间的L-R交互
    """
    if gene_pairs_df.empty or 'is_lr_pair' not in gene_pairs_df.columns or not gene_pairs_df['is_lr_pair'].any():
        print("警告: 无可用的配体-受体数据")
        return None
    
    print("创建配体-受体点图...")
    
    # 确保有聚类信息
    if 'cluster' not in coords_df.columns:
        print("警告: 未找到聚类信息，无法创建聚类间配体-受体点图")
        return None
    
    # 仅保留配体-受体对
    lr_pairs = gene_pairs_df[gene_pairs_df['is_lr_pair']].copy()
    
    if lr_pairs.empty:
        print("警告: 数据中没有识别出配体-受体对")
        return None
    
    # 添加方向：从边界spot到邻居
    lr_pairs['direction'] = lr_pairs.apply(
        lambda row: f"C{row['boundary_cluster']}→C{row['neighbor_cluster']}", 
        axis=1
    )
    
    # 计算每个方向、每个L-R对的平均注意力分数
    lr_avg = lr_pairs.groupby(['direction', 'lr_direction'])['attn_score'].agg(['mean', 'count']).reset_index()
    
    # 找出每个方向中平均分数最高的前3个L-R对
    top_lr = []
    for direction, group in lr_avg.groupby('direction'):
        top_in_direction = group.nlargest(3, 'mean')
        top_lr.append(top_in_direction)
    
    if not top_lr:
        print("警告: 无法找到顶级配体-受体对")
        return None
    
    top_lr_df = pd.concat(top_lr)
    
    # 准备点图数据
    all_directions = top_lr_df['direction'].unique()
    all_lr_pairs = top_lr_df['lr_direction'].unique()
    
    # 创建点图矩阵
    dot_matrix = np.zeros((len(all_directions), len(all_lr_pairs)))
    size_matrix = np.zeros((len(all_directions), len(all_lr_pairs)))
    
    direction_to_idx = {d: i for i, d in enumerate(all_directions)}
    lr_pair_to_idx = {p: i for i, p in enumerate(all_lr_pairs)}
    
    for _, row in top_lr_df.iterrows():
        i = direction_to_idx[row['direction']]
        j = lr_pair_to_idx[row['lr_direction']]
        dot_matrix[i, j] = row['mean']
        size_matrix[i, j] = row['count']
    
    # 绘制点图
    plt.figure(figsize=(14, len(all_directions) * 0.6 + 2))
    
    # 归一化大小矩阵
    min_size, max_size = 50, 400
    if size_matrix.max() > size_matrix.min():
        norm_size = min_size + (max_size - min_size) * (size_matrix - size_matrix.min()) / (size_matrix.max() - size_matrix.min())
    else:
        norm_size = np.ones_like(size_matrix) * min_size
    
    for i, direction in enumerate(all_directions):
        for j, lr_pair in enumerate(all_lr_pairs):
            if dot_matrix[i, j] > 0:
                plt.scatter(
                    j, i, 
                    s=norm_size[i, j],
                    c=[dot_matrix[i, j]], 
                    cmap='viridis', 
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=1
                )
    
    # 添加颜色条
    norm = Normalize(vmin=dot_matrix[dot_matrix > 0].min(), vmax=dot_matrix.max())
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Average Attention Score')
    
    # 设置坐标轴
    plt.yticks(range(len(all_directions)), all_directions)
    plt.xticks(range(len(all_lr_pairs)), all_lr_pairs, rotation=90)
    
    plt.title('Top Ligand-Receptor Interactions Between Clusters', fontsize=16)
    plt.xlabel('Ligand-Receptor Pair', fontsize=14)
    plt.ylabel('Interaction Direction', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, 'ligand_receptor_dot_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"配体-受体点图已保存到: {output_path}")
    return output_path

# ============= 分析结果缓存函数 =============
def save_analysis_results(output_dir: str, spot_to_spot: pd.DataFrame, boundary_spots: List[int], 
                         cross_cluster_ratio: Dict[int, float], flow_vectors: Dict[int, Dict], 
                         gene_pairs_df: pd.DataFrame):
    """保存分析结果到文件"""
    analysis_dir = os.path.join(output_dir, 'analysis_cache')
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("保存分析结果...")
    
    # 保存spot-to-spot注意力
    spot_to_spot.to_csv(os.path.join(analysis_dir, 'spot_to_spot_attention.csv'), index=False)
    
    # 保存边界spots信息（转换为可序列化格式）
    boundary_data = {
        'boundary_spots': [int(x) for x in boundary_spots],
        'cross_cluster_ratio': {int(k): float(v) for k, v in cross_cluster_ratio.items()}
    }
    with open(os.path.join(analysis_dir, 'boundary_spots.json'), 'w') as f:
        json.dump(boundary_data, f, indent=2)
    
    # 保存流向量（转换为可序列化格式）
    flow_vectors_serializable = {}
    for spot_idx, data in flow_vectors.items():
        flow_vectors_serializable[str(spot_idx)] = {
            'position': data['position'],
            'target_idx': int(data['target_idx']),
            'direction': data['direction'].tolist(),
            'magnitude': float(data['magnitude']),
            'target_position': data['target_position']
        }
    
    with open(os.path.join(analysis_dir, 'flow_vectors.json'), 'w') as f:
        json.dump(flow_vectors_serializable, f, indent=2)
    
    # 保存基因对数据
    if not gene_pairs_df.empty:
        gene_pairs_df.to_csv(os.path.join(analysis_dir, 'boundary_gene_pairs.csv'), index=False)
    
    print(f"分析结果已保存到: {analysis_dir}")
    return analysis_dir

def load_analysis_results(output_dir: str):
    """从文件加载分析结果"""
    analysis_dir = os.path.join(output_dir, 'analysis_cache')
    
    if not os.path.exists(analysis_dir):
        return None
    
    print("加载缓存的分析结果...")
    
    try:
        # 加载spot-to-spot注意力
        spot_to_spot = pd.read_csv(os.path.join(analysis_dir, 'spot_to_spot_attention.csv'))
        
        # 加载边界spots信息
        with open(os.path.join(analysis_dir, 'boundary_spots.json'), 'r') as f:
            boundary_data = json.load(f)
        boundary_spots = boundary_data['boundary_spots']
        cross_cluster_ratio = boundary_data['cross_cluster_ratio']
        
        # 加载流向量
        with open(os.path.join(analysis_dir, 'flow_vectors.json'), 'r') as f:
            flow_vectors_data = json.load(f)
        
        flow_vectors = {}
        for spot_idx_str, data in flow_vectors_data.items():
            spot_idx = int(spot_idx_str)
            flow_vectors[spot_idx] = {
                'position': tuple(data['position']),
                'target_idx': data['target_idx'],
                'direction': np.array(data['direction']),
                'magnitude': data['magnitude'],
                'target_position': tuple(data['target_position'])
            }
        
        # 加载基因对数据
        gene_pairs_path = os.path.join(analysis_dir, 'boundary_gene_pairs.csv')
        if os.path.exists(gene_pairs_path):
            gene_pairs_df = pd.read_csv(gene_pairs_path)
        else:
            gene_pairs_df = pd.DataFrame()
        
        print("分析结果加载成功！")
        return {
            'spot_to_spot': spot_to_spot,
            'boundary_spots': boundary_spots,
            'cross_cluster_ratio': cross_cluster_ratio,
            'flow_vectors': flow_vectors,
            'gene_pairs_df': gene_pairs_df
        }
    
    except Exception as e:
        print(f"加载分析结果时出错: {e}")
        return None

# ============= 主函数 =============
def main(data_dir: str, output_dir: str = None, force_recompute: bool = False):
    """
    主函数：加载数据并执行所有分析和可视化
    """
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 尝试加载缓存的分析结果
    analysis_results = None
    if not force_recompute:
        analysis_results = load_analysis_results(output_dir)
    
    if analysis_results is None:
        print("执行完整分析...")
        
        # 加载数据
        data = load_data(data_dir)
        attn_df = data['attn_df']
        coords_df = data['coords_df']
        lr_df = data.get('lr_df', None)
        
        print(f"读取了{len(attn_df)}条注意力数据记录和{len(coords_df)}个空间坐标")
        
        # 1. 计算Spot-to-Spot注意力
        print("计算Spot-to-Spot注意力...")
        spot_to_spot = prepare_spot_to_spot_attention(attn_df)
        
        # 2. 识别边界spots
        print("识别边界spots...")
        boundary_spots, cross_cluster_ratio = identify_boundary_spots(coords_df, spot_to_spot)
        print(f"边界spots: {boundary_spots}")
        print(f"跨聚类比例: {cross_cluster_ratio}")
        # INSERT_YOUR_CODE
        import json
        boundary_spots_path = os.path.join(output_dir, 'boundary_spots.json')
        with open(boundary_spots_path, 'w', encoding='utf-8') as f:
            # 转换NumPy类型为Python原生类型以确保JSON序列化
            boundary_spots_serializable = [int(x) for x in boundary_spots]
            cross_cluster_ratio_serializable = {int(k): float(v) for k, v in cross_cluster_ratio.items()}
            
            json.dump({
                "boundary_spots": boundary_spots_serializable,
                "cross_cluster_ratio": cross_cluster_ratio_serializable
            }, f, ensure_ascii=False, indent=2)
        print(f"已保存boundary_spots到: {boundary_spots_path}")
        
        # 3. 计算注意力流向量
        print("计算注意力流向量...")
        flow_vectors = find_attention_flow_vectors(spot_to_spot, coords_df)
        
        # 4. 如果有边界spots，分析其基因对
        if boundary_spots:
            print("分析边界spots的基因对...")
            gene_pairs_df = find_gene_pairs_for_boundary_spots(attn_df, boundary_spots, coords_df, lr_df)
        else:
            gene_pairs_df = pd.DataFrame()
            print("警告: 未找到边界spots，跳过基因对分析")
        
        # 保存分析结果
        save_analysis_results(output_dir, spot_to_spot, boundary_spots, cross_cluster_ratio, flow_vectors, gene_pairs_df)
        
        analysis_results = {
            'spot_to_spot': spot_to_spot,
            'boundary_spots': boundary_spots,
            'cross_cluster_ratio': cross_cluster_ratio,
            'flow_vectors': flow_vectors,
            'gene_pairs_df': gene_pairs_df
        }
    else:
        print("使用缓存的分析结果")
        # 重新加载坐标数据用于可视化
        data = load_data(data_dir)
        coords_df = data['coords_df']
    
    # 执行可视化
    print("\n开始生成可视化...\n")
    
    # 5.1 空间交互网络
    plot_spatial_interaction_network(coords_df, analysis_results['spot_to_spot'], 
                                   analysis_results['boundary_spots'], 
                                   analysis_results['cross_cluster_ratio'], output_dir)
    
     # 5.2 注意力流向量场
    plot_attention_flow_vectors(coords_df, analysis_results['flow_vectors'], output_dir)
    # 5.3 边界基因交互
    if not analysis_results['gene_pairs_df'].empty:
        plot_boundary_gene_interactions(analysis_results['gene_pairs_df'], coords_df, output_dir)
        
        # 5.4 配体-受体点图
        create_lr_dot_plot(analysis_results['gene_pairs_df'], coords_df, output_dir)
    
    print(f"\n所有分析和可视化已完成！结果保存在: {output_dir}")
    return output_dir


def plot_attention_flow_vectors(coords_df: pd.DataFrame, flow_vectors: Dict[int, Dict], output_dir: str):
    """
    绘制注意力流向量场（Nature风格）
    """
    print("绘制注意力流向量场...")
    
    # Nature配色方案
    nature_cmap = plt.cm.get_cmap('Set3')  # 柔和的离散色
    
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')
    
    # 绘制所有spots（更优雅的样式）
    if 'cluster' in coords_df.columns:
        # 确保有数值化的聚类标签
        if coords_df['cluster'].dtype == 'object' and 'cluster_num' not in coords_df.columns:
            unique_clusters = coords_df['cluster'].unique()
            cluster_to_num = {cluster: i for i, cluster in enumerate(unique_clusters)}
            coords_df['cluster_num'] = coords_df['cluster'].map(cluster_to_num)
        
        scatter = ax.scatter(
            coords_df['x'], coords_df['y'],
            c=coords_df['cluster_num'], cmap=nature_cmap, 
            alpha=0.6, s=40, edgecolors='white', linewidths=0.5
        )
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cluster', fontsize=11, color='#2C3E50')
        cbar.ax.tick_params(labelsize=9, colors='#2C3E50')
        cbar.outline.set_edgecolor('#DDDDDD')
        cbar.outline.set_linewidth(0.8)
    else:
        ax.scatter(
            coords_df['x'], coords_df['y'],
            c='#CCCCCC', alpha=0.6, s=40, edgecolors='white', linewidths=0.5
        )
    
    # 提取位置和方向向量
    positions = []
    directions = []
    magnitudes = []
    
    for spot_idx, data in flow_vectors.items():
        positions.append(data['position'])
        directions.append(data['direction'])
        magnitudes.append(data['magnitude'])
    
    # 转换为NumPy数组
    positions = np.array(positions)
    directions = np.array(directions)
    magnitudes = np.array(magnitudes)
    
    # 归一化箭头长度，使其适合可视化
    max_magnitude = magnitudes.max() if len(magnitudes) > 0 else 1
    norm = Normalize(vmin=0, vmax=max_magnitude)
    
    # 计算合适的缩放比例
    x_range = coords_df['x'].max() - coords_df['x'].min()
    y_range = coords_df['y'].max() - coords_df['y'].min()
    scale_factor = min(x_range, y_range) * 0.03 / max_magnitude
    
    # 绘制箭头（Nature风格：更优雅的配色和样式）
    arrow_cmap = plt.cm.get_cmap('YlOrRd')  # 黄橙红渐变，更适合Nature风格
    
    for i in range(len(positions)):
        color = arrow_cmap(norm(magnitudes[i]))
        
        # 绘制箭头（更精细的样式）
        ax.arrow(
            positions[i][0], positions[i][1],
            directions[i][0] * scale_factor, directions[i][1] * scale_factor,
            head_width=scale_factor * max_magnitude * 0.25, 
            head_length=scale_factor * max_magnitude * 0.4,
            fc=color, ec='none', alpha=0.75,
            length_includes_head=True, zorder=3
        )
    
    # 添加颜色条（优化样式）
    sm = ScalarMappable(cmap='YlOrRd', norm=norm)
    sm.set_array([])
    cbar2 = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Attention Strength', fontsize=11, color='#2C3E50')
    cbar2.ax.tick_params(labelsize=9, colors='#2C3E50')
    cbar2.outline.set_edgecolor('#DDDDDD')
    cbar2.outline.set_linewidth(0.8)
    
    # 设置标签和标题
    ax.set_xlabel('X Coordinate', fontsize=12, color='#2C3E50', fontweight='normal')
    ax.set_ylabel('Y Coordinate', fontsize=12, color='#2C3E50', fontweight='normal')
    ax.set_title('Spatial Attention Flow Vector Field', 
                fontsize=13, fontweight='bold', color='#2C3E50', pad=15)
    
    # 优化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#DDDDDD')
        spine.set_linewidth(0.8)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.2, color='#E5E5E5', zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, 'attention_flow_vectors.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # 可选：绘制流场密度图（Nature风格）
    try:
        fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')
        
        # 绘制所有spots（淡背景）
        ax.scatter(
            coords_df['x'], coords_df['y'],
            c='#E8E8E8', alpha=0.4, s=25, edgecolors='none', zorder=1
        )
        
        # 创建网格
        x = np.linspace(coords_df['x'].min(), coords_df['x'].max(), 100)
        y = np.linspace(coords_df['y'].min(), coords_df['y'].max(), 100)
        X, Y = np.meshgrid(x, y)
        
        # 准备插值数据
        points = positions
        values_x = np.array([d[0] for d in directions]) * scale_factor
        values_y = np.array([d[1] for d in directions]) * scale_factor
        
        if len(points) >= 4:  # 需要至少4个点进行插值
            # 使用径向基函数进行插值
            grid_x = griddata(points, values_x, (X, Y), method='cubic', fill_value=0)
            grid_y = griddata(points, values_y, (X, Y), method='cubic', fill_value=0)
            
            # 计算流场强度
            speed = np.sqrt(grid_x**2 + grid_y**2)
            
            # 绘制流线（使用Nature友好的配色）
            streamplot = ax.streamplot(
                X, Y, grid_x, grid_y,
                density=1.5, color=speed, cmap='YlOrRd',
                linewidth=1.2, arrowsize=1.0, zorder=2
            )
            
            # 添加颜色条
            cbar = plt.colorbar(streamplot.lines, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Flow Intensity', fontsize=11, color='#2C3E50')
            cbar.ax.tick_params(labelsize=9, colors='#2C3E50')
            cbar.outline.set_edgecolor('#DDDDDD')
            cbar.outline.set_linewidth(0.8)
            
            # 设置标签和标题
            ax.set_xlabel('X Coordinate', fontsize=12, color='#2C3E50', fontweight='normal')
            ax.set_ylabel('Y Coordinate', fontsize=12, color='#2C3E50', fontweight='normal')
            ax.set_title('Attention Flow Density Visualization', 
                        fontsize=13, fontweight='bold', color='#2C3E50', pad=15)
            
            # 优化边框
            for spine in ax.spines.values():
                spine.set_edgecolor('#DDDDDD')
                spine.set_linewidth(0.8)
            
            # 添加淡网格
            ax.grid(True, linestyle='--', alpha=0.2, color='#E5E5E5', zorder=0)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            density_path = os.path.join(output_dir, 'attention_flow_density.png')
            plt.savefig(density_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"流场密度图已保存到: {density_path}")
    except Exception as e:
        print(f"绘制流场密度图时出错: {e}")
    
    print(f"注意力流向量场图已保存到: {output_path}")
    return output_path

def load_whole_slice_data(data_dir: str) -> Dict[str, Any]:
    """加载全切片数据（优化版本）"""
    data = {}
    
    # 首先读取配置文件以获取路径信息
    config_file = os.path.join(data_dir, "export_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # 使用配置中的路径
        data['config'] = config
        
        # 加载优化的数据结构
        if 'optimized_paths' in config:
            paths = config['optimized_paths']
            
            # Spot-to-Spot注意力
            if os.path.exists(paths['spot_to_spot']):
                data['spot_to_spot'] = pd.read_parquet(paths['spot_to_spot'])
            
            # 基因对注意力
            if os.path.exists(paths['gene_pairs']):
                data['gene_pairs'] = pd.read_parquet(paths['gene_pairs'])
            
            # 空间邻居
            if os.path.exists(paths['neighbors']):
                data['spot_neighbors'] = pd.read_pickle(paths['neighbors'])
        
        # 加载空间坐标
        if 'coords_path' in config and os.path.exists(config['coords_path']):
            data['coords_df'] = pd.read_csv(config['coords_path'])
        
        # 加载AnnData
        if 'adata_path' in config and os.path.exists(config['adata_path']):
            data['adata'] = sc.read_h5ad(config['adata_path'])
        
        # 加载配体-受体数据库
        if 'lr_db_path' in config and os.path.exists(config['lr_db_path']):
            data['lr_df'] = pd.read_csv(config['lr_db_path'])
    else:
        # 无配置文件时的备用加载方式
        print("警告: 未找到配置文件，使用备用加载方式")
        
        # 找到注意力数据
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet') and 'whole_slice' in f]
        if parquet_files:
            attn_path = os.path.join(data_dir, parquet_files[0])
            data['attn_df'] = pd.read_parquet(attn_path)
        else:
            # 尝试CSV
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'whole_slice' in f]
            if csv_files:
                attn_path = os.path.join(data_dir, csv_files[0])
                data['attn_df'] = pd.read_csv(attn_path)
        
        # 空间坐标
        coords_files = [f for f in os.listdir(data_dir) if 'coordinates' in f and f.endswith('.csv')]
        if coords_files:
            data['coords_df'] = pd.read_csv(os.path.join(data_dir, coords_files[0]))
        
        # AnnData
        h5ad_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
        if h5ad_files:
            data['adata'] = sc.read_h5ad(os.path.join(data_dir, h5ad_files[0]))
        
        # 配体-受体数据库
        lr_files = [f for f in os.listdir(data_dir) if 'lr_database' in f]
        if lr_files:
            data['lr_df'] = pd.read_csv(os.path.join(data_dir, lr_files[0]))
    
    # 如果没有优化的spot_to_spot，但有原始注意力数据，则计算它
    if 'spot_to_spot' not in data and 'attn_df' in data:
        print("计算Spot-to-Spot注意力...")
        attn_df = data['attn_df']
        spot_to_spot = attn_df.groupby(['center_global_idx', 'neighbor_global_idx'])['attn_sum_norm'].sum().reset_index()
        spot_to_spot.columns = ['source_idx', 'target_idx', 'attention_weight']
        data['spot_to_spot'] = spot_to_spot
    
    # 报告加载的数据
    print("已加载以下数据:")
    for key in data.keys():
        if key == 'attn_df':
            print(f"  - 注意力数据: {len(data[key])}行")
        elif key == 'coords_df':
            print(f"  - 空间坐标: {len(data[key])}个spots")
        elif key == 'spot_to_spot':
            print(f"  - Spot-to-Spot注意力: {len(data[key])}个连接")
        elif key == 'gene_pairs':
            print(f"  - 基因对注意力: {len(data[key])}个基因对")
        elif key == 'adata':
            print(f"  - AnnData: {data[key].n_obs}个spots, {data[key].n_vars}个基因")
        elif key == 'lr_df':
            print(f"  - 配体-受体数据库: {len(data[key])}个L-R对")
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spatial Attention Visualizer')
    parser.add_argument('--data_dir', type=str, default='./PDAC/whole_slice_data_20251028_173836', help='Directory containing exported spatial data')
    parser.add_argument('--output_dir', type=str, default='./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_boundary', help='Directory to save visualizations (default: data_dir/visualizations)')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation even if cached results exist')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir, args.force_recompute)




"""
# 如果需要重新分析
python spatial_attention_visualizer.py \
    --data_dir ./whole_slice_data_20251010_193054 \
    --output_dir ./whole_slice_data_20251010_193054/visualizations \
    --force_recompute

"""