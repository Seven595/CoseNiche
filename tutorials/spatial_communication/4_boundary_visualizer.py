""" Spatial attention visualization and analysis script - Perform advanced analysis and visualization on exported spatial attention data - Implements three main visualizations: 1. Spot-levelplot 2. Spotsanalysis and gene interactions 3. global attention flow vector field """

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
# Nature
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

# of Warning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============= =============
def load_data(data_dir: str) -> Dict[str, Any]:
    """
    Load required data files
    """
    data = {}
    
    # found CSVfile
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'combined' in f]
    if not csv_files:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'comprehensive' in f]
    if not csv_files:
        # found whole_slice_attention file
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'whole_slice_attention' in f]
    
    if csv_files:
        attn_csv_path = os.path.join(data_dir, csv_files[0])
        print(f"Using attention data: {attn_csv_path}")
        data['attn_df'] = pd.read_csv(attn_csv_path)
    else:
        raise FileNotFoundError(f" in {data_dir} in not found CSVfile")
    
    # loadspatial coordinates
    coord_files = [f for f in os.listdir(data_dir) if 'coordinates' in f and f.endswith('.csv')]
    if coord_files:
        coord_csv_path = os.path.join(data_dir, coord_files[0])
        print(f"Using spatial coordinate data: {coord_csv_path}")
        data['coords_df'] = pd.read_csv(coord_csv_path)
    else:
        raise FileNotFoundError(f" in {data_dir} in not foundspatial coordinatesCSVfile")
    
    # loadAnnData (if in)
    h5ad_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
    if h5ad_files:
        h5ad_path = os.path.join(data_dir, h5ad_files[0])
        print(f"Using AnnData file: {h5ad_path}")
        data['adata'] = sc.read_h5ad(h5ad_path)
    else:
        print("not foundAnnDatafile,CSV")
        data['adata'] = None
    
    # loadligand-receptordatabase (if in)
    lr_files = [f for f in os.listdir(data_dir) if 'lr_database' in f and f.endswith('.csv')]
    if lr_files:
        lr_path = os.path.join(data_dir, lr_files[0])
        print(f"Using ligand-receptor database: {lr_path}")
        data['lr_df'] = pd.read_csv(lr_path)
    else:
        print("not foundligand-receptordatabase,L-Ranalysis")
        data['lr_df'] = None
    
    return data

def prepare_spot_to_spot_attention(attn_df: pd.DataFrame) -> pd.DataFrame:
    """ computespot to spot of total and """
    print("Computing spot-to-spot attention...")
    
    # in of column
    required_cols = ['center_global_idx', 'neighbor_global_idx', 'attn_sum_norm']
    for col in required_cols:
        if col not in attn_df.columns:
            if col == 'attn_sum_norm' and 'attn_sum' in attn_df.columns:
                # compute of total and
                attn_sum_by_center = attn_df.groupby('center_global_idx')['attn_sum'].sum()
                attn_df['attn_sum_norm'] = attn_df.apply(
                    lambda x: x['attn_sum'] / attn_sum_by_center[x['center_global_idx']] 
                    if x['center_global_idx'] in attn_sum_by_center and attn_sum_by_center[x['center_global_idx']] > 0 
                    else 0, 
                    axis=1
                )
            else:
                raise ValueError(f"Required column is missing from attention data: {col}")
    
    # compute from spot to spot of total
    spot_to_spot = attn_df.groupby(['center_global_idx', 'neighbor_global_idx'])['attn_sum_norm'].sum().reset_index()
    
    # column
    spot_to_spot.columns = ['source_idx', 'target_idx', 'attention_weight']
    
    # NaN
    spot_to_spot = spot_to_spot.dropna()
    
    print(f"Found{len(spot_to_spot)}spot-to-spot connections")
    return spot_to_spot

def identify_boundary_spots(coords_df: pd.DataFrame, spot_to_spot: pd.DataFrame, 
                           threshold: float = 0.3) -> List[int]:
    """ Identifying boundary spots (and different cluster of spots of spots) """
    print("Identifying boundary spots...")
    
    # cluster information
    if 'cluster' not in coords_df.columns:
        print("Warning: not foundcluster information,using KMeans for simple clustering")
        # perform of KMeanscluster
        kmeans = KMeans(n_clusters=7, random_state=42)
        coords_df['cluster'] = kmeans.fit_predict(coords_df[['x', 'y']])
    
    # cluster for (if no)
    if coords_df['cluster'].dtype == 'object':
        unique_clusters = coords_df['cluster'].unique()
        cluster_to_num = {cluster: i for i, cluster in enumerate(unique_clusters)}
        coords_df['cluster_num'] = coords_df['cluster'].map(cluster_to_num)
    else:
        coords_df['cluster_num'] = coords_df['cluster']
    
    # createspot_idx to cluster of ()
    idx_to_cluster = dict(zip(coords_df['spot_idx'], coords_df['cluster_num']))
    
    #  found boundary spots
    boundary_spots = []
    cross_cluster_ratio = {}
    
    # compute each spot different cluster of spots of
    for spot_idx in coords_df['spot_idx'].unique():
        if spot_idx not in idx_to_cluster:
            continue
            
        # spot of cluster
        spot_cluster = idx_to_cluster[spot_idx]
        
        # found spot of all spots
        targets = spot_to_spot[spot_to_spot['source_idx'] == spot_idx]
        
        if targets.empty:
            continue
            
        # and
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
            
            # if cross-cluster attention ratio, for spot
            if cross_ratio > threshold:
                boundary_spots.append(spot_idx)
    
    print(f"Identified{len(boundary_spots)} boundary spots (cross-cluster attention ratio > {threshold})")
    return boundary_spots, cross_cluster_ratio

def find_attention_flow_vectors(spot_to_spot: pd.DataFrame, coords_df: pd.DataFrame) -> Dict[int, Dict]:
    """ compute each spot of """
    print("Computing attention flow vectors...")
    
    # createspot_idx to of
    idx_to_coords = {}
    for _, row in coords_df.iterrows():
        idx_to_coords[row['spot_idx']] = (row['x'], row['y'])
    
    # each spot of
    flow_vectors = {}
    
    for spot_idx in coords_df['spot_idx'].unique():
        if spot_idx not in idx_to_coords:
            continue
            
        # found spot of all spots
        targets = spot_to_spot[spot_to_spot['source_idx'] == spot_idx]
        
        if targets.empty:
            continue
            
        # of
        max_target_row = targets.loc[targets['attention_weight'].idxmax()]
        max_target_idx = max_target_row['target_idx']
        
        if max_target_idx not in idx_to_coords:
            continue
            
        # compute
        source_pos = idx_to_coords[spot_idx]
        target_pos = idx_to_coords[max_target_idx]
        
        # : from
        direction = np.array([target_pos[0] - source_pos[0], target_pos[1] - source_pos[1]])
        
        # compute
        magnitude = np.linalg.norm(direction)
        
        if magnitude > 0:
            # * attention weights
            normalized_direction = direction / magnitude * max_target_row['attention_weight']
            
            flow_vectors[spot_idx] = {
                'position': source_pos,
                'target_idx': max_target_idx,
                'direction': normalized_direction,
                'magnitude': max_target_row['attention_weight'],
                'target_position': target_pos
            }
    
    print(f" for {len(flow_vectors)}spots")
    return flow_vectors

def find_gene_pairs_for_boundary_spots(attn_df: pd.DataFrame, boundary_spots: List[int], 
                                      coords_df: pd.DataFrame, lr_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """ boundary spots in of gene pairs, in of ligand-receptor """
    print("analysisboundary spots of gene pairs...")
    
    # createspot_idx to cluster of
    if 'cluster' in coords_df.columns:
        # of cluster
        if coords_df['cluster'].dtype == 'object' and 'cluster_num' not in coords_df.columns:
            unique_clusters = coords_df['cluster'].unique()
            cluster_to_num = {cluster: i for i, cluster in enumerate(unique_clusters)}
            coords_df['cluster_num'] = coords_df['cluster'].map(cluster_to_num)
        
        idx_to_cluster = dict(zip(coords_df['spot_idx'], coords_df['cluster_num']))
    else:
        print("Warning: not foundcluster information,/gene pairs")
        idx_to_cluster = {idx: 0 for idx in coords_df['spot_idx']}
    
    # ligand-receptor
    lr_pairs = {}
    if lr_df is not None:
        for _, row in lr_df.iterrows():
            lr_pairs[(row['ligand'], row['receptor'])] = True
    
    # boundary spots of gene pairs
    gene_pairs_data = []
    
    for spot_idx in boundary_spots:
        if spot_idx not in idx_to_cluster:
            continue
            
        # spot of cluster
        spot_cluster = idx_to_cluster[spot_idx]
        
        # filterspot of
        spot_data = attn_df[attn_df['center_global_idx'] == spot_idx]
        
        if spot_data.empty:
            continue
        
        # filter and
        for neighbor_idx in spot_data['neighbor_global_idx'].unique():
            if pd.isna(neighbor_idx) or neighbor_idx not in idx_to_cluster:
                continue
                
            neighbor_cluster = idx_to_cluster[int(neighbor_idx)]
            interaction_type = 'same_cluster' if neighbor_cluster == spot_cluster else 'diff_cluster'
            
            # of gene pairs
            neighbor_data = spot_data[spot_data['neighbor_global_idx'] == neighbor_idx]
            
            for _, row in neighbor_data.iterrows():
                if pd.isna(row['q_gene_symbol']) or pd.isna(row['kv_gene_symbol']) or not row['q_gene_symbol'] or not row['kv_gene_symbol']:
                    continue
                    
                # Check for ligand-receptor
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
    
    # createDataFrame
    gene_pairs_df = pd.DataFrame(gene_pairs_data)
    
    if not gene_pairs_df.empty:
        print(f"Found{len(gene_pairs_df)} gene pairs, in {gene_pairs_df['is_lr_pair'].sum()} ligand-receptor")
    else:
        print("not foundgene pairs")
    
    return gene_pairs_df

# ============= can =============
def plot_spatial_interaction_network(coords_df: pd.DataFrame, spot_to_spot: pd.DataFrame, 
                                    boundary_spots: List[int], cross_cluster_ratio: Dict[int, float],
                                    output_dir: str, percentile_threshold: int = 60):
    """ plot,region """
    print("plot...")
    
    plt.figure(figsize=(16, 14))
    
    # plot all spots
    if 'cluster' in coords_df.columns:
        # cluster for
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
    
    # boundary spots
    if boundary_spots:
        boundary_df = coords_df[coords_df['spot_idx'].isin(boundary_spots)]
        plt.scatter(
            boundary_df['x'], boundary_df['y'],
            s=100, facecolors='none', edgecolors='red', linewidth=2,
            label='Boundary Spots'
        )
    
    # createspot_idx to of
    idx_to_coords = {}
    for _, row in coords_df.iterrows():
        idx_to_coords[row['spot_idx']] = (row['x'], row['y'])
    
    # for,attention weights in before N% of
    threshold = np.percentile(spot_to_spot['attention_weight'], percentile_threshold)
    strong_connections = spot_to_spot[spot_to_spot['attention_weight'] >= threshold]
    
    print(f"attention weights in before {100-percentile_threshold}% of (: {threshold:.4f})")
    
    # plot
    for _, row in strong_connections.iterrows():
        source_idx = row['source_idx']
        target_idx = row['target_idx']
        
        if source_idx not in idx_to_coords or target_idx not in idx_to_coords:
            continue
        
        source_pos = idx_to_coords[source_idx]
        target_pos = idx_to_coords[target_idx]
        
        # compute after of and
        weight_norm = row['attention_weight'] / strong_connections['attention_weight'].max()
        line_width = max(0.5, weight_norm * 4)
        alpha = max(0.2, min(0.9, weight_norm * 1.5))
        
        # if spot of,
        if source_idx in boundary_spots:
            line_color = 'red'
        else:
            line_color = 'blue'
        
        # plot of
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
    
    # (if cluster information)
    if 'cluster' in coords_df.columns:
        cbar = plt.colorbar(scatter, orientation='vertical', pad=0.01)
        cbar.set_label('Cluster ID', fontsize=12)
    
    # plot
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
    
    # Saveimage
    output_path = os.path.join(output_dir, 'spatial_interaction_network.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"spatial interaction network plotSaved to : {output_path}")
    
    # optional:plotboundary spots of cross-cluster attention ratioplot
    if cross_cluster_ratio and len(cross_cluster_ratio) > 0:
        plt.figure(figsize=(10, 6))
        
        # for DataFrameplot
        ratio_df = pd.DataFrame({
            'spot_idx': list(cross_cluster_ratio.keys()),
            'cross_cluster_ratio': list(cross_cluster_ratio.values())
        })
        
        # Note.
        ratio_df['is_boundary'] = ratio_df['spot_idx'].isin(boundary_spots)
        
        # plot
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
        
        print(f"boundary spotsplotSaved to : {ratio_path}")
    
    return output_path

def plot_boundary_gene_interactions(gene_pairs_df: pd.DataFrame, coords_df: pd.DataFrame, output_dir: str):
    """ analysis can boundary spots of gene interactions (Nature) """
    if gene_pairs_df.empty:
        print("Warning: gene interactionsavailable")
        return None
    
    print("analysisboundary spots of gene interactions...")
    
    # Nature
    nature_colors = {
        'same_cluster': '#5BA3C7',  # Note.
        'diff_cluster': '#E87B8A'   # Note.
    }
    
    # 1. analysisvs
    interaction_counts = gene_pairs_df.groupby('interaction_type').size()
    interaction_avg_score = gene_pairs_df.groupby('interaction_type')['attn_score'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    
    # Nature of plot
    x_pos = np.arange(len(interaction_counts))
    colors = [nature_colors.get(idx, '#808080') for idx in interaction_counts.index]
    
    bars = ax.bar(x_pos, interaction_counts.values, color=colors, 
                  edgecolor='white', linewidth=1.5, alpha=0.9, width=0.6)
    
    # (of)
    for i, (v, avg) in enumerate(zip(interaction_counts.values, interaction_avg_score.values)):
        ax.text(i, v + max(interaction_counts.values) * 0.02, 
                f"{v}", 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2C3E50')
        ax.text(i, v / 2, 
                f"avg: {avg:.3f}", 
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # ()
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Same Cluster', 'Different Cluster'], fontsize=11, color='#2C3E50')
    ax.set_ylabel('Count', fontsize=12, color='#2C3E50', fontweight='normal')
    ax.set_title('Boundary Spots: Interaction Analysis', fontsize=13, fontweight='bold', 
                 color='#2C3E50', pad=15)
    
    # Optimization and
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
    
    # 2. analysisligand-receptor (Nature)
    if 'is_lr_pair' in gene_pairs_df.columns and gene_pairs_df['is_lr_pair'].any():
        # before 15 ligand-receptor (can)
        top_lr_pairs = (gene_pairs_df[gene_pairs_df['is_lr_pair']]
                        .groupby(['lr_direction', 'interaction_type'])['attn_score']
                        .mean()
                        .reset_index()
                        .sort_values('attn_score', ascending=False)
                        .head(15))
        
        fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
        
        # prepared data
        lr_pairs = top_lr_pairs['lr_direction'].unique()
        x_pos = np.arange(len(lr_pairs))
        
        # plot
        same_data = []
        diff_data = []
        for lr in lr_pairs:
            same_val = top_lr_pairs[(top_lr_pairs['lr_direction'] == lr) & 
                                   (top_lr_pairs['interaction_type'] == 'same_cluster')]['attn_score']
            diff_val = top_lr_pairs[(top_lr_pairs['lr_direction'] == lr) & 
                                   (top_lr_pairs['interaction_type'] == 'diff_cluster')]['attn_score']
            same_data.append(same_val.values[0] if len(same_val) > 0 else 0)
            diff_data.append(diff_val.values[0] if len(diff_val) > 0 else 0)
        
        # plotplot
        width = 0.35
        ax.bar(x_pos - width/2, same_data, width, label='Same Cluster',
               color=nature_colors['same_cluster'], edgecolor='white', linewidth=1, alpha=0.9)
        ax.bar(x_pos + width/2, diff_data, width, label='Different Cluster',
               color=nature_colors['diff_cluster'], edgecolor='white', linewidth=1, alpha=0.9)
        
        # and
        ax.set_xlabel('Ligand-Receptor Pair', fontsize=12, color='#2C3E50', fontweight='normal')
        ax.set_ylabel('Average Attention Score', fontsize=12, color='#2C3E50', fontweight='normal')
        ax.set_title('Top Ligand-Receptor Pairs in Boundary Interactions', 
                    fontsize=13, fontweight='bold', color='#2C3E50', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lr_pairs, rotation=45, ha='right', fontsize=9)
        
        # plot
        ax.legend(loc='upper right', frameon=True, edgecolor='#DDDDDD', 
                 fontsize=9, framealpha=1.0)
        
        # Optimization and
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
        
        print(f"ligand-receptoranalysisSaved to : {lr_path}")
    
    # 3. plot:spot each gene pairs of
    try:
        # select before 15 of gene pairs
        top_gene_pairs = (gene_pairs_df
                        .groupby(['query_gene', 'key_gene'])['attn_score']
                        .mean()
                        .reset_index()
                        .sort_values('attn_score', ascending=False)
                        .head(15))
        
        if len(top_gene_pairs) > 1:  # 2 gene pairsplotplot
            # createtable:spot x gene_pair
            pivot_data = []
            
            for _, row in top_gene_pairs.iterrows():
                q_gene = row['query_gene']
                k_gene = row['key_gene']
                gene_pair = f"{q_gene}-{k_gene}"
                
                # found all containsgene pairs of rows
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
                # for plotformat
                heatmap_data = pivot_df.pivot_table(
                    index='boundary_spot_idx', 
                    columns='gene_pair', 
                    values='attn_score', 
                    aggfunc='mean',
                    fill_value=0
                )
                
                # plotplot (Nature)
                fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
                
                # Nature of
                im = ax.imshow(heatmap_data.values, cmap='viridis', aspect='auto', 
                              interpolation='nearest')
                
                # Note.
                ax.set_xticks(np.arange(len(heatmap_data.columns)))
                ax.set_yticks(np.arange(len(heatmap_data.index)))
                ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=9)
                ax.set_yticklabels(heatmap_data.index, fontsize=9)
                
                # Note.
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Attention Score', fontsize=11, color='#2C3E50')
                cbar.ax.tick_params(labelsize=9, colors='#2C3E50')
                cbar.outline.set_edgecolor('#DDDDDD')
                cbar.outline.set_linewidth(0.8)
                
                # and
                ax.set_xlabel('Gene Pairs', fontsize=12, color='#2C3E50', fontweight='normal')
                ax.set_ylabel('Boundary Spot Index', fontsize=12, color='#2C3E50', fontweight='normal')
                ax.set_title('Gene Pair Attention Scores Across Boundary Spots', 
                           fontsize=13, fontweight='bold', color='#2C3E50', pad=15)
                
                # Optimization
                for spine in ax.spines.values():
                    spine.set_edgecolor('#DDDDDD')
                    spine.set_linewidth(0.8)
                
                plt.tight_layout()
                
                heatmap_path = os.path.join(output_dir, 'boundary_gene_pairs_heatmap.png')
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
                
                print(f"gene pairsplotSaved to : {heatmap_path}")
    except Exception as e:
        print(f"plotgene pairsplot: {e}")
    
    print(f"saved to : {count_path}")
    return count_path

def create_lr_dot_plot(gene_pairs_df: pd.DataFrame, coords_df: pd.DataFrame, output_dir: str):
    """ createligand-receptordot plot: different cluster of L-R """
    if gene_pairs_df.empty or 'is_lr_pair' not in gene_pairs_df.columns or not gene_pairs_df['is_lr_pair'].any():
        print("Warning: available of ligand-receptor")
        return None
    
    print("createligand-receptordot plot...")
    
    # cluster information
    if 'cluster' not in coords_df.columns:
        print("Warning: not foundcluster information,createclusterligand-receptordot plot")
        return None
    
    # ligand-receptor
    lr_pairs = gene_pairs_df[gene_pairs_df['is_lr_pair']].copy()
    
    if lr_pairs.empty:
        print("Warning: in no ligand-receptor")
        return None
    
    # : from spot to
    lr_pairs['direction'] = lr_pairs.apply(
        lambda row: f"C{row['boundary_cluster']}→C{row['neighbor_cluster']}", 
        axis=1
    )
    
    # compute each, each L-R of
    lr_avg = lr_pairs.groupby(['direction', 'lr_direction'])['attn_score'].agg(['mean', 'count']).reset_index()
    
    # each in of before 3 L-R
    top_lr = []
    for direction, group in lr_avg.groupby('direction'):
        top_in_direction = group.nlargest(3, 'mean')
        top_lr.append(top_in_direction)
    
    if not top_lr:
        print("Warning: found ligand-receptor")
        return None
    
    top_lr_df = pd.concat(top_lr)
    
    # dot plot
    all_directions = top_lr_df['direction'].unique()
    all_lr_pairs = top_lr_df['lr_direction'].unique()
    
    # createdot plot
    dot_matrix = np.zeros((len(all_directions), len(all_lr_pairs)))
    size_matrix = np.zeros((len(all_directions), len(all_lr_pairs)))
    
    direction_to_idx = {d: i for i, d in enumerate(all_directions)}
    lr_pair_to_idx = {p: i for i, p in enumerate(all_lr_pairs)}
    
    for _, row in top_lr_df.iterrows():
        i = direction_to_idx[row['direction']]
        j = lr_pair_to_idx[row['lr_direction']]
        dot_matrix[i, j] = row['mean']
        size_matrix[i, j] = row['count']
    
    # plotdot plot
    plt.figure(figsize=(14, len(all_directions) * 0.6 + 2))
    
    # size
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
    
    # Note.
    norm = Normalize(vmin=dot_matrix[dot_matrix > 0].min(), vmax=dot_matrix.max())
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Average Attention Score')
    
    # Note.
    plt.yticks(range(len(all_directions)), all_directions)
    plt.xticks(range(len(all_lr_pairs)), all_lr_pairs, rotation=90)
    
    plt.title('Top Ligand-Receptor Interactions Between Clusters', fontsize=16)
    plt.xlabel('Ligand-Receptor Pair', fontsize=14)
    plt.ylabel('Interaction Direction', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Saveimage
    output_path = os.path.join(output_dir, 'ligand_receptor_dot_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ligand-receptordot plotSaved to : {output_path}")
    return output_path

# ============= analysiscache =============
def save_analysis_results(output_dir: str, spot_to_spot: pd.DataFrame, boundary_spots: List[int], 
                         cross_cluster_ratio: Dict[int, float], flow_vectors: Dict[int, Dict], 
                         gene_pairs_df: pd.DataFrame):
    """Saveanalysis to file"""
    analysis_dir = os.path.join(output_dir, 'analysis_cache')
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("Saveanalysis...")
    
    # Savespot-to-spot
    spot_to_spot.to_csv(os.path.join(analysis_dir, 'spot_to_spot_attention.csv'), index=False)
    
    # Save boundary spots (for can columnformat)
    boundary_data = {
        'boundary_spots': [int(x) for x in boundary_spots],
        'cross_cluster_ratio': {int(k): float(v) for k, v in cross_cluster_ratio.items()}
    }
    with open(os.path.join(analysis_dir, 'boundary_spots.json'), 'w') as f:
        json.dump(boundary_data, f, indent=2)
    
    # save (for can columnformat)
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
    
    # Save gene pairs
    if not gene_pairs_df.empty:
        gene_pairs_df.to_csv(os.path.join(analysis_dir, 'boundary_gene_pairs.csv'), index=False)
    
    print(f"analysisSaved to : {analysis_dir}")
    return analysis_dir

def load_analysis_results(output_dir: str):
    """ from fileloadanalysis"""
    analysis_dir = os.path.join(output_dir, 'analysis_cache')
    
    if not os.path.exists(analysis_dir):
        return None
    
    print("loadCache of analysis...")
    
    try:
        # loadspot-to-spot
        spot_to_spot = pd.read_csv(os.path.join(analysis_dir, 'spot_to_spot_attention.csv'))
        
        # load boundary spots
        with open(os.path.join(analysis_dir, 'boundary_spots.json'), 'r') as f:
            boundary_data = json.load(f)
        boundary_spots = boundary_data['boundary_spots']
        cross_cluster_ratio = boundary_data['cross_cluster_ratio']
        
        # load
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
        
        # load gene pairs
        gene_pairs_path = os.path.join(analysis_dir, 'boundary_gene_pairs.csv')
        if os.path.exists(gene_pairs_path):
            gene_pairs_df = pd.read_csv(gene_pairs_path)
        else:
            gene_pairs_df = pd.DataFrame()
        
        print("analysisloadsuccessful！")
        return {
            'spot_to_spot': spot_to_spot,
            'boundary_spots': boundary_spots,
            'cross_cluster_ratio': cross_cluster_ratio,
            'flow_vectors': flow_vectors,
            'gene_pairs_df': gene_pairs_df
        }
    
    except Exception as e:
        print(f"loadanalysis: {e}")
        return None

# ============= =============
def main(data_dir: str, output_dir: str = None, force_recompute: bool = False):
    """ :loadrows all analysis and can """
    # createOutput directory
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # load cache of analysis
    analysis_results = None
    if not force_recompute:
        analysis_results = load_analysis_results(output_dir)
    
    if analysis_results is None:
        print("rowsanalysis...")
        
        # load
        data = load_data(data_dir)
        attn_df = data['attn_df']
        coords_df = data['coords_df']
        lr_df = data.get('lr_df', None)
        
        print(f"reading{len(attn_df)}attention data and {len(coords_df)} spatial coordinates")
        
        # 1. Computing spot-to-spot attention
        print("Computing spot-to-spot attention...")
        spot_to_spot = prepare_spot_to_spot_attention(attn_df)
        
        # 2. Identifying boundary spots
        print("Identifying boundary spots...")
        boundary_spots, cross_cluster_ratio = identify_boundary_spots(coords_df, spot_to_spot)
        print(f"boundary spots: {boundary_spots}")
        print(f"cluster: {cross_cluster_ratio}")
        # INSERT_YOUR_CODE
        import json
        boundary_spots_path = os.path.join(output_dir, 'boundary_spots.json')
        with open(boundary_spots_path, 'w', encoding='utf-8') as f:
            # Num Py for Python JSONcolumn
            boundary_spots_serializable = [int(x) for x in boundary_spots]
            cross_cluster_ratio_serializable = {int(k): float(v) for k, v in cross_cluster_ratio.items()}
            
            json.dump({
                "boundary_spots": boundary_spots_serializable,
                "cross_cluster_ratio": cross_cluster_ratio_serializable
            }, f, ensure_ascii=False, indent=2)
        print(f"Savedboundary_spots to : {boundary_spots_path}")
        
        # 3. Computing attention flow vectors
        print("Computing attention flow vectors...")
        flow_vectors = find_attention_flow_vectors(spot_to_spot, coords_df)
        
        # 4. if boundary spots,analysisgene pairs
        if boundary_spots:
            print("analysisboundary spots of gene pairs...")
            gene_pairs_df = find_gene_pairs_for_boundary_spots(attn_df, boundary_spots, coords_df, lr_df)
        else:
            gene_pairs_df = pd.DataFrame()
            print("Warning: not foundboundary spots,Skippinggene pairsanalysis")
        
        # Saveanalysis
        save_analysis_results(output_dir, spot_to_spot, boundary_spots, cross_cluster_ratio, flow_vectors, gene_pairs_df)
        
        analysis_results = {
            'spot_to_spot': spot_to_spot,
            'boundary_spots': boundary_spots,
            'cross_cluster_ratio': cross_cluster_ratio,
            'flow_vectors': flow_vectors,
            'gene_pairs_df': gene_pairs_df
        }
    else:
        print("cache of analysis")
        # load for can
        data = load_data(data_dir)
        coords_df = data['coords_df']
    
    # rows can
    print("\n Start can...\n")
    
    # 5.1
    plot_spatial_interaction_network(coords_df, analysis_results['spot_to_spot'], 
                                   analysis_results['boundary_spots'], 
                                   analysis_results['cross_cluster_ratio'], output_dir)
    
     # 5.2
    plot_attention_flow_vectors(coords_df, analysis_results['flow_vectors'], output_dir)
    # 5.3 gene interactions
    if not analysis_results['gene_pairs_df'].empty:
        plot_boundary_gene_interactions(analysis_results['gene_pairs_df'], coords_df, output_dir)
        
        # 5.4 ligand-receptordot plot
        create_lr_dot_plot(analysis_results['gene_pairs_df'], coords_df, output_dir)
    
    print(f"\n all analysis and can completed！save in : {output_dir}")
    return output_dir


def plot_attention_flow_vectors(coords_df: pd.DataFrame, flow_vectors: Dict[int, Dict], output_dir: str):
    """ plot (Nature) """
    print("plot...")
    
    # Nature
    nature_cmap = plt.cm.get_cmap('Set3')  # and of
    
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')
    
    # plot all spots (of)
    if 'cluster' in coords_df.columns:
        # of cluster
        if coords_df['cluster'].dtype == 'object' and 'cluster_num' not in coords_df.columns:
            unique_clusters = coords_df['cluster'].unique()
            cluster_to_num = {cluster: i for i, cluster in enumerate(unique_clusters)}
            coords_df['cluster_num'] = coords_df['cluster'].map(cluster_to_num)
        
        scatter = ax.scatter(
            coords_df['x'], coords_df['y'],
            c=coords_df['cluster_num'], cmap=nature_cmap, 
            alpha=0.6, s=40, edgecolors='white', linewidths=0.5
        )
        
        # Note.
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
    
    # and
    positions = []
    directions = []
    magnitudes = []
    
    for spot_idx, data in flow_vectors.items():
        positions.append(data['position'])
        directions.append(data['direction'])
        magnitudes.append(data['magnitude'])
    
    # for Num Py
    positions = np.array(positions)
    directions = np.array(directions)
    magnitudes = np.array(magnitudes)
    
    # , can
    max_magnitude = magnitudes.max() if len(magnitudes) > 0 else 1
    norm = Normalize(vmin=0, vmax=max_magnitude)
    
    # compute of
    x_range = coords_df['x'].max() - coords_df['x'].min()
    y_range = coords_df['y'].max() - coords_df['y'].min()
    scale_factor = min(x_range, y_range) * 0.03 / max_magnitude
    
    # plot (Nature: of and)
    arrow_cmap = plt.cm.get_cmap('YlOrRd')  # ,Nature
    
    for i in range(len(positions)):
        color = arrow_cmap(norm(magnitudes[i]))
        
        # plot (of)
        ax.arrow(
            positions[i][0], positions[i][1],
            directions[i][0] * scale_factor, directions[i][1] * scale_factor,
            head_width=scale_factor * max_magnitude * 0.25, 
            head_length=scale_factor * max_magnitude * 0.4,
            fc=color, ec='none', alpha=0.75,
            length_includes_head=True, zorder=3
        )
    
    # (Optimization)
    sm = ScalarMappable(cmap='YlOrRd', norm=norm)
    sm.set_array([])
    cbar2 = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Attention Strength', fontsize=11, color='#2C3E50')
    cbar2.ax.tick_params(labelsize=9, colors='#2C3E50')
    cbar2.outline.set_edgecolor('#DDDDDD')
    cbar2.outline.set_linewidth(0.8)
    
    # and
    ax.set_xlabel('X Coordinate', fontsize=12, color='#2C3E50', fontweight='normal')
    ax.set_ylabel('Y Coordinate', fontsize=12, color='#2C3E50', fontweight='normal')
    ax.set_title('Spatial Attention Flow Vector Field', 
                fontsize=13, fontweight='bold', color='#2C3E50', pad=15)
    
    # Optimization
    for spine in ax.spines.values():
        spine.set_edgecolor('#DDDDDD')
        spine.set_linewidth(0.8)
    
    # Note.
    ax.grid(True, linestyle='--', alpha=0.2, color='#E5E5E5', zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Saveimage
    output_path = os.path.join(output_dir, 'attention_flow_vectors.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # optional:plotplot (Nature)
    try:
        fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')
        
        # plot all spots ()
        ax.scatter(
            coords_df['x'], coords_df['y'],
            c='#E8E8E8', alpha=0.4, s=25, edgecolors='none', zorder=1
        )
        
        # create
        x = np.linspace(coords_df['x'].min(), coords_df['x'].max(), 100)
        y = np.linspace(coords_df['y'].min(), coords_df['y'].max(), 100)
        X, Y = np.meshgrid(x, y)
        
        # Note.
        points = positions
        values_x = np.array([d[0] for d in directions]) * scale_factor
        values_y = np.array([d[1] for d in directions]) * scale_factor
        
        if len(points) >= 4:  # 4 perform
            # perform
            grid_x = griddata(points, values_x, (X, Y), method='cubic', fill_value=0)
            grid_y = griddata(points, values_y, (X, Y), method='cubic', fill_value=0)
            
            # compute
            speed = np.sqrt(grid_x**2 + grid_y**2)
            
            # plot (Nature of)
            streamplot = ax.streamplot(
                X, Y, grid_x, grid_y,
                density=1.5, color=speed, cmap='YlOrRd',
                linewidth=1.2, arrowsize=1.0, zorder=2
            )
            
            # Note.
            cbar = plt.colorbar(streamplot.lines, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Flow Intensity', fontsize=11, color='#2C3E50')
            cbar.ax.tick_params(labelsize=9, colors='#2C3E50')
            cbar.outline.set_edgecolor('#DDDDDD')
            cbar.outline.set_linewidth(0.8)
            
            # and
            ax.set_xlabel('X Coordinate', fontsize=12, color='#2C3E50', fontweight='normal')
            ax.set_ylabel('Y Coordinate', fontsize=12, color='#2C3E50', fontweight='normal')
            ax.set_title('Attention Flow Density Visualization', 
                        fontsize=13, fontweight='bold', color='#2C3E50', pad=15)
            
            # Optimization
            for spine in ax.spines.values():
                spine.set_edgecolor('#DDDDDD')
                spine.set_linewidth(0.8)
            
            # Note.
            ax.grid(True, linestyle='--', alpha=0.2, color='#E5E5E5', zorder=0)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            density_path = os.path.join(output_dir, 'attention_flow_density.png')
            plt.savefig(density_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"plotSaved to : {density_path}")
    except Exception as e:
        print(f"plotplot: {e}")
    
    print(f"plotSaved to : {output_path}")
    return output_path

def load_whole_slice_data(data_dir: str) -> Dict[str, Any]:
    """load (Optimization)"""
    data = {}
    
    # reading Configuration file path
    config_file = os.path.join(data_dir, "export_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # in of path
        data['config'] = config
        
        # loadOptimization of
        if 'optimized_paths' in config:
            paths = config['optimized_paths']
            
            # Spot-to-Spot
            if os.path.exists(paths['spot_to_spot']):
                data['spot_to_spot'] = pd.read_parquet(paths['spot_to_spot'])
            
            # gene pairs
            if os.path.exists(paths['gene_pairs']):
                data['gene_pairs'] = pd.read_parquet(paths['gene_pairs'])
            
            # Note.
            if os.path.exists(paths['neighbors']):
                data['spot_neighbors'] = pd.read_pickle(paths['neighbors'])
        
        # loadspatial coordinates
        if 'coords_path' in config and os.path.exists(config['coords_path']):
            data['coords_df'] = pd.read_csv(config['coords_path'])
        
        # loadAnnData
        if 'adata_path' in config and os.path.exists(config['adata_path']):
            data['adata'] = sc.read_h5ad(config['adata_path'])
        
        # loadligand-receptordatabase
        if 'lr_db_path' in config and os.path.exists(config['lr_db_path']):
            data['lr_df'] = pd.read_csv(config['lr_db_path'])
    else:
        # Configuration file of load
        print("Warning: not found Configuration file,load")
        
        #  found attention data
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet') and 'whole_slice' in f]
        if parquet_files:
            attn_path = os.path.join(data_dir, parquet_files[0])
            data['attn_df'] = pd.read_parquet(attn_path)
        else:
            # CSV
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'whole_slice' in f]
            if csv_files:
                attn_path = os.path.join(data_dir, csv_files[0])
                data['attn_df'] = pd.read_csv(attn_path)
        
        # spatial coordinates
        coords_files = [f for f in os.listdir(data_dir) if 'coordinates' in f and f.endswith('.csv')]
        if coords_files:
            data['coords_df'] = pd.read_csv(os.path.join(data_dir, coords_files[0]))
        
        # AnnData
        h5ad_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
        if h5ad_files:
            data['adata'] = sc.read_h5ad(os.path.join(data_dir, h5ad_files[0]))
        
        # ligand-receptordatabase
        lr_files = [f for f in os.listdir(data_dir) if 'lr_database' in f]
        if lr_files:
            data['lr_df'] = pd.read_csv(os.path.join(data_dir, lr_files[0]))
    
    # if no Optimization of spot_to_spot,attention data,compute
    if 'spot_to_spot' not in data and 'attn_df' in data:
        print("Computing spot-to-spot attention...")
        attn_df = data['attn_df']
        spot_to_spot = attn_df.groupby(['center_global_idx', 'neighbor_global_idx'])['attn_sum_norm'].sum().reset_index()
        spot_to_spot.columns = ['source_idx', 'target_idx', 'attention_weight']
        data['spot_to_spot'] = spot_to_spot
    
    # load of
    print("load:")
    for key in data.keys():
        if key == 'attn_df':
            print(f"  - attention data: {len(data[key])}rows")
        elif key == 'coords_df':
            print(f"  - spatial coordinates: {len(data[key])} spots")
        elif key == 'spot_to_spot':
            print(f" - Spot-to-Spot: {len(data[key])} ")
        elif key == 'gene_pairs':
            print(f" - gene pairs: {len(data[key])} gene pairs")
        elif key == 'adata':
            print(f"  - AnnData: {data[key].n_obs} spots, {data[key].n_vars}genes")
        elif key == 'lr_df':
            print(f" - ligand-receptordatabase: {len(data[key])} L-R")
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spatial Attention Visualizer')
    parser.add_argument('--data_dir', type=str, default='./PDAC/whole_slice_data_20251028_173836', help='Directory containing exported spatial data')
    parser.add_argument('--output_dir', type=str, default='./PDAC/whole_slice_data_20251028_173836/spatial_attention_visualizer_boundary', help='Directory to save visualizations (default: data_dir/visualizations)')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation even if cached results exist')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir, args.force_recompute)




""" # if analysis python spatial_attention_visualizer.py \ --data_dir./whole_slice_data_20251010_193054 \ --output_dir./whole_slice_data_20251010_193054/visualizations \ --force_recompute """