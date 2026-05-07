"""
数据和基因检查工具
在运行完整分析前，快速检查：
1. 数据文件是否存在
2. HBRC marker基因在数据中的覆盖率
3. 数据基本统计信息
"""

import os
import glob
import json
import pandas as pd
import scanpy as sc
import numpy as np

# HBRC marker基因（与主脚本相同）
HBRC_MARKER_GENES = {
    "B cell": ["MS4A1", "CD79A", "CD79B", "BANK1"],
    "NK cell": ["NKG7", "PRF1", "GNLY", "KLRD1"],
    "T cell": ["CD3D", "CD3E", "TRAC", "CD2"],
    "fibroblast": ["COL1A1", "DCN", "COL1A2", "LUM"],
    "luminal cell": ["KRT8", "EPCAM", "KRT18", "MUC1"],
    "luminal progenitor": ["KRT23", "KIT", "KRT8", "KRT18"],
    "lymphatic endothelial cell": ["PROX1", "PDPN", "LYVE1", "FLT4"],
    "macrophage/DC/monocyte": ["LYZ", "LST1", "MRC1", "FCGR3A"],
    "muscle cell": ["ACTA2", "TAGLN", "MYH11", "MYL9"],
    "myoepithelial cell": ["KRT14", "KRT5", "ACTA2", "TP63"],
    "pDC": ["GZMB", "IRF7", "TCF4", "CLEC4C"],
    "plasma cell": ["MZB1", "IGHG1", "SDC1", "IGKC"],
    "vascular endothelial cell": ["PECAM1", "VWF", "KDR", "ENG"],
}

def find_latest_data_dir():
    """查找最新的数据目录"""
    base_dir = "/home/junning/projectnvme/ST/project-20-contrast-organ/Analysis/spatial_communications"
    data_dirs = glob.glob(os.path.join(base_dir, "HBRC1/whole_slice_data_*"))
    
    if data_dirs:
        latest = sorted(data_dirs)[-1]
        return latest
    return None

def check_files(data_dir):
    """检查必需文件是否存在"""
    print("=" * 70)
    print("1. 检查数据文件")
    print("=" * 70)
    
    required_files = {
        "配置文件": "export_config.json",
        "空间坐标": "spatial_coordinates.csv",
        "表达数据": "adata_with_metadata.h5ad",
        "Attention数据": ["whole_slice_attention_*.csv", "whole_slice_attention_*.parquet"]
    }
    
    all_exist = True
    
    for name, patterns in required_files.items():
        if isinstance(patterns, list):
            # 检查多个可能的文件
            found = False
            for pattern in patterns:
                files = glob.glob(os.path.join(data_dir, pattern))
                if files:
                    print(f"✓ {name}: {os.path.basename(files[0])}")
                    found = True
                    break
            if not found:
                print(f"✗ {name}: 未找到")
                all_exist = False
        else:
            # 检查单个文件
            filepath = os.path.join(data_dir, patterns)
            if os.path.exists(filepath):
                print(f"✓ {name}: {patterns}")
            else:
                print(f"✗ {name}: {patterns} (未找到)")
                all_exist = False
    
    print()
    return all_exist

def check_marker_genes(data_dir):
    """检查marker基因在数据中的覆盖率"""
    print("=" * 70)
    print("2. 检查HBRC Marker基因覆盖率")
    print("=" * 70)
    
    # 加载adata
    adata_path = os.path.join(data_dir, "adata_with_metadata.h5ad")
    if not os.path.exists(adata_path):
        print("✗ 无法加载表达数据")
        return
    
    print(f"加载表达数据: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  - Spots数量: {adata.n_obs}")
    print(f"  - 基因数量: {adata.n_vars}")
    print()
    
    # 检查每种细胞类型的marker
    total_markers = 0
    found_markers = 0
    
    results = []
    
    for cell_type, genes in HBRC_MARKER_GENES.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        total_markers += len(genes)
        found_markers += len(valid_genes)
        
        coverage = len(valid_genes) / len(genes) * 100 if genes else 0
        
        results.append({
            "细胞类型": cell_type,
            "总marker数": len(genes),
            "找到数": len(valid_genes),
            "覆盖率": f"{coverage:.0f}%",
            "缺失基因": [g for g in genes if g not in adata.var_names]
        })
        
        status = "✓" if len(valid_genes) > 0 else "✗"
        print(f"{status} {cell_type:30s} : {len(valid_genes):2d}/{len(genes):2d} ({coverage:5.1f}%)")
        if len(valid_genes) < len(genes) and len(genes) - len(valid_genes) <= 2:
            missing = [g for g in genes if g not in adata.var_names]
            print(f"    缺失: {', '.join(missing)}")
    
    print()
    print(f"总计: {found_markers}/{total_markers} 个marker基因存在 ({found_markers/total_markers*100:.1f}%)")
    print()
    
    return results, adata

def check_attention_data(data_dir):
    """检查attention数据统计信息"""
    print("=" * 70)
    print("3. Attention数据统计")
    print("=" * 70)
    
    # 查找attention文件
    attn_files = glob.glob(os.path.join(data_dir, "whole_slice_attention_*.csv"))
    if not attn_files:
        attn_files = glob.glob(os.path.join(data_dir, "whole_slice_attention_*.parquet"))
    
    if not attn_files:
        print("✗ 未找到attention数据文件")
        return
    
    attn_path = attn_files[0]
    print(f"加载attention数据: {os.path.basename(attn_path)}")
    
    try:
        if attn_path.endswith('.parquet'):
            attn_df = pd.read_parquet(attn_path)
        else:
            attn_df = pd.read_csv(attn_path)
        
        print(f"  - 总记录数: {len(attn_df):,}")
        print(f"  - 中心spot数: {attn_df['center_global_idx'].nunique():,}")
        print(f"  - 邻居spot数: {attn_df['neighbor_global_idx'].nunique():,}")
        
        if 'kv_gene_symbol' in attn_df.columns:
            # 过滤空字符串
            kv_genes = attn_df[attn_df['kv_gene_symbol'].str.len() > 0]['kv_gene_symbol']
            print(f"  - KV基因数: {kv_genes.nunique():,}")
            
            # Top KV基因
            top_kv = attn_df.groupby('kv_gene_symbol')['attn_score'].sum().sort_values(ascending=False).head(10)
            print("\n  Top 10 KV基因（按attention总和）:")
            for gene, score in top_kv.items():
                if gene and len(str(gene)) > 0:
                    print(f"    {gene:15s} : {score:.4f}")
        
        if 'q_gene_symbol' in attn_df.columns:
            q_genes = attn_df[attn_df['q_gene_symbol'].str.len() > 0]['q_gene_symbol']
            print(f"\n  - Query基因数: {q_genes.nunique():,}")
        
        print()
        
    except Exception as e:
        print(f"✗ 加载attention数据时出错: {e}")
        return

def check_cluster_info(data_dir):
    """检查聚类信息"""
    print("=" * 70)
    print("4. 聚类信息")
    print("=" * 70)
    
    coords_path = os.path.join(data_dir, "spatial_coordinates.csv")
    if not os.path.exists(coords_path):
        print("✗ 未找到空间坐标文件")
        return
    
    coords_df = pd.read_csv(coords_path)
    
    if 'cluster' in coords_df.columns:
        n_clusters = coords_df['cluster'].nunique()
        print(f"聚类数量: {n_clusters}")
        print("\n各聚类spot数量:")
        
        cluster_counts = coords_df['cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = count / len(coords_df) * 100
            print(f"  Cluster {cluster}: {count:5d} spots ({percentage:5.1f}%)")
    else:
        print("警告: 空间坐标文件中没有聚类信息")
    
    print()

def main():
    """主函数"""
    print("\n")
    print("=" * 70)
    print("数据和基因检查工具")
    print("=" * 70)
    print()
    
    # 查找数据目录
    data_dir = find_latest_data_dir()
    
    if data_dir is None:
        print("✗ 未找到数据目录")
        print("\n请先运行 1_enhanced_spatial_data_exporter.py 生成数据")
        return
    
    print(f"数据目录: {data_dir}")
    print()
    
    # 执行各项检查
    files_ok = check_files(data_dir)
    
    if not files_ok:
        print("\n⚠ 部分必需文件缺失，请检查数据目录")
        return
    
    # marker_results, adata = check_marker_genes(data_dir)
    check_attention_data(data_dir)
    check_cluster_info(data_dir)
    
    # 总结
    print("=" * 70)
    print("检查完成")
    print("=" * 70)
    print()
    print("如果上述检查都通过，可以运行完整分析:")
    print("  bash run_kv_analysis.sh")
    print("或")
    print("  python 2_kv_gene_attention_expression_analysis.py")
    print()

if __name__ == "__main__":
    main()

