"""
分析KV基因在不同区域的attention和表达模式
- 统计代表性基因作为KV基因在不同区域获得的attention
- 统计这些基因在不同区域的表达水平
- 绘制对比boxplot，探索attention和表达的生物学联系
"""
try:
    from .utils_2_kv_gene_attention_expression_analysis_pdac import *
except ImportError:
    # 如果相对导入失败，使用绝对导入（直接运行脚本时）
    from utils_2_kv_gene_attention_expression_analysis_pdac import *



if __name__ == "__main__":
    # 直接运行分析，使用脚本开头定义的参数
    DATA_DIR = "./Colorectal_cancer_whole/whole_slice_data_20251102_120239"

    # 输出目录（默认在DATA_DIR下创建analysis子目录）
    OUTPUT_DIR = DATA_DIR + "/kv_gene_attention_expression_analysis"

    # PDAC特异性marker基因（已去重整理）
    MARKER_GENES = {
            # 导管细胞 - 仅保留特异性marker
            "Ductal - terminal ductal like": ["CFTR"],  # KRT19, KRT7, MUC1, EPCAM移至通用上皮类别
            "Ductal - CRISP3 high/centroacinar like": ["CRISP3", "GP2"],  # REG1A, PRSS1移至腺泡/胰腺酶类别
            "Ductal - MHC Class II": [],  # HLA-DRA, HLA-DRB1, CD74, KRT19移至免疫和上皮类别，此类别合并到免疫
            
            # 腺泡细胞和胰腺酶（合并去重）
            "Acinar cells": ["PRSS1", "PRSS2","PRSS3", "CTRB1", "CTRB2", "CPA1", "CPB1", "CEL", "SPINK1", "AMY2A", 
                            "REG1A", "REG1B", "REG3A", "PNLIP"],
            
            # 胰腺激素
            "Pancreatic hormones": ["INS", "GCG", "SST", "PPY","MMP7","MMP9"],
            
            # 癌细胞克隆 - 仅保留特异性marker
            "Cancer clone A": ["KRAS", "S100P"],  # MUC1, EPCAM, KRT19移至上皮
            "Cancer clone B": ["MSLN"],  # KRAS, MUC1, CEACAM5, S100A4移至其他类别
            
            # 免疫细胞
            "T cells": ["CD3D", "CD3E", "CD2", "TRAC", "IL7R"],
            "B cells": ["MS4A1", "CD79A", "CD79B", "BANK1"],
            "Monocytes": ["LYZ", "S100A8", "S100A9", "CD14", "FCGR3A"],
            "Macrophages": ["CD68", "CD163", "MRC1", "MSR1", "C1QA", "C1QB"],  # 合并C1Q家族
            "NK cells": ["NKG7", "GNLY", "PRF1", "KLRD1", "NCAM1"],
            "Dendritic cells": ["CLEC9A", "XCR1", "BATF3", "IRF8"],
            "Mast cells": ["TPSAB1", "CPA3", "KIT", "MS4A2"],
            
            # MHC II类分子（免疫相关）
            "MHC Class II": ["HLA-DRA", "HLA-DRB1", "CD74"],
            
            # 基质细胞 - ECM和结构蛋白
            "Fibroblasts": ["COL1A1", "COL1A2", "COL3A1", "FN1", "DCN", "LUM", "VIM", 
                            "SPARC", "POSTN", "THBS2", "MGP"],
            "Cancer-associated fibroblasts": ["FAP", "PDGFRA", "PDGFRB", "ACTA2"],  # 去重FAP, COL1A1, ACTA2
            "Pancreatic stellate cells": ["RGS5", "DES"],  # ACTA2, PDGFRB已在CAF中
            
            # 内皮细胞
            "Endothelial cells": ["PECAM1", "VWF", "CDH5", "KDR", "ENG", "LAMB3", "LAMC2"],
            
            # 上皮标志物（合并去重）
            "Epithelial markers": ["KRT7", "KRT8", "KRT18", "KRT19", "EPCAM", "MUC1", 
                                "CEACAM5", "CEACAM6", "CLDN4", "TFF1", "DMBT1"],
            
            # 信号分子和生长因子（S100家族单独列出）
            "S100 family": ["S100A4", "S100A6", "S100A10", "S100A11", "S100B", "S100P"],
            "Growth factors & signaling": ["SPP1", "TGFBI", "IGFBP7", "GDF15", "CTGF", "CYR61", 
                                        "SERPINE1", "PLAU"],
            
            # 免疫和炎症因子
            "Cytokines & chemokines": ["APOE", "CXCL8", "CXCL12", "CCL2", "CCL11", "IL1B"],
            
            # 代谢相关
            "Metabolism": ["ALDOA", "ENO1", "GAPDH", "PGK1", "LDHA", "PKM", 
                        "PFKP", "TPI1", "ACSS1", "ACOX1"],
            
            # 核糖体和翻译
            "Ribosomal": ["RPL34P33", "RPL34P34", "RPS2", "RPS18", "RPS27", 
                        "EEF1A1", "TMSB4X", "H3F3B", "HIST1H4C"],
            
            # 神经相关
            "Schwann cells": ["MPZ", "PLP1", "SOX10"],  # S100B移至S100家族
        }




    # 绘图参数
    TOP_N_GENES_PLOT = 20  # boxplot中显示的基因数量
    N_COMPARISON_GENES = 125  # 生成详细对比图的基因数量
    PLOT_ALL_GENES = True  # 是否绘制所有候选基因的单一基因对比图（True=全部，False=只绘制前N_COMPARISON_GENES个）

    try:
        output_dir = main()
        print("\n" + "=" * 70)
        print("✓ 分析成功完成！")
        print(f"✓ 结果目录: {output_dir}")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ 分析过程中出现错误:")
        print(f"  {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        raise

