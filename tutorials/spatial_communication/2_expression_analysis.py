""" Analyze attention and expression patterns of KV genes across regions - Summarize attention received by representative KV genes across regions - Summarize expression levels of these genes across regions - plotboxplot,attention and table of """
try:
    from .utils_2_kv_gene_attention_expression_analysis_pdac import *
except ImportError:
    # if failed, (when running the script directly)
    from utils_2_kv_gene_attention_expression_analysis_pdac import *



if __name__ == "__main__":
    # Run the analysis directly,using parameters defined at the top of the script
    DATA_DIR = "./Colorectal_cancer_whole/whole_slice_data_20251102_120239"

    # Output directory (created by default as an analysis subdirectory under DATA_DIR)
    OUTPUT_DIR = DATA_DIR + "/kv_gene_attention_expression_analysis"

    # PDACspecific marker genes ()
    MARKER_GENES = {
            # Ductal cells - keep only specific markers
            "Ductal - terminal ductal like": ["CFTR"],  # KRT19, KRT7, MUC1, EPCAMmoved to the general epithelial category
            "Ductal - CRISP3 high/centroacinar like": ["CRISP3", "GP2"],  # REG1A, PRSS1moved to the acinar/pancreatic enzyme category
            "Ductal - MHC Class II": [],  # HLA-DRA, HLA-DRB1, CD74, KRT19immune and,this category is merged into immune
            
            # Acinar cells and pancreatic enzymes (merged and deduplicated)
            "Acinar cells": ["PRSS1", "PRSS2","PRSS3", "CTRB1", "CTRB2", "CPA1", "CPB1", "CEL", "SPINK1", "AMY2A", 
                            "REG1A", "REG1B", "REG3A", "PNLIP"],
            
            # Pancreatic hormones
            "Pancreatic hormones": ["INS", "GCG", "SST", "PPY","MMP7","MMP9"],
            
            # Cancer cell clones - keep only specific markers
            "Cancer clone A": ["KRAS", "S100P"],  # MUC1, EPCAM, KRT19moved to epithelial
            "Cancer clone B": ["MSLN"],  # KRAS, MUC1, CEACAM5, S100A4moved to other categories
            
            # Immune cells
            "T cells": ["CD3D", "CD3E", "CD2", "TRAC", "IL7R"],
            "B cells": ["MS4A1", "CD79A", "CD79B", "BANK1"],
            "Monocytes": ["LYZ", "S100A8", "S100A9", "CD14", "FCGR3A"],
            "Macrophages": ["CD68", "CD163", "MRC1", "MSR1", "C1QA", "C1QB"],  # merged C1Q family
            "NK cells": ["NKG7", "GNLY", "PRF1", "KLRD1", "NCAM1"],
            "Dendritic cells": ["CLEC9A", "XCR1", "BATF3", "IRF8"],
            "Mast cells": ["TPSAB1", "CPA3", "KIT", "MS4A2"],
            
            # MHC class II molecules (immune-related)
            "MHC Class II": ["HLA-DRA", "HLA-DRB1", "CD74"],
            
            # Stromal cells - ECM and
            "Fibroblasts": ["COL1A1", "COL1A2", "COL3A1", "FN1", "DCN", "LUM", "VIM", 
                            "SPARC", "POSTN", "THBS2", "MGP"],
            "Cancer-associated fibroblasts": ["FAP", "PDGFRA", "PDGFRB", "ACTA2"],  # FAP, COL1A1, ACTA2
            "Pancreatic stellate cells": ["RGS5", "DES"],  # ACTA2, PDGFRB in CAF in
            
            # Endothelial cells
            "Endothelial cells": ["PECAM1", "VWF", "CDH5", "KDR", "ENG", "LAMB3", "LAMC2"],
            
            # Epithelial markers (merged and deduplicated)
            "Epithelial markers": ["KRT7", "KRT8", "KRT18", "KRT19", "EPCAM", "MUC1", 
                                "CEACAM5", "CEACAM6", "CLDN4", "TFF1", "DMBT1"],
            
            # Signaling molecules and growth factors (S100listed separately)
            "S100 family": ["S100A4", "S100A6", "S100A10", "S100A11", "S100B", "S100P"],
            "Growth factors & signaling": ["SPP1", "TGFBI", "IGFBP7", "GDF15", "CTGF", "CYR61", 
                                        "SERPINE1", "PLAU"],
            
            # Immune and inflammatory factors
            "Cytokines & chemokines": ["APOE", "CXCL8", "CXCL12", "CCL2", "CCL11", "IL1B"],
            
            # Metabolism-related
            "Metabolism": ["ALDOA", "ENO1", "GAPDH", "PGK1", "LDHA", "PKM", 
                        "PFKP", "TPI1", "ACSS1", "ACOX1"],
            
            # Ribosome and translation
            "Ribosomal": ["RPL34P33", "RPL34P34", "RPS2", "RPS18", "RPS27", 
                        "EEF1A1", "TMSB4X", "H3F3B", "HIST1H4C"],
            
            # Neural-related
            "Schwann cells": ["MPZ", "PLP1", "SOX10"],  # S100BS100
        }




    # Plotting parameters
    TOP_N_GENES_PLOT = 20  # boxplot in number of genes shown
    N_COMPARISON_GENES = 125  # number of genes for detailed comparison plots
    PLOT_ALL_GENES = True  # whether to plot single-gene comparison charts for all candidate genes (True=all,False=only plot the firstN_COMPARISON_GENES )

    try:
        output_dir = main()
        print("\n" + "=" * 70)
        print("✓ Analysis completed successfully！")
        print(f"✓ Result directory: {output_dir}")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ An error occurred during analysis:")
        print(f"  {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        raise

