#!/usr/bin/env python3
"""
Spatial Deconvolution Visualization

This module provides comprehensive visualization tools for spatial transcriptomics
deconvolution results from CoseNiche. It generates multiple plot types including:
- Stacked bar charts showing cell type composition per spatial domain
- Pie charts displaying average cell type fractions
- Spatial pie plots overlaying composition on tissue coordinates

The visualization follows Nature journal style guidelines with publication-quality
output (300+ DPI, color-blind friendly palettes, proper typography).

Usage:
    python 2_plot_composition.py --config config_pdac.yaml
    python 2_plot_composition.py --deconv-file results/deconv.h5ad --sc-file results/sc.h5ad

Author: CoseNiche Team
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import yaml

# Try to import utils from parent directory
try:
    from utils_plot import project_cell_to_spot
except ImportError:
    print("Warning: utils_plot not found. Spatial projection may not work.")


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class PlotConfig:
    """
    Configuration for publication-quality plotting.
    
    Follows Nature journal style guidelines:
    - Arial/Helvetica fonts (5-9pt)
    - 89mm (single column) or 183mm (double column) width
    - 300-1000 DPI resolution
    - Color-blind friendly palettes
    """
    
    # Input/Output paths
    data_dir: str = "."
    output_dir: str = "results/plots"
    deconv_file: str = "deconvolution_result.h5ad"
    sc_file: str = "sc_reference.h5ad"
    
    # Font configuration (Nature style)
    font_family: str = "Arial"
    font_tiny: int = 8
    font_small: int = 9
    font_medium: int = 10
    font_large: int = 13
    font_xlarge: int = 15
    
    # Figure sizes (inches, Nature journal standard)
    dpi: int = 300
    figsize_heatmap: Tuple[float, float] = (7.2, 4.5)  # 183mm double column
    figsize_umap: Tuple[float, float] = (3.5, 3.5)     # 89mm single column
    figsize_pie: Tuple[float, float] = (3.5, 3.5)
    figsize_stacked: Tuple[float, float] = (7.2, 3.5)
    
    # Color palettes (color-blind friendly)
    cmap_heatmap: str = "RdYlBu_r"
    cmap_celltype: str = "tab20"
    
    # Line and edge properties
    line_width: float = 0.5
    line_width_thick: float = 1.0
    spine_width: float = 0.5
    
    # Transparency and styling
    alpha_scatter: float = 0.7
    alpha_bar: float = 0.9
    edge_color_scatter: str = "none"
    edge_color_bar: str = "white"
    
    # Processing parameters
    retain_percent: float = 0.1
    
    # Domain ordering (if specified)
    domain_order: Optional[List[str]] = None
    
    # Nature style parameters
    use_grid: bool = True
    grid_alpha: float = 0.3
    grid_style: str = "--"
    legend_frameon: bool = False
    tight_layout: bool = True
    
    def __post_init__(self):
        """Initialize matplotlib style after dataclass creation."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._setup_matplotlib_style()
    
    def _get_available_font(self) -> str:
        """
        Get available font with automatic fallback.
        Priority: Arial > Liberation Sans > DejaVu Sans > sans-serif
        """
        import matplotlib.font_manager as fm
        
        font_preferences = [
            self.font_family,
            'Liberation Sans',
            'DejaVu Sans',
            'Helvetica',
            'sans-serif'
        ]
        
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        
        for font in font_preferences:
            if font in available_fonts:
                if font != self.font_family:
                    print(f"ℹ Font '{self.font_family}' not found, using '{font}'")
                return font
        
        return 'sans-serif'
    
    def _setup_matplotlib_style(self):
        """Configure matplotlib for Nature journal style."""
        available_font = self._get_available_font()
        
        plt.rcParams.update({
            # Fonts
            'font.family': available_font,
            'font.size': self.font_medium,
            
            # Axes
            'axes.linewidth': self.spine_width,
            'axes.labelsize': self.font_medium,
            'axes.titlesize': self.font_large,
            'axes.labelweight': 'normal',
            'axes.titleweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': '#000000',
            'axes.facecolor': 'white',
            
            # Ticks
            'xtick.major.width': self.spine_width,
            'ytick.major.width': self.spine_width,
            'xtick.labelsize': self.font_small,
            'ytick.labelsize': self.font_small,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            
            # Legend
            'legend.fontsize': self.font_small,
            'legend.frameon': self.legend_frameon,
            'legend.edgecolor': 'none',
            
            # Lines
            'lines.linewidth': self.line_width,
            'patch.linewidth': self.line_width,
            
            # Grid
            'grid.alpha': self.grid_alpha,
            'grid.linestyle': self.grid_style,
            'grid.linewidth': self.line_width * 0.8,
            
            # Saving
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none',
            
            # PDF output
            'pdf.fonttype': 42,  # TrueType fonts (editable)
            'ps.fonttype': 42,
        })
    
    def get_nature_colors(self, n_colors: int) -> List[str]:
        """
        Get Nature-style color palette (color-blind friendly).
        
        Parameters
        ----------
        n_colors : int
            Number of colors needed
            
        Returns
        -------
        colors : list of str
            Hex color codes
        """
        # Nature-recommended color-blind friendly palette
        colors = [
            '#3C5488',  # Deep blue
            '#E64B35',  # Warm red
            '#4DBBD5',  # Sky blue
            '#00A087',  # Cyan-green
            '#F39B7F',  # Soft orange
            '#91D1C2',  # Glacier blue
            '#8491B4',  # Gray-purple
            '#7E6148',  # Soft brown
            '#B09C85',  # Beige-brown
            '#E64B35',  # Warm red (repeat)
            '#DF8F44',  # Amber
            '#B2473B',  # Brick red
            '#73C0DE',  # Lake blue
            '#8D9440',  # Olive green
            '#C3A29E',  # Sand pink
            '#FFDC91',  # Light apricot
            '#9A8F97',  # Warm gray-purple
            '#5A8A8A',  # Sea pine green
            '#C28E9B',  # Rose gray
            '#6C6F7D',  # Smoke gray
        ]
        
        # Cycle if needed
        if n_colors > len(colors):
            colors = colors * (n_colors // len(colors) + 1)
        
        return colors[:n_colors]


# =============================================================================
# Utility Functions
# =============================================================================

def sanitize_filename(name: str) -> str:
    """
    Clean filename by removing or replacing illegal characters.
    
    Parameters
    ----------
    name : str
        Original filename
        
    Returns
    -------
    safe_name : str
        Sanitized filename
    """
    unsafe_chars = str.maketrans({
        '/': '_', '\\': '_', ':': '_', '*': '_',
        '?': '_', '"': '_', '<': '_', '>': '_', '|': '_',
        ' ': '_'
    })
    
    safe_name = name.translate(unsafe_chars)
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip('_')
    
    return safe_name


def extract_main_type(domain_name: str) -> str:
    """
    Extract main domain type by removing numeric suffixes.
    
    Parameters
    ----------
    domain_name : str
        Domain name (e.g., "Tumor_1", "Normal_2")
        
    Returns
    -------
    main_type : str
        Main type without suffix (e.g., "Tumor", "Normal")
    """
    return re.sub(r'_\d+$', '', str(domain_name))


def build_domain_groups(
    domains: pd.Series,
    order: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Group domains by main type and optionally reorder.
    
    Parameters
    ----------
    domains : pd.Series
        Domain labels for each spot
    order : list of str, optional
        Desired ordering of main domain types
        
    Returns
    -------
    group_to_domains : dict
        Mapping from main type to list of specific domains
    """
    domains = domains.astype(str)
    
    # Group by main type
    group_to_domains = {}
    for d in domains.unique():
        g = extract_main_type(d)
        group_to_domains.setdefault(g, []).append(d)
    
    # Sort within groups by numeric suffix
    for g in group_to_domains:
        group_to_domains[g].sort(
            key=lambda x: int(m.group(1)) if (m := re.search(r'_(\d+)$', x)) else 0
        )
    
    # Reorder by specified order if provided
    if order:
        ordered_dict = {}
        for main_type in order:
            if main_type in group_to_domains:
                ordered_dict[main_type] = group_to_domains[main_type]
        
        # Add any remaining types not in order
        for main_type, domains_list in group_to_domains.items():
            if main_type not in ordered_dict:
                ordered_dict[main_type] = domains_list
        
        return ordered_dict
    
    return group_to_domains


def normalize_celltype_name(celltype_name: str) -> str:
    """
    Normalize cell type name by removing numeric suffixes.
    
    For example, "T cell.151" -> "T cell"
    
    Parameters
    ----------
    celltype_name : str
        Original cell type name
        
    Returns
    -------
    normalized : str
        Normalized cell type name
    """
    return re.sub(r'\.\d+$', '', str(celltype_name))


# =============================================================================
# Data Loading and Processing
# =============================================================================

class DeconvolutionData:
    """
    Data loader and processor for deconvolution results.
    
    Attributes
    ----------
    config : PlotConfig
        Configuration object
    adata_deconv : AnnData
        Spatial deconvolution results
    adata_sc : AnnData
        Single-cell reference data
    adata_out : AnnData
        Processed spatial data with projections
    df_projection : pd.DataFrame
        Cell type proportion matrix (spots × cell types)
    cell_types : np.ndarray
        Array of unique cell type names
    domains : pd.Series
        Domain labels for each spot
    """
    
    def __init__(self, config: PlotConfig):
        self.config = config
        self.adata_deconv = None
        self.adata_sc = None
        self.adata_out = None
        self.df_projection = None
        self.cell_types = None
        self.domains = None
    
    def load_data(self) -> None:
        """Load deconvolution and single-cell reference data."""
        print("=" * 80)
        print("Loading data...")
        print("=" * 80)
        
        deconv_path = Path(self.config.data_dir) / self.config.deconv_file
        sc_path = Path(self.config.data_dir) / self.config.sc_file
        
        if not deconv_path.exists():
            raise FileNotFoundError(f"Deconvolution file not found: {deconv_path}")
        if not sc_path.exists():
            raise FileNotFoundError(f"Single-cell file not found: {sc_path}")
        
        try:
            self.adata_deconv = sc.read_h5ad(deconv_path)
            self.adata_sc = sc.read_h5ad(sc_path)
            print(f"✓ Loaded deconvolution data: {deconv_path.name}")
            print(f"  Spots: {self.adata_deconv.n_obs}, Genes: {self.adata_deconv.n_vars}")
            print(f"✓ Loaded single-cell data: {sc_path.name}")
            print(f"  Cells: {self.adata_sc.n_obs}, Genes: {self.adata_sc.n_vars}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")
    
    def preprocess_celltypes(self) -> None:
        """Normalize cell type annotations."""
        if 'cell_type' not in self.adata_sc.obs.columns:
            print("⚠ No 'cell_type' column found in single-cell data")
            return
        
        print("\nNormalizing cell type annotations...")
        print(f"  Original unique cell types: {self.adata_sc.obs['cell_type'].nunique()}")
        
        # Backup original
        self.adata_sc.obs['cell_type_original'] = self.adata_sc.obs['cell_type'].copy()
        
        # Normalize
        self.adata_sc.obs['cell_type'] = self.adata_sc.obs['cell_type'].apply(
            normalize_celltype_name
        )
        
        print(f"  Normalized unique cell types: {self.adata_sc.obs['cell_type'].nunique()}")
        
        # Print statistics
        ct_counts = self.adata_sc.obs['cell_type'].value_counts()
        print("\n  Cell type distribution:")
        for ct, count in ct_counts.items():
            pct = count / len(self.adata_sc) * 100
            print(f"    {ct:40s}: {count:6d} ({pct:5.1f}%)")
    
    def process_data(self) -> None:
        """
        Process and project single-cell data onto spatial spots.
        
        This function requires project_cell_to_spot from utils_plot module.
        """
        print("\nProjecting single-cell data to spatial spots...")
        
        try:
            from utils_plot import project_cell_to_spot
            
            self.adata_out, self.df_projection, self.cell_types = project_cell_to_spot(
                self.adata_deconv,
                self.adata_sc,
                retain_percent=self.config.retain_percent
            )
            
            print(f"✓ Projection completed")
            print(f"  Spatial spots: {self.adata_out.n_obs}")
            print(f"  Genes: {self.adata_out.n_vars}")
            print(f"  Cell types: {len(self.cell_types)}")
            
        except ImportError:
            # Fallback: use cell_type_proportions if available
            print("⚠ utils_plot not found, using existing proportions")
            
            if 'cell_type_proportions' in self.adata_deconv.obsm:
                self.adata_out = self.adata_deconv
                self.df_projection = self.adata_deconv.obsm['cell_type_proportions']
                self.cell_types = self.df_projection.columns.values
                
                print(f"✓ Using existing proportions")
                print(f"  Spatial spots: {self.adata_out.n_obs}")
                print(f"  Cell types: {len(self.cell_types)}")
            else:
                raise RuntimeError("No projection function or existing proportions found")
        
        # Get domain information
        domain_key = "ground_truth" if "ground_truth" in self.adata_out.obs.columns else "refined_label"
        
        if domain_key not in self.adata_out.obs.columns:
            raise ValueError("Missing 'ground_truth' or 'refined_label' column in data")
        
        self.domains = self.adata_out.obs[domain_key].astype(str)
        print(f"  Using domain column: '{domain_key}'")
        print(f"  Unique domains: {self.domains.nunique()}")


# =============================================================================
# Visualization Classes
# =============================================================================

class StackedBarPlotter:
    """
    Create stacked bar charts and pie charts for cell type composition.
    
    Generates multiple plot types:
    - Per-domain stacked bar charts (one bar per spot)
    - Per-domain pie charts (average composition)
    - Domain-averaged bar charts
    - All-spots overview plots
    """
    
    def __init__(self, config: PlotConfig, data: DeconvolutionData):
        self.config = config
        self.data = data
        self.celltype_colors = None
    
    def plot_all(self):
        """Generate all stacked bar and pie chart variants."""
        print("\n" + "=" * 80)
        print("Generating stacked bar charts and pie charts...")
        print("=" * 80)
        
        # Build domain groups
        domain_groups = build_domain_groups(
            self.data.domains,
            order=self.config.domain_order
        )
        
        # Prepare colors
        self.celltype_colors = self._prepare_celltype_colors()
        
        # Plot per-domain charts
        print("\nGenerating per-domain charts...")
        for grp, dom_list in domain_groups.items():
            print(f"  Processing group: {grp}")
            
            for dom in dom_list:
                spot_idx = self.data.domains[self.data.domains == dom].index
                
                if len(spot_idx) == 0:
                    continue
                
                safe_dom = sanitize_filename(dom)
                
                # Stacked bar chart
                bar_file = Path(self.config.output_dir) / f"stacked_bar_{safe_dom}.png"
                self._plot_domain_stacked_bar(spot_idx, dom, bar_file)
                
                # Pie chart
                pie_file = Path(self.config.output_dir) / f"pie_{safe_dom}.png"
                self._plot_domain_pie(spot_idx, dom, pie_file)
        
        # Domain-averaged bar chart
        print("\nGenerating domain-averaged bar chart...")
        avg_file = Path(self.config.output_dir) / "domain_average_bar.png"
        self._plot_domain_average_bar(domain_groups, avg_file)
        
        # All spots by domain
        print("\nGenerating all-spots overview...")
        all_spots_file = Path(self.config.output_dir) / "all_spots_by_domain.png"
        self._plot_all_spots_by_domain(domain_groups, all_spots_file)
        
        print(f"\n✓ All charts saved to: {self.config.output_dir}")
    
    def _prepare_celltype_colors(self) -> Dict[str, str]:
        """Create color mapping for cell types."""
        cell_types = sorted(self.data.cell_types)
        
        if len(cell_types) <= 10:
            colors = self.config.get_nature_colors(len(cell_types))
            return {ct: colors[i] for i, ct in enumerate(cell_types)}
        else:
            # Use tab20 for >10 types
            cmap = plt.colormaps.get_cmap(self.config.cmap_celltype)
            return {ct: cmap(i % cmap.N) for i, ct in enumerate(cell_types)}
    
    def _plot_domain_stacked_bar(self, spot_idx, domain: str, output_file: Path):
        """
        Plot stacked bar chart for a single domain.
        
        Parameters
        ----------
        spot_idx : array-like
            Indices of spots in this domain
        domain : str
            Domain name
        output_file : Path
            Output file path
        """
        # Get composition matrix
        df_domain = self.data.df_projection.loc[spot_idx, :]
        df_domain = df_domain.div(df_domain.sum(axis=1), axis=0).fillna(0)
        
        # Sort by cell types
        cell_types = sorted(self.data.cell_types)
        df_domain = df_domain[cell_types]
        
        # Create figure
        n_spots = len(spot_idx)
        fig, ax = plt.subplots(figsize=self.config.figsize_stacked, dpi=100)
        
        # Plot stacked bars
        x_pos = np.arange(n_spots)
        bottom = np.zeros(n_spots)
        
        for ct in cell_types:
            values = df_domain[ct].values
            
            if values.sum() > 0:
                ax.bar(
                    x_pos, values, bottom=bottom,
                    color=self.celltype_colors[ct],
                    width=0.85,
                    label=ct,
                    edgecolor=self.config.edge_color_bar,
                    linewidth=self.config.line_width * 0.3,
                    alpha=self.config.alpha_bar,
                    rasterized=True
                )
                bottom += values
        
        # Styling
        ax.set_title(domain, fontsize=self.config.font_large, fontweight='bold', pad=8)
        ax.set_xlabel("Spots", fontsize=self.config.font_medium)
        ax.set_ylabel("Cell Fraction", fontsize=self.config.font_medium)
        ax.set_xlim(-0.5, n_spots - 0.5)
        ax.set_ylim(0, 1.05)
        
        # X-axis ticks
        if n_spots <= 20:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(spot_idx, rotation=45, ha='right',
                              fontsize=self.config.font_tiny)
        else:
            ax.set_xticks([])
            ax.set_xlabel(f"Spots (n={n_spots})", fontsize=self.config.font_medium)
        
        # Y-axis ticks
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        
        # Grid
        if self.config.use_grid:
            ax.grid(axis='y', alpha=self.config.grid_alpha,
                   linestyle=self.config.grid_style, zorder=0)
            ax.set_axisbelow(True)
        
        # Legend
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            frameon=self.config.legend_frameon,
            fontsize=self.config.font_tiny,
            title="Cell Types",
            title_fontsize=self.config.font_small
        )
        
        # Save
        if self.config.tight_layout:
            plt.tight_layout()
        fig.savefig(output_file, dpi=self.config.dpi, format='png')
        plt.close(fig)
    
    def _plot_domain_pie(self, spot_idx, domain: str, output_file: Path):
        """
        Plot pie chart showing average composition for a domain.
        
        Parameters
        ----------
        spot_idx : array-like
            Indices of spots in this domain
        domain : str
            Domain name
        output_file : Path
            Output file path
        """
        # Get average composition
        df_domain = self.data.df_projection.loc[spot_idx, :]
        avg_composition = df_domain.mean(axis=0)
        
        # Sort by cell types
        cell_types = sorted(self.data.cell_types)
        avg_composition = avg_composition[cell_types]
        
        # Filter small fractions
        min_fraction = 0.02
        major_celltypes = avg_composition[avg_composition >= min_fraction]
        other_fraction = avg_composition[avg_composition < min_fraction].sum()
        
        if other_fraction > 0:
            major_celltypes = pd.concat([
                major_celltypes,
                pd.Series({'Other': other_fraction})
            ])
        
        # Normalize and sort
        major_celltypes = major_celltypes / major_celltypes.sum()
        major_celltypes = major_celltypes.sort_values(ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize_pie, dpi=100)
        
        # Prepare data
        labels = list(major_celltypes.index)
        sizes = major_celltypes.values
        colors = [self.celltype_colors.get(ct, '#CCCCCC') for ct in labels]
        
        # Plot pie
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            colors=colors,
            autopct=lambda pct: f'{pct:.0f}%' if pct > 3 else '',
            startangle=90,
            counterclock=False,
            wedgeprops=dict(
                linewidth=self.config.line_width,
                edgecolor='white',
                antialiased=True
            ),
            textprops=dict(
                fontsize=self.config.font_tiny,
                fontweight='normal',
                color='black'
            )
        )
        
        # Style percentage labels
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        # Ensure circular
        ax.axis('equal')
        ax.set_ylim(-1.05, 1.15)
        
        # Title
        ax.text(0, 1.08, domain, fontsize=self.config.font_large,
               fontweight='bold', ha='center', va='bottom')
        
        # Legend
        legend_labels = [f"{label} ({size*100:.1f}%)" 
                        for label, size in zip(labels, sizes)]
        
        ax.legend(
            wedges, legend_labels,
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            frameon=self.config.legend_frameon,
            fontsize=self.config.font_tiny,
            title="Cell Types",
            title_fontsize=self.config.font_small
        )
        
        # Save
        if self.config.tight_layout:
            plt.tight_layout(pad=0.1)
        fig.savefig(output_file, dpi=self.config.dpi, format='png')
        plt.close(fig)
    
    def _plot_domain_average_bar(self, domain_groups: Dict, output_file: Path):
        """
        Plot bar chart with one bar per domain showing average composition.
        
        Parameters
        ----------
        domain_groups : dict
            Mapping from group to list of domains
        output_file : Path
            Output file path
        """
        # Collect average compositions
        domain_names = []
        avg_compositions = []
        
        for grp, dom_list in domain_groups.items():
            for dom in dom_list:
                spot_idx = self.data.domains[self.data.domains == dom].index
                
                if len(spot_idx) == 0:
                    continue
                
                df_domain = self.data.df_projection.loc[spot_idx, :]
                avg_comp = df_domain.mean(axis=0)
                
                domain_names.append(dom)
                avg_compositions.append(avg_comp)
        
        if not domain_names:
            print("  ⚠ No domains to plot")
            return
        
        # Create DataFrame
        df_avg = pd.DataFrame(avg_compositions, index=domain_names)
        df_avg = df_avg.div(df_avg.sum(axis=1), axis=0).fillna(0)
        
        # Sort by cell types
        cell_types = sorted(self.data.cell_types)
        df_avg = df_avg[cell_types]
        
        # Create figure
        n_domains = len(domain_names)
        fig_width = max(8, min(14, n_domains * 0.5 + 3))
        fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=100)
        
        # Plot stacked bars
        x_pos = np.arange(n_domains)
        bottom = np.zeros(n_domains)
        
        for ct in cell_types:
            values = df_avg[ct].values
            
            if values.sum() > 0:
                ax.bar(
                    x_pos, values, bottom=bottom,
                    color=self.celltype_colors[ct],
                    width=0.8,
                    label=ct,
                    edgecolor=self.config.edge_color_bar,
                    linewidth=self.config.line_width * 0.3,
                    alpha=self.config.alpha_bar,
                    rasterized=True
                )
                bottom += values
        
        # Styling
        ax.set_title("Average Cell Type Composition per Domain",
                    fontsize=self.config.font_large, fontweight='bold', pad=12)
        ax.set_xlabel("Domains", fontsize=self.config.font_medium)
        ax.set_ylabel("Cell Fraction", fontsize=self.config.font_medium)
        ax.set_xlim(-0.5, n_domains - 0.5)
        ax.set_ylim(0, 1.05)
        
        # X-axis ticks
        ax.set_xticks(x_pos)
        ax.set_xticklabels(domain_names, rotation=45, ha='right',
                          fontsize=self.config.font_small)
        
        # Y-axis ticks
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        
        # Grid
        if self.config.use_grid:
            ax.grid(axis='y', alpha=self.config.grid_alpha,
                   linestyle=self.config.grid_style, zorder=0)
            ax.set_axisbelow(True)
        
        # Legend
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            frameon=self.config.legend_frameon,
            fontsize=self.config.font_tiny,
            title="Cell Types",
            title_fontsize=self.config.font_small
        )
        
        # Save
        if self.config.tight_layout:
            plt.tight_layout()
        fig.savefig(output_file, dpi=self.config.dpi, format='png', bbox_inches='tight')
        plt.close(fig)
    
    def _plot_all_spots_by_domain(self, domain_groups: Dict, output_file: Path):
        """
        Plot all spots in one large stacked bar chart, grouped by domain.
        
        Parameters
        ----------
        domain_groups : dict
            Mapping from group to list of domains
        output_file : Path
            Output file path
        """
        # Collect all spots ordered by domain
        all_spot_indices = []
        domain_boundaries = []
        current_pos = 0
        
        for grp, dom_list in domain_groups.items():
            for dom in dom_list:
                spot_idx = self.data.domains[self.data.domains == dom].index
                
                if len(spot_idx) == 0:
                    continue
                
                all_spot_indices.extend(spot_idx)
                domain_boundaries.append((dom, current_pos, current_pos + len(spot_idx)))
                current_pos += len(spot_idx)
        
        if not all_spot_indices:
            print("  ⚠ No spots to plot")
            return
        
        # Get composition matrix
        df_all_spots = self.data.df_projection.loc[all_spot_indices, :]
        df_all_spots = df_all_spots.div(df_all_spots.sum(axis=1), axis=0).fillna(0)
        
        # Sort by cell types
        cell_types = sorted(self.data.cell_types)
        df_all_spots = df_all_spots[cell_types]
        
        # Create figure
        n_spots = len(all_spot_indices)
        fig_width = max(15, min(50, n_spots * 0.03 + 5))
        fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=100)
        
        # Plot stacked bars
        x_pos = np.arange(n_spots)
        bottom = np.zeros(n_spots)
        
        for ct in cell_types:
            values = df_all_spots[ct].values
            
            if values.sum() > 0:
                ax.bar(
                    x_pos, values, bottom=bottom,
                    color=self.celltype_colors[ct],
                    width=1.0,
                    label=ct,
                    edgecolor='none',
                    alpha=self.config.alpha_bar,
                    rasterized=True
                )
                bottom += values
        
        # Styling
        ax.set_title("Cell Type Composition - All Spots by Domain",
                    fontsize=self.config.font_large, fontweight='bold', pad=12)
        ax.set_xlabel("Spots (grouped by domain)", fontsize=self.config.font_medium)
        ax.set_ylabel("Cell Fraction", fontsize=self.config.font_medium)
        ax.set_xlim(-0.5, n_spots - 0.5)
        ax.set_ylim(0, 1.05)
        
        # Add domain separators and labels
        for i, (dom, start, end) in enumerate(domain_boundaries):
            if i > 0:
                ax.axvline(x=start - 0.5, color='black', linewidth=1.5,
                          linestyle='-', alpha=0.7, zorder=10)
            
            # Add domain label
            mid = (start + end - 1) / 2
            if end - start >= 3:
                display_dom = dom if len(dom) <= 15 else dom[:13] + ".."
                ax.text(
                    mid, -0.08, display_dom,
                    ha='center', va='top',
                    fontsize=self.config.font_tiny,
                    transform=ax.get_xaxis_transform(),
                    rotation=0 if end - start > 30 else 45
                )
        
        # Hide x-axis ticks
        ax.set_xticks([])
        
        # Y-axis ticks
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        
        # Grid
        if self.config.use_grid:
            ax.grid(axis='y', alpha=self.config.grid_alpha,
                   linestyle=self.config.grid_style, zorder=0)
            ax.set_axisbelow(True)
        
        # Legend
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            frameon=self.config.legend_frameon,
            fontsize=self.config.font_tiny,
            title="Cell Types",
            title_fontsize=self.config.font_small
        )
        
        # Save
        if self.config.tight_layout:
            plt.tight_layout()
        fig.savefig(output_file, dpi=self.config.dpi, format='png', bbox_inches='tight')
        plt.close(fig)


class SpatialPiePlotter:
    """
    Create spatial pie plots overlaying cell type composition on tissue coordinates.
    
    Each spot is represented as a pie chart showing its cell type composition,
    positioned at the spot's spatial coordinates.
    """
    
    def __init__(self, config: PlotConfig, data: DeconvolutionData):
        self.config = config
        self.data = data
        self.celltype_colors = None
    
    def plot_all(self, top_k: int = 8):
        """
        Generate spatial pie plots.
        
        Parameters
        ----------
        top_k : int, default=8
            Number of top cell types to show per spot (others grouped as "Other")
        """
        print("\n" + "=" * 80)
        print("Generating spatial pie plots...")
        print("=" * 80)
        
        # Prepare colors
        self.celltype_colors = self._prepare_celltype_colors()
        
        # Plot full slice
        print("\nGenerating full-slice spatial pie plot...")
        idx_all = self.data.adata_out.obs_names
        output_file = Path(self.config.output_dir) / "spatial_pie_full.png"
        self._plot_spatial_pie(idx_all, "Full Slice", top_k, output_file)
        
        print(f"\n✓ Spatial pie plots saved to: {self.config.output_dir}")
    
    def _prepare_celltype_colors(self) -> Dict[str, str]:
        """Create color mapping for cell types."""
        cell_types = sorted(self.data.cell_types)
        
        if len(cell_types) <= 10:
            colors = self.config.get_nature_colors(len(cell_types))
            color_dict = {ct: colors[i] for i, ct in enumerate(cell_types)}
        else:
            cmap = plt.colormaps.get_cmap(self.config.cmap_celltype)
            color_dict = {ct: cmap(i % cmap.N) for i, ct in enumerate(cell_types)}
        
        # Add "Other" color
        color_dict["Other"] = "#DDDDDD"
        
        return color_dict
    
    def _estimate_radius(self, coords: np.ndarray) -> float:
        """
        Estimate pie chart radius based on nearest neighbor distances.
        
        Parameters
        ----------
        coords : np.ndarray
            Spatial coordinates (n_spots × 2)
            
        Returns
        -------
        radius : float
            Estimated radius for pie charts
        """
        n_spots = len(coords)
        if n_spots < 2:
            return 50.0
        
        # Use KNN to estimate spacing
        n_neighbors = min(7, n_spots)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
        dists, _ = nbrs.kneighbors(coords)
        
        # Median of 1-3 nearest neighbor distances
        nn_med = np.median(dists[:, 1:min(4, n_neighbors)])
        
        if not np.isfinite(nn_med) or nn_med <= 0:
            # Fallback: based on coordinate range
            nn_med = (np.ptp(coords[:, 0]) + np.ptp(coords[:, 1])) / 200.0
        
        # Radius = 40% of spacing to avoid overlap
        return 0.40 * float(nn_med)
    
    def _prepare_spot_data(self, idx_grp, top_k: int) -> List[List[Tuple[str, float]]]:
        """
        Prepare cell type composition data for each spot.
        
        Parameters
        ----------
        idx_grp : array-like
            Spot indices
        top_k : int
            Number of top cell types to retain
            
        Returns
        -------
        spot_data : list of list of tuples
            For each spot, list of (cell_type, fraction) pairs
        """
        df_spots = self.data.df_projection.loc[idx_grp, :]
        cell_types = sorted(self.data.cell_types)
        
        spot_data = []
        for i in range(len(idx_grp)):
            vals = df_spots.iloc[i][cell_types].values
            
            # Get top k
            top_indices = np.argsort(-vals)[:top_k]
            kept_vals = vals[top_indices]
            kept_types = [cell_types[idx] for idx in top_indices]
            
            # Calculate "Other"
            other_val = vals[[j for j in range(len(vals)) if j not in top_indices]].sum()
            
            # Normalize
            total = kept_vals.sum() + other_val
            if total > 0:
                kept_vals = kept_vals / total
                other_val = other_val / total
            
            # Build data
            spot_type_vals = [(t, v) for t, v in zip(kept_types, kept_vals) if v > 1e-6]
            if other_val > 1e-6:
                spot_type_vals.append(("Other", other_val))
            
            spot_data.append(spot_type_vals)
        
        return spot_data
    
    def _plot_spatial_pie(self, idx_grp, title: str, top_k: int, output_file: Path):
        """
        Plot spatial pie chart visualization.
        
        Parameters
        ----------
        idx_grp : array-like
            Spot indices to plot
        title : str
            Plot title
        top_k : int
            Number of top cell types per spot
        output_file : Path
            Output file path
        """
        # Get spatial coordinates
        coords = np.asarray(self.data.adata_out.obsm["spatial"][
            self.data.adata_out.obs_names.isin(idx_grp)
        ], dtype=float)
        
        # Prepare data
        spot_data = self._prepare_spot_data(idx_grp, top_k)
        
        # Estimate radius
        radius = self._estimate_radius(coords)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
        
        # Draw pie chart for each spot
        for i, spot_type_vals in enumerate(spot_data):
            if not spot_type_vals:
                continue
            
            cx, cy = coords[i, 0], coords[i, 1]
            theta = 90.0  # Start at top
            
            # Sort by value (largest first)
            ordered = sorted(spot_type_vals, key=lambda x: x[1], reverse=True)
            
            for z, (cell_type, value) in enumerate(ordered, start=1):
                if value <= 0:
                    continue
                
                theta1 = theta
                theta2 = theta + 360.0 * float(value)
                
                color = self.celltype_colors.get(cell_type, "#BBBBBB")
                
                wedge = Wedge(
                    (cx, cy), radius,
                    theta1, theta2,
                    facecolor=color,
                    edgecolor='white',
                    linewidth=self.config.line_width * 0.5,
                    alpha=self.config.alpha_bar,
                    zorder=10 + z,
                    antialiased=True
                )
                ax.add_patch(wedge)
                
                theta = theta2
        
        # Styling
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(title, fontsize=self.config.font_large,
                    fontweight='bold', pad=10)
        
        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.margins(0.02)
        
        # Create legend
        all_types = set()
        for spot in spot_data:
            for t, _ in spot:
                if t != "Other":
                    all_types.add(t)
        all_types = sorted(list(all_types))
        
        # Limit legend size
        max_legend = min(15, len(all_types))
        handles = [
            Patch(facecolor=self.celltype_colors.get(t, "#777777"),
                 edgecolor='none', label=t)
            for t in all_types[:max_legend]
        ]
        
        # Add "Other" if present
        if any("Other" in [t for t, _ in spot] for spot in spot_data):
            handles.append(
                Patch(facecolor=self.celltype_colors["Other"],
                     edgecolor='none', label="Other")
            )
        
        # Calculate columns
        n_items = len(handles)
        if n_items <= 4:
            ncol = n_items
        elif n_items <= 8:
            ncol = 4
        elif n_items <= 12:
            ncol = 4
        else:
            ncol = 5
        
        # Add legend
        ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            frameon=False,
            fontsize=12,
            title=" ",
            title_fontsize=25,
            ncol=ncol,
            handlelength=1.5,
            handleheight=1.3,
            labelspacing=0.3,
            columnspacing=1.0,
            borderpad=0.3,
            handletextpad=0.5
        )
        
        # Save
        if self.config.tight_layout:
            plt.tight_layout(pad=0.5, rect=[0, 0.05, 1, 1])
        fig.savefig(output_file, dpi=self.config.dpi, format='png', bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# Main Function
# =============================================================================

def load_config_file(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
        
    Returns
    -------
    config : dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Visualize spatial deconvolution results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Configuration YAML file')
    
    # Data arguments
    parser.add_argument('--deconv-file', type=str, help='Deconvolution results h5ad file')
    parser.add_argument('--sc-file', type=str, help='Single-cell reference h5ad file')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                       help='Output directory for plots')
    
    # Plotting parameters
    parser.add_argument('--plot-types', type=str, nargs='+',
                       choices=['bar', 'pie', 'spatial'],
                       default=['bar', 'pie', 'spatial'],
                       help='Types of plots to generate')
    parser.add_argument('--spatial-top-k', type=int, default=8,
                       help='Number of top cell types for spatial pie plots')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output resolution (DPI)')
    parser.add_argument('--retain-percent', type=float, default=0.1,
                       help='Percentage of cells to retain in projection')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        config_dict = load_config_file(args.config)
        
        # Extract relevant parameters
        if 'visualization' in config_dict:
            viz_config = config_dict['visualization']
            if not args.output_dir and 'plot_output_dir' in viz_config:
                args.output_dir = viz_config['plot_output_dir']
            if 'dpi' in viz_config:
                args.dpi = viz_config['dpi']
            if 'spatial_pie' in viz_config:
                args.spatial_top_k = viz_config['spatial_pie'].get('top_k', 8)
        
        # Data paths from main config
        if not args.deconv_file and 'output_dir' in config_dict:
            args.deconv_file = str(Path(config_dict['output_dir']) / 'deconvolution_result.h5ad')
        if not args.sc_file and 'output_dir' in config_dict:
            args.sc_file = str(Path(config_dict['output_dir']) / 'sc_reference.h5ad')
    
    # Validate required arguments
    if not args.deconv_file:
        raise ValueError("Must provide --deconv-file (or --config with output_dir)")
    if not args.sc_file:
        raise ValueError("Must provide --sc-file (or --config with output_dir)")
    
    # Create configuration
    config = PlotConfig(
        deconv_file=args.deconv_file,
        sc_file=args.sc_file,
        output_dir=args.output_dir,
        dpi=args.dpi,
        retain_percent=args.retain_percent
    )
    
    print("=" * 80)
    print("CoseNiche Spatial Deconvolution Visualization")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Deconvolution file: {args.deconv_file}")
    print(f"  Single-cell file:   {args.sc_file}")
    print(f"  Output directory:   {args.output_dir}")
    print(f"  DPI:                {args.dpi}")
    print(f"  Plot types:         {', '.join(args.plot_types)}")
    
    # Load and process data
    data = DeconvolutionData(config)
    data.load_data()
    data.preprocess_celltypes()
    data.process_data()
    
    # Generate plots
    if 'bar' in args.plot_types or 'pie' in args.plot_types:
        plotter = StackedBarPlotter(config, data)
        plotter.plot_all()
    
    if 'spatial' in args.plot_types:
        spatial_plotter = SpatialPiePlotter(config, data)
        spatial_plotter.plot_all(top_k=args.spatial_top_k)
    
    print("\n" + "=" * 80)
    print("✓ Visualization completed successfully!")
    print(f"Results saved in: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
