"""
Visualization utilities for CoseNiche tutorials.

Provides plotting style setup, color palettes, and figure saving utilities.
"""

from pathlib import Path
from typing import Union, Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def setup_plotting_style(style: str = 'nature', 
                        dpi: int = 300,
                        font_family: str = 'Arial') -> None:
    """
    Set up matplotlib plotting style.
    
    Parameters
    ----------
    style : str, optional
        Style preset ('nature', 'seaborn', 'default') (default: 'nature')
    dpi : int, optional
        Figure DPI for saving (default: 300)
    font_family : str, optional
        Font family name (default: 'Arial')
        Falls back to 'DejaVu Sans' if Arial not available
        
    Examples
    --------
    >>> setup_plotting_style('nature', dpi=600)
    >>> fig, ax = plt.subplots()
    """
    # Try to use requested font, fall back if not available
    import matplotlib.font_manager as fm
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    font_preferences = [font_family, 'Liberation Sans', 'DejaVu Sans', 'Helvetica', 'sans-serif']
    selected_font = 'sans-serif'
    
    for font in font_preferences:
        if font in available_fonts or font == 'sans-serif':
            selected_font = font
            break
    
    if selected_font != font_family:
        print(f"Font '{font_family}' not available, using '{selected_font}'")
    
    if style == 'nature':
        # Nature journal style
        plt.rcParams.update({
            # Font
            'font.family': selected_font,
            'font.size': 10,
            
            # Axes
            'axes.linewidth': 0.5,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'axes.labelweight': 'normal',
            'axes.titleweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': '#000000',
            'axes.facecolor': 'white',
            
            # Ticks
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            
            # Legend
            'legend.fontsize': 9,
            'legend.frameon': False,
            
            # Lines
            'lines.linewidth': 1.0,
            'patch.linewidth': 0.5,
            
            # Grid
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'grid.linewidth': 0.4,
            
            # Figure
            'figure.dpi': 100,
            'figure.facecolor': 'white',
            
            # Saving
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'savefig.facecolor': 'white',
            
            # PDF
            'pdf.fonttype': 42,  # TrueType fonts (editable)
            'ps.fonttype': 42,
        })
    
    elif style == 'seaborn':
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['savefig.dpi'] = dpi
    
    else:  # default
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['savefig.dpi'] = dpi


def get_nature_colors(n_colors: int, 
                     palette: str = 'categorical',
                     alpha: Optional[float] = None) -> List:
    """
    Get Nature-style color palette (colorblind-friendly).
    
    Parameters
    ----------
    n_colors : int
        Number of colors needed
    palette : str, optional
        Palette type ('categorical', 'sequential') (default: 'categorical')
    alpha : float, optional
        Transparency level (0-1), None for opaque
        
    Returns
    -------
    colors : list
        List of color values
        
    Examples
    --------
    >>> colors = get_nature_colors(5)
    >>> colors = get_nature_colors(10, alpha=0.7)
    """
    if palette == 'categorical':
        # Colorblind-friendly categorical palette
        base_colors = [
            '#3C5488',  # Deep blue
            '#E64B35',  # Warm red
            '#4DBBD5',  # Sky blue
            '#00A087',  # Teal green
            '#F39B7F',  # Soft orange
            '#91D1C2',  # Glacier blue
            '#8491B4',  # Gray purple
            '#7E6148',  # Soft brown
            '#B09C85',  # Tan
            '#DF8F44',  # Amber
            '#B2473B',  # Brick red
            '#73C0DE',  # Lake blue
            '#8D9440',  # Olive green
            '#C3A29E',  # Sand pink
            '#FFDC91',  # Light apricot
            '#9A8F97',  # Warm gray purple
            '#5A8A8A',  # Sea green
            '#C28E9B',  # Rose gray
            '#6C6F7D',  # Smoky blue gray
        ]
    
    elif palette == 'sequential':
        # Sequential palette (viridis-like)
        cmap = plt.colormaps.get_cmap('viridis')
        base_colors = [mpl.colors.to_hex(cmap(i / (n_colors - 1))) 
                      for i in range(n_colors)]
    
    else:
        raise ValueError(f"Unknown palette: {palette}")
    
    # Extend if needed
    if n_colors > len(base_colors) and palette == 'categorical':
        base_colors = base_colors * (n_colors // len(base_colors) + 1)
    
    colors = base_colors[:n_colors]
    
    # Apply alpha if specified
    if alpha is not None:
        colors = [mpl.colors.to_rgba(c, alpha=alpha) for c in colors]
    
    return colors


def save_figure(fig, 
               output_path: Union[str, Path],
               dpi: int = 300,
               formats: Optional[List[str]] = None,
               close: bool = True) -> None:
    """
    Save figure in multiple formats.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    output_path : str or Path
        Base output path (without extension)
    dpi : int, optional
        Resolution in dots per inch (default: 300)
    formats : list of str, optional
        File formats to save (['png', 'pdf', 'svg'])
        If None, infers from output_path extension or uses ['png']
    close : bool, optional
        If True, close figure after saving (default: True)
        
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> save_figure(fig, 'plot', formats=['png', 'pdf'])
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine formats
    if formats is None:
        if output_path.suffix:
            # Use extension from path
            formats = [output_path.suffix.lstrip('.')]
            output_base = output_path.with_suffix('')
        else:
            # Default to PNG
            formats = ['png']
            output_base = output_path
    else:
        output_base = output_path.with_suffix('')
    
    # Save in each format
    for fmt in formats:
        save_path = output_base.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=dpi, format=fmt, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    # Close figure
    if close:
        plt.close(fig)


def create_colormap_from_colors(colors: List, 
                                name: str = 'custom',
                                n_bins: int = 256) -> mpl.colors.LinearSegmentedColormap:
    """
    Create a colormap from a list of colors.
    
    Parameters
    ----------
    colors : list
        List of color specifications
    name : str, optional
        Name for the colormap (default: 'custom')
    n_bins : int, optional
        Number of discrete colors (default: 256)
        
    Returns
    -------
    cmap : LinearSegmentedColormap
        Custom colormap
        
    Examples
    --------
    >>> colors = ['#3C5488', '#FFFFFF', '#E64B35']
    >>> cmap = create_colormap_from_colors(colors, 'blue_white_red')
    >>> plt.imshow(data, cmap=cmap)
    """
    return mpl.colors.LinearSegmentedColormap.from_list(name, colors, N=n_bins)


def adjust_text_positions(texts, 
                         ax,
                         avoid_points: Optional[np.ndarray] = None) -> None:
    """
    Adjust text label positions to avoid overlap (basic implementation).
    
    Parameters
    ----------
    texts : list of matplotlib.text.Text
        Text objects to adjust
    ax : matplotlib.axes.Axes
        Axes containing the texts
    avoid_points : np.ndarray, optional
        Points to avoid (shape: n_points × 2)
        
    Notes
    -----
    This is a basic implementation. For advanced label placement,
    consider using the 'adjustText' package.
    
    Examples
    --------
    >>> texts = [ax.text(x, y, label) for x, y, label in zip(xs, ys, labels)]
    >>> adjust_text_positions(texts, ax)
    """
    # Basic implementation - just ensures texts are within axis bounds
    # For production use, consider: pip install adjustText
    # from adjustText import adjust_text
    # adjust_text(texts, ax=ax, avoid_points=avoid_points)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    for text in texts:
        x, y = text.get_position()
        
        # Keep within bounds
        x = np.clip(x, xlim[0], xlim[1])
        y = np.clip(y, ylim[0], ylim[1])
        
        text.set_position((x, y))
