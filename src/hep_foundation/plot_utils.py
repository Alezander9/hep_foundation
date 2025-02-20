"""
Plotting utilities and theme settings for consistent, publication-quality visualizations.
Provides color palettes, sizing guidelines, and helper functions for scientific plots.
"""

import matplotlib.pyplot as plt
from typing import List, Tuple

# Disable LaTeX rendering by default
plt.rcParams['text.usetex'] = False

# ============================================================================
# Color Palettes
# ============================================================================

# High contrast colors for complex plots with multiple overlapping elements
# Use when data series need to be clearly distinguished in the same plot space
HIGH_CONTRAST_COLORS: List[str] = [
    'dodgerblue',    # Strong blue
    'crimson',       # Deep red
    'forestgreen',   # Rich green
    'darkorange',    # Bright orange
    'dimgrey',       # Neutral grey
    'purple',        # Deep purple
    'orchid'         # Light purple
]

# Aesthetic gradient for simple plots or subplots
# Use when data is spatially separated or for sequential/progressive data
AESTHETIC_COLORS: List[Tuple[float, float, float]] = [
    (4/256, 87/256, 172/256),    # Deep blue
    (48/256, 143/256, 172/256),  # Light blue
    (55/256, 189/256, 121/256),  # Bright green
    (167/256, 226/256, 55/256),  # Light green
    (244/256, 230/256, 4/256)    # Yellow
]

# ============================================================================
# Figure Sizing and Text Parameters
# ============================================================================

# Standard figure sizes (in inches)
SINGLE_COLUMN_WIDTH = 8.5    # Width for single-column journal figures
DOUBLE_COLUMN_WIDTH = 12.0   # Width for double-column journal figures
GOLDEN_RATIO = 1.618        # Aesthetic ratio for figure dimensions

# Font sizes optimized for readability in printed journals
FONT_SIZES = {
    'tiny': 8,
    'small': 10,
    'normal': 12,
    'large': 14,
    'xlarge': 16,
    'huge': 18
}

# Line widths and marker sizes
LINE_WIDTHS = {
    'thin': 0.5,
    'normal': 1.0,
    'thick': 2.0,
    'heavy': 3.0
}

MARKER_SIZES = {
    'tiny': 2,
    'small': 4,
    'normal': 6,
    'large': 8,
    'xlarge': 10
}

# ============================================================================
# Style Configuration
# ============================================================================

def set_science_style(use_tex: bool = True) -> None:
    """Configure matplotlib for scientific publication plots"""
    plt.style.use('seaborn-v0_8-paper')
    
    # Enable LaTeX rendering
    if use_tex:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{lmodern}',
            'font.family': 'serif',
            'font.serif': ['Latin Modern Roman'],
        })
    
    # Update default parameters for publication quality
    plt.rcParams.update({
        'font.size': FONT_SIZES['normal'],
        'axes.labelsize': FONT_SIZES['large'],
        'axes.titlesize': FONT_SIZES['xlarge'],
        'xtick.labelsize': FONT_SIZES['normal'],
        'ytick.labelsize': FONT_SIZES['normal'],
        'legend.fontsize': FONT_SIZES['normal'],
        'figure.dpi': 300
    })

def get_figure_size(width: str = 'single', ratio: float = None) -> Tuple[float, float]:
    """
    Get recommended figure dimensions for publication
    
    Args:
        width: 'single' or 'double' for column width
        ratio: Optional custom aspect ratio (default: golden ratio)
    
    Returns:
        Tuple of (width, height) in inches
    """
    w = SINGLE_COLUMN_WIDTH if width == 'single' else DOUBLE_COLUMN_WIDTH
    r = ratio if ratio is not None else GOLDEN_RATIO
    return (w, w/r)

def get_color_cycle(palette: str = 'high_contrast', n: int = None) -> List:
    """
    Get a color cycle for plotting multiple data series
    
    Args:
        palette: 'high_contrast' or 'aesthetic'
        n: Number of colors needed (if None, returns full palette)
    
    Returns:
        List of colors
    """
    colors = HIGH_CONTRAST_COLORS if palette == 'high_contrast' else AESTHETIC_COLORS
    if n is not None:
        # Cycle colors if more are needed than available
        return [colors[i % len(colors)] for i in range(n)]
    return colors 