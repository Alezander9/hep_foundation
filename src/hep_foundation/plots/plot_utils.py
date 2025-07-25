"""
Plotting utilities and theme settings for consistent, publication-quality visualizations.
Provides color palettes, sizing guidelines, and helper functions for scientific plots.

The module includes:
- HIGH_CONTRAST_COLORS: Extended 15-color palette for complex multi-series plots
- AESTHETIC_COLORS: 5-color gradient palette for simple plots
- Color cycling functions with automatic palette management
- Visualization utilities to preview color palettes
- Line style management for model type differentiation
- Publication-ready styling functions
"""

import logging

import matplotlib.pyplot as plt

# Disable LaTeX rendering by default
plt.rcParams["text.usetex"] = False

# ============================================================================
# Color Palettes
# ============================================================================

# High contrast colors for complex plots with multiple overlapping elements
# Use when data series need to be clearly distinguished in the same plot space
HIGH_CONTRAST_COLORS: list[str] = [
    "dodgerblue",  # Strong blue #0370DB
    "crimson",  # Deep red #BA0020
    "forestgreen",  # Rich green #076C07
    "darkorange",  # Bright orange #DB6D00
    "dimgrey",  # Neutral grey #4B4B4B
    "purple",  # Deep purple #610061
    "orchid",  # Light purple #B752B4
    # Extensions for plots with >7 data series
    "darkturquoise",  # Bright cyan/teal #00CED1
    "chocolate",  # Rich brown #D2691E
    "hotpink",  # Vibrant pink #FF69B4
    "gold",  # Bright yellow-orange #FFD700
    "darkslategray",  # Dark blue-grey #2F4F4F
    "mediumseagreen",  # Medium green #3CB371
    "indianred",  # Muted red #CD5C5C
    "plum",  # Light purple-pink #DDA0DD
]

# Aesthetic gradient for simple plots or subplots
# Use when data is spatially separated or for sequential/progressive data
AESTHETIC_COLORS: list[tuple[float, float, float]] = [
    (4 / 256, 87 / 256, 172 / 256),  # Deep blue #0457AC
    (48 / 256, 143 / 256, 172 / 256),  # Light blue #308FAC
    (55 / 256, 189 / 256, 121 / 256),  # Bright green #37BD79
    (167 / 256, 226 / 256, 55 / 256),  # Light green #A7E237
    (244 / 256, 230 / 256, 4 / 256),  # Yellow #F4E604
]

# ============================================================================
# Figure Sizing and Text Parameters
# ============================================================================

# Standard figure sizes (in inches)
SINGLE_COLUMN_WIDTH = 8.5  # Width for single-column journal figures
DOUBLE_COLUMN_WIDTH = 12.0  # Width for double-column journal figures
GOLDEN_RATIO = 1.618  # Aesthetic ratio for figure dimensions

# Font sizes optimized for readability in printed journals
FONT_SIZES = {
    "tiny": 8,
    "small": 10,
    "normal": 12,
    "large": 14,
    "xlarge": 16,
    "huge": 18,
}

# Line widths and marker sizes
LINE_WIDTHS = {"thin": 0.5, "normal": 1.0, "thick": 2.0, "heavy": 3.0}

# Line styles for systematic model type differentiation
LINE_STYLES = {
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dashdot": "-.",
    "densely_dashed": (0, (5, 2)),
    "densely_dotted": (0, (1, 1)),
    "loosely_dashed": (0, (5, 10)),
    "loosely_dotted": (0, (1, 10)),
}

# Model type to line style mapping for systematic plotting
MODEL_LINE_STYLES = {
    "from_scratch": "-",  # Solid line
    "fine_tuned": "--",  # Dashed line
    "fixed_encoder": ":",  # Dotted line
    "fixed": ":",  # Alternative name for fixed encoder
}

# Additional distinguishable line styles for reference
# These provide more options if you need to distinguish more model types
EXTENDED_LINE_STYLES = {
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dashdot": "-.",  # Dash-dot line (alternating dash and dot)
    "densely_dashed": (0, (5, 2)),  # Tightly spaced dashes
    "densely_dotted": (0, (1, 1)),  # Tightly spaced dots
    "loosely_dashed": (0, (5, 10)),  # Widely spaced dashes
    "loosely_dotted": (0, (1, 10)),  # Widely spaced dots
}


def get_model_line_style(model_type: str) -> str:
    """
    Get the line style for a given model type.

    Args:
        model_type: Model type string (e.g., 'from_scratch', 'fine_tuned', 'fixed_encoder')

    Returns:
        Matplotlib line style string
    """
    return MODEL_LINE_STYLES.get(model_type.lower(), "-")  # Default to solid line


def get_available_line_styles() -> dict:
    """
    Get all available line styles for plotting.

    Returns:
        Dictionary of line style names and their matplotlib codes
    """
    return EXTENDED_LINE_STYLES.copy()


def print_line_style_reference():
    """
    Print a reference of all available line styles.
    Useful for choosing additional line styles for new model types.
    """
    print("Available Line Styles:")
    print("=" * 40)
    print("Currently used in MODEL_LINE_STYLES:")
    for model, style in MODEL_LINE_STYLES.items():
        style_name = next(
            (name for name, code in EXTENDED_LINE_STYLES.items() if code == style),
            "unknown",
        )
        print(f"  {model:<15} : {style:<10} ({style_name})")

    print("\nAll available line styles:")
    for name, code in EXTENDED_LINE_STYLES.items():
        print(f"  {name:<15} : {code}")

    print("\nNote: The most distinguishable line styles are:")
    print("  - Solid line (-)")
    print("  - Dashed line (--)")
    print("  - Dotted line (:)")
    print("  - Dash-dot line (-.) - alternating dash and dot")
    print("  - Densely dashed - tightly spaced dashes")
    print("  - Loosely dashed - widely spaced dashes")


MARKER_SIZES = {"tiny": 2, "small": 4, "normal": 6, "large": 8, "xlarge": 10}

# ============================================================================
# Style Configuration
# ============================================================================


def set_science_style(use_tex: bool = False) -> None:
    """Configure matplotlib for scientific publication plots"""
    plt.style.use("seaborn-v0_8-paper")

    # Disable LaTeX warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # Enable LaTeX rendering only if explicitly requested
    if use_tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{lmodern}",
                "font.family": "serif",
                "font.serif": ["Latin Modern Roman"],
            }
        )

    # Update default parameters for publication quality
    plt.rcParams.update(
        {
            "font.size": FONT_SIZES["normal"],
            "axes.labelsize": FONT_SIZES["large"],
            "axes.titlesize": FONT_SIZES["xlarge"],
            "xtick.labelsize": FONT_SIZES["normal"],
            "ytick.labelsize": FONT_SIZES["normal"],
            "legend.fontsize": FONT_SIZES["normal"],
            "figure.dpi": 300,
        }
    )


def get_figure_size(width: str = "single", ratio: float = None) -> tuple[float, float]:
    """
    Get recommended figure dimensions for publication

    Args:
        width: 'single' or 'double' for column width
        ratio: Optional custom aspect ratio (default: golden ratio)

    Returns:
        Tuple of (width, height) in inches
    """
    w = SINGLE_COLUMN_WIDTH if width == "single" else DOUBLE_COLUMN_WIDTH
    r = ratio if ratio is not None else GOLDEN_RATIO
    return (w, w / r)


def get_color_cycle(palette: str = "high_contrast", n: int = None) -> list:
    """
    Get a color cycle for plotting multiple data series

    Args:
        palette: 'high_contrast' or 'aesthetic'
        n: Number of colors needed (if None, returns full palette)

    Returns:
        List of colors
    """
    colors = HIGH_CONTRAST_COLORS if palette == "high_contrast" else AESTHETIC_COLORS
    if n is not None:
        # Cycle colors if more are needed than available
        return [colors[i % len(colors)] for i in range(n)]
    return colors
