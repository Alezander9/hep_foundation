"""
Plotting utilities and theme settings for consistent, publication-quality visualizations.
Provides color palettes, sizing guidelines, and helper functions for scientific plots.
"""

import logging
from pathlib import Path

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


# TODO: make this not specific to the two regression models, but a general function for any amount of training histories
def plot_combined_training_histories(
    # baseline_epoch_history: dict,
    # encoded_epoch_history: dict,
    histories: dict,  # Dict[str, Dict] -> {label: epoch_history_dict}
    output_path: Path,
    metrics_to_plot: list[str] = ["loss", "val_loss"],
    metric_labels: dict[str, str] = {
        "loss": "Train Loss",
        "val_loss": "Validation Loss",
    },
    title: str = "Combined Model Training History",
    y_label: str = "Loss (log scale)",
    y_log_scale: bool = True,
) -> None:
    """
    Plots specified metrics from multiple training histories on the same axes.

    Args:
        histories: Dictionary where keys are model labels (str) and values are
                   training history dictionaries structured as {'epoch_str': {'metric': value}}.
        output_path: Path object where the combined plot PDF will be saved.
        metrics_to_plot: List of metric keys (e.g., 'loss', 'val_loss') to plot.
        metric_labels: Dictionary mapping metric keys to display labels for the legend.
        title: Title for the plot.
        y_label: Label for the Y-axis.
        y_log_scale: Whether to use a logarithmic scale for the Y-axis.
    """
    logging.info(f"Generating combined training history plot at: {output_path}")

    def transform_history(epoch_history: dict) -> dict:
        """Transforms epoch-based history dict to metric-based list dict."""
        if not epoch_history:
            return {}

        # Infer metrics from the first epoch
        first_epoch_key = next(iter(epoch_history.keys()))
        metrics = list(epoch_history[first_epoch_key].keys())

        transformed = {metric: [] for metric in metrics}
        num_epochs = len(epoch_history)

        for epoch in range(num_epochs):
            epoch_str = str(epoch)
            if epoch_str in epoch_history:
                for metric in metrics:
                    # Append value if metric exists for this epoch, else None or NaN? Let's use value or None.
                    transformed[metric].append(epoch_history[epoch_str].get(metric))
            else:
                # Handle missing epochs if necessary, though unlikely with standard training
                for metric in metrics:
                    transformed[metric].append(None)

        # Filter out metrics that are all None
        transformed = {
            k: v for k, v in transformed.items() if not all(x is None for x in v)
        }
        return transformed

    try:
        # baseline_history = transform_history(baseline_epoch_history)
        # encoded_history = transform_history(encoded_epoch_history)

        transformed_histories = {
            label: transform_history(epoch_hist)
            for label, epoch_hist in histories.items()
        }

        set_science_style(use_tex=False)
        plt.figure(figsize=get_figure_size("single", ratio=1.2))
        ax = plt.gca()
        colors = get_color_cycle("high_contrast", n=len(histories))

        # metrics_to_plot = [('loss', 'Train Loss'), ('val_loss', 'Validation Loss')]
        # Default linestyles - could be made configurable if needed
        line_styles = {
            metrics_to_plot[i]: ("-" if i % 2 == 0 else "--")
            for i in range(len(metrics_to_plot))
        }

        for idx, (model_label, history) in enumerate(transformed_histories.items()):
            if not history:
                logging.warning(
                    f"Skipping plot for '{model_label}' due to empty history."
                )
                continue

            # Find the first metric actually present to determine epoch range
            epochs = []
            for metric in metrics_to_plot:
                if metric in history and history[metric]:
                    epochs = range(len(history[metric]))
                    break
            if not any(epochs):  # Check if epochs list is still empty
                logging.warning(
                    f"Skipping plot for '{model_label}' as no metrics to plot were found in history."
                )
                continue

            for metric in metrics_to_plot:
                if (
                    metric in history and history[metric]
                ):  # Ensure metric exists and has data
                    display_label = metric_labels.get(
                        metric, metric
                    )  # Use provided label or default to key
                    linestyle = line_styles.get(metric, "-")  # Default to solid line
                    ax.plot(
                        epochs,
                        history[metric],
                        label=f"{model_label} {display_label}",
                        color=colors[idx % len(colors)],
                        linestyle=linestyle,
                        linewidth=LINE_WIDTHS["thick"],
                    )

        # Plot Baseline
        # if baseline_history:
        #     epochs_baseline = range(len(baseline_history.get('loss', [])))
        #     for i, (metric, label) in enumerate(metrics_to_plot):
        #         if metric in baseline_history:
        #             ax.plot(
        #                 epochs_baseline,
        #                 baseline_history[metric],
        #                 label=f"Baseline {label}",
        #                 color=colors[0], # Use first color for baseline
        #                 linestyle=line_styles[metric],
        #                 linewidth=LINE_WIDTHS['thick']
        #             )

        # # Plot Encoded
        # if encoded_history:
        #     epochs_encoded = range(len(encoded_history.get('loss', [])))
        #     for i, (metric, label) in enumerate(metrics_to_plot):
        #          if metric in encoded_history:
        #             ax.plot(
        #                 epochs_encoded,
        #                 encoded_history[metric],
        #                 label=f"Encoded {label}",
        #                 color=colors[1], # Use second color for encoded
        #                 linestyle=line_styles[metric],
        #                 linewidth=LINE_WIDTHS['thick']
        #             )

        if y_log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=FONT_SIZES["large"])
        # ax.set_ylabel('Loss (log scale)', fontsize=FONT_SIZES['large'])
        ax.set_ylabel(y_label, fontsize=FONT_SIZES["large"])
        # ax.set_title('Baseline vs Encoded Model Training History', fontsize=FONT_SIZES['xlarge'])
        ax.set_title(title, fontsize=FONT_SIZES["xlarge"])
        ax.legend(fontsize=FONT_SIZES["normal"], loc="best")  # Changed loc to 'best'
        ax.grid(
            True, which="both", linestyle="--", linewidth=LINE_WIDTHS["thin"], alpha=0.6
        )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Successfully saved combined plot to {output_path}")

    except Exception as e:
        logging.error(f"Failed to create combined training history plot: {e}")
        import traceback

        logging.error(traceback.format_exc())
