import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

from hep_foundation.utils import plot_utils

from hep_foundation.config.logging_config import get_logger

def create_plot_from_hist_data(
    hist_data_path: Union[str, Path],
    output_plot_path: Union[str, Path],
    title_prefix: str = "Recreated",
):
    """
    Recreates a feature distribution plot from saved histogram data.

    Args:
        hist_data_path: Path to the JSON file containing histogram data.
        output_plot_path: Path to save the recreated PNG plot.
        title_prefix: Prefix for the main plot title.
    """
    logger = get_logger(__name__)
    hist_data_path = Path(hist_data_path)
    output_plot_path = Path(output_plot_path)

    if not hist_data_path.exists():
        self.logger.error(f"Histogram data file not found: {hist_data_path}")
        return

    try:
        with open(hist_data_path, 'r') as f_json:
            loaded_hist_data = json.load(f_json)
    except Exception as e:
        self.logger.error(f"Failed to load histogram data from {hist_data_path}: {e}")
        return

    if not loaded_hist_data:
        self.logger.warning(f"No histogram data found in {hist_data_path}. Cannot create plot.")
        return

    self.logger.info(f"Recreating plot from {hist_data_path} to {output_plot_path}")
    plot_utils.set_science_style(use_tex=False)

    # Determine order: N_Tracks_per_Event first, then others sorted for consistency
    feature_names_ordered = []
    if "N_Tracks_per_Event" in loaded_hist_data:
        feature_names_ordered.append("N_Tracks_per_Event")
    
    # Add other features, sorted, excluding N_Tracks if already added
    other_features = sorted([name for name in loaded_hist_data if name != "N_Tracks_per_Event"])
    feature_names_ordered.extend(other_features)
    
    total_plots = len(feature_names_ordered)
    if total_plots == 0:
        self.logger.warning("No features to plot from loaded data.")
        return

    ncols = max(1, int(math.ceil(math.sqrt(total_plots))))
    nrows = max(1, int(math.ceil(total_plots / ncols)))

    target_ratio = 16 / 9
    fig_width, fig_height = plot_utils.get_figure_size(width="double", ratio=target_ratio)
    if nrows > 3:
        fig_height *= (nrows / 3.0)**0.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    plot_idx = 0
    colors = plot_utils.get_color_cycle(palette="high_contrast", n=total_plots)

    for feature_name in feature_names_ordered:
        if plot_idx >= len(axes): break
        ax = axes[plot_idx]
        data = loaded_hist_data.get(feature_name)

        if not data or not data.get("counts") or not data.get("bin_edges"):
            self.logger.warning(f"Missing counts or bin_edges for feature '{feature_name}'. Skipping.")
            ax.set_title(f"{feature_name}\n(Data Missing)", fontsize=plot_utils.FONT_SIZES["small"])
            plot_idx += 1
            continue
        
        counts = np.array(data["counts"])
        bin_edges = np.array(data["bin_edges"])

        if counts.size == 0 or bin_edges.size == 0:
            ax.set_title(f"{feature_name}\n(No Data in file)", fontsize=plot_utils.FONT_SIZES["small"])
        else:
            # plt.stairs expects edges to be len(values)+1. Counts are values.
            # The `density=True` from original hist means counts are already normalized.
            ax.stairs(counts, bin_edges, fill=True, color=colors[plot_idx % len(colors)], alpha=0.7, linewidth=0) # linewidth=0 mimics stepfilled
            
            if feature_name == "N_Tracks_per_Event":
                ax.set_title("N Tracks per Event", fontsize=plot_utils.FONT_SIZES["small"])
                ax.set_xlabel("Number of Selected Tracks")
                ax.set_ylabel("Density of Events")
            else:
                ax.set_title(feature_name, fontsize=plot_utils.FONT_SIZES["small"])
                ax.set_ylabel("Density") # Default for others

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=plot_utils.FONT_SIZES['tiny'])
        ax.set_yscale('log')
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    main_title = f"{title_prefix} Feature Distributions from Saved Data"
    if hist_data_path.name:
            main_title += f"\n(Source: {hist_data_path.name})"

    fig.suptitle(main_title, fontsize=plot_utils.FONT_SIZES['large'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjusted rect for potentially longer title
    
    try:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Successfully recreated plot and saved to {output_plot_path}")
    except Exception as e_save:
        self.logger.error(f"Failed to save recreated plot to {output_plot_path}: {e_save}")
        plt.close(fig) # Ensure figure is closed even if save fails


if __name__ == "__main__":
    create_plot_from_hist_data(
        hist_data_path="data/hist_data.json",
        output_plot_path="data/plot.png"
    )