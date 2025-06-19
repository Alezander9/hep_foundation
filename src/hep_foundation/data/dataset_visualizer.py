from pathlib import Path
from typing import Union, Optional
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from hep_foundation.utils import plot_utils
from hep_foundation.config.logging_config import get_logger

def create_plot_from_hist_data(
    hist_data_paths: Union[list[Union[str, Path]], Union[str, Path]],
    output_plot_path: Union[str, Path],
    legend_labels: Optional[list[str]] = None,
    title_prefix: str = "Feature",
):
    """
    Creates a feature distribution plot from saved histogram data.
    Can overlay data from multiple histogram files.

    Args:
        hist_data_paths: Path or list of paths to JSON files containing histogram data.
        output_plot_path: Path to save the PNG plot.
        legend_labels: Optional list of labels for the legend. If provided, must match the number of hist_data_paths.
        title_prefix: Prefix for the main plot title.
    """
    logger = get_logger(__name__)
    
    if not isinstance(hist_data_paths, list):
        hist_data_paths = [hist_data_paths]
    
    hist_data_paths = [Path(p) for p in hist_data_paths]
    output_plot_path = Path(output_plot_path)

    loaded_hist_data_list = []
    effective_legend_labels = []

    if legend_labels and len(legend_labels) != len(hist_data_paths):
        logger.warning("Number of legend_labels does not match number of hist_data_paths. Using filenames as labels.")
        legend_labels = None

    for idx, hist_file_path in enumerate(hist_data_paths):
        if not hist_file_path.exists():
            logger.error(f"Histogram data file not found: {hist_file_path}. Skipping this file.")
            continue
        try:
            with open(hist_file_path, 'r') as f_json:
                data = json.load(f_json)
                if not data:
                    logger.warning(f"No histogram data found in {hist_file_path}. Skipping this file.")
                    continue
                loaded_hist_data_list.append(data)
                if legend_labels:
                    effective_legend_labels.append(legend_labels[idx])
                else:
                    effective_legend_labels.append(hist_file_path.stem.replace("_hist_data", ""))
        except Exception as e:
            logger.error(f"Failed to load histogram data from {hist_file_path}: {e}. Skipping this file.")
            continue

    if not loaded_hist_data_list:
        logger.error("No valid histogram data loaded. Cannot create plot.")
        return

    logger.info(f"Creating plot from {len(loaded_hist_data_list)} data file(s) to {output_plot_path}")
    plot_utils.set_science_style(use_tex=False)

    first_dataset_hist_data = loaded_hist_data_list[0]
    feature_names_ordered = []
    if "N_Tracks_per_Event" in first_dataset_hist_data:
        feature_names_ordered.append("N_Tracks_per_Event")
    other_features = sorted([name for name in first_dataset_hist_data if name != "N_Tracks_per_Event"])
    feature_names_ordered.extend(other_features)
    
    total_plots = len(feature_names_ordered)
    if total_plots == 0:
        logger.warning("No features to plot from loaded data.")
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
    dataset_colors = plot_utils.get_color_cycle(palette="high_contrast", n=len(loaded_hist_data_list))

    for feature_name in feature_names_ordered:
        if plot_idx >= len(axes): break
        ax = axes[plot_idx]

        has_data_for_feature = False
        for i, hist_data_set in enumerate(loaded_hist_data_list):
            data = hist_data_set.get(feature_name)
            if not data or not data.get("counts") or not data.get("bin_edges"):
                logger.warning(f"Missing counts or bin_edges for feature '{feature_name}' in dataset {i}. Skipping this entry for the feature.")
                continue
            
            counts = np.array(data["counts"])
            bin_edges = np.array(data["bin_edges"])

            if counts.size == 0 or bin_edges.size == 0:
                continue
            
            has_data_for_feature = True
            color = dataset_colors[i % len(dataset_colors)]
            label = effective_legend_labels[i]

            if i == 0:  # First dataset (background)
                ax.stairs(counts, bin_edges, fill=True, color=color, 
                          alpha=0.7, linewidth=plot_utils.LINE_WIDTHS["normal"], label=label)
            else:  # Subsequent datasets (signals)
                ax.stairs(counts, bin_edges, fill=False, color=color,
                          linewidth=plot_utils.LINE_WIDTHS["thick"], label=label)

        if not has_data_for_feature:
            ax.set_title(f"{feature_name}\n(No Data in any file)", fontsize=plot_utils.FONT_SIZES["small"])
        else:
            if feature_name == "N_Tracks_per_Event":
                ax.set_title("N Tracks per Event", fontsize=plot_utils.FONT_SIZES["small"])
            else:
                ax.set_title(feature_name, fontsize=plot_utils.FONT_SIZES["small"])

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=plot_utils.FONT_SIZES['tiny'])
        ax.set_yscale('log')

        # Set y-axis formatter for clean power-of-10 labels
        ax.yaxis.set_major_formatter(mticker.LogFormatterExponent())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # Control scientific notation rendering and offset text for the x-axis
        ax.ticklabel_format(style='sci', scilimits=(-3,4), axis='x', useMathText=False)
        
        # Ensure offset text (e.g., 1e-5 at the end of an axis) is small for the x-axis
        offset_text_x = ax.xaxis.get_offset_text()
        offset_text_x.set_fontsize(plot_utils.FONT_SIZES['tiny'])

        # For the y-axis (log scale), LogFormatter handles its own ticks.
        # We avoid using ticklabel_format and get_offset_text for it here to prevent conflicts.

        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    fig.supylabel("Density (log scale)", fontsize=plot_utils.FONT_SIZES['small'])

    if effective_legend_labels and loaded_hist_data_list:
        proxy_handles = []
        for i in range(len(effective_legend_labels)):
            color_index = i % len(dataset_colors)
            current_color = dataset_colors[color_index]
            if i == 0: # First dataset legend entry (solid)
                proxy_handles.append(plt.Rectangle((0, 0), 1, 1, 
                                                   facecolor=current_color,
                                                   alpha=0.7,
                                                   edgecolor=current_color, # Match facecolor for solid look
                                                   linewidth=plot_utils.LINE_WIDTHS["normal"]))
            else: # Subsequent dataset legend entries (outline)
                 proxy_handles.append(plt.Line2D([0], [0], color=current_color, 
                                                linewidth=plot_utils.LINE_WIDTHS["thick"]))
        
        fig.legend(proxy_handles, effective_legend_labels, 
                   loc='lower center', 
                   bbox_to_anchor=(0.5, 0.01), 
                   ncol=min(len(effective_legend_labels), 4), 
                   fontsize=plot_utils.FONT_SIZES['tiny'])

    if len(hist_data_paths) > 1:
        main_title = f"{title_prefix} Distributions Comparison"
    else:
        main_title = f"{title_prefix} Distributions"

    fig.suptitle(main_title, fontsize=plot_utils.FONT_SIZES['large'])
    plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])
    
    try:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Successfully created plot and saved to {output_plot_path}")
    except Exception as e_save:
        logger.error(f"Failed to save created plot to {output_plot_path}: {e_save}")
        plt.close(fig)


def create_combined_two_panel_loss_plot_from_json(
    recon_json_paths: list[Union[str, Path]],
    kl_json_paths: list[Union[str, Path]],
    output_plot_path: Union[str, Path],
    legend_labels: Optional[list[str]] = None,
    title_prefix: str = "Loss Distributions",
):
    """
    Creates a two-panel loss distribution plot (reconstruction + KL divergence) from saved JSON data.
    Left panel shows reconstruction loss, right panel shows KL divergence.
    
    Args:
        recon_json_paths: List of paths to JSON files containing reconstruction loss data.
        kl_json_paths: List of paths to JSON files containing KL divergence loss data.
        output_plot_path: Path to save the PNG plot.
        legend_labels: Optional list of labels for the legend. If provided, must match the number of paths.
        title_prefix: Prefix for the main plot title.
    """
    logger = get_logger(__name__)
    
    recon_json_paths = [Path(p) for p in recon_json_paths]
    kl_json_paths = [Path(p) for p in kl_json_paths]
    output_plot_path = Path(output_plot_path)

    if len(recon_json_paths) != len(kl_json_paths):
        logger.error("Number of reconstruction and KL divergence JSON files must match")
        return

    # Load reconstruction loss data
    recon_data_list = []
    effective_legend_labels = []
    
    for idx, json_path in enumerate(recon_json_paths):
        if not json_path.exists():
            logger.error(f"Reconstruction loss JSON file not found: {json_path}. Skipping this file.")
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if "reconstruction" not in data:
                    logger.warning(f"Reconstruction loss data not found in {json_path}. Skipping this file.")
                    continue
                
                loss_data = data["reconstruction"]
                if not loss_data.get("counts") or not loss_data.get("bin_edges"):
                    logger.warning(f"Missing counts or bin_edges for reconstruction in {json_path}. Skipping this file.")
                    continue
                
                recon_data_list.append(loss_data)
                if legend_labels and idx < len(legend_labels):
                    effective_legend_labels.append(legend_labels[idx])
                else:
                    # Extract signal name from filename
                    filename_parts = json_path.stem.split('_')
                    if len(filename_parts) >= 3:
                        signal_name = filename_parts[2]  # loss_distributions_{signal_name}_reconstruction_data
                    else:
                        signal_name = json_path.stem
                    effective_legend_labels.append(signal_name)
        except Exception as e:
            logger.error(f"Failed to load reconstruction loss data from {json_path}: {e}. Skipping this file.")
            continue

    # Load KL divergence loss data
    kl_data_list = []
    
    for idx, json_path in enumerate(kl_json_paths):
        if not json_path.exists():
            logger.error(f"KL divergence loss JSON file not found: {json_path}. Skipping this file.")
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if "kl_divergence" not in data:
                    logger.warning(f"KL divergence loss data not found in {json_path}. Skipping this file.")
                    continue
                
                loss_data = data["kl_divergence"]
                if not loss_data.get("counts") or not loss_data.get("bin_edges"):
                    logger.warning(f"Missing counts or bin_edges for KL divergence in {json_path}. Skipping this file.")
                    continue
                
                kl_data_list.append(loss_data)
        except Exception as e:
            logger.error(f"Failed to load KL divergence loss data from {json_path}: {e}. Skipping this file.")
            continue

    if not recon_data_list or not kl_data_list:
        logger.error("No valid loss distribution data loaded. Cannot create plot.")
        return

    if len(recon_data_list) != len(kl_data_list):
        logger.error("Mismatch between number of reconstruction and KL divergence datasets after loading")
        return

    logger.info(f"Creating combined two-panel loss distribution plot from {len(recon_data_list)} dataset(s) to {output_plot_path}")
    
    # Set up the two-panel plot
    plot_utils.set_science_style(use_tex=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_utils.get_figure_size("double"))
    
    dataset_colors = plot_utils.get_color_cycle(palette="high_contrast", n=len(recon_data_list))
    
    # Plot reconstruction loss (left panel)
    for i, loss_data in enumerate(recon_data_list):
        counts = np.array(loss_data["counts"])
        bin_edges = np.array(loss_data["bin_edges"])
        
        if counts.size == 0 or bin_edges.size == 0:
            continue
            
        color = dataset_colors[i % len(dataset_colors)]
        label = effective_legend_labels[i]
        
        if i == 0:  # First dataset (background) - solid fill
            ax1.stairs(counts, bin_edges, fill=True, color=color, 
                      alpha=0.7, linewidth=plot_utils.LINE_WIDTHS["normal"], label=label)
        else:  # Subsequent datasets (signals) - outline only
            ax1.stairs(counts, bin_edges, fill=False, color=color,
                      linewidth=plot_utils.LINE_WIDTHS["thick"], label=label)
    
    # Plot KL divergence (right panel)
    for i, loss_data in enumerate(kl_data_list):
        counts = np.array(loss_data["counts"])
        bin_edges = np.array(loss_data["bin_edges"])
        
        if counts.size == 0 or bin_edges.size == 0:
            continue
            
        color = dataset_colors[i % len(dataset_colors)]
        label = effective_legend_labels[i]
        
        if i == 0:  # First dataset (background) - solid fill
            ax2.stairs(counts, bin_edges, fill=True, color=color, 
                      alpha=0.7, linewidth=plot_utils.LINE_WIDTHS["normal"], label=label)
        else:  # Subsequent datasets (signals) - outline only
            ax2.stairs(counts, bin_edges, fill=False, color=color,
                      linewidth=plot_utils.LINE_WIDTHS["thick"], label=label)
    
    # Format the plots
    ax1.set_xlabel("Reconstruction Loss", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.set_ylabel("Density", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.set_title("Reconstruction Loss", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.legend(fontsize=plot_utils.FONT_SIZES["normal"])
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel("KL Divergence", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.set_ylabel("Density", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.set_title("KL Divergence", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.legend(fontsize=plot_utils.FONT_SIZES["normal"])
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Main title
    fig.suptitle(f"{title_prefix}: Background vs Signals", fontsize=plot_utils.FONT_SIZES["xlarge"])
    plt.tight_layout()
    
    # Save the plot
    try:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Successfully created combined two-panel loss distribution plot and saved to {output_plot_path}")
    except Exception as e:
        logger.error(f"Failed to save combined loss distribution plot to {output_plot_path}: {e}")
        plt.close(fig)


def create_combined_roc_curves_plot_from_json(
    recon_roc_json_paths: list[Union[str, Path]],
    kl_roc_json_paths: list[Union[str, Path]],
    output_plot_path: Union[str, Path],
    legend_labels: Optional[list[str]] = None,
    title_prefix: str = "ROC Curves",
):
    """
    Creates a two-panel ROC curves plot (reconstruction + KL divergence) from saved JSON data.
    Left panel shows reconstruction loss ROC curves, right panel shows KL divergence ROC curves.
    Each curve represents one signal's performance vs background.
    
    Args:
        recon_roc_json_paths: List of paths to JSON files containing reconstruction ROC curve data.
        kl_roc_json_paths: List of paths to JSON files containing KL divergence ROC curve data.
        output_plot_path: Path to save the PNG plot.
        legend_labels: Optional list of labels for the legend. If provided, must match the number of paths.
        title_prefix: Prefix for the main plot title.
    """
    logger = get_logger(__name__)
    
    recon_roc_json_paths = [Path(p) for p in recon_roc_json_paths]
    kl_roc_json_paths = [Path(p) for p in kl_roc_json_paths]
    output_plot_path = Path(output_plot_path)

    if len(recon_roc_json_paths) != len(kl_roc_json_paths):
        logger.error("Number of reconstruction and KL divergence ROC JSON files must match")
        return

    # Load reconstruction ROC data
    recon_roc_data_list = []
    effective_legend_labels = []
    
    for idx, json_path in enumerate(recon_roc_json_paths):
        if not json_path.exists():
            logger.error(f"Reconstruction ROC JSON file not found: {json_path}. Skipping this file.")
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if "reconstruction" not in data:
                    logger.warning(f"Reconstruction ROC data not found in {json_path}. Skipping this file.")
                    continue
                
                roc_data = data["reconstruction"]
                if not roc_data.get("fpr") or not roc_data.get("tpr"):
                    logger.warning(f"Missing fpr or tpr for reconstruction in {json_path}. Skipping this file.")
                    continue
                
                recon_roc_data_list.append(roc_data)
                if legend_labels and idx < len(legend_labels):
                    effective_legend_labels.append(legend_labels[idx])
                else:
                    # Extract signal name from filename
                    filename_parts = json_path.stem.split('_')
                    if len(filename_parts) >= 3:
                        signal_name = filename_parts[2]  # roc_curves_{signal_name}_reconstruction_data
                    else:
                        signal_name = json_path.stem
                    effective_legend_labels.append(signal_name)
        except Exception as e:
            logger.error(f"Failed to load reconstruction ROC data from {json_path}: {e}. Skipping this file.")
            continue

    # Load KL divergence ROC data
    kl_roc_data_list = []
    
    for idx, json_path in enumerate(kl_roc_json_paths):
        if not json_path.exists():
            logger.error(f"KL divergence ROC JSON file not found: {json_path}. Skipping this file.")
            continue
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if "kl_divergence" not in data:
                    logger.warning(f"KL divergence ROC data not found in {json_path}. Skipping this file.")
                    continue
                
                roc_data = data["kl_divergence"]
                if not roc_data.get("fpr") or not roc_data.get("tpr"):
                    logger.warning(f"Missing fpr or tpr for KL divergence in {json_path}. Skipping this file.")
                    continue
                
                kl_roc_data_list.append(roc_data)
        except Exception as e:
            logger.error(f"Failed to load KL divergence ROC data from {json_path}: {e}. Skipping this file.")
            continue

    if not recon_roc_data_list or not kl_roc_data_list:
        logger.error("No valid ROC curve data loaded. Cannot create plot.")
        return

    if len(recon_roc_data_list) != len(kl_roc_data_list):
        logger.error("Mismatch between number of reconstruction and KL divergence ROC datasets after loading")
        return

    logger.info(f"Creating combined two-panel ROC curves plot from {len(recon_roc_data_list)} dataset(s) to {output_plot_path}")
    
    # Set up the two-panel plot
    plot_utils.set_science_style(use_tex=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_utils.get_figure_size("double"))
    
    dataset_colors = plot_utils.get_color_cycle(palette="high_contrast", n=len(recon_roc_data_list))
    
    # Plot reconstruction ROC curves (left panel)
    for i, roc_data in enumerate(recon_roc_data_list):
        fpr = np.array(roc_data["fpr"])
        tpr = np.array(roc_data["tpr"])
        auc_value = roc_data.get("auc", 0.0)
        
        if fpr.size == 0 or tpr.size == 0:
            continue
            
        color = dataset_colors[i % len(dataset_colors)]
        label = f"{effective_legend_labels[i]} (AUC = {auc_value:.3f})"
        
        ax1.plot(fpr, tpr, color=color, linewidth=plot_utils.LINE_WIDTHS["thick"], label=label)
    
    # Plot KL divergence ROC curves (right panel)
    for i, roc_data in enumerate(kl_roc_data_list):
        fpr = np.array(roc_data["fpr"])
        tpr = np.array(roc_data["tpr"])
        auc_value = roc_data.get("auc", 0.0)
        
        if fpr.size == 0 or tpr.size == 0:
            continue
            
        color = dataset_colors[i % len(dataset_colors)]
        label = f"{effective_legend_labels[i]} (AUC = {auc_value:.3f})"
        
        ax2.plot(fpr, tpr, color=color, linewidth=plot_utils.LINE_WIDTHS["thick"], label=label)
    
    # Add diagonal reference line to both panels
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=plot_utils.LINE_WIDTHS["normal"])
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=plot_utils.LINE_WIDTHS["normal"])
    
    # Format the plots
    ax1.set_xlabel("False Positive Rate", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.set_ylabel("True Positive Rate", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.set_title("Reconstruction Loss", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.legend(fontsize=plot_utils.FONT_SIZES["normal"], loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    
    ax2.set_xlabel("False Positive Rate", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.set_ylabel("True Positive Rate", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.set_title("KL Divergence", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.legend(fontsize=plot_utils.FONT_SIZES["normal"], loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    
    # Main title
    fig.suptitle(f"{title_prefix}: Signal Performance", fontsize=plot_utils.FONT_SIZES["xlarge"])
    plt.tight_layout()
    
    # Save the plot
    try:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Successfully created combined two-panel ROC curves plot and saved to {output_plot_path}")
    except Exception as e:
        logger.error(f"Failed to save combined ROC curves plot to {output_plot_path}: {e}")
        plt.close(fig)
