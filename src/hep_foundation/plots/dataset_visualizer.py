import json
import math
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from hep_foundation.config.logging_config import get_logger
from hep_foundation.plots import plot_utils


def format_event_count(event_count: int) -> str:
    """Format event count with order of magnitude notation (k, M, etc.)"""
    if event_count >= 1_000_000:
        return f"{event_count / 1_000_000:.1f}M"
    elif event_count >= 1_000:
        return f"{event_count / 1_000:.1f}k"
    else:
        return str(event_count)


def get_event_count_from_dataset_info(hist_data_path: Path) -> Optional[int]:
    """Extract event count from dataset metadata file."""
    try:
        dataset_info_path = hist_data_path.parent.parent / "_dataset_info.json"
        with open(dataset_info_path) as f:
            data = json.load(f)
        return data["processing_stats"]["total_stats"]["processed_events"]
    except (FileNotFoundError, KeyError):
        return None


def get_specific_signal_event_count(
    hist_data_path: Path, signal_name: str
) -> Optional[int]:
    """Extract event count for a specific signal from dataset metadata."""
    try:
        dataset_info_path = hist_data_path.parent.parent / "_dataset_info.json"
        with open(dataset_info_path) as f:
            data = json.load(f)
        return data["processing_stats"]["signal_stats"][signal_name]["processed_events"]
    except (FileNotFoundError, KeyError):
        return None


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
        logger.warning(
            "Number of legend_labels does not match number of hist_data_paths. Using filenames as labels."
        )
        legend_labels = None

    for idx, hist_file_path in enumerate(hist_data_paths):
        if not hist_file_path.exists():
            logger.error(
                f"Histogram data file not found: {hist_file_path}. Skipping this file."
            )
            continue
        try:
            with open(hist_file_path) as f_json:
                data = json.load(f_json)
                if not data:
                    logger.warning(
                        f"No histogram data found in {hist_file_path}. Skipping this file."
                    )
                    continue
                loaded_hist_data_list.append(data)
                if legend_labels:
                    # Use provided legend labels but enhance with event count
                    base_label = legend_labels[idx]
                    event_count = get_event_count_from_dataset_info(hist_file_path)

                    if event_count is not None:
                        event_str = format_event_count(event_count)
                        enhanced_label = f"{base_label} ({event_str} events)"
                    else:
                        enhanced_label = base_label
                    effective_legend_labels.append(enhanced_label)
                else:
                    # Auto-generate labels with event counts
                    base_name = hist_file_path.stem.replace("_hist_data", "")

                    # Extract signal name for specific signal counts
                    signal_name = None
                    if "_dataset_features" in base_name:
                        signal_name = base_name.replace("_dataset_features", "")
                        if signal_name in ["atlas", "background"]:
                            signal_name = None

                    # Get event count (signal-specific or general)
                    event_count = (
                        get_specific_signal_event_count(hist_file_path, signal_name)
                        if signal_name
                        else get_event_count_from_dataset_info(hist_file_path)
                    )

                    if event_count is not None:
                        event_str = format_event_count(event_count)
                        enhanced_label = f"{base_name} ({event_str} events)"
                    else:
                        enhanced_label = base_name
                    effective_legend_labels.append(enhanced_label)
        except Exception as e:
            logger.error(
                f"Failed to load histogram data from {hist_file_path}: {e}. Skipping this file."
            )
            continue

    if not loaded_hist_data_list:
        logger.error("No valid histogram data loaded. Cannot create plot.")
        return

    logger.info(
        f"Creating plot from {len(loaded_hist_data_list)} data file(s) to {output_plot_path}"
    )
    plot_utils.set_science_style(use_tex=False)

    first_dataset_hist_data = loaded_hist_data_list[0]
    feature_names_ordered = []
    if "N_Tracks_per_Event" in first_dataset_hist_data:
        feature_names_ordered.append("N_Tracks_per_Event")
    # Filter out metadata and other non-feature keys when collecting feature names
    other_features = sorted(
        [
            name
            for name in first_dataset_hist_data
            if name != "N_Tracks_per_Event" and not name.startswith("_")
        ]
    )
    feature_names_ordered.extend(other_features)

    total_plots = len(feature_names_ordered)
    if total_plots == 0:
        logger.warning("No features to plot from loaded data.")
        return

    ncols = max(1, int(math.ceil(math.sqrt(total_plots))))
    nrows = max(1, int(math.ceil(total_plots / ncols)))

    target_ratio = 16 / 9
    fig_width, fig_height = plot_utils.get_figure_size(
        width="double", ratio=target_ratio
    )
    if nrows > 3:
        fig_height *= (nrows / 3.0) ** 0.5

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), squeeze=False
    )
    axes = axes.flatten()
    plot_idx = 0
    dataset_colors = plot_utils.get_color_cycle(
        palette="high_contrast", n=len(loaded_hist_data_list)
    )

    for feature_name in feature_names_ordered:
        if plot_idx >= len(axes):
            break
        ax = axes[plot_idx]

        has_data_for_feature = False
        for i, hist_data_set in enumerate(loaded_hist_data_list):
            data = hist_data_set.get(feature_name)
            if not data or not data.get("counts") or not data.get("bin_edges"):
                logger.warning(
                    f"Missing counts or bin_edges for feature '{feature_name}' in dataset {i}. Skipping this entry for the feature."
                )
                continue

            counts = np.array(data["counts"])
            bin_edges = np.array(data["bin_edges"])

            if counts.size == 0 or bin_edges.size == 0:
                continue

            has_data_for_feature = True
            color = dataset_colors[i % len(dataset_colors)]
            label = effective_legend_labels[i]

            if i == 0:  # First dataset (background)
                ax.stairs(
                    counts,
                    bin_edges,
                    fill=True,
                    color=color,
                    alpha=0.7,
                    linewidth=plot_utils.LINE_WIDTHS["normal"],
                    label=label,
                )
            else:  # Subsequent datasets (signals)
                ax.stairs(
                    counts,
                    bin_edges,
                    fill=False,
                    color=color,
                    linewidth=plot_utils.LINE_WIDTHS["thick"],
                    label=label,
                )

        if not has_data_for_feature:
            ax.set_title(
                f"{feature_name}\n(No Data in any file)",
                fontsize=plot_utils.FONT_SIZES["small"],
            )
        else:
            if feature_name == "N_Tracks_per_Event":
                ax.set_title(
                    "N Tracks per Event", fontsize=plot_utils.FONT_SIZES["small"]
                )
            else:
                ax.set_title(feature_name, fontsize=plot_utils.FONT_SIZES["small"])

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(
            axis="both", which="major", labelsize=plot_utils.FONT_SIZES["tiny"]
        )
        ax.set_yscale("log")

        # Set y-axis formatter for clean power-of-10 labels
        ax.yaxis.set_major_formatter(mticker.LogFormatterExponent())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # Control scientific notation rendering and offset text for the x-axis
        ax.ticklabel_format(style="sci", scilimits=(-3, 4), axis="x", useMathText=False)

        # Ensure offset text (e.g., 1e-5 at the end of an axis) is small for the x-axis
        offset_text_x = ax.xaxis.get_offset_text()
        offset_text_x.set_fontsize(plot_utils.FONT_SIZES["tiny"])

        # For the y-axis (log scale), LogFormatter handles its own ticks.
        # We avoid using ticklabel_format and get_offset_text for it here to prevent conflicts.

        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis("off")

    fig.supylabel("Density (log scale)", fontsize=plot_utils.FONT_SIZES["small"])

    if effective_legend_labels and loaded_hist_data_list:
        proxy_handles = []
        for i in range(len(effective_legend_labels)):
            color_index = i % len(dataset_colors)
            current_color = dataset_colors[color_index]
            if i == 0:  # First dataset legend entry (solid)
                proxy_handles.append(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=current_color,
                        alpha=0.7,
                        edgecolor=current_color,  # Match facecolor for solid look
                        linewidth=plot_utils.LINE_WIDTHS["normal"],
                    )
                )
            else:  # Subsequent dataset legend entries (outline)
                proxy_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=current_color,
                        linewidth=plot_utils.LINE_WIDTHS["thick"],
                    )
                )

        fig.legend(
            proxy_handles,
            effective_legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(effective_legend_labels), 4),
            fontsize=plot_utils.FONT_SIZES["tiny"],
        )

    if len(hist_data_paths) > 1:
        main_title = f"{title_prefix} Distributions Comparison"
    else:
        main_title = f"{title_prefix} Distributions"

    fig.suptitle(main_title, fontsize=plot_utils.FONT_SIZES["large"])
    plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])

    try:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
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
    log_scale: bool = True,
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
        log_scale: Whether to use log scale for y-axis. Default is True.
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
            logger.error(
                f"Reconstruction loss JSON file not found: {json_path}. Skipping this file."
            )
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
                if "reconstruction" not in data:
                    logger.warning(
                        f"Reconstruction loss data not found in {json_path}. Skipping this file."
                    )
                    continue

                loss_data = data["reconstruction"]
                if not loss_data.get("counts") or not loss_data.get("bin_edges"):
                    logger.warning(
                        f"Missing counts or bin_edges for reconstruction in {json_path}. Skipping this file."
                    )
                    continue

                recon_data_list.append(loss_data)
                if legend_labels and idx < len(legend_labels):
                    base_label = legend_labels[idx]
                else:
                    # Extract signal name from filename
                    filename_parts = json_path.stem.split("_")
                    if len(filename_parts) >= 3:
                        signal_name = filename_parts[
                            2
                        ]  # loss_distributions_{signal_name}_reconstruction_data
                    else:
                        signal_name = json_path.stem
                    base_label = signal_name

                # Try to load corresponding ROC data to get AUC values for enhanced legend
                enhanced_label = base_label
                try:
                    # Construct ROC data path by replacing pattern in filename
                    # loss_distributions_{signal_name}_reconstruction_data.json -> roc_curves_{signal_name}_reconstruction_data.json
                    roc_path = json_path.parent / json_path.name.replace(
                        "loss_distributions_", "roc_curves_"
                    )

                    if roc_path.exists():
                        with open(roc_path) as roc_f:
                            roc_data = json.load(roc_f)
                            if (
                                "reconstruction" in roc_data
                                and "auc" in roc_data["reconstruction"]
                            ):
                                auc_value = roc_data["reconstruction"]["auc"]
                                enhanced_label = f"{base_label} (AUC = {auc_value:.3f})"
                except Exception as e:
                    logger.debug(
                        f"Could not load ROC data for AUC enhancement from {json_path.name}: {e}"
                    )
                    # Continue with base label if ROC data loading fails

                effective_legend_labels.append(enhanced_label)
        except Exception as e:
            logger.error(
                f"Failed to load reconstruction loss data from {json_path}: {e}. Skipping this file."
            )
            continue

    # Load KL divergence loss data
    kl_data_list = []

    for idx, json_path in enumerate(kl_json_paths):
        if not json_path.exists():
            logger.error(
                f"KL divergence loss JSON file not found: {json_path}. Skipping this file."
            )
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
                if "kl_divergence" not in data:
                    logger.warning(
                        f"KL divergence loss data not found in {json_path}. Skipping this file."
                    )
                    continue

                loss_data = data["kl_divergence"]
                if not loss_data.get("counts") or not loss_data.get("bin_edges"):
                    logger.warning(
                        f"Missing counts or bin_edges for KL divergence in {json_path}. Skipping this file."
                    )
                    continue

                kl_data_list.append(loss_data)
        except Exception as e:
            logger.error(
                f"Failed to load KL divergence loss data from {json_path}: {e}. Skipping this file."
            )
            continue

    if not recon_data_list or not kl_data_list:
        logger.error("No valid loss distribution data loaded. Cannot create plot.")
        return

    if len(recon_data_list) != len(kl_data_list):
        logger.error(
            "Mismatch between number of reconstruction and KL divergence datasets after loading"
        )
        return

    scale_suffix = "log_scale" if log_scale else "linear_scale"
    logger.info(
        f"Creating combined two-panel loss distribution plot ({scale_suffix}) from {len(recon_data_list)} dataset(s) to {output_plot_path}"
    )

    # Set up the two-panel plot
    plot_utils.set_science_style(use_tex=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_utils.get_figure_size("double"))

    dataset_colors = plot_utils.get_color_cycle(
        palette="high_contrast", n=len(recon_data_list)
    )

    # Plot reconstruction loss (left panel)
    for i, loss_data in enumerate(recon_data_list):
        counts = np.array(loss_data["counts"])
        bin_edges = np.array(loss_data["bin_edges"])

        if counts.size == 0 or bin_edges.size == 0:
            continue

        color = dataset_colors[i % len(dataset_colors)]
        label = effective_legend_labels[i]

        if i == 0:  # First dataset (background) - solid fill
            ax1.stairs(
                counts,
                bin_edges,
                fill=True,
                color=color,
                alpha=0.7,
                linewidth=plot_utils.LINE_WIDTHS["normal"],
                label=label,
            )
        else:  # Subsequent datasets (signals) - outline only
            ax1.stairs(
                counts,
                bin_edges,
                fill=False,
                color=color,
                linewidth=plot_utils.LINE_WIDTHS["thick"],
                label=label,
            )

    # Plot KL divergence (right panel)
    for i, loss_data in enumerate(kl_data_list):
        counts = np.array(loss_data["counts"])
        bin_edges = np.array(loss_data["bin_edges"])

        if counts.size == 0 or bin_edges.size == 0:
            continue

        color = dataset_colors[i % len(dataset_colors)]
        label = effective_legend_labels[i]

        if i == 0:  # First dataset (background) - solid fill
            ax2.stairs(
                counts,
                bin_edges,
                fill=True,
                color=color,
                alpha=0.7,
                linewidth=plot_utils.LINE_WIDTHS["normal"],
                label=label,
            )
        else:  # Subsequent datasets (signals) - outline only
            ax2.stairs(
                counts,
                bin_edges,
                fill=False,
                color=color,
                linewidth=plot_utils.LINE_WIDTHS["thick"],
                label=label,
            )

    # Format the plots
    ax1.set_xlabel("Reconstruction Loss", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.set_ylabel("Density", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.set_title("Reconstruction Loss", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.grid(True, alpha=0.3)
    if log_scale:
        ax1.set_yscale("log")

    ax2.set_xlabel("KL Divergence", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.set_ylabel("Density", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.set_title("KL Divergence", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.grid(True, alpha=0.3)
    if log_scale:
        ax2.set_yscale("log")

    # Create shared legend below both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(labels), 4),  # Arrange horizontally, max 4 columns
        fontsize=plot_utils.FONT_SIZES["normal"],
    )

    # Main title
    scale_text = "Log Scale" if log_scale else "Linear Scale"
    fig.suptitle(
        f"{title_prefix}: Background vs Signals ({scale_text})",
        fontsize=plot_utils.FONT_SIZES["xlarge"],
    )
    plt.tight_layout(
        rect=[0, 0.12, 1, 0.95]
    )  # Leave space for legend below and title above

    # Save the plot
    try:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(
            f"Successfully created {scale_suffix} loss distribution plot: {output_plot_path}"
        )
    except Exception as e:
        logger.error(
            f"Failed to save combined loss distribution plot to {output_plot_path}: {e}"
        )
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
        logger.error(
            "Number of reconstruction and KL divergence ROC JSON files must match"
        )
        return

    # Load reconstruction ROC data
    recon_roc_data_list = []
    effective_legend_labels = []

    for idx, json_path in enumerate(recon_roc_json_paths):
        if not json_path.exists():
            logger.error(
                f"Reconstruction ROC JSON file not found: {json_path}. Skipping this file."
            )
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
                if "reconstruction" not in data:
                    logger.warning(
                        f"Reconstruction ROC data not found in {json_path}. Skipping this file."
                    )
                    continue

                roc_data = data["reconstruction"]
                if not roc_data.get("fpr") or not roc_data.get("tpr"):
                    logger.warning(
                        f"Missing fpr or tpr for reconstruction in {json_path}. Skipping this file."
                    )
                    continue

                recon_roc_data_list.append(roc_data)
                if legend_labels and idx < len(legend_labels):
                    effective_legend_labels.append(legend_labels[idx])
                else:
                    # Extract signal name from filename
                    filename_parts = json_path.stem.split("_")
                    if len(filename_parts) >= 3:
                        signal_name = filename_parts[
                            2
                        ]  # roc_curves_{signal_name}_reconstruction_data
                    else:
                        signal_name = json_path.stem
                    effective_legend_labels.append(signal_name)
        except Exception as e:
            logger.error(
                f"Failed to load reconstruction ROC data from {json_path}: {e}. Skipping this file."
            )
            continue

    # Load KL divergence ROC data
    kl_roc_data_list = []

    for idx, json_path in enumerate(kl_roc_json_paths):
        if not json_path.exists():
            logger.error(
                f"KL divergence ROC JSON file not found: {json_path}. Skipping this file."
            )
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
                if "kl_divergence" not in data:
                    logger.warning(
                        f"KL divergence ROC data not found in {json_path}. Skipping this file."
                    )
                    continue

                roc_data = data["kl_divergence"]
                if not roc_data.get("fpr") or not roc_data.get("tpr"):
                    logger.warning(
                        f"Missing fpr or tpr for KL divergence in {json_path}. Skipping this file."
                    )
                    continue

                kl_roc_data_list.append(roc_data)
        except Exception as e:
            logger.error(
                f"Failed to load KL divergence ROC data from {json_path}: {e}. Skipping this file."
            )
            continue

    if not recon_roc_data_list or not kl_roc_data_list:
        logger.error("No valid ROC curve data loaded. Cannot create plot.")
        return

    if len(recon_roc_data_list) != len(kl_roc_data_list):
        logger.error(
            "Mismatch between number of reconstruction and KL divergence ROC datasets after loading"
        )
        return

    logger.info(
        f"Creating combined two-panel ROC curves plot from {len(recon_roc_data_list)} dataset(s) to {output_plot_path}"
    )

    # Set up the two-panel plot
    plot_utils.set_science_style(use_tex=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_utils.get_figure_size("double"))

    dataset_colors = plot_utils.get_color_cycle(
        palette="high_contrast", n=len(recon_roc_data_list)
    )

    # Plot reconstruction ROC curves (left panel)
    for i, roc_data in enumerate(recon_roc_data_list):
        fpr = np.array(roc_data["fpr"])
        tpr = np.array(roc_data["tpr"])
        auc_value = roc_data.get("auc", 0.0)

        if fpr.size == 0 or tpr.size == 0:
            continue

        color = dataset_colors[i % len(dataset_colors)]
        label = f"{effective_legend_labels[i]} (AUC = {auc_value:.3f})"

        ax1.plot(
            fpr,
            tpr,
            color=color,
            linewidth=plot_utils.LINE_WIDTHS["thick"],
            label=label,
        )

    # Plot KL divergence ROC curves (right panel)
    for i, roc_data in enumerate(kl_roc_data_list):
        fpr = np.array(roc_data["fpr"])
        tpr = np.array(roc_data["tpr"])
        auc_value = roc_data.get("auc", 0.0)

        if fpr.size == 0 or tpr.size == 0:
            continue

        color = dataset_colors[i % len(dataset_colors)]
        label = f"{effective_legend_labels[i]} (AUC = {auc_value:.3f})"

        ax2.plot(
            fpr,
            tpr,
            color=color,
            linewidth=plot_utils.LINE_WIDTHS["thick"],
            label=label,
        )

    # Add diagonal reference line to both panels
    ax1.plot(
        [0, 1], [0, 1], "k--", alpha=0.5, linewidth=plot_utils.LINE_WIDTHS["normal"]
    )
    ax2.plot(
        [0, 1], [0, 1], "k--", alpha=0.5, linewidth=plot_utils.LINE_WIDTHS["normal"]
    )

    # Format the plots
    ax1.set_xlabel("False Positive Rate", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.set_ylabel("True Positive Rate", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.set_title("Reconstruction Loss", fontsize=plot_utils.FONT_SIZES["large"])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])

    ax2.set_xlabel("False Positive Rate", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.set_ylabel("True Positive Rate", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.set_title("KL Divergence", fontsize=plot_utils.FONT_SIZES["large"])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])

    # Create shared legend below both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(
            len(labels), 3
        ),  # Arrange horizontally, max 3 columns for ROC (labels are longer with AUC)
        fontsize=plot_utils.FONT_SIZES["normal"],
    )

    # Main title
    fig.suptitle(
        f"{title_prefix}: Signal Performance", fontsize=plot_utils.FONT_SIZES["xlarge"]
    )
    plt.tight_layout(
        rect=[0, 0.15, 1, 0.95]
    )  # Leave more space for legend below (AUC labels are longer)

    # Save the plot
    try:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.error(
            f"Failed to save combined ROC curves plot to {output_plot_path}: {e}"
        )
        plt.close(fig)
