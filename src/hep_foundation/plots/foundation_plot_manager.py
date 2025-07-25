import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from hep_foundation.config.logging_config import get_logger
from hep_foundation.plots.plot_utils import (
    FONT_SIZES,
    LINE_WIDTHS,
    MODEL_LINE_STYLES,
    get_color_cycle,
    get_figure_size,
    set_science_style,
)


class FoundationPlotManager:
    """
    Handles plotting functionality for foundation model evaluations.

    This class encapsulates all plotting logic for:
    - Data efficiency plots for regression and signal classification
    - Training history comparison plots
    - Label distribution analysis plots
    """

    def __init__(self):
        """Initialize the foundation plot manager."""
        self.logger = get_logger(__name__)

        # Set up plotting style
        set_science_style(use_tex=False)
        self.colors = get_color_cycle("high_contrast")

    # DATA EFFICIENCY PLOT CODE
    def create_data_efficiency_plot(
        self,
        results: dict[str, list[float]],
        output_path: Path,
        plot_type: str = "regression",
        metric_name: str = "Test Loss (MSE)",
        total_test_events: int = 0,
        signal_key: Optional[str] = None,
        title_suffix: str = "",
    ) -> bool:
        """
        Create a data efficiency plot comparing different model types.

        Args:
            results: Dictionary containing data_sizes and performance metrics for each model
            output_path: Path where to save the plot
            plot_type: Type of plot ("regression", "classification_loss", "classification_accuracy")
            metric_name: Name of the metric being plotted
            total_test_events: Total number of test events for labeling
            signal_key: Signal key for classification plots
            title_suffix: Additional suffix for the plot title

        Returns:
            bool: True if plot was created successfully, False otherwise
        """
        try:
            plt.figure(figsize=get_figure_size("single", ratio=1.2))

            # Determine plot scale and y-axis limits
            if plot_type == "classification_accuracy":
                plot_func = plt.semilogx
                ylim = (0, 1)
                legend_loc = "lower right"
            else:
                plot_func = plt.loglog
                ylim = None
                legend_loc = "upper right"

            # Plot the three models
            plot_func(
                results["data_sizes"],
                results.get(
                    "From_Scratch",
                    results.get(
                        "From_Scratch_loss", results.get("From_Scratch_accuracy", [])
                    ),
                ),
                "o-",
                color=self.colors[0],
                linewidth=LINE_WIDTHS["thick"],
                markersize=8,
                label="From Scratch",
            )
            plot_func(
                results["data_sizes"],
                results.get(
                    "Fine_Tuned",
                    results.get(
                        "Fine_Tuned_loss", results.get("Fine_Tuned_accuracy", [])
                    ),
                ),
                "s-",
                color=self.colors[1],
                linewidth=LINE_WIDTHS["thick"],
                markersize=8,
                label="Fine-Tuned",
            )
            plot_func(
                results["data_sizes"],
                results.get(
                    "Fixed_Encoder",
                    results.get(
                        "Fixed_Encoder_loss", results.get("Fixed_Encoder_accuracy", [])
                    ),
                ),
                "^-",
                color=self.colors[2],
                linewidth=LINE_WIDTHS["thick"],
                markersize=8,
                label="Fixed Encoder",
            )

            plt.xlabel("Number of Training Events", fontsize=FONT_SIZES["large"])

            # Format y-axis label
            test_events_label = (
                f" - over {total_test_events:,} events" if total_test_events > 0 else ""
            )
            plt.ylabel(
                f"{metric_name}{test_events_label}", fontsize=FONT_SIZES["large"]
            )

            # Create title
            if plot_type == "regression":
                title = "Data Efficiency: Foundation Model Benefits"
            elif plot_type == "classification_loss":
                signal_part = f"\n(Signal: {signal_key})" if signal_key else ""
                title = f"Signal Classification Data Efficiency: Loss{signal_part}"
            elif plot_type == "classification_accuracy":
                signal_part = f"\n(Signal: {signal_key})" if signal_key else ""
                title = f"Signal Classification Data Efficiency: Accuracy{signal_part}"
            else:
                title = f"Data Efficiency{title_suffix}"

            plt.title(title, fontsize=FONT_SIZES["xlarge"])
            plt.legend(fontsize=FONT_SIZES["normal"], loc=legend_loc)
            plt.grid(True, alpha=0.3, which="both")

            if ylim:
                plt.ylim(ylim)

            # Save plot
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Data efficiency plot saved to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create data efficiency plot: {str(e)}")
            return False

    # TRAINING HISTORY COMPARISON PLOT CODE
    def _create_training_history_comparison_plot(
        self,
        training_history_json_paths,
        output_plot_path,
        legend_labels: Optional[list[str]] = None,
        title_prefix: str = "Training History",
        metrics_to_plot: Optional[list[str]] = None,
        validation_only: bool = False,
        handle_outliers: bool = True,
        outlier_percentile: float = 80.0,
    ):
        """
        Creates a training history plot from saved training history JSON data.
        Can overlay data from multiple training runs.

        Args:
            training_history_json_paths: Path or list of paths to JSON files containing training history data.
            output_plot_path: Path to save the PNG plot.
            legend_labels: Optional list of labels for the legend. If provided, must match the number of paths.
            title_prefix: Prefix for the main plot title.
            metrics_to_plot: Optional list of specific metrics to plot. If None, plots all loss metrics.
            validation_only: If True, only plots validation metrics (no training metrics).
            handle_outliers: If True, creates a 1x2 subplot with full range and cropped view.
            outlier_percentile: Percentile threshold for cropping the zoomed view (default: 95.0).
        """
        if not isinstance(training_history_json_paths, list):
            training_history_json_paths = [training_history_json_paths]

        training_history_json_paths = [Path(p) for p in training_history_json_paths]
        output_plot_path = Path(output_plot_path)

        loaded_training_data_list = []
        effective_legend_labels = []

        if legend_labels and len(legend_labels) != len(training_history_json_paths):
            self.logger.warning(
                "Number of legend_labels does not match number of training_history_json_paths. Using model names as labels."
            )
            legend_labels = None

        for idx, json_path in enumerate(training_history_json_paths):
            if not json_path.exists():
                self.logger.error(
                    f"Training history JSON file not found: {json_path}. Skipping this file."
                )
                continue
            try:
                with open(json_path) as f:
                    data = json.load(f)
                    if not data or "history" not in data:
                        self.logger.warning(
                            f"No training history data found in {json_path}. Skipping this file."
                        )
                        continue
                    loaded_training_data_list.append(data)

                    if legend_labels:
                        # Use provided legend labels without adding epoch counts
                        base_label = legend_labels[idx]
                        effective_legend_labels.append(base_label)
                    else:
                        # Auto-generate labels from model names without epoch counts
                        model_name = data.get("model_name", json_path.stem)
                        effective_legend_labels.append(model_name)
            except Exception as e:
                self.logger.error(
                    f"Failed to load training history data from {json_path}: {e}. Skipping this file."
                )
                continue

        if not loaded_training_data_list:
            self.logger.error(
                "No valid training history data loaded. Cannot create plot."
            )
            return

        self.logger.info(
            f"Creating training history plot from {len(loaded_training_data_list)} training run(s) to {output_plot_path}"
        )

        # Set up plotting style
        set_science_style(use_tex=False)

        # Determine metrics to plot
        first_history = loaded_training_data_list[0]["history"]

        if metrics_to_plot is None:
            # Auto-select metrics based on validation_only flag
            all_metrics = list(first_history.keys())
            metrics_to_plot = []

            if validation_only:
                # Only add validation losses
                for metric in all_metrics:
                    if metric.startswith("val_") and "loss" in metric.lower():
                        metrics_to_plot.append(metric)
            else:
                # Add training losses
                for metric in all_metrics:
                    if "loss" in metric.lower() and not metric.startswith(
                        ("val_", "test_")
                    ):
                        metrics_to_plot.append(metric)

                # Add validation losses if they exist
                for metric in all_metrics:
                    if metric.startswith("val_") and "loss" in metric.lower():
                        metrics_to_plot.append(metric)

        # Filter out metrics that don't exist in all datasets
        available_metrics = []
        for metric in metrics_to_plot:
            if all(metric in data["history"] for data in loaded_training_data_list):
                available_metrics.append(metric)
            else:
                self.logger.warning(
                    f"Metric '{metric}' not available in all training histories, skipping."
                )

        metrics_to_plot = available_metrics

        if not metrics_to_plot:
            self.logger.error("No common metrics found across all training histories.")
            return

        # Collect all loss values to determine outlier threshold if needed
        all_loss_values = []
        if handle_outliers:
            for training_data in loaded_training_data_list:
                history = training_data["history"]
                for metric in metrics_to_plot:
                    if metric in history:
                        all_loss_values.extend(history[metric])

            if all_loss_values:
                outlier_threshold = np.percentile(all_loss_values, outlier_percentile)
                self.logger.info(
                    f"Outlier threshold ({outlier_percentile}th percentile): {outlier_threshold:.6f}"
                )
            else:
                self.logger.warning(
                    "No loss values found for outlier detection, using single plot"
                )
                handle_outliers = False

        # Create figure - either single plot or 1x2 subplot
        if handle_outliers and all_loss_values:
            # Use double-column width for 1x2 subplot with slightly less height to prevent it being too tall
            fig, (ax_full, ax_cropped) = plt.subplots(
                1, 2, figsize=get_figure_size("double", ratio=2.0)
            )
            axes = [ax_full, ax_cropped]
            panel_titles = [
                "Full Range (All Data)",
                f"Cropped View ({outlier_percentile:.0f}th Percentile)",
            ]
        else:
            fig, ax = plt.subplots(figsize=get_figure_size("single", ratio=1.2))
            axes = [ax]
            panel_titles = [None]

        # Parse legend labels to extract dataset sizes and model types for systematic styling
        def parse_label(label):
            """Parse label to extract dataset size and model type"""
            # Handle labels like "From Scratch (10k)", "Fine Tuned (5k)", "Fixed Encoder (2k)"
            import re

            # Extract dataset size from parentheses
            dataset_size_match = re.search(r"\(([^)]+)\)", label)
            dataset_size = (
                dataset_size_match.group(1) if dataset_size_match else "unknown"
            )

            # Extract model type from the beginning of the label
            label_lower = label.lower()
            if "from scratch" in label_lower:
                model_type = "from_scratch"
            elif "fine tuned" in label_lower or "fine-tuned" in label_lower:
                model_type = "fine_tuned"
            elif "fixed encoder" in label_lower or "fixed_encoder" in label_lower:
                model_type = "fixed_encoder"
            else:
                model_type = "unknown"

            return dataset_size, model_type

        # Parse all labels to get unique dataset sizes and model types
        dataset_sizes = []
        model_types = []
        label_info = []

        for label in effective_legend_labels:
            dataset_size, model_type = parse_label(label)
            label_info.append((dataset_size, model_type))
            if dataset_size not in dataset_sizes:
                dataset_sizes.append(dataset_size)
            if model_type not in model_types:
                model_types.append(model_type)

        # Sort dataset sizes for consistent color assignment
        # Handle both numeric and string sizes (e.g., "10k", "5k", "2k")
        def sort_key(size):
            if size == "unknown":
                return float("inf")
            # Convert sizes like "10k" to 10000 for sorting
            if size.endswith("k"):
                return int(size[:-1]) * 1000
            elif size.isdigit():
                return int(size)
            else:
                return float("inf")

        dataset_sizes.sort(key=sort_key)

        # Create color mapping: one color per dataset size
        colors = get_color_cycle("high_contrast", n=len(dataset_sizes))
        size_to_color = {size: colors[i] for i, size in enumerate(dataset_sizes)}

        # Create line style mapping: one line style per model type
        model_to_linestyle = MODEL_LINE_STYLES

        # Plot each training run with systematic styling on each axis
        for ax_idx, current_ax in enumerate(axes):
            for train_idx, training_data in enumerate(loaded_training_data_list):
                history = training_data["history"]
                label = effective_legend_labels[train_idx]

                # Get dataset size and model type for this training run
                dataset_size, model_type = label_info[train_idx]

                # Get color based on dataset size and line style based on model type
                base_color = size_to_color.get(
                    dataset_size, colors[0]
                )  # fallback to first color
                base_linestyle = model_to_linestyle.get(
                    model_type, "-"
                )  # fallback to solid line

                if validation_only:
                    # Plot only validation metrics with clean labels (no "- validation" suffix)
                    val_metrics = [m for m in metrics_to_plot if m.startswith("val_")]
                    for metric_idx, metric in enumerate(val_metrics):
                        if metric in history:
                            values = history[metric]
                            epochs = list(range(1, len(values) + 1))

                            # Use clean label without "- validation" suffix
                            # Only add label for first axis to avoid legend duplication
                            line_label = f"{label}" if ax_idx == 0 else None
                            current_ax.plot(
                                epochs,
                                values,
                                color=base_color,
                                linewidth=LINE_WIDTHS["thick"],
                                linestyle=base_linestyle,
                                label=line_label,
                            )
                else:
                    # Plot training metrics (using base line style)
                    train_metrics = [
                        m
                        for m in metrics_to_plot
                        if not m.startswith(("val_", "test_"))
                    ]
                    for metric_idx, metric in enumerate(train_metrics):
                        if metric in history:
                            values = history[metric]
                            epochs = list(range(1, len(values) + 1))

                            line_label = (
                                f"{label}"
                                if len(train_metrics) == 1
                                else f"{label} - {metric}"
                            )
                            # Only add label for first axis to avoid legend duplication
                            line_label = line_label if ax_idx == 0 else None
                            current_ax.plot(
                                epochs,
                                values,
                                color=base_color,
                                linewidth=LINE_WIDTHS["thick"],
                                linestyle=base_linestyle,
                                label=line_label,
                            )

                    # Plot validation metrics (using base line style - systematic approach)
                    val_metrics = [m for m in metrics_to_plot if m.startswith("val_")]
                    for metric_idx, metric in enumerate(val_metrics):
                        if metric in history:
                            values = history[metric]
                            epochs = list(range(1, len(values) + 1))

                            # Use same color and line style for validation (systematic approach)
                            line_label = (
                                f"{label} - {metric}"
                                if len(val_metrics) > 1
                                else f"{label} - validation"
                            )
                            # Only add label for first axis to avoid legend duplication
                            line_label = line_label if ax_idx == 0 else None
                            current_ax.plot(
                                epochs,
                                values,
                                color=base_color,
                                linewidth=LINE_WIDTHS["thick"],
                                linestyle=base_linestyle,
                                label=line_label,
                            )

        # Format each axis
        for ax_idx, current_ax in enumerate(axes):
            current_ax.set_xlabel("Epoch", fontsize=FONT_SIZES["large"])
            current_ax.set_ylabel("Loss (log scale)", fontsize=FONT_SIZES["large"])
            current_ax.set_yscale("log")
            current_ax.grid(True, alpha=0.3, which="both")

            # Set panel-specific title
            if len(axes) > 1:
                if ax_idx == 0:
                    current_ax.set_title(
                        f"{title_prefix}\n{panel_titles[ax_idx]}",
                        fontsize=FONT_SIZES["large"],
                    )
                else:
                    current_ax.set_title(
                        panel_titles[ax_idx], fontsize=FONT_SIZES["large"]
                    )
                    # Set y-limit for cropped view
                    current_ax.set_ylim(top=outlier_threshold)
            else:
                current_ax.set_title(title_prefix, fontsize=FONT_SIZES["xlarge"])

        # Create custom legend with dataset sizes (colors) and model types (line styles)
        # Create legend elements for dataset sizes (colors)
        size_legend_elements = []
        for size in dataset_sizes:
            if size != "unknown":
                color = size_to_color[size]
                size_legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linewidth=LINE_WIDTHS["thick"],
                        linestyle="-",
                        label=f"{size} events",
                    )
                )

        # Create legend elements for model types (line styles)
        model_legend_elements = []
        for model_type in model_types:
            if model_type != "unknown":
                linestyle = model_to_linestyle.get(model_type, "-")
                # Use a neutral color (gray) for line style legend
                display_name = model_type.replace("_", " ").title()
                model_legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        linewidth=LINE_WIDTHS["thick"],
                        linestyle=linestyle,
                        label=display_name,
                    )
                )

        # Combine legend elements with section headers
        legend_elements = []

        # Add dataset size section
        if size_legend_elements:
            legend_elements.append(
                Line2D([0], [0], color="none", label="Dataset Sizes:")
            )
            legend_elements.extend(size_legend_elements)

        # Add model type section
        if model_legend_elements:
            legend_elements.append(Line2D([0], [0], color="none", label="Model Types:"))
            legend_elements.extend(model_legend_elements)

        # Create the legend (only on first axis for multi-panel plots)
        legend_ax = axes[0]
        legend = legend_ax.legend(
            handles=legend_elements,
            fontsize=FONT_SIZES["normal"],
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # Style section headers: make them bold and hide their lines
        legend_texts = legend.get_texts()
        legend_lines = legend.get_lines()

        for text, line in zip(legend_texts, legend_lines):
            text_content = text.get_text()
            if text_content in ["Dataset Sizes:", "Model Types:"]:
                text.set_weight("bold")
                text.set_color("black")
                line.set_visible(False)

        # Save the plot
        try:
            output_plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            self.logger.info(
                f"Successfully created training history plot and saved to {output_plot_path}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to save training history plot to {output_plot_path}: {e}"
            )
            plt.close(fig)

    def create_training_history_comparison_plot_from_directory(
        self,
        training_histories_dir: Path,
        output_path: Path,
        title_prefix: str = "Model Training Comparison",
        validation_only: bool = True,
        handle_outliers: bool = True,
        outlier_percentile: float = 95.0,
    ) -> bool:
        """
        Create a combined training history plot comparing different models from a directory.

        This method discovers training history JSON files in a directory, sorts them by model type
        and data size, then delegates to the core plotting function.

        Args:
            training_histories_dir: Directory containing training history JSON files
            output_path: Path where to save the plot
            title_prefix: Prefix for the plot title
            validation_only: Whether to show only validation loss
            handle_outliers: If True, creates a 1x2 subplot with full range and cropped view
            outlier_percentile: Percentile threshold for cropping the zoomed view (default: 95.0)

        Returns:
            bool: True if plot was created successfully, False otherwise
        """
        try:
            if not training_histories_dir.exists():
                self.logger.warning(
                    f"Training histories directory not found: {training_histories_dir}"
                )
                return False

            training_history_files = list(
                training_histories_dir.glob("training_history_*.json")
            )
            if not training_history_files:
                self.logger.info("No training history files found for combined plot")
                return False

            self.logger.info("Creating combined training history plot...")

            # Sort files to ensure consistent ordering and group by model type and data size
            sorted_files = []
            sorted_labels = []

            # Group by model type and data size
            for model_name in ["From_Scratch", "Fine_Tuned", "Fixed_Encoder"]:
                matching_files = [
                    f for f in training_history_files if model_name in f.name
                ]

                # Sort by data size (extract from filename)
                def extract_data_size(filename):
                    # Extract data size from filename like "From_Scratch_10k"
                    for part in filename.stem.split("_"):
                        if part.endswith("k"):
                            return int(part[:-1]) * 1000
                        elif part.isdigit():
                            return int(part)
                    return 0

                matching_files.sort(key=extract_data_size)

                for file in matching_files:
                    sorted_files.append(file)
                    # Extract model name and data size for label
                    model_display_name = model_name.replace("_", " ")
                    # Extract data size from filename
                    data_size_str = None
                    for part in file.stem.split("_"):
                        if part.endswith("k") or part.isdigit():
                            data_size_str = part
                            break

                    if data_size_str:
                        sorted_labels.append(f"{model_display_name} ({data_size_str})")
                    else:
                        sorted_labels.append(model_display_name)

            if sorted_files:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self._create_training_history_comparison_plot(
                    training_history_json_paths=sorted_files,
                    output_plot_path=output_path,
                    legend_labels=sorted_labels,
                    title_prefix=title_prefix,
                    validation_only=validation_only,
                    handle_outliers=handle_outliers,
                    outlier_percentile=outlier_percentile,
                )
                self.logger.info(
                    f"Combined training history plot saved to: {output_path}"
                )
                return True
            else:
                self.logger.warning(
                    "Could not find expected training history files for combined plot"
                )
                return False

        except Exception as e:
            self.logger.error(
                f"Failed to create training history comparison plot: {str(e)}"
            )
            return False

    # LABEL HISTOGRAM PLOT CODE
    def _save_label_histogram_data(
        self,
        histogram_data: dict,
        output_path: Path,
        metadata_suffix: str = "",
    ):
        """
        Save histogram data to a JSON file.

        Args:
            histogram_data: Dictionary containing histogram data
            output_path: Path where to save the data
            metadata_suffix: Additional metadata information
        """
        try:
            # Ensure the output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create final output structure
            output_data = {}
            for key, value in histogram_data.items():
                if key.startswith("_"):
                    # Skip internal metadata keys
                    continue
                output_data[key] = value

            # Add global metadata if provided
            if metadata_suffix:
                output_data["_metadata"] = {
                    "description": metadata_suffix,
                    "timestamp": str(
                        output_path.stat().st_mtime
                        if output_path.exists()
                        else "unknown"
                    ),
                    "num_variables": len(output_data),
                }

            # Save to JSON
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            self.logger.info(f"Saved histogram data to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save label histogram data: {str(e)}")

    def _create_label_histogram_data(
        self,
        labels: np.ndarray,
        num_bins: int = 50,
        sample_size: Optional[int] = None,
        label_name: str = "labels",
        predefined_bin_edges: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Create histogram data from label arrays.

        Args:
            labels: Array of label values
            num_bins: Number of histogram bins (ignored if predefined_bin_edges is provided)
            sample_size: Optional limit on number of samples to use
            label_name: Name for this label variable
            predefined_bin_edges: Optional predefined bin edges for coordinated binning

        Returns:
            Dictionary with histogram data for the label variable
        """
        try:
            # Convert to numpy array if not already
            labels_array = np.array(labels)

            # Limit sample size if requested
            if sample_size and len(labels_array) > sample_size:
                labels_array = labels_array[:sample_size]

            # Remove any NaN or infinite values
            valid_mask = np.isfinite(labels_array)
            labels_clean = labels_array[valid_mask]

            if len(labels_clean) == 0:
                return {
                    label_name: {
                        "counts": [],
                        "bin_edges": [],
                        "metadata": f"No valid data for {label_name}",
                    }
                }

            # Create histogram using predefined bin edges or compute new ones
            if predefined_bin_edges is not None:
                counts, bin_edges = np.histogram(
                    labels_clean, bins=predefined_bin_edges
                )
            else:
                counts, bin_edges = np.histogram(labels_clean, bins=num_bins)

            return {
                label_name: {
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                    "metadata": f"Histogram for {label_name} with {len(labels_clean)} samples",
                }
            }

        except Exception as e:
            self.logger.error(
                f"Failed to create histogram data for {label_name}: {str(e)}"
            )
            return {
                label_name: {
                    "counts": [],
                    "bin_edges": [],
                    "metadata": f"Error creating histogram for {label_name}",
                }
            }

    def _extract_labels_from_dataset(
        self,
        dataset,
        max_samples: int = 5000,
        task_config=None,
    ):
        """
        Extract labels from a TensorFlow dataset for analysis.

        This method handles both single and multiple label variables and provides
        appropriate data structures for histogram generation.

        Args:
            dataset: TensorFlow dataset containing (features, labels) pairs
            max_samples: Maximum number of samples to extract
            task_config: TaskConfig to extract individual label variable names

        Returns:
            For single label: np.ndarray of shape (n_samples,)
            For multiple labels: dict mapping variable names to np.ndarray
        """
        try:
            labels_list = []
            samples_collected = 0

            for batch_idx, batch in enumerate(dataset):
                if samples_collected >= max_samples:
                    break

                if isinstance(batch, tuple) and len(batch) == 2:
                    features, labels = batch

                    # Convert to numpy
                    try:
                        # Handle case where labels might be a tuple of data structures
                        if isinstance(labels, tuple):
                            # Try to get the actual label data from the tuple
                            # Common patterns: (label_data,) or (label_data, metadata)
                            labels_data = None
                            for item in labels:
                                if hasattr(item, "shape") and hasattr(item, "numpy"):
                                    labels_data = item
                                    break
                                elif (
                                    hasattr(item, "shape")
                                    and len(getattr(item, "shape", [])) >= 2
                                ):
                                    labels_data = item
                                    break

                            if labels_data is None:
                                # If no tensor-like object found, try the first element
                                labels_data = labels[0] if len(labels) > 0 else labels

                            labels = labels_data

                        if hasattr(labels, "numpy"):
                            labels_np = labels.numpy()
                        else:
                            labels_np = np.array(labels)

                        # Handle different label shapes
                        if labels_np.ndim == 3 and labels_np.shape[0] == 1:
                            # Shape like (1, 800, 3) - remove the first dimension
                            labels_np = labels_np.squeeze(axis=0)

                        if labels_np.ndim == 2:
                            # Shape like (batch_size, num_variables) or (num_samples, num_variables)
                            batch_size = labels_np.shape[0]
                            num_variables = labels_np.shape[1]
                        elif labels_np.ndim == 1:
                            # Shape like (batch_size,) - single variable
                            batch_size = labels_np.shape[0]
                        else:
                            continue

                        # Calculate how many samples to take from this batch
                        samples_to_take = min(
                            batch_size, max_samples - samples_collected
                        )

                        # Take the samples
                        batch_labels = labels_np[:samples_to_take]
                        labels_list.append(batch_labels)
                        samples_collected += samples_to_take

                    except Exception:
                        continue

                else:
                    self.logger.warning(
                        f"Unexpected batch structure in dataset: {type(batch)}"
                    )

            if not labels_list:
                return {}

            # Concatenate all collected labels
            all_labels = np.concatenate(labels_list, axis=0)

            # Handle different label structures
            if all_labels.ndim == 1:
                # Single variable case
                return all_labels
            elif all_labels.ndim == 2:
                # Multiple variables case
                num_variables = all_labels.shape[1]

                # Extract variable names using a robust approach with multiple fallback strategies
                variable_names = self._extract_variable_names_robust(
                    task_config, num_variables
                )

                # Create dict mapping variable names to their data
                result_dict = {}
                for i, var_name in enumerate(variable_names):
                    var_data = all_labels[:, i]
                    result_dict[var_name] = var_data

                return result_dict

            else:
                self.logger.warning(
                    f"Unexpected label array dimensions: {all_labels.shape}"
                )
                return {}

        except Exception as e:
            self.logger.error(f"Failed to extract labels from dataset: {str(e)}")
            return {}

    def _extract_variable_names_robust(
        self, task_config, num_variables: int
    ) -> list[str]:
        """
        Extract variable names using multiple robust strategies with systematic fallbacks.

        Args:
            task_config: TaskConfig object (may be None)
            num_variables: Number of variables detected in the data

        Returns:
            List of variable names, always exactly num_variables long
        """
        # Strategy 1: Try to convert task_config to dict and extract names
        try:
            if task_config and hasattr(task_config, "to_dict"):
                task_dict = task_config.to_dict()

                if "labels" in task_dict and task_dict["labels"]:
                    first_label = task_dict["labels"][0]
                    if (
                        "feature_array_aggregators" in first_label
                        and first_label["feature_array_aggregators"]
                    ):
                        first_agg = first_label["feature_array_aggregators"][0]
                        if "input_branches" in first_agg:
                            branch_names = []
                            for branch_info in first_agg["input_branches"]:
                                if (
                                    isinstance(branch_info, dict)
                                    and "branch" in branch_info
                                ):
                                    if (
                                        isinstance(branch_info["branch"], dict)
                                        and "name" in branch_info["branch"]
                                    ):
                                        branch_names.append(
                                            branch_info["branch"]["name"]
                                        )
                                    elif hasattr(branch_info["branch"], "name"):
                                        branch_names.append(branch_info["branch"].name)
                                elif isinstance(branch_info, str):
                                    branch_names.append(branch_info)

                            if len(branch_names) == num_variables:
                                return branch_names
        except Exception:
            pass

        # Strategy 2: Try direct object navigation (original approach)
        try:
            if task_config and hasattr(task_config, "labels") and task_config.labels:
                first_label_config = task_config.labels[0]

                if (
                    hasattr(first_label_config, "feature_array_aggregators")
                    and first_label_config.feature_array_aggregators
                ):
                    first_aggregator = first_label_config.feature_array_aggregators[0]

                    if hasattr(first_aggregator, "input_branches"):
                        branch_names = []
                        for branch_selector in first_aggregator.input_branches:
                            if hasattr(branch_selector, "branch") and hasattr(
                                branch_selector.branch, "name"
                            ):
                                branch_names.append(branch_selector.branch.name)

                        if len(branch_names) == num_variables:
                            return branch_names
        except Exception:
            pass

        # Strategy 3: Use intelligent fallback based on common patterns
        # Common patterns for 3-variable cases (most common in HEP)
        if num_variables == 3:
            # Check if this looks like MET data (common case)
            fallback_names = [
                "MET_Core_AnalysisMETAuxDyn.mpx",
                "MET_Core_AnalysisMETAuxDyn.mpy",
                "MET_Core_AnalysisMETAuxDyn.sumet",
            ]
        elif num_variables == 4:
            # Common 4-vector pattern
            fallback_names = [
                "variable_pt",
                "variable_eta",
                "variable_phi",
                "variable_m",
            ]
        else:
            # Generic pattern
            fallback_names = [f"variable_{i}" for i in range(num_variables)]

        return fallback_names

    def create_label_distribution_analysis(
        self,
        test_dataset,
        label_distributions_dir: Path,
        task_config=None,
        max_samples: int = 5000,
        num_bins: int = 50,
    ) -> Optional[dict]:
        """
        Create label distribution analysis for regression tasks.

        Args:
            test_dataset: TensorFlow dataset containing test labels
            label_distributions_dir: Directory to save label distribution data
            task_config: TaskConfig to extract individual label variable names
            max_samples: Maximum number of samples to use for analysis
            num_bins: Number of bins for histogram

        Returns:
            Dict containing bin edges metadata for each variable, or None if failed
        """
        try:
            self.logger.info("Creating label distribution analysis...")
            label_distributions_dir.mkdir(parents=True, exist_ok=True)

            # Sample actual test labels
            self.logger.info("Sampling actual test labels for distribution analysis...")
            actual_test_labels = self._extract_labels_from_dataset(
                test_dataset, max_samples=max_samples, task_config=task_config
            )

            if isinstance(actual_test_labels, dict):
                # Multiple label variables
                if len(actual_test_labels) == 0:
                    self.logger.warning(
                        "No test labels extracted for histogram analysis"
                    )
                    return None

                # Create histogram data for each variable
                all_hist_data = {}
                bin_edges_metadata = {}

                for var_name, var_data in actual_test_labels.items():
                    self.logger.info(f"Creating histogram for variable: {var_name}")

                    var_hist_data = self._create_label_histogram_data(
                        var_data, num_bins=num_bins, label_name=var_name
                    )

                    # Add this variable's data to the combined structure
                    all_hist_data.update(var_hist_data)

                    # Extract bin edges for this variable
                    if var_name in var_hist_data:
                        bin_edges_metadata[var_name] = var_hist_data[var_name][
                            "bin_edges"
                        ]

                # Save the combined histogram data
                self._save_label_histogram_data(
                    all_hist_data,
                    label_distributions_dir / "actual_test_labels_hist.json",
                    "Actual test labels from regression dataset (multiple variables)",
                )

                # Save bin edges metadata for coordinated binning
                bin_edges_metadata_path = (
                    label_distributions_dir / "label_bin_edges_metadata.json"
                )

                # Save bin edges metadata directly
                try:
                    bin_edges_metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(bin_edges_metadata_path, "w") as f:
                        json.dump(bin_edges_metadata, f, indent=2)
                    self.logger.info(
                        f"Saved bin edges metadata to: {bin_edges_metadata_path}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to save bin edges metadata: {str(e)}")

                self.logger.info(
                    f"Label distribution analysis completed for {len(actual_test_labels)} variables"
                )
                return bin_edges_metadata
            else:
                self.logger.warning(
                    "Expected dictionary result from _extract_labels_from_dataset for multiple variables"
                )
                return None

        except Exception as e:
            self.logger.error(f"Failed to create label distribution analysis: {str(e)}")
            return None

    def save_model_predictions_histogram(
        self,
        predictions: np.ndarray,
        model_name: str,
        data_size: int,
        label_distributions_dir: Path,
        bin_edges_metadata: Optional[dict] = None,
        max_samples: int = 500,
        label_variable_names: Optional[list[str]] = None,
    ) -> bool:
        """
        Save histogram data for model predictions.

        Args:
            predictions: Array of model predictions (1D for single variable, 2D for multiple variables)
            model_name: Name of the model
            data_size: Size of training data used
            label_distributions_dir: Directory to save histogram data
            bin_edges_metadata: Metadata for coordinated binning
            max_samples: Maximum number of samples to include
            label_variable_names: Names of label variables (for multi-variable predictions)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if len(predictions) == 0:
                self.logger.warning(f"No predictions provided for {model_name}")
                return False

            # Limit samples if needed
            predictions_array = np.array(predictions[:max_samples])

            # Check if we have multiple variables (2D array with multiple columns)
            if predictions_array.ndim == 2 and predictions_array.shape[1] > 1:
                # Multiple variables
                if label_variable_names is None:
                    label_variable_names = [
                        f"variable_{i}" for i in range(predictions_array.shape[1])
                    ]

                # Create combined histogram data for all variables
                all_hist_data = {}

                for i, var_name in enumerate(label_variable_names):
                    if i < predictions_array.shape[1]:
                        var_predictions = predictions_array[:, i]

                        # Use predefined bin edges if available for coordinated binning
                        predefined_bin_edges = None
                        if bin_edges_metadata and var_name in bin_edges_metadata:
                            predefined_bin_edges = np.array(
                                bin_edges_metadata[var_name]
                            )
                            self.logger.info(
                                f"Using coordinated binning for {model_name} predictions on variable {var_name}"
                            )

                        var_hist_data = self._create_label_histogram_data(
                            var_predictions,
                            num_bins=50,  # This will be ignored if predefined_bin_edges is provided
                            label_name=var_name,
                            predefined_bin_edges=predefined_bin_edges,
                        )

                        # Add this variable's data to the combined structure
                        all_hist_data.update(var_hist_data)

                # Save the combined histogram data
                pred_hist_filename = f"{model_name}_predictions_hist.json"
                label_distributions_dir.mkdir(parents=True, exist_ok=True)

                self._save_label_histogram_data(
                    all_hist_data,
                    label_distributions_dir / pred_hist_filename,
                    f"Predictions from {model_name} model trained with {data_size} samples (multiple variables)",
                )

                self.logger.info(
                    f"Saved {len(predictions_array)} predictions for {model_name} across {len(label_variable_names)} variables"
                )
                return True
            else:
                # For single variable predictions, we'd handle them here, but this case isn't being used in the current code
                return False

        except Exception as e:
            self.logger.error(
                f"Failed to save predictions histogram for {model_name}: {str(e)}"
            )
            return False

    def create_label_distribution_comparison_plot_with_subplots(
        self,
        evaluation_dir: Path,
        data_sizes: list[int],
        label_variable_names: list[str],
        physlite_plot_labels: Optional[dict] = None,
    ) -> bool:
        """
        Create a subplot-based comparison plot of label distributions with one subplot per variable.

        Args:
            evaluation_dir: Directory containing the evaluation results
            data_sizes: List of data sizes used in evaluation
            label_variable_names: List of label variable names
            physlite_plot_labels: Dictionary mapping branch names to display labels

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import math

            from matplotlib.lines import Line2D

            self.logger.info(
                "Creating label distribution comparison plot with subplots..."
            )

            label_distributions_dir = evaluation_dir / "label_distributions"

            # Load actual test labels histogram
            actual_labels_path = (
                label_distributions_dir / "actual_test_labels_hist.json"
            )
            if not actual_labels_path.exists():
                self.logger.warning(
                    f"Actual labels histogram not found: {actual_labels_path}"
                )
                return False

            with open(actual_labels_path) as f:
                actual_hist_data = json.load(f)

            # Set up plotting style
            set_science_style(use_tex=False)

            # Create subplot layout
            total_plots = len(label_variable_names)
            if total_plots == 0:
                self.logger.warning("No label variables to plot.")
                return False

            ncols = max(1, int(math.ceil(math.sqrt(total_plots))))
            nrows = max(1, int(math.ceil(total_plots / ncols)))

            # Create figure with appropriate size
            target_ratio = 16 / 9
            fig_width, fig_height = get_figure_size("double", ratio=target_ratio)
            if nrows > 3:
                fig_height *= (nrows / 3.0) ** 0.5

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(fig_width, fig_height), squeeze=False
            )
            axes = axes.flatten()

            # Color and style setup
            model_names = ["From_Scratch", "Fine_Tuned", "Fixed_Encoder"]
            colors = get_color_cycle("high_contrast", n=len(data_sizes))
            size_to_color = {
                size: colors[i] for i, size in enumerate(sorted(data_sizes))
            }

            # Line style mapping
            model_to_linestyle = {
                "From_Scratch": "-",
                "Fine_Tuned": "--",
                "Fixed_Encoder": ":",
            }

            plot_idx = 0
            for var_name in label_variable_names:
                if plot_idx >= len(axes):
                    break

                ax = axes[plot_idx]
                has_data_for_variable = False

                # Plot actual labels as a gray, filled histogram
                if var_name in actual_hist_data:
                    actual_data = actual_hist_data[var_name]
                    actual_bin_edges = np.array(actual_data["bin_edges"])
                    actual_counts = np.array(actual_data["counts"])

                    if len(actual_counts) > 0 and np.sum(actual_counts > 0) > 0:
                        ax.stairs(
                            actual_counts,
                            actual_bin_edges,
                            fill=True,
                            color="gray",
                            alpha=0.6,
                            linewidth=LINE_WIDTHS["normal"],
                            label="Actual Test Labels",
                        )
                        has_data_for_variable = True

                # Plot predictions for each model and data size
                for model_name in model_names:
                    for data_size in data_sizes:
                        data_size_label = (
                            f"{data_size // 1000}k"
                            if data_size >= 1000
                            else str(data_size)
                        )

                        pred_hist_path = (
                            label_distributions_dir
                            / f"{model_name}_{data_size_label}_predictions_hist.json"
                        )

                        if pred_hist_path.exists():
                            with open(pred_hist_path) as f:
                                pred_hist_data = json.load(f)

                            if var_name in pred_hist_data:
                                pred_data = pred_hist_data[var_name]
                                pred_counts = np.array(pred_data["counts"])
                                np.array(pred_data["bin_edges"])

                                # Only plot if there are non-zero values
                                if len(pred_counts) > 0 and np.sum(pred_counts > 0) > 0:
                                    plot_color = size_to_color.get(data_size, colors[0])
                                    plot_linestyle = model_to_linestyle.get(
                                        model_name, "-"
                                    )

                                    # Use actual bin edges for consistency (coordinated binning)
                                    if var_name in actual_hist_data:
                                        actual_bin_edges = np.array(
                                            actual_hist_data[var_name]["bin_edges"]
                                        )
                                        ax.stairs(
                                            pred_counts,
                                            actual_bin_edges,
                                            fill=False,
                                            color=plot_color,
                                            linewidth=LINE_WIDTHS["thick"],
                                            linestyle=plot_linestyle,
                                            alpha=0.8,
                                        )
                                        has_data_for_variable = True

                # Set subplot title
                if has_data_for_variable:
                    # Get display name from physlite_plot_labels if available
                    display_name = var_name
                    if physlite_plot_labels and var_name in physlite_plot_labels:
                        display_name = physlite_plot_labels[var_name]
                    elif var_name.startswith("MET_Core_AnalysisMETAuxDyn."):
                        # Handle missing sumet label
                        if var_name.endswith(".sumet"):
                            display_name = "MET Sum ET"
                        else:
                            display_name = var_name.split(".")[
                                -1
                            ]  # Use the last part as fallback

                    ax.set_title(display_name, fontsize=FONT_SIZES["small"])
                else:
                    ax.set_title(f"{var_name}\n(No Data)", fontsize=FONT_SIZES["small"])

                # Set common subplot formatting
                ax.grid(True, alpha=0.3, which="both")
                ax.set_yscale("log")
                ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tiny"])

                plot_idx += 1

            # Hide unused subplots
            for i in range(plot_idx, len(axes)):
                axes[i].axis("off")

            # Create legend
            legend_elements = []

            # Add actual test labels
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    linewidth=LINE_WIDTHS["thick"],
                    alpha=0.8,
                    label="Actual Test Labels",
                )
            )

            # Add dataset size section
            legend_elements.append(
                Line2D([0], [0], color="none", label="Dataset Sizes:")
            )
            for data_size in sorted(data_sizes):
                data_size_label = (
                    f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                )
                color = size_to_color.get(data_size, colors[0])
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linewidth=LINE_WIDTHS["thick"],
                        linestyle="-",
                        label=f"{data_size_label} events",
                    )
                )

            # Add model type section
            legend_elements.append(Line2D([0], [0], color="none", label="Model Types:"))
            for model_name in model_names:
                linestyle = model_to_linestyle.get(model_name, "-")
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        linewidth=LINE_WIDTHS["thick"],
                        linestyle=linestyle,
                        label=model_name.replace("_", " "),
                    )
                )

            # Create the legend
            legend = fig.legend(
                handles=legend_elements,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(len(legend_elements), 4),
                fontsize=FONT_SIZES["tiny"],
                frameon=True,
                fancybox=True,
                shadow=True,
            )

            # Style section headers
            legend_texts = legend.get_texts()
            legend_lines = legend.get_lines()
            for text, line in zip(legend_texts, legend_lines):
                text_content = text.get_text()
                if text_content in ["Dataset Sizes:", "Model Types:"]:
                    text.set_weight("bold")
                    text.set_color("black")
                    line.set_visible(False)

            # Set overall labels and title
            fig.supylabel("Density (log scale)", fontsize=FONT_SIZES["small"])
            fig.suptitle(
                "Label Distribution Comparison: Actual vs. Predicted",
                fontsize=FONT_SIZES["large"],
            )

            # Adjust layout and save
            plt.tight_layout(rect=[0.05, 0.08, 1, 0.95])

            plot_path = evaluation_dir / "label_distribution_comparison.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(
                f"Saved label distribution comparison plot to: {plot_path}"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to create label distribution comparison plot with subplots: {str(e)}"
            )
            return False
