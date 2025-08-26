import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from hep_foundation.config.logging_config import get_logger
from hep_foundation.plots.histogram_manager import HistogramManager
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
        self.histogram_manager = HistogramManager()

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
        ax=None,
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
            ax: Optional matplotlib axis to plot on. If None, creates a new figure.
        """
        if not isinstance(training_history_json_paths, list):
            training_history_json_paths = [training_history_json_paths]

        training_history_json_paths = [Path(p) for p in training_history_json_paths]
        if output_plot_path is not None:
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

        # Create figure or use provided axis
        if ax is None:
            set_science_style(use_tex=False)
            fig, ax = plt.subplots(figsize=get_figure_size("single", ratio=1.2))
            create_new_figure = True
        else:
            create_new_figure = False

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

        # Plot each training run with systematic styling
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
                for metric in val_metrics:
                    if metric in history:
                        values = history[metric]
                        epochs = list(range(1, len(values) + 1))

                        ax.plot(
                            epochs,
                            values,
                            color=base_color,
                            linewidth=LINE_WIDTHS["thick"],
                            linestyle=base_linestyle,
                            label=label,
                        )
            else:
                # Plot training metrics (using base line style)
                train_metrics = [
                    m for m in metrics_to_plot if not m.startswith(("val_", "test_"))
                ]
                for metric in train_metrics:
                    if metric in history:
                        values = history[metric]
                        epochs = list(range(1, len(values) + 1))

                        line_label = (
                            f"{label}"
                            if len(train_metrics) == 1
                            else f"{label} - {metric}"
                        )
                        ax.plot(
                            epochs,
                            values,
                            color=base_color,
                            linewidth=LINE_WIDTHS["thick"],
                            linestyle=base_linestyle,
                            label=line_label,
                        )

                # Plot validation metrics (using base line style - systematic approach)
                val_metrics = [m for m in metrics_to_plot if m.startswith("val_")]
                for metric in val_metrics:
                    if metric in history:
                        values = history[metric]
                        epochs = list(range(1, len(values) + 1))

                        # Use same color and line style for validation (systematic approach)
                        line_label = (
                            f"{label} - {metric}"
                            if len(val_metrics) > 1
                            else f"{label} - validation"
                        )
                        ax.plot(
                            epochs,
                            values,
                            color=base_color,
                            linewidth=LINE_WIDTHS["thick"],
                            linestyle=base_linestyle,
                            label=line_label,
                        )

        # Format axis
        ax.set_xlabel("Epoch", fontsize=FONT_SIZES["large"])
        ax.set_ylabel("Loss (log scale)", fontsize=FONT_SIZES["large"])
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

        # Set title only if we created a new figure
        if create_new_figure:
            ax.set_title(title_prefix, fontsize=FONT_SIZES["xlarge"])

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

        # Create the legend
        legend = ax.legend(
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

        # Save the plot only if we created a new figure and have an output path
        if create_new_figure and output_plot_path is not None:
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

    def _create_data_efficiency_plot_on_axis(
        self,
        ax,
        results: dict[str, list[float]],
        plot_type: str = "regression",
        metric_name: str = "Test Loss (MSE)",
        total_test_events: int = 0,
        signal_key: Optional[str] = None,
    ):
        """
        Create a data efficiency plot on the provided axis.
        """
        # Determine plot scale and y-axis limits
        if plot_type == "classification_accuracy":
            plot_func = ax.semilogx
            ylim = (0, 1)
            legend_loc = "lower right"
        else:
            plot_func = ax.loglog
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
                results.get("Fine_Tuned_loss", results.get("Fine_Tuned_accuracy", [])),
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

        ax.set_xlabel("Number of Training Events", fontsize=FONT_SIZES["large"])

        # Format y-axis label
        test_events_label = (
            f" - over {total_test_events:,} events" if total_test_events > 0 else ""
        )
        ax.set_ylabel(f"{metric_name}{test_events_label}", fontsize=FONT_SIZES["large"])

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
            title = "Data Efficiency"

        ax.set_title(title, fontsize=FONT_SIZES["large"])
        ax.legend(fontsize=FONT_SIZES["normal"], loc=legend_loc)
        ax.grid(True, alpha=0.3, which="both")

        if ylim:
            ax.set_ylim(ylim)

    def create_combined_downstream_evaluation_plot(
        self,
        training_histories_dir: Path,
        results: dict[str, list[float]],
        output_path: Path,
        plot_type: str = "regression",
        metric_name: str = "Test Loss (MSE)",
        total_test_events: int = 0,
        signal_key: Optional[str] = None,
        title_prefix: str = "Downstream Task Evaluation",
    ) -> bool:
        """
        Create a combined two-panel plot with training history comparison and data efficiency.

        Args:
            training_histories_dir: Directory containing training history JSON files
            results: Dictionary containing data_sizes and performance metrics for each model
            output_path: Path where to save the plot
            plot_type: Type of plot ("regression", "classification_loss", "classification_accuracy")
            metric_name: Name of the metric being plotted
            total_test_events: Total number of test events for labeling
            signal_key: Signal key for classification plots
            title_prefix: Prefix for the overall plot title

        Returns:
            bool: True if plot was created successfully, False otherwise
        """
        try:
            set_science_style(use_tex=False)
            fig, (ax_efficiency, ax_training) = plt.subplots(
                1, 2, figsize=get_figure_size("double", ratio=2.0)
            )

            # Panel 1: Data Efficiency Plot
            self._create_data_efficiency_plot_on_axis(
                ax_efficiency,
                results,
                plot_type,
                metric_name,
                total_test_events,
                signal_key,
            )

            # Panel 2: Training History Comparison
            if training_histories_dir.exists():
                training_history_files = list(
                    training_histories_dir.glob("training_history_*.json")
                )
                if training_history_files:
                    # Sort files to ensure consistent ordering
                    sorted_files = []
                    sorted_labels = []

                    for model_name in ["From_Scratch", "Fine_Tuned", "Fixed_Encoder"]:
                        matching_files = [
                            f for f in training_history_files if model_name in f.name
                        ]

                        def extract_data_size(filename):
                            for part in filename.stem.split("_"):
                                if part.endswith("k"):
                                    return int(part[:-1]) * 1000
                                elif part.isdigit():
                                    return int(part)
                            return 0

                        matching_files.sort(key=extract_data_size)

                        for file in matching_files:
                            sorted_files.append(file)
                            model_display_name = model_name.replace("_", " ")
                            data_size_str = None
                            for part in file.stem.split("_"):
                                if part.endswith("k") or part.isdigit():
                                    data_size_str = part
                                    break

                            if data_size_str:
                                sorted_labels.append(
                                    f"{model_display_name} ({data_size_str})"
                                )
                            else:
                                sorted_labels.append(model_display_name)

                    if sorted_files:
                        self._create_training_history_comparison_plot(
                            training_history_json_paths=sorted_files,
                            output_plot_path=None,
                            legend_labels=sorted_labels,
                            title_prefix="Training History",
                            validation_only=True,
                            ax=ax_training,
                        )

            # Set overall title
            fig.suptitle(title_prefix, fontsize=FONT_SIZES["xlarge"], y=0.98)

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Make room for suptitle

            # Save plot
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            self.logger.info(
                f"Combined downstream evaluation plot saved to: {output_path}"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to create combined downstream evaluation plot: {str(e)}"
            )
            return False

    # LABEL HISTOGRAM PLOT CODE
    def create_label_distribution_comparison_plot_with_subplots(
        self,
        evaluation_dir: Path,
        data_sizes: list[int],
        label_variable_names: list[str],
        physlite_plot_labels: Optional[dict] = None,
    ) -> bool:
        """
        Create a combined two-panel plot with actual vs predicted (left) and differences (right).

        Args:
            evaluation_dir: Directory containing the evaluation results
            data_sizes: List of data sizes used in evaluation
            label_variable_names: List of label variable names
            physlite_plot_labels: Dictionary mapping branch names to display labels

        Returns:
            bool: True if plot was created successfully, False otherwise
        """
        return self._create_combined_label_distribution_plot(
            evaluation_dir,
            data_sizes,
            label_variable_names,
            physlite_plot_labels,
        )

    def _create_combined_label_distribution_plot(
        self,
        evaluation_dir: Path,
        data_sizes: list[int],
        label_variable_names: list[str],
        physlite_plot_labels: Optional[dict] = None,
    ) -> bool:
        """
        Create a combined two-panel plot with actual vs predicted (left) and differences (right).

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

            self.logger.info("Creating combined label distribution plot...")

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

            actual_hist_data, _ = self.histogram_manager.load_hist_file(
                actual_labels_path
            )

            # Set up plotting style
            set_science_style(use_tex=False)

            # Calculate subplot layout for each panel
            total_plots = len(label_variable_names)
            if total_plots == 0:
                self.logger.warning("No label variables to plot.")
                return False

            ncols = max(1, int(math.ceil(math.sqrt(total_plots))))
            nrows = max(1, int(math.ceil(total_plots / ncols)))

            # Create figure with two main panels side by side, each containing subplots
            # Use wider aspect ratio so each subplot can be approximately square
            fig_width, fig_height = get_figure_size("double", ratio=2.0)
            if nrows > 3:
                fig_height *= (nrows / 3.0) ** 0.5

            # Create nested gridspec: 1x2 main grid, each cell contains nrows x ncols subplots
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Create two separate subplot grids
            gs_main = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
            gs_left = gs_main[0].subgridspec(nrows, ncols, hspace=0.4, wspace=0.3)
            gs_right = gs_main[1].subgridspec(nrows, ncols, hspace=0.4, wspace=0.3)

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

            # Create subplots for both panels
            for plot_idx, var_name in enumerate(label_variable_names):
                if plot_idx >= nrows * ncols:
                    break

                row = plot_idx // ncols
                col = plot_idx % ncols

                # Left panel: Actual vs Predicted (log scale)
                ax_left = fig.add_subplot(gs_left[row, col])
                has_data_left = False

                # Plot actual labels as gray filled histogram
                if var_name in actual_hist_data:
                    actual_data = actual_hist_data[var_name]
                    actual_bin_edges = np.array(actual_data["bin_edges"])
                    actual_counts = np.array(actual_data["counts"])

                    if len(actual_counts) > 0 and np.sum(actual_counts > 0) > 0:
                        ax_left.stairs(
                            actual_counts,
                            actual_bin_edges,
                            fill=True,
                            color="gray",
                            alpha=0.6,
                            linewidth=LINE_WIDTHS["normal"],
                            label="Actual" if plot_idx == 0 else "",
                        )
                        has_data_left = True

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
                            pred_hist_data, _ = self.histogram_manager.load_hist_file(
                                pred_hist_path
                            )

                            if var_name in pred_hist_data:
                                pred_data = pred_hist_data[var_name]
                                pred_counts = np.array(pred_data["counts"])

                                if len(pred_counts) > 0 and np.sum(pred_counts > 0) > 0:
                                    plot_color = size_to_color.get(data_size, colors[0])
                                    plot_linestyle = model_to_linestyle.get(
                                        model_name, "-"
                                    )

                                    # Use actual bin edges for consistency
                                    if var_name in actual_hist_data:
                                        actual_bin_edges = np.array(
                                            actual_hist_data[var_name]["bin_edges"]
                                        )
                                        ax_left.stairs(
                                            pred_counts,
                                            actual_bin_edges,
                                            fill=False,
                                            color=plot_color,
                                            linewidth=LINE_WIDTHS["thick"],
                                            linestyle=plot_linestyle,
                                            alpha=0.8,
                                        )
                                        has_data_left = True

                # Right panel: Differences (linear scale)
                ax_right = fig.add_subplot(gs_right[row, col])
                has_data_right = False

                # Plot differences for each model and data size
                for model_name in model_names:
                    for data_size in data_sizes:
                        data_size_label = (
                            f"{data_size // 1000}k"
                            if data_size >= 1000
                            else str(data_size)
                        )

                        # Look for difference histogram files
                        diff_hist_path = (
                            label_distributions_dir
                            / f"{model_name}_{data_size_label}_diff_predictions_hist.json"
                        )

                        if diff_hist_path.exists():
                            diff_hist_data, _ = self.histogram_manager.load_hist_file(
                                diff_hist_path
                            )

                            if var_name in diff_hist_data:
                                diff_data = diff_hist_data[var_name]
                                diff_counts = np.array(diff_data["counts"])
                                diff_bin_edges = np.array(diff_data["bin_edges"])

                                if len(diff_counts) > 0 and np.any(
                                    np.abs(diff_counts) > 1e-6
                                ):
                                    plot_color = size_to_color.get(data_size, colors[0])
                                    plot_linestyle = model_to_linestyle.get(
                                        model_name, "-"
                                    )

                                    ax_right.stairs(
                                        diff_counts,
                                        diff_bin_edges,
                                        fill=False,
                                        color=plot_color,
                                        linewidth=LINE_WIDTHS["thick"],
                                        linestyle=plot_linestyle,
                                        alpha=0.8,
                                    )
                                    has_data_right = True

                # Add reference line at y=0 for differences
                if has_data_right:
                    ax_right.axhline(
                        y=0,
                        color="black",
                        linestyle="-",
                        alpha=0.3,
                        linewidth=LINE_WIDTHS["normal"],
                    )

                # Set subplot titles
                display_name = var_name
                if physlite_plot_labels and var_name in physlite_plot_labels:
                    display_name = physlite_plot_labels[var_name]
                elif var_name.startswith("MET_Core_AnalysisMETAuxDyn."):
                    if var_name.endswith(".sumet"):
                        display_name = "MET Sum ET"
                    else:
                        display_name = var_name.split(".")[-1]

                if has_data_left or has_data_right:
                    ax_left.set_title(display_name, fontsize=FONT_SIZES["small"])
                    ax_right.set_title(display_name, fontsize=FONT_SIZES["small"])
                else:
                    ax_left.set_title(
                        f"{display_name}\n(No Data)", fontsize=FONT_SIZES["small"]
                    )
                    ax_right.set_title(
                        f"{display_name}\n(No Data)", fontsize=FONT_SIZES["small"]
                    )

                # Format axes
                ax_left.set_yscale("log")
                ax_left.grid(True, alpha=0.3, which="both")
                ax_left.tick_params(
                    axis="both", which="major", labelsize=FONT_SIZES["tiny"]
                )

                ax_right.set_yscale("linear")
                ax_right.grid(True, alpha=0.3, which="both")
                ax_right.tick_params(
                    axis="both", which="major", labelsize=FONT_SIZES["tiny"]
                )

            # Create shared legend
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

            # Add zero reference line for differences
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=LINE_WIDTHS["normal"],
                    alpha=0.3,
                    label="Zero Reference",
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
                ncol=min(len(legend_elements), 6),
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

            # Set overall title
            fig.suptitle(
                "Label Distribution Comparison & Differences",
                fontsize=FONT_SIZES["xlarge"],
                y=0.98,
            )

            # Adjust layout and save
            plt.subplots_adjust(top=0.88, bottom=0.12, left=0.05, right=0.95)

            # Save the plot
            plot_path = evaluation_dir / "label_distribution_combined_analysis.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            self.logger.info(f"Combined label distribution plot saved to: {plot_path}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to create combined label distribution plot: {str(e)}"
            )
            return False
