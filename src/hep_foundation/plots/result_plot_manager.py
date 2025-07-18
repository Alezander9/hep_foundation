import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.lines import Line2D

from hep_foundation.config.logging_config import get_logger
from hep_foundation.plots import plot_utils


class ResultPlotManager:
    """
    Manager for creating result comparison plots including label distributions
    and training history comparisons.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def create_label_histogram_data(
        self,
        labels: np.ndarray,
        num_bins: int = 50,
        sample_size: Optional[int] = None,
        label_name: str = "labels",
        predefined_bin_edges: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Create histogram data for labels similar to dataset_manager's histogram functions.

        Args:
            labels: Array of label values to histogram
            num_bins: Number of histogram bins (ignored if predefined_bin_edges provided)
            sample_size: Optional number of samples to use (random sampling if provided)
            label_name: Name for the label in the histogram data
            predefined_bin_edges: Optional predefined bin edges for consistent binning

        Returns:
            Dictionary containing histogram data in the same format as dataset_manager
        """
        try:
            # Sample if requested
            if sample_size and len(labels) > sample_size:
                indices = np.random.choice(len(labels), size=sample_size, replace=False)
                labels_to_hist = labels[indices]
            else:
                labels_to_hist = labels

            # Create histogram with predefined or auto-generated bins
            if predefined_bin_edges is not None:
                # Use predefined bin edges for consistent binning
                counts, bin_edges = np.histogram(
                    labels_to_hist, bins=predefined_bin_edges, density=True
                )
                self.logger.info(
                    f"Used predefined bin edges for {label_name} histogram"
                )
            else:
                # Auto-generate bin edges
                counts, bin_edges = np.histogram(
                    labels_to_hist, bins=num_bins, density=True
                )
                self.logger.info(f"Generated new bin edges for {label_name} histogram")

            # Create histogram data structure
            histogram_data = {
                label_name: {
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                },
                "_metadata": {
                    "total_samples": len(labels_to_hist),
                    "original_total": len(labels),
                    "num_bins": len(bin_edges) - 1,
                    "label_name": label_name,
                    "used_predefined_bins": predefined_bin_edges is not None,
                },
            }

            return histogram_data

        except Exception as e:
            self.logger.error(f"Failed to create label histogram data: {str(e)}")
            return {}

    def save_label_histogram_data(
        self,
        histogram_data: dict,
        output_path: Path,
        metadata_suffix: str = "",
    ):
        """
        Save label histogram data to JSON file.

        Args:
            histogram_data: Histogram data dictionary
            output_path: Path to save the JSON file
            metadata_suffix: Optional suffix to add to metadata
        """
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Add any additional metadata
            if metadata_suffix:
                histogram_data["_metadata"]["description"] = metadata_suffix

            # Save to JSON
            with open(output_path, "w") as f:
                json.dump(histogram_data, f, indent=2)

            self.logger.info(f"Saved label histogram data to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save label histogram data: {str(e)}")

    def save_bin_edges_metadata(
        self,
        histogram_data: dict,
        metadata_path: Path,
    ):
        """
        Save bin edges metadata from histogram data for coordinated binning.

        Args:
            histogram_data: Histogram data dictionary containing bin edges
            metadata_path: Path to save the bin edges metadata JSON file
        """
        try:
            # Ensure directory exists
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract bin edges from histogram data
            bin_edges_metadata = {}
            for key, data in histogram_data.items():
                if (
                    not key.startswith("_")
                    and isinstance(data, dict)
                    and "bin_edges" in data
                ):
                    bin_edges_metadata[key] = data["bin_edges"]

            if bin_edges_metadata:
                # Save to JSON
                with open(metadata_path, "w") as f:
                    json.dump(bin_edges_metadata, f, indent=2)

                self.logger.info(f"Saved bin edges metadata to: {metadata_path}")
            else:
                self.logger.warning("No bin edges found in histogram data to save")

        except Exception as e:
            self.logger.error(f"Failed to save bin edges metadata: {str(e)}")

    def load_bin_edges_metadata(
        self,
        metadata_path: Path,
    ) -> Optional[dict]:
        """
        Load bin edges metadata from JSON file for coordinated binning.

        Args:
            metadata_path: Path to the bin edges metadata JSON file

        Returns:
            Dictionary mapping label names to their bin edges, or None if loading fails
        """
        try:
            if not metadata_path.exists():
                self.logger.info(f"Bin edges metadata file not found: {metadata_path}")
                return None

            with open(metadata_path) as f:
                bin_edges_metadata = json.load(f)

            self.logger.info(f"Loaded bin edges metadata from: {metadata_path}")
            return bin_edges_metadata

        except Exception as e:
            self.logger.error(f"Failed to load bin edges metadata: {str(e)}")
            return None

    def extract_labels_from_dataset(
        self,
        dataset: tf.data.Dataset,
        max_samples: int = 5000,
    ) -> np.ndarray:
        """
        Extract labels from a TensorFlow dataset.

        Args:
            dataset: TensorFlow dataset containing (features, labels) pairs
            max_samples: Maximum number of samples to extract

        Returns:
            Numpy array of extracted labels
        """
        try:
            labels_list = []
            samples_collected = 0

            for batch in dataset:
                if isinstance(batch, tuple) and len(batch) == 2:
                    _, labels_batch = batch

                    # Handle different types of labels_batch (tensor, tuple, list, etc.)
                    try:
                        if hasattr(labels_batch, "numpy"):
                            # TensorFlow tensor
                            labels_np = labels_batch.numpy()
                        elif isinstance(labels_batch, (tuple, list)):
                            # Tuple or list - convert to numpy
                            labels_np = np.array(labels_batch)
                        else:
                            # Already numpy array or other type
                            labels_np = np.array(labels_batch)

                        # Flatten if needed
                        if labels_np.ndim > 1:
                            labels_np = labels_np.flatten()

                        labels_list.extend(labels_np)
                        samples_collected += len(labels_np)

                        if samples_collected >= max_samples:
                            break

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to process batch labels: {str(e)}, type: {type(labels_batch)}"
                        )
                        continue

            # Convert to numpy array and limit to max_samples
            labels_array = np.array(labels_list[:max_samples])
            self.logger.info(f"Extracted {len(labels_array)} labels from dataset")

            return labels_array

        except Exception as e:
            self.logger.error(f"Failed to extract labels from dataset: {str(e)}")
            return np.array([])

    def create_label_distribution_comparison_plot(
        self,
        regression_dir: Path,
        data_sizes: list,
    ):
        """
        Create comparison plots for label distributions across data sizes and models.

        Args:
            regression_dir: Directory containing label distribution data
            data_sizes: List of data sizes used in evaluation
        """
        try:
            plot_utils.set_science_style(use_tex=False)

            label_distributions_dir = regression_dir / "label_distributions"

            # Load actual test labels histogram
            actual_labels_path = (
                label_distributions_dir / "actual_test_labels_hist.json"
            )
            if not actual_labels_path.exists():
                self.logger.warning(
                    f"Actual labels histogram not found: {actual_labels_path}"
                )
                return

            with open(actual_labels_path) as f:
                actual_hist_data = json.load(f)

            # Get the label name (first non-metadata key)
            label_name = next(
                (key for key in actual_hist_data if not key.startswith("_")), None
            )

            if not label_name:
                self.logger.error("No label data found in actual labels histogram")
                return

            actual_data = actual_hist_data[label_name]
            actual_bin_edges = np.array(actual_data["bin_edges"])
            actual_counts = np.array(actual_data["counts"])

            # Debug: Print actual data info
            self.logger.info(f"DEBUG - Label name: {label_name}")
            self.logger.info(
                f"DEBUG - Actual bin edges shape: {actual_bin_edges.shape}, range: [{actual_bin_edges[0]:.3f}, {actual_bin_edges[-1]:.3f}]"
            )
            self.logger.info(
                f"DEBUG - Actual counts shape: {actual_counts.shape}, non-zero bins: {np.sum(actual_counts > 0)}"
            )
            self.logger.info(
                f"DEBUG - Actual counts range: [{np.min(actual_counts):.6f}, {np.max(actual_counts):.6f}]"
            )

            # Create the plot
            fig, ax = plt.subplots(
                figsize=plot_utils.get_figure_size("wide", ratio=1.0)
            )

            model_names = ["From_Scratch", "Fine_Tuned", "Fixed_Encoder"]

            # Create color mapping: one color per dataset size (consistent with training history plots)
            colors = plot_utils.get_color_cycle("high_contrast", n=len(data_sizes))
            size_to_color = {
                size: colors[i] for i, size in enumerate(sorted(data_sizes))
            }

            # Create line style mapping: one line style per model type (consistent with training history plots)
            model_to_linestyle = {
                "From_Scratch": plot_utils.MODEL_LINE_STYLES.get("from_scratch", "-"),
                "Fine_Tuned": plot_utils.MODEL_LINE_STYLES.get("fine_tuned", "--"),
                "Fixed_Encoder": plot_utils.MODEL_LINE_STYLES.get("fixed_encoder", ":"),
            }

            # Debug: Track what we're plotting
            plots_added = 0

            # Plot actual labels as a gray, filled histogram
            self.logger.info(
                f"DEBUG - Plotting actual labels histogram with {len(actual_counts)} bins"
            )
            ax.stairs(
                actual_counts,
                actual_bin_edges,
                fill=True,
                color="gray",
                alpha=0.6,
                linewidth=plot_utils.LINE_WIDTHS["normal"],
                label="Actual Test Labels",
            )
            plots_added += 1

            # Plot predictions for each model and data size
            for model_name in model_names:
                for data_size in data_sizes:
                    data_size_label = (
                        f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                    )

                    pred_hist_path = (
                        label_distributions_dir
                        / f"{model_name}_{data_size_label}_predictions_hist.json"
                    )

                    if pred_hist_path.exists():
                        self.logger.debug(f"Found prediction file: {pred_hist_path}")
                        with open(pred_hist_path) as f:
                            pred_hist_data = json.load(f)

                        if label_name in pred_hist_data:
                            pred_data = pred_hist_data[label_name]
                            pred_counts = np.array(pred_data["counts"])
                            pred_bin_edges = np.array(pred_data["bin_edges"])

                            # Debug: Print prediction data info
                            self.logger.info(
                                f"Processing {model_name}_{data_size_label}:"
                            )
                            self.logger.info(
                                f"  Pred counts shape: {pred_counts.shape}, non-zero bins: {np.sum(pred_counts > 0)}"
                            )
                            self.logger.info(
                                f"  Pred counts range: [{np.min(pred_counts):.6f}, {np.max(pred_counts):.6f}]"
                            )
                            self.logger.info(
                                f"  Pred bin edges range: [{pred_bin_edges[0]:.3f}, {pred_bin_edges[-1]:.3f}]"
                            )

                            # Check if bin edges are consistent (should be since we now use coordinated binning)
                            if len(pred_bin_edges) != len(actual_bin_edges):
                                self.logger.warning(
                                    f"Bin edge length mismatch for {model_name}_{data_size_label}: {len(pred_bin_edges)} vs {len(actual_bin_edges)}"
                                )
                                continue

                            if not np.allclose(
                                pred_bin_edges, actual_bin_edges, rtol=1e-6
                            ):
                                self.logger.warning(
                                    f"Bin edges not matching for {model_name}_{data_size_label}, may indicate inconsistent binning"
                                )

                            # Use predicted counts directly (no rebinning needed with coordinated binning)
                            plot_counts = pred_counts

                            # Debug: Print final plot data info
                            self.logger.info(
                                f"  Plot counts shape: {plot_counts.shape}, non-zero bins: {np.sum(plot_counts > 0)}"
                            )
                            self.logger.info(
                                f"  Plot counts range: [{np.min(plot_counts):.6f}, {np.max(plot_counts):.6f}]"
                            )

                            # Only plot if there are non-zero values
                            if np.sum(plot_counts > 0) > 0:
                                # Get color based on dataset size and line style based on model type
                                plot_color = size_to_color.get(data_size, colors[0])
                                plot_linestyle = model_to_linestyle.get(model_name, "-")

                                # Plot as an outline using actual bin edges
                                self.logger.info(
                                    f"  Plotting with color={plot_color}, linestyle={plot_linestyle}"
                                )
                                ax.stairs(
                                    plot_counts,
                                    actual_bin_edges,
                                    fill=False,
                                    color=plot_color,
                                    linewidth=plot_utils.LINE_WIDTHS["thick"],
                                    linestyle=plot_linestyle,
                                    alpha=0.8,  # Use consistent alpha for better visibility
                                )
                                plots_added += 1
                            else:
                                self.logger.info(
                                    f"  Skipping plot for {model_name}_{data_size_label} - no non-zero values"
                                )
                        else:
                            self.logger.info(
                                f"  Label '{label_name}' not found in prediction data"
                            )
                    else:
                        self.logger.info(f"Missing prediction file: {pred_hist_path}")

            # Create dataset size legend elements
            size_legend_elements = []
            for data_size in sorted(data_sizes):
                data_size_label = (
                    f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                )
                color = size_to_color.get(data_size, colors[0])
                size_legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linewidth=plot_utils.LINE_WIDTHS["thick"],
                        linestyle="-",
                        label=f"{data_size_label} events",
                    )
                )

            # Create model type legend elements
            model_legend_elements = []
            for model_name in model_names:
                linestyle = model_to_linestyle.get(model_name, "-")
                model_legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="gray",
                        linewidth=plot_utils.LINE_WIDTHS["thick"],
                        linestyle=linestyle,
                        label=model_name.replace("_", " "),
                    )
                )

            # Combine legend elements with section headers (same structure as training history)
            legend_elements = []

            # Add actual test labels first (use Line2D for consistency with other elements)
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    linewidth=plot_utils.LINE_WIDTHS["thick"]
                    * 3,  # Extra thick for visibility
                    linestyle="-",
                    alpha=0.8,
                    label="Actual Test Labels",
                )
            )

            # Add dataset size section
            if size_legend_elements:
                legend_elements.append(
                    Line2D([0], [0], color="none", label="Dataset Sizes:")
                )
                legend_elements.extend(size_legend_elements)

            # Add model type section
            if model_legend_elements:
                legend_elements.append(
                    Line2D([0], [0], color="none", label="Model Types:")
                )
                legend_elements.extend(model_legend_elements)

            # Create the legend
            legend = ax.legend(
                handles=legend_elements,
                fontsize=plot_utils.FONT_SIZES["normal"],
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

            ax.set_xlabel("Label Values", fontsize=plot_utils.FONT_SIZES["large"])
            ax.set_ylabel("Density", fontsize=plot_utils.FONT_SIZES["large"])
            ax.set_title(
                "Label Distribution Comparison: Actual vs. Predicted",
                fontsize=plot_utils.FONT_SIZES["xlarge"],
            )
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

            # Debug: Final plot info
            y_min, y_max = ax.get_ylim()
            self.logger.debug("Final plot info:")
            self.logger.debug(f"  Total plots added: {plots_added}")
            self.logger.debug(f"  Y-axis range: [{y_min:.6f}, {y_max:.6f}]")
            self.logger.debug(f"  Number of artists: {len(ax.get_children())}")

            # Save plot
            plot_path = regression_dir / "label_distribution_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(
                f"Saved label distribution comparison plot to: {plot_path}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to create label distribution comparison plot: {str(e)}"
            )

    def create_training_history_comparison_plot(
        self,
        training_history_json_paths: Union[list[Union[str, Path]], Union[str, Path]],
        output_plot_path: Union[str, Path],
        legend_labels: Optional[list[str]] = None,
        title_prefix: str = "Training History",
        metrics_to_plot: Optional[list[str]] = None,
        validation_only: bool = False,
        handle_outliers: bool = True,
        outlier_percentile: float = 95.0,
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
        plot_utils.set_science_style(use_tex=False)

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
                1, 2, figsize=plot_utils.get_figure_size("double", ratio=2.0)
            )
            axes = [ax_full, ax_cropped]
            panel_titles = [
                "Full Range (All Data)",
                f"Cropped View ({outlier_percentile:.0f}th Percentile)",
            ]
        else:
            fig, ax = plt.subplots(
                figsize=plot_utils.get_figure_size("single", ratio=1.2)
            )
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
        colors = plot_utils.get_color_cycle("high_contrast", n=len(dataset_sizes))
        size_to_color = {size: colors[i] for i, size in enumerate(dataset_sizes)}

        # Create line style mapping: one line style per model type
        model_to_linestyle = plot_utils.MODEL_LINE_STYLES

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
                                linewidth=plot_utils.LINE_WIDTHS["thick"],
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
                                linewidth=plot_utils.LINE_WIDTHS["thick"],
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
                                linewidth=plot_utils.LINE_WIDTHS["thick"],
                                linestyle=base_linestyle,
                                label=line_label,
                            )

        # Format each axis
        for ax_idx, current_ax in enumerate(axes):
            current_ax.set_xlabel("Epoch", fontsize=plot_utils.FONT_SIZES["large"])
            current_ax.set_ylabel(
                "Loss (log scale)", fontsize=plot_utils.FONT_SIZES["large"]
            )
            current_ax.set_yscale("log")
            current_ax.grid(True, alpha=0.3, which="both")

            # Set panel-specific title
            if len(axes) > 1:
                if ax_idx == 0:
                    current_ax.set_title(
                        f"{title_prefix}\n{panel_titles[ax_idx]}",
                        fontsize=plot_utils.FONT_SIZES["large"],
                    )
                else:
                    current_ax.set_title(
                        panel_titles[ax_idx], fontsize=plot_utils.FONT_SIZES["large"]
                    )
                    # Set y-limit for cropped view
                    current_ax.set_ylim(top=outlier_threshold)
            else:
                current_ax.set_title(
                    title_prefix, fontsize=plot_utils.FONT_SIZES["xlarge"]
                )

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
                        linewidth=plot_utils.LINE_WIDTHS["thick"],
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
                        linewidth=plot_utils.LINE_WIDTHS["thick"],
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
            fontsize=plot_utils.FONT_SIZES["normal"],
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
