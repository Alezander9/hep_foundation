import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from hep_foundation.config.logging_config import get_logger
from hep_foundation.plots.plot_utils import (
    FONT_SIZES,
    LINE_WIDTHS,
    get_color_cycle,
    get_figure_size,
    set_science_style,
)
from hep_foundation.plots.result_plot_manager import ResultPlotManager


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
        self.plot_manager = ResultPlotManager()

        # Set up plotting style
        set_science_style(use_tex=False)
        self.colors = get_color_cycle("high_contrast")

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
                self.plot_manager.create_training_history_comparison_plot(
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
            actual_test_labels = self.plot_manager.extract_labels_from_dataset(
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

                    var_hist_data = self.plot_manager.create_label_histogram_data(
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
                self.plot_manager.save_label_histogram_data(
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
                    "Expected dictionary result from extract_labels_from_dataset for multiple variables"
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

                        var_hist_data = self.plot_manager.create_label_histogram_data(
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

                self.plot_manager.save_label_histogram_data(
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

    def create_label_distribution_comparison_plot(
        self,
        evaluation_dir: Path,
        data_sizes: list[int],
    ) -> bool:
        """
        Create a comparison plot of label distributions.

        Args:
            evaluation_dir: Directory containing the evaluation results
            data_sizes: List of data sizes used in evaluation

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info("Creating label distribution comparison plot...")
            return self.plot_manager.create_label_distribution_comparison_plot(
                evaluation_dir, data_sizes
            )
        except Exception as e:
            self.logger.error(
                f"Failed to create label distribution comparison plot: {str(e)}"
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

    def log_evaluation_summary(
        self,
        results: dict[str, list[float]],
        evaluation_type: str = "regression",
        signal_key: Optional[str] = None,
    ) -> None:
        """
        Log a summary of evaluation results.

        Args:
            results: Dictionary containing evaluation results
            evaluation_type: Type of evaluation ("regression" or "signal_classification")
            signal_key: Signal key for classification evaluations
        """
        try:
            self.logger.info("=" * 100)
            if evaluation_type == "regression":
                self.logger.info("Regression Evaluation Results Summary")
            elif evaluation_type == "signal_classification":
                self.logger.info("Signal Classification Evaluation Results Summary")
                if signal_key:
                    self.logger.info(f"Signal Dataset: {signal_key}")
            self.logger.info("=" * 100)

            data_sizes = results.get("data_sizes", [])

            for i, data_size in enumerate(data_sizes):
                self.logger.info(f"Training Events: {data_size}")

                if evaluation_type == "regression":
                    self.logger.info(
                        f"  From Scratch:  {results['From_Scratch'][i]:.6f}"
                    )
                    self.logger.info(f"  Fine-Tuned:    {results['Fine_Tuned'][i]:.6f}")
                    self.logger.info(
                        f"  Fixed Encoder: {results['Fixed_Encoder'][i]:.6f}"
                    )

                    # Calculate improvement ratios
                    if results["From_Scratch"][i] > 0:
                        ft_improvement = (
                            (results["From_Scratch"][i] - results["Fine_Tuned"][i])
                            / results["From_Scratch"][i]
                            * 100
                        )
                        fx_improvement = (
                            (results["From_Scratch"][i] - results["Fixed_Encoder"][i])
                            / results["From_Scratch"][i]
                            * 100
                        )
                        self.logger.info(
                            f"  Fine-Tuned improvement: {ft_improvement:.1f}%"
                        )
                        self.logger.info(f"  Fixed improvement: {fx_improvement:.1f}%")

                elif evaluation_type == "signal_classification":
                    self.logger.info(
                        f"  From Scratch:  Loss: {results['From_Scratch_loss'][i]:.6f}, "
                        f"Accuracy: {results['From_Scratch_accuracy'][i]:.6f}"
                    )
                    self.logger.info(
                        f"  Fine-Tuned:    Loss: {results['Fine_Tuned_loss'][i]:.6f}, "
                        f"Accuracy: {results['Fine_Tuned_accuracy'][i]:.6f}"
                    )
                    self.logger.info(
                        f"  Fixed Encoder: Loss: {results['Fixed_Encoder_loss'][i]:.6f}, "
                        f"Accuracy: {results['Fixed_Encoder_accuracy'][i]:.6f}"
                    )

                    # Calculate improvement ratios for accuracy
                    scratch_acc = results["From_Scratch_accuracy"][i]
                    if scratch_acc < 1.0:  # Avoid division issues
                        ft_acc_improvement = (
                            (results["Fine_Tuned_accuracy"][i] - scratch_acc)
                            / (1.0 - scratch_acc)
                            * 100
                        )
                        fx_acc_improvement = (
                            (results["Fixed_Encoder_accuracy"][i] - scratch_acc)
                            / (1.0 - scratch_acc)
                            * 100
                        )
                        self.logger.info(
                            f"  Fine-Tuned accuracy improvement: {ft_acc_improvement:.1f}%"
                        )
                        self.logger.info(
                            f"  Fixed accuracy improvement: {fx_acc_improvement:.1f}%"
                        )

                self.logger.info("")

        except Exception as e:
            self.logger.error(f"Failed to log evaluation summary: {str(e)}")
