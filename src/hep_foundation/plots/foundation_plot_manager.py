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

    def create_training_history_comparison_plot(
        self,
        training_histories_dir: Path,
        output_path: Path,
        title_prefix: str = "Model Training Comparison",
        validation_only: bool = True,
    ) -> bool:
        """
        Create a combined training history plot comparing different models.

        Args:
            training_histories_dir: Directory containing training history JSON files
            output_path: Path where to save the plot
            title_prefix: Prefix for the plot title
            validation_only: Whether to show only validation loss

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
        max_samples: int = 5000,
        num_bins: int = 50,
    ) -> Optional[dict]:
        """
        Create label distribution analysis for regression tasks.

        Args:
            test_dataset: TensorFlow dataset containing test labels
            label_distributions_dir: Directory to save label distribution data
            max_samples: Maximum number of samples to use for analysis
            num_bins: Number of bins for histogram

        Returns:
            Dict containing bin edges metadata, or None if failed
        """
        try:
            self.logger.info("Creating label distribution analysis...")
            label_distributions_dir.mkdir(parents=True, exist_ok=True)

            # Sample actual test labels
            self.logger.info("Sampling actual test labels for distribution analysis...")
            actual_test_labels = self.plot_manager.extract_labels_from_dataset(
                test_dataset, max_samples=max_samples
            )

            if len(actual_test_labels) == 0:
                self.logger.warning("No test labels extracted for histogram analysis")
                return None

            actual_labels_hist_data = self.plot_manager.create_label_histogram_data(
                actual_test_labels, num_bins=num_bins, label_name="test_labels"
            )

            self.plot_manager.save_label_histogram_data(
                actual_labels_hist_data,
                label_distributions_dir / "actual_test_labels_hist.json",
                "Actual test labels from regression dataset",
            )

            # Save bin edges metadata for coordinated binning
            bin_edges_metadata_path = (
                label_distributions_dir / "label_bin_edges_metadata.json"
            )
            self.plot_manager.save_bin_edges_metadata(
                actual_labels_hist_data, bin_edges_metadata_path
            )

            # Load the bin edges for use in prediction histograms
            bin_edges_metadata = self.plot_manager.load_bin_edges_metadata(
                bin_edges_metadata_path
            )

            self.logger.info("Label distribution analysis completed")
            return bin_edges_metadata

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
    ) -> bool:
        """
        Save histogram data for model predictions.

        Args:
            predictions: Array of model predictions
            model_name: Name of the model
            data_size: Size of training data used
            label_distributions_dir: Directory to save histogram data
            bin_edges_metadata: Metadata for coordinated binning
            max_samples: Maximum number of samples to include

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if len(predictions) == 0:
                self.logger.warning(f"No predictions provided for {model_name}")
                return False

            # Limit samples if needed
            predictions_array = np.array(predictions[:max_samples])

            # Use predefined bin edges if available for coordinated binning
            predefined_bin_edges = None
            if bin_edges_metadata and "test_labels" in bin_edges_metadata:
                predefined_bin_edges = np.array(bin_edges_metadata["test_labels"])
                self.logger.info(
                    f"Using coordinated binning for {model_name} predictions"
                )

            predictions_hist_data = self.plot_manager.create_label_histogram_data(
                predictions_array,
                num_bins=50,  # This will be ignored if predefined_bin_edges is provided
                label_name="test_labels",
                predefined_bin_edges=predefined_bin_edges,
            )

            pred_hist_filename = f"{model_name}_predictions_hist.json"
            label_distributions_dir.mkdir(parents=True, exist_ok=True)

            self.plot_manager.save_label_histogram_data(
                predictions_hist_data,
                label_distributions_dir / pred_hist_filename,
                f"Predictions from {model_name} model trained with {data_size} samples",
            )

            self.logger.info(
                f"Saved {len(predictions_array)} predictions for {model_name}"
            )
            return True

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
