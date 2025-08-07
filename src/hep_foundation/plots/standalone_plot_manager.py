"""
Standalone Plot Manager for HEP Foundation Pipeline.

This module provides comprehensive plotting functionality for standalone DNN regression tasks,
including training history, prediction analysis, and error analysis visualizations.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from hep_foundation.config.logging_config import get_logger

# Import plot utilities (assuming they exist in the project)
try:
    from hep_foundation.plots.plot_utils import (
        FONT_SIZES,
        LINE_WIDTHS,
        MARKER_SIZES,
        get_color_cycle,
        get_figure_size,
        set_science_style,
    )
except ImportError:
    # Fallback constants if plot_utils not available
    FONT_SIZES = {
        "tiny": 8,
        "small": 10,
        "normal": 12,
        "large": 14,
        "xlarge": 16,
    }
    LINE_WIDTHS = {"thin": 1, "thick": 2, "very_thick": 3}
    MARKER_SIZES = {"tiny": 2, "small": 4, "normal": 6, "large": 8}

    def get_color_cycle(style="default", n_colors=8):
        return plt.cm.tab10(np.linspace(0, 1, n_colors))

    def get_figure_size(size_type="single", ratio=1.0):
        base_width = 8 if size_type == "single" else 12
        return (base_width, base_width * ratio)

    def set_science_style(use_tex=False):
        plt.style.use(
            "seaborn-v0_8"
            if hasattr(plt.style, "available") and "seaborn-v0_8" in plt.style.available
            else "default"
        )


class StandalonePlotManager:
    """
    Plot manager for standalone DNN regression experiments.

    Provides comprehensive visualization capabilities for training monitoring,
    prediction analysis, and error analysis specific to standalone regression tasks.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        set_science_style(use_tex=False)

    def create_training_history_plot(
        self,
        history_data: Dict[str, List[float]],
        output_path: Path,
        title_prefix: str = "Standalone DNN Training",
    ) -> None:
        """
        Create comprehensive training history plot with loss, metrics, and learning rate.

        Args:
            history_data: Dictionary containing training history
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
        """
        try:
            # Extract data
            epochs = list(range(1, len(history_data.get("loss", [])) + 1))
            if not epochs:
                self.logger.warning("No training history data available")
                return

            # Determine number of subplots based on available data
            has_val_data = any(key.startswith("val_") for key in history_data.keys())
            has_lr_data = "lr" in history_data
            n_subplots = 2 + (1 if has_lr_data else 0)

            # Create figure
            fig, axes = plt.subplots(
                n_subplots, 1, figsize=get_figure_size("single", ratio=1.2), sharex=True
            )
            if n_subplots == 1:
                axes = [axes]

            colors = get_color_cycle("high_contrast", 4)

            # Plot 1: Loss
            ax_idx = 0
            if "loss" in history_data:
                axes[ax_idx].plot(
                    epochs,
                    history_data["loss"],
                    color=colors[0],
                    linewidth=LINE_WIDTHS["thick"],
                    label="Training Loss",
                    linestyle="-",
                )
                if "val_loss" in history_data:
                    axes[ax_idx].plot(
                        epochs,
                        history_data["val_loss"],
                        color=colors[1],
                        linewidth=LINE_WIDTHS["thick"],
                        label="Validation Loss",
                        linestyle="--",
                    )
                axes[ax_idx].set_ylabel("Loss", fontsize=FONT_SIZES["normal"])
                axes[ax_idx].legend(fontsize=FONT_SIZES["small"])
                axes[ax_idx].grid(True, alpha=0.3)
                axes[ax_idx].set_yscale("log")
                ax_idx += 1

            # Plot 2: Metrics (MSE, MAE)
            metric_names = [
                key
                for key in history_data.keys()
                if key in ["mse", "mae"]
                or key.startswith("val_")
                and key[4:] in ["mse", "mae"]
            ]
            if metric_names:
                for i, metric in enumerate(["mse", "mae"]):
                    if metric in history_data:
                        axes[ax_idx].plot(
                            epochs,
                            history_data[metric],
                            color=colors[i % len(colors)],
                            linewidth=LINE_WIDTHS["thick"],
                            label=f"Training {metric.upper()}",
                            linestyle="-",
                        )
                    val_metric = f"val_{metric}"
                    if val_metric in history_data:
                        axes[ax_idx].plot(
                            epochs,
                            history_data[val_metric],
                            color=colors[i % len(colors)],
                            linewidth=LINE_WIDTHS["thick"],
                            label=f"Validation {metric.upper()}",
                            linestyle="--",
                        )

                axes[ax_idx].set_ylabel("Metrics", fontsize=FONT_SIZES["normal"])
                axes[ax_idx].legend(fontsize=FONT_SIZES["small"])
                axes[ax_idx].grid(True, alpha=0.3)
                axes[ax_idx].set_yscale("log")
                ax_idx += 1

            # Plot 3: Learning Rate (if available)
            if has_lr_data and "lr" in history_data:
                axes[ax_idx].plot(
                    epochs,
                    history_data["lr"],
                    color=colors[3],
                    linewidth=LINE_WIDTHS["thick"],
                    label="Learning Rate",
                    linestyle="-",
                )
                axes[ax_idx].set_ylabel("Learning Rate", fontsize=FONT_SIZES["normal"])
                axes[ax_idx].set_yscale("log")
                axes[ax_idx].legend(fontsize=FONT_SIZES["small"])
                axes[ax_idx].grid(True, alpha=0.3)

            # Set common x-axis label
            axes[-1].set_xlabel("Epoch", fontsize=FONT_SIZES["normal"])

            # Set title
            fig.suptitle(
                f"{title_prefix} - Training History", fontsize=FONT_SIZES["large"]
            )

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Training history plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create training history plot: {e}")

    def create_data_efficiency_plot(
        self,
        results_data: Dict[str, Any],
        output_path: Path,
        title_prefix: str = "Standalone DNN Data Efficiency",
    ) -> None:
        """
        Create data efficiency plot showing model performance vs training data size.

        Args:
            results_data: Dictionary containing evaluation results for different data sizes
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
        """
        try:
            # Extract data sizes and corresponding metrics
            data_sizes = []
            test_losses = []

            for data_size, metrics in results_data.items():
                if isinstance(data_size, (int, str)) and str(data_size).isdigit():
                    data_sizes.append(int(data_size))
                    test_losses.append(
                        metrics.get("test_loss", metrics.get("test_mse", 0.0))
                    )

            if not data_sizes:
                self.logger.warning("No data efficiency results available")
                return

            # Sort by data size
            sorted_data = sorted(zip(data_sizes, test_losses))
            data_sizes, test_losses = zip(*sorted_data)

            # Create plot
            fig, ax = plt.subplots(figsize=get_figure_size("single", ratio=0.8))

            colors = get_color_cycle("high_contrast", 1)
            ax.plot(
                data_sizes,
                test_losses,
                color=colors[0],
                linewidth=LINE_WIDTHS["thick"],
                marker="o",
                markersize=MARKER_SIZES["normal"],
                label="Standalone DNN",
            )

            # Formatting
            ax.set_xlabel("Training Data Size", fontsize=FONT_SIZES["normal"])
            ax.set_ylabel("Test Loss (MSE)", fontsize=FONT_SIZES["normal"])
            ax.set_title(f"{title_prefix}", fontsize=FONT_SIZES["large"])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=FONT_SIZES["small"])

            # Log scale for better visualization
            ax.set_xscale("log")
            ax.set_yscale("log")

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Data efficiency plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create data efficiency plot: {e}")

    def create_prediction_quality_plot(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_path: Path,
        title_prefix: str = "Prediction Quality Analysis",
        sample_size: int = 1000,
    ) -> None:
        """
        Create comprehensive prediction quality analysis plot.

        Args:
            predictions: Model predictions
            targets: True target values
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            sample_size: Number of samples to plot (for performance)
        """
        try:
            # Sample data if too large
            if len(predictions) > sample_size:
                indices = np.random.choice(len(predictions), sample_size, replace=False)
                pred_sample = predictions[indices]
                target_sample = targets[indices]
            else:
                pred_sample = predictions
                target_sample = targets

            # Create 2x2 subplot
            fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", ratio=1.0))
            colors = get_color_cycle("high_contrast", 3)

            # Plot 1: Predictions vs Targets scatter
            axes[0, 0].scatter(
                target_sample,
                pred_sample,
                alpha=0.6,
                s=MARKER_SIZES["small"],
                color=colors[0],
            )
            # Perfect prediction line
            min_val = min(np.min(target_sample), np.min(pred_sample))
            max_val = max(np.max(target_sample), np.max(pred_sample))
            axes[0, 0].plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=LINE_WIDTHS["thick"],
                label="Perfect Prediction",
            )

            axes[0, 0].set_xlabel("True Values", fontsize=FONT_SIZES["small"])
            axes[0, 0].set_ylabel("Predictions", fontsize=FONT_SIZES["small"])
            axes[0, 0].set_title(
                "Predictions vs True Values", fontsize=FONT_SIZES["normal"]
            )
            axes[0, 0].legend(fontsize=FONT_SIZES["tiny"])
            axes[0, 0].grid(True, alpha=0.3)

            # Calculate R²
            r2 = stats.pearsonr(target_sample.flatten(), pred_sample.flatten())[0] ** 2
            axes[0, 0].text(
                0.05,
                0.95,
                f"R² = {r2:.3f}",
                transform=axes[0, 0].transAxes,
                fontsize=FONT_SIZES["small"],
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Plot 2: Residuals histogram
            residuals = pred_sample - target_sample
            axes[0, 1].hist(
                residuals.flatten(), bins=50, alpha=0.7, color=colors[1], density=True
            )
            axes[0, 1].axvline(
                0, color="r", linestyle="--", linewidth=LINE_WIDTHS["thick"]
            )
            axes[0, 1].set_xlabel(
                "Residuals (Pred - True)", fontsize=FONT_SIZES["small"]
            )
            axes[0, 1].set_ylabel("Density", fontsize=FONT_SIZES["small"])
            axes[0, 1].set_title(
                "Residuals Distribution", fontsize=FONT_SIZES["normal"]
            )
            axes[0, 1].grid(True, alpha=0.3)

            # Add residual statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            axes[0, 1].text(
                0.05,
                0.95,
                f"μ = {mean_residual:.3e}\nσ = {std_residual:.3e}",
                transform=axes[0, 1].transAxes,
                fontsize=FONT_SIZES["tiny"],
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Plot 3: Residuals vs Predictions
            axes[1, 0].scatter(
                pred_sample.flatten(),
                residuals.flatten(),
                alpha=0.6,
                s=MARKER_SIZES["small"],
                color=colors[2],
            )
            axes[1, 0].axhline(
                0, color="r", linestyle="--", linewidth=LINE_WIDTHS["thick"]
            )
            axes[1, 0].set_xlabel("Predictions", fontsize=FONT_SIZES["small"])
            axes[1, 0].set_ylabel("Residuals", fontsize=FONT_SIZES["small"])
            axes[1, 0].set_title(
                "Residuals vs Predictions", fontsize=FONT_SIZES["normal"]
            )
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Q-Q plot for residual normality
            from scipy import stats

            stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title(
                "Q-Q Plot (Residual Normality)", fontsize=FONT_SIZES["normal"]
            )
            axes[1, 1].grid(True, alpha=0.3)

            # Overall title
            fig.suptitle(f"{title_prefix}", fontsize=FONT_SIZES["large"])

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Prediction quality plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create prediction quality plot: {e}")

    def create_error_analysis_plot(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        output_path: Path,
        title_prefix: str = "Error Analysis",
        n_bins: int = 50,
    ) -> None:
        """
        Create detailed error analysis plots.

        Args:
            predictions: Model predictions
            targets: True target values
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            n_bins: Number of bins for histograms
        """
        try:
            # Calculate errors
            absolute_errors = np.abs(predictions - targets)
            relative_errors = np.abs(predictions - targets) / (np.abs(targets) + 1e-8)
            squared_errors = (predictions - targets) ** 2

            # Create 2x2 subplot
            fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", ratio=1.0))
            colors = get_color_cycle("high_contrast", 4)

            # Plot 1: Absolute errors histogram
            axes[0, 0].hist(
                absolute_errors.flatten(),
                bins=n_bins,
                alpha=0.7,
                color=colors[0],
                density=True,
            )
            axes[0, 0].set_xlabel("Absolute Error", fontsize=FONT_SIZES["small"])
            axes[0, 0].set_ylabel("Density", fontsize=FONT_SIZES["small"])
            axes[0, 0].set_title(
                "Absolute Error Distribution", fontsize=FONT_SIZES["normal"]
            )
            axes[0, 0].grid(True, alpha=0.3)

            # Add statistics
            mae = np.mean(absolute_errors)
            axes[0, 0].axvline(
                mae,
                color="r",
                linestyle="--",
                linewidth=LINE_WIDTHS["thick"],
                label=f"MAE = {mae:.3e}",
            )
            axes[0, 0].legend(fontsize=FONT_SIZES["tiny"])

            # Plot 2: Relative errors histogram
            # Clip extreme relative errors for better visualization
            rel_errors_clipped = np.clip(
                relative_errors, 0, np.percentile(relative_errors, 95)
            )
            axes[0, 1].hist(
                rel_errors_clipped.flatten(),
                bins=n_bins,
                alpha=0.7,
                color=colors[1],
                density=True,
            )
            axes[0, 1].set_xlabel("Relative Error", fontsize=FONT_SIZES["small"])
            axes[0, 1].set_ylabel("Density", fontsize=FONT_SIZES["small"])
            axes[0, 1].set_title(
                "Relative Error Distribution", fontsize=FONT_SIZES["normal"]
            )
            axes[0, 1].grid(True, alpha=0.3)

            # Add statistics
            mape = np.mean(relative_errors) * 100
            axes[0, 1].axvline(
                np.mean(rel_errors_clipped),
                color="r",
                linestyle="--",
                linewidth=LINE_WIDTHS["thick"],
                label=f"MAPE = {mape:.1f}%",
            )
            axes[0, 1].legend(fontsize=FONT_SIZES["tiny"])

            # Plot 3: Error vs True Values
            axes[1, 0].scatter(
                targets.flatten(),
                absolute_errors.flatten(),
                alpha=0.5,
                s=MARKER_SIZES["tiny"],
                color=colors[2],
            )
            axes[1, 0].set_xlabel("True Values", fontsize=FONT_SIZES["small"])
            axes[1, 0].set_ylabel("Absolute Error", fontsize=FONT_SIZES["small"])
            axes[1, 0].set_title("Error vs True Values", fontsize=FONT_SIZES["normal"])
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Cumulative error distribution
            sorted_errors = np.sort(absolute_errors.flatten())
            cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            axes[1, 1].plot(
                sorted_errors,
                cumulative,
                color=colors[3],
                linewidth=LINE_WIDTHS["thick"],
            )
            axes[1, 1].set_xlabel("Absolute Error", fontsize=FONT_SIZES["small"])
            axes[1, 1].set_ylabel(
                "Cumulative Probability", fontsize=FONT_SIZES["small"]
            )
            axes[1, 1].set_title(
                "Cumulative Error Distribution", fontsize=FONT_SIZES["normal"]
            )
            axes[1, 1].grid(True, alpha=0.3)

            # Add percentile lines
            for percentile in [50, 90, 95]:
                error_val = np.percentile(sorted_errors, percentile)
                axes[1, 1].axvline(
                    error_val,
                    linestyle="--",
                    alpha=0.7,
                    label=f"{percentile}th percentile",
                )
            axes[1, 1].legend(fontsize=FONT_SIZES["tiny"])

            # Overall title
            fig.suptitle(f"{title_prefix}", fontsize=FONT_SIZES["large"])

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Error analysis plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create error analysis plot: {e}")

    def create_multi_size_comparison_plot(
        self,
        multi_size_data: Dict[int, Dict[str, np.ndarray]],
        output_path: Path,
        title_prefix: str = "Multi-Size Prediction Comparison",
        max_samples_per_plot: int = 500,
    ) -> None:
        """
        Create comparison plot showing prediction quality across different data sizes.

        Args:
            multi_size_data: Dictionary with data_size -> {predictions, targets}
            output_path: Path to save the plot
            title_prefix: Prefix for the plot title
            max_samples_per_plot: Maximum samples per subplot
        """
        try:
            data_sizes = sorted(multi_size_data.keys())
            n_sizes = len(data_sizes)

            if n_sizes == 0:
                self.logger.warning("No multi-size data available")
                return

            # Create subplot grid
            n_cols = min(n_sizes, 4)
            n_rows = (n_sizes + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=get_figure_size("double", ratio=0.8)
            )
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            colors = get_color_cycle("high_contrast", n_sizes)

            for i, data_size in enumerate(data_sizes):
                if i >= len(axes):
                    break

                data = multi_size_data[data_size]
                predictions = data["predictions"]
                targets = data["targets"]

                # Sample data for performance
                if len(predictions) > max_samples_per_plot:
                    indices = np.random.choice(
                        len(predictions), max_samples_per_plot, replace=False
                    )
                    pred_sample = predictions[indices]
                    target_sample = targets[indices]
                else:
                    pred_sample = predictions
                    target_sample = targets

                # Scatter plot
                axes[i].scatter(
                    target_sample,
                    pred_sample,
                    alpha=0.6,
                    s=MARKER_SIZES["tiny"],
                    color=colors[i % len(colors)],
                )

                # Perfect prediction line
                min_val = min(np.min(target_sample), np.min(pred_sample))
                max_val = max(np.max(target_sample), np.max(pred_sample))
                axes[i].plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "r--",
                    linewidth=LINE_WIDTHS["thin"],
                    alpha=0.7,
                )

                # Calculate R²
                r2 = (
                    stats.pearsonr(target_sample.flatten(), pred_sample.flatten())[0]
                    ** 2
                )

                # Format data size label
                size_label = (
                    f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                )
                axes[i].set_title(
                    f"{size_label} events (R²={r2:.3f})", fontsize=FONT_SIZES["small"]
                )
                axes[i].grid(True, alpha=0.3)

                if i >= (n_rows - 1) * n_cols:  # Bottom row
                    axes[i].set_xlabel("True Values", fontsize=FONT_SIZES["small"])
                if i % n_cols == 0:  # Left column
                    axes[i].set_ylabel("Predictions", fontsize=FONT_SIZES["small"])

            # Hide unused subplots
            for i in range(n_sizes, len(axes)):
                axes[i].set_visible(False)

            # Overall title
            fig.suptitle(f"{title_prefix}", fontsize=FONT_SIZES["large"])

            # Adjust layout and save
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Multi-size comparison plot saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create multi-size comparison plot: {e}")
