import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import TaskConfig
from hep_foundation.config.training_config import TrainingConfig
from hep_foundation.data.dataset_manager import DatasetManager
from hep_foundation.models.base_model import BaseModel
from hep_foundation.models.variational_autoencoder import (
    VAEConfig,
    VariationalAutoEncoder,
)
from hep_foundation.plots.dataset_visualizer import (
    create_combined_roc_curves_plot_from_json,
    create_combined_two_panel_loss_plot_from_json,
)


class AnomalyDetectionEvaluator:
    """Class for evaluating trained models with various tests"""

    def __init__(
        self,
        model: BaseModel = None,
        test_dataset: tf.data.Dataset = None,
        signal_datasets: Optional[dict[str, tf.data.Dataset]] = None,
        experiment_id: str = None,
        base_path: Path = Path("experiments"),
        processed_datasets_dir: str = None,
    ):
        """
        Initialize the model tester

        Args:
            model: Trained model to evaluate (optional, can be loaded later)
            test_dataset: Dataset of background events for testing (optional, can be loaded later)
            signal_datasets: Dictionary of signal datasets for comparison (optional, can be loaded later)
            experiment_id: ID of the experiment (e.g. '001_vae_test')
            base_path: Base path where experiments are stored
            processed_datasets_dir: Directory for processed datasets (needed for loading datasets)
        """
        self.model = model
        self.test_dataset = test_dataset
        self.signal_datasets = signal_datasets or {}
        self.processed_datasets_dir = (
            Path(processed_datasets_dir) if processed_datasets_dir else None
        )

        # Setup paths
        self.base_path = Path(base_path)
        if experiment_id is None:
            raise ValueError("experiment_id must be provided")

        self.experiment_path = self.base_path / experiment_id
        self.testing_path = self.experiment_path / "testing"
        self.testing_path.mkdir(parents=True, exist_ok=True)

        # Setup self.logger
        self.logger = get_logger(__name__)

        # Load experiment info from the new file structure (only if it exists)
        self.experiment_info_path = self.experiment_path / "_experiment_info.json"
        self.experiment_info = None
        if self.experiment_info_path.exists():
            with open(self.experiment_info_path) as f:
                self.experiment_info = json.load(f)
        else:
            self.logger.warning(
                f"No experiment info found at {self.experiment_info_path} - will be created if needed"
            )

    def _calculate_losses(
        self, dataset: tf.data.Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate reconstruction (MSE) and KL losses for a dataset

        Args:
            dataset: Dataset to evaluate

        Returns:
            Tuple of (reconstruction_losses, kl_losses) arrays
        """
        if not isinstance(self.model, VariationalAutoEncoder):
            raise ValueError("Model must be a VariationalAutoEncoder")

        reconstruction_losses = []
        kl_losses = []

        # Log dataset info
        total_batches = 0
        total_events = 0

        self.logger.info("Calculating losses for dataset...")
        for batch in dataset:
            total_batches += 1

            # Handle both (features, labels) and features-only cases
            if isinstance(batch, tuple):
                features, _ = batch  # We only need features for reconstruction
                total_events += features.shape[0]
                # Ensure input batch is float32
                features = tf.cast(features, tf.float32)
            else:
                features = batch
                total_events += features.shape[0]
                # Ensure input batch is float32
                features = tf.cast(features, tf.float32)

                # Get encoder outputs
            z_mean, z_log_var, z = self.model.encoder(features)

            # Get reconstructions
            reconstructions = self.model.decoder(z)

            # Use static shape to avoid retracing
            input_shape = tf.shape(features)
            flat_inputs = tf.reshape(features, [input_shape[0], -1])
            flat_reconstructions = tf.reshape(reconstructions, [input_shape[0], -1])

            # Ensure both tensors are the same type before subtraction
            flat_inputs = tf.cast(flat_inputs, tf.float32)
            flat_reconstructions = tf.cast(flat_reconstructions, tf.float32)

            # Calculate losses per event (not taking the mean) with bounds for signal data
            recon_losses_batch = tf.reduce_sum(
                tf.square(flat_inputs - flat_reconstructions), axis=1
            ).numpy()

            # Clip reconstruction losses to prevent extreme values from out-of-distribution signal data
            recon_losses_batch = np.clip(recon_losses_batch, 0.0, 1e8)

            # Calculate KL losses using softplus for numerical stability
            # Use softplus instead of exp to prevent overflow
            variance = tf.nn.softplus(z_log_var) + 1e-6

            # Calculate KL divergence with numerical stability
            kl_losses_batch = (
                -0.5
                * tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - variance,
                    axis=1,
                ).numpy()
            )

            # Clip final KL losses to prevent extreme values from signal data
            kl_losses_batch = np.clip(kl_losses_batch, 0.0, 1e6)

            # Check for NaN/Inf in final loss calculations
            invalid_recon = np.sum(~np.isfinite(recon_losses_batch))
            invalid_kl = np.sum(~np.isfinite(kl_losses_batch))

            if invalid_recon > 0 or invalid_kl > 0:
                self.logger.warning(
                    f"Batch {total_batches}: Found extreme loss values (likely out-of-distribution signal data)"
                )
                self.logger.warning(
                    f"  Reconstruction loss - NaN: {np.sum(np.isnan(recon_losses_batch))}, Inf: {np.sum(np.isinf(recon_losses_batch))}"
                )
                self.logger.warning(
                    f"  KL loss - NaN: {np.sum(np.isnan(kl_losses_batch))}, Inf: {np.sum(np.isinf(kl_losses_batch))}"
                )

                # Log valid data ranges for context
                valid_recon = recon_losses_batch[np.isfinite(recon_losses_batch)]
                valid_kl = kl_losses_batch[np.isfinite(kl_losses_batch)]
                if len(valid_recon) > 0:
                    self.logger.warning(
                        f"  Valid reconstruction loss range: [{np.min(valid_recon):.3e}, {np.max(valid_recon):.3e}]"
                    )
                if len(valid_kl) > 0:
                    self.logger.warning(
                        f"  Valid KL loss range: [{np.min(valid_kl):.3e}, {np.max(valid_kl):.3e}]"
                    )

                self.logger.warning(
                    "  → This is expected for signal data that differs significantly from background training data"
                )
                self.logger.warning(
                    "  → Invalid values have been clipped and will be filtered in metrics calculation"
                )

            # Append individual event losses
            reconstruction_losses.extend(recon_losses_batch.tolist())
            kl_losses.extend(kl_losses_batch.tolist())

        self.logger.info("Dataset stats:")
        self.logger.info(f"  Total batches: {total_batches}")
        self.logger.info(f"  Total events: {total_events}")
        self.logger.info(
            f"  Events per batch: {total_events / total_batches if total_batches > 0 else 0:.1f}"
        )

        return np.array(reconstruction_losses), np.array(kl_losses)

    def _calculate_separation_metrics(
        self, background_losses: np.ndarray, signal_losses: np.ndarray, loss_type: str
    ) -> dict:
        """
        Calculate metrics for separation between background and signal

        Args:
            background_losses: Array of losses for background events
            signal_losses: Array of losses for signal events
            loss_type: String identifier for the type of loss

        Returns:
            Dictionary of separation metrics
        """
        # Check for NaN/Inf values and filter them out
        bg_nan_count = np.sum(np.isnan(background_losses))
        bg_inf_count = np.sum(np.isinf(background_losses))
        sig_nan_count = np.sum(np.isnan(signal_losses))
        sig_inf_count = np.sum(np.isinf(signal_losses))

        total_invalid = bg_nan_count + bg_inf_count + sig_nan_count + sig_inf_count

        if total_invalid > 0:
            self.logger.warning(
                f"Found {total_invalid} invalid values in {loss_type} losses!"
            )
            self.logger.warning(
                f"  Background: {bg_nan_count} NaN + {bg_inf_count} Inf = {bg_nan_count + bg_inf_count}"
            )
            self.logger.warning(
                f"  Signal: {sig_nan_count} NaN + {sig_inf_count} Inf = {sig_nan_count + sig_inf_count}"
            )

            # Filter out invalid values
            background_losses_clean = background_losses[np.isfinite(background_losses)]
            signal_losses_clean = signal_losses[np.isfinite(signal_losses)]

            self.logger.warning(
                f"After filtering: Background {len(background_losses_clean)}/{len(background_losses)}, Signal {len(signal_losses_clean)}/{len(signal_losses)}"
            )

            # Check if we have enough valid data left
            if len(background_losses_clean) == 0 or len(signal_losses_clean) == 0:
                self.logger.error(
                    f"No valid {loss_type} losses remaining after filtering NaN/Inf!"
                )
                # Return metrics with default values
                return {
                    "background_mean": 0.0,
                    "background_std": 0.0,
                    "signal_mean": 0.0,
                    "signal_std": 0.0,
                    "separation": 0.0,
                    "roc_auc": 0.5,  # Random performance
                    "roc_curve": {"fpr": [0, 1], "tpr": [0, 1], "thresholds": [1, 0]},
                    "data_quality_warning": f"All {loss_type} losses were NaN/Inf - using default values",
                }
        else:
            # No invalid values, use original arrays
            background_losses_clean = background_losses
            signal_losses_clean = signal_losses

        # Calculate basic statistics using cleaned data
        bg_std = float(np.std(background_losses_clean))
        sig_std = float(np.std(signal_losses_clean))

        # Calculate separation with protection against zero variance
        mean_diff = abs(np.mean(signal_losses_clean) - np.mean(background_losses_clean))
        std_sum_squared = sig_std**2 + bg_std**2

        if std_sum_squared == 0.0:
            # Both distributions have zero variance (all values identical)
            if mean_diff == 0.0:
                separation = 0.0  # Identical distributions
            else:
                separation = float("inf")  # Perfect separation with no overlap
            self.logger.warning(
                f"Zero variance detected in {loss_type} separation calculation: "
                f"bg_std={bg_std}, sig_std={sig_std}, mean_diff={mean_diff}. "
                f"Setting separation to {'0.0' if separation == 0.0 else 'inf'}."
            )
        else:
            separation = float(mean_diff / np.sqrt(std_sum_squared))

        metrics = {
            "background_mean": float(np.mean(background_losses_clean)),
            "background_std": bg_std,
            "signal_mean": float(np.mean(signal_losses_clean)),
            "signal_std": sig_std,
            "separation": separation,
        }

        # Add data quality info if filtering was needed
        if total_invalid > 0:
            metrics["data_quality_info"] = {
                "original_background_count": len(background_losses),
                "original_signal_count": len(signal_losses),
                "filtered_background_count": len(background_losses_clean),
                "filtered_signal_count": len(signal_losses_clean),
                "background_nan_count": int(bg_nan_count),
                "background_inf_count": int(bg_inf_count),
                "signal_nan_count": int(sig_nan_count),
                "signal_inf_count": int(sig_inf_count),
            }

        # Add ROC curve metrics using cleaned data
        from sklearn.metrics import auc, roc_curve

        labels = np.concatenate(
            [np.zeros(len(background_losses_clean)), np.ones(len(signal_losses_clean))]
        )
        scores = np.concatenate([background_losses_clean, signal_losses_clean])

        # Final safety check - this should not happen but just in case
        if not np.all(np.isfinite(scores)):
            self.logger.error("BUG: Still have NaN/Inf in scores after filtering!")
            valid_indices = np.isfinite(scores)
            scores = scores[valid_indices]
            labels = labels[valid_indices]

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        metrics["roc_auc"] = float(roc_auc)

        # ownsample the ROC curve to reduce size
        # Only include a maximum of 20 points to keep the JSON file manageable
        if len(fpr) > 20:
            # Get indices for approximately 20 evenly spaced points
            indices = np.linspace(0, len(fpr) - 1, 20).astype(int)
            # Make sure to always include the endpoints
            if indices[0] != 0:
                indices[0] = 0
            if indices[-1] != len(fpr) - 1:
                indices[-1] = len(fpr) - 1

            metrics["roc_curve"] = {
                "fpr": fpr[indices].tolist(),
                "tpr": tpr[indices].tolist(),
                "thresholds": thresholds[indices].tolist()
                if len(thresholds) > 0
                else [],
            }
        else:
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }

        return metrics

    def _save_loss_distribution_data(
        self,
        losses: np.ndarray,
        loss_type: str,
        signal_name: str,
        plot_data_dir: Path,
        bin_edges: np.ndarray = None,
    ) -> Path:
        """
        Save loss distribution data as JSON for later combined plotting.

        Args:
            losses: Array of loss values
            loss_type: Type of loss ('reconstruction' or 'kl_divergence')
            signal_name: Name of the signal dataset ('background' for background data)
            plot_data_dir: Directory to save the JSON file
            bin_edges: Optional pre-calculated bin edges for coordinated binning

        Returns:
            Path to the saved JSON file
        """
        # Calculate histogram data
        if losses.size == 0:
            counts = []
            if bin_edges is not None:
                bin_edges_list = bin_edges.tolist()
            else:
                bin_edges_list = []
        else:
            if bin_edges is not None:
                # Use provided bin edges for coordinated binning
                counts, _ = np.histogram(losses, bins=bin_edges, density=True)
                bin_edges_list = bin_edges.tolist()
            else:
                # Fallback to individual percentile-based range
                p0_1, p99_9 = np.percentile(losses, [0.1, 99.9])
                if p0_1 == p99_9:
                    p0_1 -= 0.5
                    p99_9 += 0.5
                plot_range = (p0_1, p99_9)

                counts, calculated_bin_edges = np.histogram(
                    losses, bins=50, range=plot_range, density=True
                )
                bin_edges_list = calculated_bin_edges.tolist()

        # Create JSON data structure
        loss_data = {
            loss_type: {
                "counts": counts.tolist() if hasattr(counts, "tolist") else counts,
                "bin_edges": bin_edges_list,
                "n_events": len(losses),
                "mean": float(np.mean(losses)) if losses.size > 0 else 0.0,
                "std": float(np.std(losses)) if losses.size > 0 else 0.0,
            }
        }

        # Save to JSON file
        json_filename = f"loss_distributions_{signal_name}_{loss_type}_data.json"
        json_path = plot_data_dir / json_filename

        with open(json_path, "w") as f:
            json.dump(loss_data, f, indent=2)

        return json_path

    def _save_roc_curve_data(
        self,
        bg_losses: np.ndarray,
        sig_losses: np.ndarray,
        loss_type: str,
        signal_name: str,
        plot_data_dir: Path,
    ) -> Path:
        """
        Save ROC curve data as JSON for later combined plotting.

        Args:
            bg_losses: Background loss values
            sig_losses: Signal loss values
            loss_type: Type of loss ('reconstruction' or 'kl_divergence')
            signal_name: Name of the signal dataset
            plot_data_dir: Directory to save the JSON file

        Returns:
            Path to the saved JSON file
        """
        from sklearn.metrics import auc, roc_curve

        # Create labels and scores for ROC calculation
        labels = np.concatenate([np.zeros(len(bg_losses)), np.ones(len(sig_losses))])
        scores = np.concatenate([bg_losses, sig_losses])

        # Filter out NaN and Inf values for ROC calculation
        valid_mask = np.isfinite(scores)
        if not np.any(valid_mask):
            self.logger.warning(
                f"All {loss_type} scores are NaN/Inf for {signal_name}! "
                f"Cannot calculate ROC curve, using default values."
            )
            # Return default ROC curve (diagonal line)
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            thresholds = np.array([np.inf, -np.inf])
            roc_auc = 0.5
        else:
            # Use only valid data points
            valid_labels = labels[valid_mask]
            valid_scores = scores[valid_mask]

            # Log if we filtered out any values
            n_invalid = np.sum(~valid_mask)
            if n_invalid > 0:
                self.logger.warning(
                    f"Filtered out {n_invalid} invalid values from {loss_type} scores for {signal_name} "
                    f"before ROC calculation (kept {np.sum(valid_mask)} valid values)"
                )

            # Calculate ROC curve with valid data
            fpr, tpr, thresholds = roc_curve(valid_labels, valid_scores)
            roc_auc = auc(fpr, tpr)

        # Downsample to reduce file size (keep max 50 points)
        if len(fpr) > 50:
            indices = np.linspace(0, len(fpr) - 1, 50).astype(int)
            # Always include endpoints
            if indices[0] != 0:
                indices[0] = 0
            if indices[-1] != len(fpr) - 1:
                indices[-1] = len(fpr) - 1
            fpr = fpr[indices]
            tpr = tpr[indices]
            thresholds = thresholds[indices] if len(thresholds) > 0 else []

        # Create JSON data structure
        roc_data = {
            loss_type: {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist() if len(thresholds) > 0 else [],
                "auc": float(roc_auc),
                "n_background": len(bg_losses),
                "n_signal": len(sig_losses),
            }
        }

        # Save to JSON file
        json_filename = f"roc_curves_{signal_name}_{loss_type}_data.json"
        json_path = plot_data_dir / json_filename

        with open(json_path, "w") as f:
            json.dump(roc_data, f, indent=2)

        self.logger.info(
            f"Saved {loss_type} ROC curve data for {signal_name} to {json_path}"
        )
        return json_path

    def _create_combined_loss_distribution_plots(
        self,
        bg_recon_losses: np.ndarray,
        bg_kl_losses: np.ndarray,
        signal_loss_data: dict,
        test_dir: Path,
    ) -> None:
        """
        Create combined loss distribution plots for all signals together.

        Args:
            bg_recon_losses: Background reconstruction losses
            bg_kl_losses: Background KL divergence losses
            signal_loss_data: Dictionary mapping signal names to their loss arrays
            test_dir: Base testing directory (contains plot_data/ and plots/ subdirectories)
        """
        self.logger.info("Creating combined loss distribution plots...")

        # Create plot_data and plots directories
        plot_data_dir = test_dir / "plot_data"
        plot_data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = test_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Calculate global bin edges for coordinated binning
        self.logger.info(
            "Calculating global bin edges for coordinated loss distribution binning..."
        )

        # NEW APPROACH: Calculate individual percentiles and take maximum
        self.logger.info(
            "Using individual dataset percentiles to ensure all data ranges are captured..."
        )

        # Calculate 99.9th percentile for background
        bg_recon_p99_9 = np.percentile(bg_recon_losses, 99.9)
        bg_kl_p99_9 = np.percentile(bg_kl_losses, 99.9)

        # Calculate 99.9th percentile for each signal and track the maximum
        signal_recon_p99_9_values = [bg_recon_p99_9]
        signal_kl_p99_9_values = [bg_kl_p99_9]

        for signal_name, (sig_recon_losses, sig_kl_losses) in signal_loss_data.items():
            sig_recon_p99_9 = np.percentile(sig_recon_losses, 99.9)
            sig_kl_p99_9 = np.percentile(sig_kl_losses, 99.9)

            signal_recon_p99_9_values.append(sig_recon_p99_9)
            signal_kl_p99_9_values.append(sig_kl_p99_9)

            self.logger.info(
                f"{signal_name} - Reconstruction 99.9th percentile: {sig_recon_p99_9:.3f}, "
                f"KL 99.9th percentile: {sig_kl_p99_9:.3f}"
            )

        # Use maximum of all individual 99.9th percentiles
        recon_p99_9 = max(signal_recon_p99_9_values)
        kl_p99_9 = max(signal_kl_p99_9_values)

        # Calculate 0.1st percentile from combined data (this should be fine as it's typically near zero)
        all_recon_losses = [bg_recon_losses]
        for signal_name, (sig_recon_losses, _) in signal_loss_data.items():
            all_recon_losses.append(sig_recon_losses)
        combined_recon_losses = np.concatenate(all_recon_losses)

        all_kl_losses = [bg_kl_losses]
        for signal_name, (_, sig_kl_losses) in signal_loss_data.items():
            all_kl_losses.append(sig_kl_losses)
        combined_kl_losses = np.concatenate(all_kl_losses)

        recon_p0_1 = np.percentile(combined_recon_losses, 0.1)
        kl_p0_1 = np.percentile(combined_kl_losses, 0.1)

        # Handle edge case where percentiles are equal
        if recon_p0_1 == recon_p99_9:
            recon_p0_1 -= 0.5
            recon_p99_9 += 0.5
        if kl_p0_1 == kl_p99_9:
            kl_p0_1 -= 0.5
            kl_p99_9 += 0.5

        recon_bin_edges = np.linspace(recon_p0_1, recon_p99_9, 51)  # 50 bins
        kl_bin_edges = np.linspace(kl_p0_1, kl_p99_9, 51)  # 50 bins

        self.logger.info(
            f"Background - Reconstruction 99.9th percentile: {bg_recon_p99_9:.3f}, KL 99.9th percentile: {bg_kl_p99_9:.3f}"
        )
        self.logger.info(
            f"Final bin ranges - Reconstruction: [{recon_p0_1:.3f}, {recon_p99_9:.3f}], "
            f"KL divergence: [{kl_p0_1:.3f}, {kl_p99_9:.3f}]"
        )
        self.logger.info(
            "Using maximum individual percentiles to ensure signal data is not truncated (covers 99.8% of background + full range of signals)"
        )

        # Save bin edges metadata for reference
        bin_edges_metadata = {
            "reconstruction_loss": {
                "bin_edges": recon_bin_edges.tolist(),
                "percentile_range": [0.1, 99.9],
                "data_range": [float(recon_p0_1), float(recon_p99_9)],
                "n_bins": len(recon_bin_edges) - 1,
                "total_events": len(combined_recon_losses),
            },
            "kl_divergence": {
                "bin_edges": kl_bin_edges.tolist(),
                "percentile_range": [0.1, 99.9],
                "data_range": [float(kl_p0_1), float(kl_p99_9)],
                "n_bins": len(kl_bin_edges) - 1,
                "total_events": len(combined_kl_losses),
            },
            "timestamp": str(datetime.now()),
            "datasets": ["background"] + list(signal_loss_data.keys()),
        }

        bin_edges_metadata_path = plot_data_dir / "loss_bin_edges_metadata.json"
        with open(bin_edges_metadata_path, "w") as f:
            json.dump(bin_edges_metadata, f, indent=2)
        self.logger.info(f"Saved loss bin edges metadata to {bin_edges_metadata_path}")

        # Save background loss distribution data with coordinated bin edges
        bg_recon_json = self._save_loss_distribution_data(
            bg_recon_losses,
            "reconstruction",
            "background",
            plot_data_dir,
            recon_bin_edges,
        )
        bg_kl_json = self._save_loss_distribution_data(
            bg_kl_losses, "kl_divergence", "background", plot_data_dir, kl_bin_edges
        )

        # Save signal loss distribution data and collect paths
        signal_recon_jsons = []
        signal_kl_jsons = []
        signal_legend_labels = []

        # Save ROC curve data for each signal
        signal_recon_roc_jsons = []
        signal_kl_roc_jsons = []

        for signal_name, (sig_recon_losses, sig_kl_losses) in signal_loss_data.items():
            # Save loss distribution data with coordinated bin edges
            sig_recon_json = self._save_loss_distribution_data(
                sig_recon_losses,
                "reconstruction",
                signal_name,
                plot_data_dir,
                recon_bin_edges,
            )
            sig_kl_json = self._save_loss_distribution_data(
                sig_kl_losses, "kl_divergence", signal_name, plot_data_dir, kl_bin_edges
            )

            signal_recon_jsons.append(sig_recon_json)
            signal_kl_jsons.append(sig_kl_json)
            signal_legend_labels.append(signal_name)

            # Save ROC curve data
            sig_recon_roc_json = self._save_roc_curve_data(
                bg_recon_losses,
                sig_recon_losses,
                "reconstruction",
                signal_name,
                plot_data_dir,
            )
            sig_kl_roc_json = self._save_roc_curve_data(
                bg_kl_losses, sig_kl_losses, "kl_divergence", signal_name, plot_data_dir
            )

            signal_recon_roc_jsons.append(sig_recon_roc_json)
            signal_kl_roc_jsons.append(sig_kl_roc_json)

        if signal_recon_jsons and signal_kl_jsons:
            recon_json_paths = [bg_recon_json] + signal_recon_jsons
            kl_json_paths = [bg_kl_json] + signal_kl_jsons
            legend_labels = ["Background"] + signal_legend_labels

            # Create log scale version
            combined_plot_path_log = plots_dir / "combined_loss_distributions_log.png"
            create_combined_two_panel_loss_plot_from_json(
                recon_json_paths=recon_json_paths,
                kl_json_paths=kl_json_paths,
                output_plot_path=str(combined_plot_path_log),
                legend_labels=legend_labels,
                title_prefix="Loss Distributions",
                log_scale=True,
            )
            self.logger.info(
                f"Saved combined two-panel loss distribution plot (log scale) to {combined_plot_path_log}"
            )

            # Create linear scale version
            combined_plot_path_linear = (
                plots_dir / "combined_loss_distributions_linear.png"
            )
            create_combined_two_panel_loss_plot_from_json(
                recon_json_paths=recon_json_paths,
                kl_json_paths=kl_json_paths,
                output_plot_path=str(combined_plot_path_linear),
                legend_labels=legend_labels,
                title_prefix="Loss Distributions",
                log_scale=False,
            )
            self.logger.info(
                f"Saved combined two-panel loss distribution plot (linear scale) to {combined_plot_path_linear}"
            )

        # Create combined ROC curves plot

        if signal_recon_roc_jsons and signal_kl_roc_jsons:
            combined_roc_plot_path = plots_dir / "combined_roc_curves.png"
            create_combined_roc_curves_plot_from_json(
                recon_roc_json_paths=signal_recon_roc_jsons,
                kl_roc_json_paths=signal_kl_roc_jsons,
                output_plot_path=str(combined_roc_plot_path),
                legend_labels=signal_legend_labels,
                title_prefix="ROC Curves",
            )
            self.logger.info(
                f"Saved combined ROC curves plot to {combined_roc_plot_path}"
            )

    def run_anomaly_detection_test(self) -> dict:
        """
        Evaluate model's anomaly detection capabilities

        Compares reconstruction error (MSE) and KL divergence distributions
        between background and signal datasets.

        Returns:
            Dictionary containing test metrics and results
        """
        self.logger.info("Running anomaly detection test...")

        if not isinstance(self.model, VariationalAutoEncoder):
            raise ValueError("Anomaly detection test requires a VariationalAutoEncoder")

        # Create testing/anomaly_detection directory
        test_dir = self.testing_path / "anomaly_detection"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Log dataset info before testing
        self.logger.info("Dataset information before testing:")
        for batch in self.test_dataset:
            if isinstance(batch, tuple):
                features, labels = batch
                self.logger.info("Background test dataset batch shapes:")
                self.logger.info(f"  Features: {features.shape}")
                self.logger.info(f"  Labels: {labels.shape}")
            else:
                self.logger.info(f"Background test dataset batch shape: {batch.shape}")
            break

        for signal_name, signal_dataset in self.signal_datasets.items():
            for batch in signal_dataset:
                if isinstance(batch, tuple):
                    features, labels = batch
                    self.logger.info(f"{signal_name} signal dataset batch shapes:")
                    self.logger.info(f"  Features: {features.shape}")
                    self.logger.info(f"  Labels: {labels.shape}")
                else:
                    self.logger.info(
                        f"{signal_name} signal dataset batch shape: {batch.shape}"
                    )
                break

        # Calculate losses for background dataset
        self.logger.info("Calculating losses for background dataset...")
        bg_recon_losses, bg_kl_losses = self._calculate_losses(self.test_dataset)

        # Calculate losses for each signal dataset and store for combined plotting
        signal_results = {}
        signal_loss_data = {}  # For combined plotting

        for signal_name, signal_dataset in self.signal_datasets.items():
            self.logger.info(f"Calculating losses for signal dataset: {signal_name}")
            sig_recon_losses, sig_kl_losses = self._calculate_losses(signal_dataset)

            # Store for combined plotting
            signal_loss_data[signal_name] = (sig_recon_losses, sig_kl_losses)

            # Calculate separation metrics
            recon_metrics = self._calculate_separation_metrics(
                bg_recon_losses, sig_recon_losses, "reconstruction"
            )
            kl_metrics = self._calculate_separation_metrics(
                bg_kl_losses, sig_kl_losses, "kl_divergence"
            )

            signal_results[signal_name] = {
                "reconstruction_metrics": recon_metrics,
                "kl_divergence_metrics": kl_metrics,
                "n_events": len(sig_recon_losses),
            }

        # Create combined loss distribution plots (replaces individual plots)
        if signal_loss_data:
            self._create_combined_loss_distribution_plots(
                bg_recon_losses,
                bg_kl_losses,
                signal_loss_data,
                test_dir,
            )

        # Prepare test results
        test_results = {
            "anomaly_detection": {
                "timestamp": str(datetime.now()),
                "background_events": len(bg_recon_losses),
                "signal_results": signal_results,
                "plots_directory": str(test_dir / "plots"),
                "data_directory": str(test_dir / "plot_data"),
            }
        }

        return test_results

    def _load_model_from_foundation_path(
        self,
        foundation_model_path: Path,
        task_config: TaskConfig,
    ) -> VariationalAutoEncoder:
        """
        Load a trained VAE model from a foundation model path.

        Args:
            foundation_model_path: Path to the foundation model directory
            task_config: Task configuration for input shape derivation

        Returns:
            Loaded VariationalAutoEncoder model
        """
        model_weights_path = (
            foundation_model_path / "models" / "foundation_model" / "full_model"
        )
        config_path = foundation_model_path / "_experiment_config.yaml"

        if not model_weights_path.exists():
            model_weights_path_h5 = model_weights_path.with_suffix(".weights.h5")
            if model_weights_path_h5.exists():
                model_weights_path = model_weights_path_h5
                self.logger.info("Found model weights with .h5 extension.")
            else:
                raise FileNotFoundError(
                    f"Foundation model weights not found at: {model_weights_path} or {model_weights_path_h5}"
                )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Foundation model config not found at: {config_path}"
            )

        # Load original model configuration
        self.logger.info(f"Loading original model config from: {config_path}")
        from hep_foundation.config.config_loader import PipelineConfigLoader

        config_loader = PipelineConfigLoader()
        original_experiment_data = config_loader.load_config(config_path)

        # Get the VAE model config from the YAML structure
        if "foundation_model_training" in original_experiment_data:
            foundation_config = original_experiment_data["foundation_model_training"]
            original_model_config = {
                "model_type": foundation_config["model"]["model_type"],
                "architecture": foundation_config["model"]["architecture"],
                "hyperparameters": foundation_config["model"]["hyperparameters"],
            }
        elif (
            "models" in original_experiment_data
            and "vae" in original_experiment_data["models"]
        ):
            # Backward compatibility with old format
            original_model_config = original_experiment_data["models"]["vae"]
        else:
            raise ValueError(f"Could not find VAE model config in: {config_path}")

        if (
            not original_model_config
            or original_model_config.get("model_type") != "variational_autoencoder"
        ):
            raise ValueError(
                f"Loaded config is not for a variational_autoencoder: {config_path}"
            )

        # Ensure input_shape is present in the loaded config
        if "input_shape" not in original_model_config["architecture"]:
            self.logger.warning(
                "Input shape missing in loaded model config, deriving from task_config."
            )
            input_shape = (task_config.input.get_total_feature_size(),)
            if input_shape[0] is None or input_shape[0] <= 0:
                raise ValueError(
                    "Could not determine a valid input shape from task_config."
                )
            original_model_config["architecture"]["input_shape"] = list(input_shape)

        # Ensure hyperparameters like beta_schedule are present
        if "hyperparameters" not in original_model_config:
            original_model_config["hyperparameters"] = {}
        if "beta_schedule" not in original_model_config["hyperparameters"]:
            self.logger.warning(
                "beta_schedule missing in loaded config, using default."
            )
            original_model_config["hyperparameters"]["beta_schedule"] = {
                "start": 0.0,
                "end": 1.0,
                "warmup_epochs": 0,
                "cycle_epochs": 0,
            }

        # Create VAE model (building will happen when needed)
        vae_config = VAEConfig(
            model_type=original_model_config["model_type"],
            architecture=original_model_config["architecture"],
            hyperparameters=original_model_config["hyperparameters"],
        )
        model = VariationalAutoEncoder(config=vae_config)

        # Build model to load weights (for evaluation, we can build directly)
        input_shape = tuple(original_model_config["architecture"]["input_shape"])
        model.build(input_shape)

        # Load weights
        self.logger.info(f"Loading model weights from: {model_weights_path}")
        try:
            model.model.load_weights(str(model_weights_path)).expect_partial()
            self.logger.info("VAE model loaded successfully.")
            model.model.summary(print_fn=self.logger.info)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")

        return model

    def _load_datasets_for_anomaly_detection(
        self,
        dataset_config: DatasetConfig,
        eval_batch_size: int,
        delete_catalogs: bool,
    ) -> tuple[tf.data.Dataset, dict[str, tf.data.Dataset]]:
        """
        Load background and signal datasets for anomaly detection.

        Args:
            dataset_config: Dataset configuration
            eval_batch_size: Batch size for evaluation
            delete_catalogs: Whether to delete catalogs after processing

        Returns:
            Tuple of (test_dataset, signal_datasets)
        """
        if not self.processed_datasets_dir:
            raise ValueError("processed_datasets_dir must be set to load datasets")

        data_manager = DatasetManager(base_dir=self.processed_datasets_dir)

        # Load test dataset (background)
        self.logger.info("Loading background (test) dataset...")
        _, _, test_dataset = data_manager.load_atlas_datasets(
            dataset_config=dataset_config,
            validation_fraction=0.0,
            test_fraction=1.0,
            batch_size=eval_batch_size,
            shuffle_buffer=dataset_config.shuffle_buffer,
            include_labels=False,
            delete_catalogs=delete_catalogs,
        )
        background_dataset_id_for_plots = data_manager.get_current_dataset_id()
        self.logger.info("Loaded background (test) dataset.")

        # Determine background histogram data path for comparison plot
        background_hist_data_path_for_comparison = None
        if dataset_config.plot_distributions and background_dataset_id_for_plots:
            background_plot_data_dir = (
                data_manager.get_dataset_dir(background_dataset_id_for_plots)
                / "plot_data"
            )
            potential_background_hist_path = (
                background_plot_data_dir / "atlas_dataset_features_hist_data.json"
            )
            if potential_background_hist_path.exists():
                background_hist_data_path_for_comparison = (
                    potential_background_hist_path
                )
            else:
                self.logger.warning(
                    f"Background histogram data for comparison not found at {potential_background_hist_path}. Comparison plot may be skipped by DatasetManager."
                )

        # Load signal datasets
        signal_datasets = {}
        if dataset_config.signal_keys:
            self.logger.info(
                "Loading signal datasets for evaluation and comparison plotting..."
            )
            signal_datasets = data_manager.load_signal_datasets(
                dataset_config=dataset_config,
                batch_size=eval_batch_size,
                include_labels=False,
                background_hist_data_path=background_hist_data_path_for_comparison,
                split=False,  # Anomaly detection uses full signal datasets
            )
            self.logger.info(
                f"Loaded {len(signal_datasets)} signal datasets for evaluation."
            )
        else:
            self.logger.warning(
                "No signal keys provided. Skipping signal dataset loading for evaluation."
            )

        return test_dataset, signal_datasets

    def evaluate_anomaly_detection(
        self,
        dataset_config: DatasetConfig,
        task_config: TaskConfig,
        delete_catalogs: bool = True,
        foundation_model_path: str = None,
        vae_training_config: TrainingConfig = None,
        eval_batch_size: int = 1024,
    ):
        """
        Evaluate a trained foundation model (VAE) for anomaly detection.

        Loads a pre-trained VAE, test/signal datasets, performs evaluation,
        and saves results directly in the foundation model's experiment directory.
        """

        if not foundation_model_path:
            self.logger.error(
                "Foundation model path must be provided for anomaly evaluation."
            )
            return False

        foundation_model_path = Path(foundation_model_path)

        try:
            # Use eval_batch_size or derive from vae_training_config if provided
            batch_size = eval_batch_size
            if vae_training_config:
                try:
                    batch_size = vae_training_config.batch_size
                    self.logger.info(
                        f"Using batch size from provided VAE training config: {batch_size}"
                    )
                except AttributeError:
                    self.logger.warning(
                        "Provided vae_training_config lacks batch_size, using default."
                    )
            else:
                self.logger.info(f"Using default evaluation batch size: {batch_size}")

            # Load model from foundation path
            self.logger.info("Loading VAE model from foundation model path...")
            self.model = self._load_model_from_foundation_path(
                foundation_model_path, task_config
            )

            # Load datasets
            self.logger.info("Loading datasets for anomaly detection...")
            self.test_dataset, self.signal_datasets = (
                self._load_datasets_for_anomaly_detection(
                    dataset_config, batch_size, delete_catalogs
                )
            )

            # Run the anomaly detection evaluation using the existing method
            self.logger.info("Running anomaly detection evaluation...")
            self.run_anomaly_detection_test()

            # Display Results
            self.logger.info("=" * 100)
            self.logger.info("Anomaly Detection Results")
            self.logger.info("=" * 100)

            foundation_experiment_id = foundation_model_path.name
            self.logger.info(f"Foundation Model ID: {foundation_experiment_id}")
            self.logger.info(
                "Anomaly detection evaluation completed. Results are available in the testing directory."
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Anomaly detection evaluation failed: {type(e).__name__}: {str(e)}"
            )
            self.logger.exception("Detailed traceback:")
            return False
