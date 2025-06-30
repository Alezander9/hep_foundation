import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hep_foundation.config.logging_config import get_logger
from hep_foundation.models.base_model import BaseModel
from hep_foundation.models.dnn_predictor import DNNPredictor


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for clean self.logger output"""

    def __init__(self, epochs: int):
        super().__init__()
        self.epochs = epochs
        self.epoch_start_time = None

        # Setup self.logger
        self.logger = get_logger(__name__)

    def on_train_begin(self, logs=None):
        self.logger.info(f"Starting training for {self.epochs} epochs")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.logger.info(f"Starting epoch {epoch + 1}/{self.epochs}")

    def on_epoch_end(self, epoch, logs=None):
        time_taken = time.time() - self.epoch_start_time
        metrics = " - ".join(f"{k}: {v:.6f}" for k, v in logs.items())
        self.logger.info(
            f"Epoch {epoch + 1}/{self.epochs} completed in {time_taken:.1f}s - {metrics}"
        )


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(
        self,
        model: BaseModel,
        training_config: dict,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: Optional[tf.keras.losses.Loss] = None,
    ):
        self.logger = get_logger(__name__)
        self.model = model
        self.config = training_config

        # Set up training parameters
        self.batch_size = training_config.get("batch_size", 32)
        self.epochs = training_config.get("epochs", 10)
        self.validation_split = training_config.get("validation_split", 0.2)

        # Set up optimizer and loss
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            learning_rate=training_config.get("learning_rate", 0.001)
        )
        self.loss = loss or tf.keras.losses.MeanSquaredError()

        # Training history
        self.history = None
        self.metrics_history = {}

        self.training_start_time = None
        self.training_end_time = None

    def compile_model(self):
        """Compile the model with optimizer and loss"""
        if self.model.model is None:
            raise ValueError("Model not built yet")

        # Import here to avoid circular imports
        from hep_foundation.models.base_model import CustomKerasModelWrapper

        # Check if this is a predictor model (either DNNPredictor or CustomKerasModelWrapper for regression)
        is_predictor = isinstance(self.model, DNNPredictor)

        # For CustomKerasModelWrapper, check if it's being used for regression or classification
        if isinstance(self.model, CustomKerasModelWrapper):
            model_name = getattr(self.model, "name", "").lower()
            is_predictor = any(
                term in model_name
                for term in [
                    "regressor",
                    "predictor",
                    "classifier",
                    "from_scratch",
                    "fine_tuned",
                    "fixed_encoder",
                ]
            )

        # Different compilation settings based on model type
        if is_predictor:
            # Check if this is a classification model
            model_name = getattr(self.model, "name", "").lower()
            is_classifier = "classifier" in model_name

            if is_classifier:
                # Binary classification compilation
                self.model.model.compile(
                    optimizer=self.optimizer,
                    loss="binary_crossentropy",
                    metrics=[
                        "binary_accuracy",
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                    ],
                    run_eagerly=True,
                )
            else:
                # Regression compilation
                self.model.model.compile(
                    optimizer=self.optimizer,
                    loss="mse",
                    metrics=["mse", "mae"],  # Add mean absolute error for regression
                    run_eagerly=True,
                )
        else:
            # Original compilation for autoencoders
            self.model.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=["mse"],
                run_eagerly=True,
            )

    def prepare_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Prepare dataset based on model type

        Args:
            dataset: Input dataset that may contain features and labels

        Returns:
            Prepared dataset with correct input/target structure
        """
        # Import here to avoid circular imports
        from hep_foundation.models.base_model import CustomKerasModelWrapper

        # Check if this is a predictor model (either DNNPredictor or CustomKerasModelWrapper for regression)
        is_predictor = isinstance(self.model, DNNPredictor)

        # For CustomKerasModelWrapper, check if it's being used for regression or classification
        # We can identify this by checking if the model name contains regression/classification-related terms
        if isinstance(self.model, CustomKerasModelWrapper):
            model_name = getattr(self.model, "name", "").lower()
            is_predictor = any(
                term in model_name
                for term in [
                    "regressor",
                    "predictor",
                    "classifier",
                    "from_scratch",
                    "fine_tuned",
                    "fixed_encoder",
                ]
            )

        if is_predictor:
            # For predictor models, use features as input and select correct label as target
            def prepare_predictor_data(
                features, labels, *args
            ):  # *args is for compatibility with the dataset which also comes with an index argument
                # Handle both tuple and non-tuple inputs
                if isinstance(features, (tf.Tensor, np.ndarray)):
                    input_features = features
                else:
                    input_features = features[0]

                # Select the correct label set based on label_index
                # For CustomKerasModelWrapper, we assume label_index 0 (first label set)
                if hasattr(self.model, "label_index"):
                    label_index = self.model.label_index
                else:
                    label_index = 0  # Default for CustomKerasModelWrapper

                if isinstance(labels, (list, tuple)):
                    target_labels = labels[label_index]
                else:
                    target_labels = labels

                return input_features, target_labels

            return dataset.map(prepare_predictor_data)
        else:
            # For autoencoders, use input as both input and target
            return dataset.map(
                lambda x, y: (x, x)
                if isinstance(x, (tf.Tensor, np.ndarray))
                else (x[0], x[0])
            )

    def _update_metrics_history(self, epoch_metrics: dict) -> None:
        """Update metrics history with new epoch results"""
        for metric_name, value in epoch_metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(float(value))

    def get_training_summary(self) -> dict[str, Any]:
        """Get comprehensive training summary with all metrics and history"""
        if not self.metrics_history:
            return {
                "training_config": self.config,
                "metrics": {},
                "history": {},
                "training_duration": 0.0,
                "epochs_completed": 0,
            }

        # Calculate training duration
        training_duration = 0.0
        if self.training_start_time and self.training_end_time:
            training_duration = (
                self.training_end_time - self.training_start_time
            ).total_seconds()

        # Get the latest metrics for each metric type
        final_metrics = {
            metric: values[-1] for metric, values in self.metrics_history.items()
        }

        # Convert history to epoch-based format
        epoch_history = {}
        for epoch in range(len(next(iter(self.metrics_history.values())))):
            epoch_history[str(epoch)] = {
                metric: values[epoch] for metric, values in self.metrics_history.items()
            }

        return {
            "training_config": self.config,
            "epochs_completed": len(epoch_history),
            "training_duration": training_duration,
            "final_metrics": final_metrics,
            "history": epoch_history,
        }

    def train(
        self,
        dataset: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset] = None,
        callbacks: list[tf.keras.callbacks.Callback] = None,
        plot_training: bool = False,
        plot_result: bool = False,
        plots_dir: Optional[Path] = None,
        plot_training_title: Optional[str] = None,
        plot_result_title: Optional[str] = None,
        test_dataset: Optional[tf.data.Dataset] = None,
    ) -> dict[str, Any]:
        """Train with enhanced metrics tracking and optional plotting"""
        self.logger.info("Starting training with metrics tracking:")

        # Record start time
        self.training_start_time = datetime.now()

        if plot_training and plots_dir is None:
            plots_dir = Path("experiments/plots")

        if plot_training:
            plots_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Will save training plots to: {plots_dir}")

        if self.model.model is None:
            raise ValueError("Model not built yet")

        self.logger.info("Preparing datasets for training...")

        try:
            # Prepare training and validation datasets
            train_dataset = self.prepare_dataset(dataset)

            if validation_data is not None:
                validation_data = self.prepare_dataset(validation_data)

            # Log dataset size and shapes for debugging
            self.logger.info(f"Training dataset batches: {train_dataset.cardinality()}")
            for batch in train_dataset.take(1):
                features, targets = batch
                self.logger.info("Training dataset batch shapes:")
                self.logger.info(f"  Features: {features.shape}")
                self.logger.info(f"  Targets: {targets.shape}")
                break

        except Exception as e:
            self.logger.error(f"Error preparing datasets: {str(e)}")
            raise

        # Compile model
        self.compile_model()

        # Setup callbacks
        if callbacks is None:
            callbacks = []

        # Add our custom progress callback
        callbacks.append(TrainingProgressCallback(epochs=self.epochs))

        # Train the model
        history = self.model.model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=True,
            verbose=0,  # Turn off default progress bar
        )

        # Record end time
        self.training_end_time = datetime.now()

        # Update model's history if it's a VAE
        if hasattr(self.model, "_history"):
            self.model._history = history.history

        # Update metrics history
        for metric, values in history.history.items():
            self.metrics_history[metric] = [float(v) for v in values]

        self.logger.info("Training completed. Final metrics:")
        for metric, values in self.metrics_history.items():
            self.logger.info(f"  {metric}: {values[-1]:.6f}")

        # After training completes, create plots if requested
        if plot_training:
            self.logger.info("Generating training plots...")
            self._create_training_plots(plots_dir, plot_training_title)

        if plot_result and test_dataset is not None:
            self.logger.info("Generating result plots...")
            self._create_result_plots(plots_dir, plot_result_title, test_dataset)

        return self.get_training_summary()

    def _create_training_plots(
        self, plots_dir: Path, plot_training_title: Optional[str] = None
    ):
        """Create standard training plots with simple formatting"""
        self.logger.info(f"Creating training plots in: {plots_dir.absolute()}")
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            from hep_foundation.utils.plot_utils import (
                FONT_SIZES,
                LINE_WIDTHS,
                get_color_cycle,
                get_figure_size,
                set_science_style,
            )

            set_science_style(use_tex=False)

            plt.figure(figsize=get_figure_size("single", ratio=1.2))
            history = self.metrics_history
            colors = get_color_cycle("high_contrast")
            color_idx = 0

            # Plot metrics with simple labels
            for metric in history.keys():
                if "loss" in metric.lower() and not metric.lower().startswith(
                    ("val_", "test_")
                ):
                    label = metric.replace("_", " ").title()
                    plt.plot(
                        history[metric],
                        label=label,
                        color=colors[color_idx % len(colors)],
                        linewidth=LINE_WIDTHS["thick"],
                    )
                    color_idx += 1

            plt.yscale("log")  # Set y-axis to logarithmic scale
            plt.xlabel("Epoch", fontsize=FONT_SIZES["large"])
            plt.ylabel("Loss (log)", fontsize=FONT_SIZES["large"])

            # Use custom title if provided, otherwise use default
            title = plot_training_title if plot_training_title else "Training History"
            plt.title(title, fontsize=FONT_SIZES["xlarge"])

            plt.legend(fontsize=FONT_SIZES["normal"], loc="upper right")
            plt.grid(
                True, alpha=0.3, which="both"
            )  # Grid lines for both major and minor ticks

            plt.savefig(
                plots_dir / "training_history.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            # Let the model create any model-specific plots
            if hasattr(self.model, "create_plots"):
                self.model.create_plots(plots_dir)

            self.logger.info(f"Plots saved to: {plots_dir}")

        except Exception as e:
            self.logger.error(f"Error creating plots: {str(e)}")
            import traceback

            traceback.print_exc()

    def _create_result_plots(
        self,
        plots_dir: Path,
        plot_result_title: Optional[str] = None,
        test_dataset: tf.data.Dataset = None,
    ):
        """Create histogram plots of model results on test data with sampling"""
        self.logger.info(f"Creating result plots in: {plots_dir.absolute()}")
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            from hep_foundation.utils.plot_utils import (
                FONT_SIZES,
                get_color_cycle,
                get_figure_size,
                set_science_style,
            )

            set_science_style(use_tex=False)

            # Sample up to 5000 events from test dataset for plotting
            sample_size = 5000
            self.logger.info(f"Sampling up to {sample_size} events for result plotting")

            # Prepare test dataset
            prepared_test_dataset = self.prepare_dataset(test_dataset)

            # Collect error metrics per event (not all raw values)
            error_values = []
            sample_count = 0

            for batch in prepared_test_dataset:
                if sample_count >= sample_size:
                    break

                features, labels = batch

                # Get predictions from model
                batch_predictions = self.model.model.predict(features, verbose=0)

                # Calculate how many events to take from this batch
                batch_size = len(batch_predictions)
                remaining_samples = sample_size - sample_count
                events_to_sample = min(batch_size, remaining_samples)

                # Take subset of batch (events, not flattened values)
                pred_subset = batch_predictions[:events_to_sample]
                target_subset = labels[:events_to_sample]

                # Convert to numpy if needed
                if hasattr(pred_subset, "numpy"):
                    pred_subset = pred_subset.numpy()
                if hasattr(target_subset, "numpy"):
                    target_subset = target_subset.numpy()

                # Compute error per event (single value per event, not per feature)
                if (
                    pred_subset.shape == target_subset.shape
                    and len(pred_subset.shape) > 1
                ):
                    # Autoencoder case: compute reconstruction error per event (mean across features)
                    event_errors = np.mean(np.abs(pred_subset - target_subset), axis=1)
                else:
                    # Regression/classification case: compute error per event
                    if len(pred_subset.shape) > 1 and pred_subset.shape[1] > 1:
                        # Multi-output case: compute mean error per event
                        event_errors = np.mean(
                            np.abs(pred_subset - target_subset), axis=1
                        )
                    else:
                        # Single output case
                        event_errors = np.abs(
                            pred_subset.flatten() - target_subset.flatten()
                        )

                error_values.extend(event_errors)
                sample_count += events_to_sample

            error_values = np.array(error_values)

            # For compatibility with existing code, also store some sample predictions/targets
            # But limit these to a smaller subset to avoid huge files
            max_raw_samples = 1000
            predictions = []
            targets = []
            raw_sample_count = 0

            for batch in prepared_test_dataset:
                if raw_sample_count >= max_raw_samples:
                    break

                features, labels = batch
                batch_predictions = self.model.model.predict(features, verbose=0)

                events_to_sample = min(
                    len(batch_predictions), max_raw_samples - raw_sample_count
                )

                # Convert to numpy if needed before flattening
                pred_sample = batch_predictions[:events_to_sample]
                target_sample = labels[:events_to_sample]

                if hasattr(pred_sample, "numpy"):
                    pred_sample = pred_sample.numpy()
                if hasattr(target_sample, "numpy"):
                    target_sample = target_sample.numpy()

                predictions.extend(pred_sample.flatten())
                targets.extend(target_sample.flatten())
                raw_sample_count += events_to_sample

            predictions = np.array(predictions)
            targets = np.array(targets)

            self.logger.info(
                f"Collected {len(error_values)} events for result plotting"
            )

            # Use pre-computed error values for plotting
            metric_values = error_values

            # Determine metric name and plot details based on model type
            model_name = getattr(self.model, "name", "").lower()
            is_classifier = "classifier" in model_name
            is_autoencoder = any(
                keyword in model_name for keyword in ["vae", "autoencoder", "encoder"]
            )

            if is_autoencoder:
                metric_name = "Reconstruction Error (Mean Absolute Error)"
                plot_title = (
                    plot_result_title
                    if plot_result_title
                    else "Model Results: Reconstruction Error"
                )
                filename = "result_reconstruction_error"
            elif is_classifier:
                metric_name = "Classification Error"
                plot_title = (
                    plot_result_title
                    if plot_result_title
                    else "Model Results: Classification Error"
                )
                filename = "result_classification_error"
            else:
                metric_name = "Absolute Error"
                plot_title = (
                    plot_result_title
                    if plot_result_title
                    else "Model Results: Absolute Error"
                )
                filename = "result_absolute_error"

            # Create histogram
            plt.figure(figsize=get_figure_size("single", ratio=1.2))
            colors = get_color_cycle("high_contrast")

            # Create histogram with appropriate number of bins
            n_bins = min(50, max(10, len(metric_values) // 20))  # Adaptive bin count
            counts, bin_edges, patches = plt.hist(
                metric_values,
                bins=n_bins,
                color=colors[0],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            plt.xlabel(metric_name, fontsize=FONT_SIZES["large"])
            plt.ylabel("Count", fontsize=FONT_SIZES["large"])
            plt.title(plot_title, fontsize=FONT_SIZES["xlarge"])
            plt.grid(True, alpha=0.3)

            # Add statistics text
            mean_val = np.mean(metric_values)
            std_val = np.std(metric_values)
            median_val = np.median(metric_values)

            stats_text = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMedian: {median_val:.3f}\nSamples: {len(metric_values)}"
            plt.text(
                0.98,
                0.98,
                stats_text,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                fontsize=FONT_SIZES["small"],
            )

            # Save plot
            plot_path = plots_dir / f"{filename}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Save raw data like in dataset distribution plots
            raw_data = {
                "metric_values": metric_values.tolist(),
                "sample_predictions": predictions[
                    :1000
                ].tolist(),  # Only save limited sample for reference
                "sample_targets": targets[
                    :1000
                ].tolist(),  # Only save limited sample for reference
                "bin_edges": bin_edges.tolist(),
                "counts": counts.tolist(),
                "statistics": {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "median": float(median_val),
                    "min": float(np.min(metric_values)),
                    "max": float(np.max(metric_values)),
                    "sample_size": len(metric_values),
                    "metric_name": metric_name,
                },
                "metadata": {
                    "model_name": getattr(self.model, "name", "unknown"),
                    "is_classifier": is_classifier,
                    "is_autoencoder": is_autoencoder,
                    "plot_title": plot_title,
                    "creation_date": str(datetime.now()),
                    "note": "sample_predictions and sample_targets are limited to 1000 samples for file size",
                },
            }

            # Save raw data to JSON
            raw_data_path = plots_dir / f"{filename}_data.json"
            with open(raw_data_path, "w") as f:
                json.dump(raw_data, f, indent=2)

            self.logger.info(f"Result plot saved to: {plot_path}")
            self.logger.info(f"Raw result data saved to: {raw_data_path}")
            self.logger.info(
                f"Result statistics - Mean: {mean_val:.3f}, Std: {std_val:.3f}, Median: {median_val:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Error creating result plots: {str(e)}")
            import traceback

            traceback.print_exc()

    def evaluate(self, dataset: tf.data.Dataset) -> dict[str, float]:
        """Evaluate with enhanced metrics tracking"""
        if self.model.model is None:
            raise ValueError("Model not built yet")

        try:
            # Prepare test dataset
            test_dataset = self.prepare_dataset(dataset)

            # Evaluate and get results
            results = self.model.model.evaluate(
                test_dataset, return_dict=True, verbose=0
            )

            # Add test_ prefix to metrics
            test_metrics = {"test_" + k: float(v) for k, v in results.items()}

            self.logger.info("Evaluation metrics:")
            for metric, value in test_metrics.items():
                self.logger.info(f"  {metric}: {value:.6f}")

            # Store test metrics in history
            self.metrics_history.update(test_metrics)

            return test_metrics

        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
