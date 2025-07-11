import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from hep_foundation.config.logging_config import get_logger
from hep_foundation.models.base_model import BaseModel
from hep_foundation.models.dnn_predictor import DNNPredictor


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for clean self.logger output"""

    def __init__(self, epochs: int, verbosity: str = "full"):
        """
        Initialize the callback.

        Args:
            epochs: Total number of epochs
            verbosity: Logging verbosity level
                - "full": Log every epoch (default)
                - "summary": Log only start, every 10th epoch, and final epoch
                - "minimal": Log only start and final epoch
                - "silent": No logging output during training
        """
        super().__init__()
        self.epochs = epochs
        self.verbosity = verbosity
        self.epoch_start_time = None

        # Setup self.logger
        self.logger = get_logger(__name__)

    def on_train_begin(self, logs=None):
        if self.verbosity != "silent":
            self.logger.info(f"Starting training for {self.epochs} epochs")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        # Only log epoch start for full verbosity
        if self.verbosity == "full":
            self.logger.info(f"Starting epoch {epoch + 1}/{self.epochs}")

    def on_epoch_end(self, epoch, logs=None):
        time_taken = time.time() - (self.epoch_start_time or time.time())

        should_log = False
        if self.verbosity == "full":
            should_log = True
        elif self.verbosity == "summary":
            # Log every 10th epoch and the final epoch
            should_log = (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs
        elif self.verbosity == "minimal":
            # Log only the final epoch
            should_log = (epoch + 1) == self.epochs
        # For "silent", never log individual epochs

        if should_log and logs is not None:
            metrics = " - ".join(f"{k}: {v:.6f}" for k, v in logs.items())
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs} completed in {time_taken:.1f}s - {metrics}"
            )

    def on_train_end(self, logs=None):
        if self.verbosity != "silent":
            self.logger.info(f"Training completed after {self.epochs} epochs")


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

        # Enable mixed precision for A100 GPUs (1.5-2x speedup)
        if training_config.get("mixed_precision", True):
            mixed_precision.set_global_policy("mixed_float16")
            self.logger.info("Enabled mixed precision training (mixed_float16)")
        else:
            # Reset to default policy when mixed precision is disabled
            mixed_precision.set_global_policy("float32")
            self.logger.info("Mixed precision disabled in configuration")

        # Check for multi-GPU setup
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 1 and training_config.get("multi_gpu", False):
            self.strategy = tf.distribute.MirroredStrategy()
            self.logger.info(
                f"Enabled multi-GPU training with {len(gpus)} GPUs using MirroredStrategy"
            )
        else:
            self.strategy = None
            if len(gpus) > 1:
                self.logger.info(
                    f"Found {len(gpus)} GPUs but multi-GPU training disabled in config"
                )

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
                )
            else:
                # Regression compilation
                self.model.model.compile(
                    optimizer=self.optimizer,
                    loss="mse",
                    metrics=["mse", "mae"],  # Add mean absolute error for regression
                )
        else:
            # Original compilation for autoencoders
            self.model.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=["mse"],
            )

    def build_and_compile_model(self, input_shape: tuple):
        """Build and compile the model within the strategy scope for multi-GPU training"""
        # Import here to avoid circular imports
        from hep_foundation.models.base_model import CustomKerasModelWrapper

        # Check if model is already built
        if self.model.model is not None:
            self.logger.info("Model already built, proceeding with compilation...")
            self.compile_model()
            return

        # Build the model
        self.logger.info(f"Building model with input shape: {input_shape}")
        self.model.build(input_shape)

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
                )
            else:
                # Regression compilation
                self.model.model.compile(
                    optimizer=self.optimizer,
                    loss="mse",
                    metrics=["mse", "mae"],  # Add mean absolute error for regression
                )
        else:
            # Original compilation for autoencoders
            self.model.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=["mse"],
            )

        self.logger.info("Model built and compiled successfully")

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

            dataset = dataset.map(
                prepare_predictor_data, num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            # For autoencoders, use input as both input and target
            dataset = dataset.map(
                lambda x, y: (x, x)
                if isinstance(x, (tf.Tensor, np.ndarray))
                else (x[0], x[0]),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # Apply performance optimizations
        dataset = dataset.cache()  # Cache dataset in memory
        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )  # Prefetch next batch while training

        return dataset

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
        verbose_training: str = "full",
    ) -> dict[str, Any]:
        """
        Train with enhanced metrics tracking and optional plotting

        Args:
            dataset: Training dataset
            validation_data: Validation dataset (optional)
            callbacks: List of Keras callbacks (optional)
            plot_training: Whether to create training plots
            plot_result: Whether to create result plots
            plots_dir: Directory to save plots
            plot_training_title: Custom title for training plots
            plot_result_title: Custom title for result plots
            test_dataset: Test dataset for result plots
            verbose_training: Training verbosity level
                - "full": Log every epoch (default)
                - "summary": Log every 10th epoch and final epoch
                - "minimal": Log only final epoch
                - "silent": Log only start and end of training

        Returns:
            Training summary dictionary
        """
        self.logger.info("Starting training with metrics tracking:")

        # Record start time
        self.training_start_time = datetime.now()

        # Set up plots directory with defaults if needed
        if plot_training and plots_dir is None:
            plots_dir = Path("experiments/plots")
        elif plot_result and plots_dir is None:
            plots_dir = Path("experiments/plots")

        if plot_training:
            plots_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Will save training plots to: {plots_dir}")

        self.logger.info("Preparing datasets for training...")

        try:
            # Prepare training and validation datasets
            train_dataset = self.prepare_dataset(dataset)

            if validation_data is not None:
                validation_data = self.prepare_dataset(validation_data)

            # Determine input shape from the dataset
            input_shape = None
            for batch in train_dataset.take(1):
                features, targets = batch
                # Get the shape without the batch dimension
                input_shape = features.shape[1:]
                self.logger.info("Training dataset batch shapes:")
                self.logger.info(f"  Features: {features.shape}")
                self.logger.info(f"  Targets: {targets.shape}")
                self.logger.info(f"  Inferred input shape: {input_shape}")
                break

            if input_shape is None:
                raise ValueError("Could not determine input shape from dataset")

            # Log dataset size for debugging
            self.logger.info(f"Training dataset batches: {train_dataset.cardinality()}")

        except Exception as e:
            self.logger.error(f"Error preparing datasets: {str(e)}")
            raise

        # Build and compile model (with multi-GPU strategy if available)
        if self.strategy:
            with self.strategy.scope():
                self.build_and_compile_model(input_shape)
        else:
            self.build_and_compile_model(input_shape)

        # Setup callbacks
        if callbacks is None:
            callbacks = []

        # Add our custom progress callback
        callbacks.append(
            TrainingProgressCallback(epochs=self.epochs, verbosity=verbose_training)
        )

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
        if plot_training and plots_dir is not None:
            self.logger.info("Generating training plots...")
            self._create_training_plots(plots_dir, plot_training_title)

        if plot_result and test_dataset is not None and plots_dir is not None:
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
        try:
            # Prepare test dataset
            test_dataset = self.prepare_dataset(dataset)

            # If model is not built yet, build and compile it
            if self.model.model is None:
                # Determine input shape from the dataset
                input_shape = None
                for batch in test_dataset.take(1):
                    features, targets = batch
                    input_shape = features.shape[1:]
                    self.logger.info(
                        f"Building model for evaluation with input shape: {input_shape}"
                    )
                    break

                if input_shape is None:
                    raise ValueError("Could not determine input shape from dataset")

                # Build and compile model (with multi-GPU strategy if available)
                if self.strategy:
                    with self.strategy.scope():
                        self.build_and_compile_model(input_shape)
                else:
                    self.build_and_compile_model(input_shape)

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
