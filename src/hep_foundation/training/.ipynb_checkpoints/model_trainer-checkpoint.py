import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
            # Check for NaN/Inf in training metrics (consistent with evaluation)
            invalid_metrics = []
            for key, value in logs.items():
                if value is None or not isinstance(value, (int, float, np.number)):
                    invalid_metrics.append(f"{key}={value}")
                elif not np.isfinite(value):
                    invalid_metrics.append(f"{key}={value}")

            if invalid_metrics:
                self.logger.warning(
                    f"Epoch {epoch + 1}: Found invalid metrics: {', '.join(invalid_metrics)}"
                )

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
            learning_rate=training_config.get("learning_rate", 0.001),
            clipnorm=training_config.get("gradient_clip_norm", 1.0),
        )
        self.loss = loss or tf.keras.losses.MeanSquaredError()

        # Log gradient clipping configuration
        gradient_clip_norm = training_config.get("gradient_clip_norm", 1.0)
        self.logger.info(f"Gradient clipping enabled with norm={gradient_clip_norm}")

        # Training history
        self.history = None
        self.metrics_history = {}

        self.training_start_time = None
        self.training_end_time = None
        self._pending_save_info = None  # For consolidated save after evaluation

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
        # Handle both list values (training metrics) and single values (test metrics)
        final_metrics = {}
        for metric, values in self.metrics_history.items():
            if isinstance(values, list):
                final_metrics[metric] = values[-1] if values else 0
            else:
                # Single value (e.g., test metrics)
                final_metrics[metric] = values

        # Convert history to epoch-based format (only for list-type metrics)
        epoch_history = {}
        list_metrics = {
            k: v for k, v in self.metrics_history.items() if isinstance(v, list)
        }

        if list_metrics:
            num_epochs = len(next(iter(list_metrics.values())))
            for epoch in range(num_epochs):
                epoch_history[str(epoch)] = {
                    metric: values[epoch] for metric, values in list_metrics.items()
                }

        return {
            "training_config": self.config,
            "epochs_completed": len(epoch_history) if epoch_history else 0,
            "training_duration": training_duration,
            "final_metrics": final_metrics,
            "history": epoch_history,
        }

    def _save_training_history(
        self,
        training_history_dir: Path,
        model_name: str,
        dataset_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> Path:
        """
        Save training history to JSON file with metadata.

        Args:
            training_history_dir: Directory to save training history files
            model_name: Name of the model being trained
            dataset_id: Optional dataset identifier
            experiment_id: Optional experiment identifier

        Returns:
            Path to the saved JSON file
        """
        training_history_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_history_{model_name}_{timestamp}.json"
        json_path = training_history_dir / filename

        # Prepare training history data
        training_summary = self.get_training_summary()

        # Add metadata
        training_data = {
            "model_name": model_name,
            "model_type": getattr(self.model, "model_type", "unknown"),
            "training_config": training_summary["training_config"],
            "epochs_completed": training_summary["epochs_completed"],
            "training_duration": training_summary["training_duration"],
            "history": self.metrics_history,  # Use raw metrics history instead of epoch-based
            "final_metrics": training_summary["final_metrics"],
            "metadata": {
                "creation_date": str(datetime.now()),
                "dataset_id": dataset_id,
                "experiment_id": experiment_id,
                "tensorflow_version": tf.__version__,
                "total_parameters": self.model.model.count_params()
                if self.model.model
                else None,
            },
        }

        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        training_data = convert_numpy_types(training_data)

        # Save to JSON
        with open(json_path, "w") as f:
            json.dump(training_data, f, indent=2)
        return json_path

    def save_consolidated_training_history(self) -> Optional[Path]:
        """
        Save consolidated training history including test metrics to a single training_history.json file.
        This should be called after both training and evaluation are complete.

        Returns:
            Path to the saved JSON file, or None if no pending save info
        """
        if not self._pending_save_info:
            self.logger.warning("No pending save info - call train() first")
            return None

        training_history_dir = self._pending_save_info["training_history_dir"]
        model_name = self._pending_save_info["model_name"]
        dataset_id = self._pending_save_info["dataset_id"]
        experiment_id = self._pending_save_info["experiment_id"]

        training_history_dir.mkdir(parents=True, exist_ok=True)

        # Use simple filename without timestamp
        json_path = training_history_dir / "training_history.json"

        # Prepare consolidated training history data
        training_summary = self.get_training_summary()

        # Prepare consolidated data with ALL metrics
        consolidated_data = {
            "model_name": model_name,
            "model_type": getattr(self.model, "model_type", "unknown"),
            "training_config": training_summary["training_config"],
            "epochs_completed": training_summary["epochs_completed"],
            "training_duration": training_summary["training_duration"],
            "history": self.metrics_history,  # Contains ALL metrics including test metrics
            "final_metrics": training_summary[
                "final_metrics"
            ],  # Contains final values of all metrics
            "metadata": {
                "creation_date": str(datetime.now()),
                "dataset_id": dataset_id,
                "experiment_id": experiment_id,
                "tensorflow_version": tf.__version__,
                "total_parameters": self.model.model.count_params()
                if self.model.model
                else None,
            },
        }

        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        consolidated_data = convert_numpy_types(consolidated_data)

        # Save to JSON
        with open(json_path, "w") as f:
            json.dump(consolidated_data, f, indent=2)

        self.logger.info(f"Consolidated training history saved to: {json_path}")

        # Clear pending save info
        self._pending_save_info = None

        return json_path

    def train(
        self,
        dataset: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset] = None,
        callbacks: list[tf.keras.callbacks.Callback] = None,
        training_history_dir: Optional[Path] = None,
        model_name: str = "model",
        dataset_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        verbose_training: str = "full",
        save_individual_history: bool = False,
    ) -> dict[str, Any]:
        """
        Train with enhanced metrics tracking and automatic history saving

        Args:
            dataset: Training dataset
            validation_data: Validation dataset (optional)
            callbacks: List of Keras callbacks (optional)
            training_history_dir: Directory to save training history JSON files (optional)
            model_name: Name of the model for metadata and filename
            dataset_id: Optional dataset identifier for metadata
            experiment_id: Optional experiment identifier for metadata
            verbose_training: Training verbosity level
                - "full": Log every epoch (default)
                - "summary": Log every 10th epoch and final epoch
                - "minimal": Log only final epoch
                - "silent": Log only start and end of training
            save_individual_history: If True, save individual timestamped training history file immediately.
                                   If False, defer saving for later consolidated save. Default False.

        Returns:
            Training summary dictionary
        """
        self.logger.info("Starting training with metrics tracking:")

        # Record start time
        self.training_start_time = datetime.now()

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
                self.logger.info(
                    f"Training dataset batch shapes: Features: {features.shape}, Targets: {targets.shape}, Inferred input shape: {input_shape}"
                )
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

        # Check for any invalid final metrics (consistent with evaluation)
        invalid_final_metrics = []
        for metric, values in self.metrics_history.items():
            if values:
                final_value = values[-1]
                if final_value is None or not isinstance(
                    final_value, (int, float, np.number)
                ):
                    invalid_final_metrics.append(f"{metric}={final_value}")
                elif not np.isfinite(final_value):
                    invalid_final_metrics.append(f"{metric}={final_value}")

        if invalid_final_metrics:
            self.logger.warning(
                f"Training completed with invalid final metrics: {', '.join(invalid_final_metrics)}"
            )

        final_metrics_str = ", ".join(
            f"{metric}: {values[-1]:.6f}"
            for metric, values in self.metrics_history.items()
        )
        self.logger.info(f"Training completed. Final metrics: {final_metrics_str}")

        # Handle training history saving based on save_individual_history parameter
        if training_history_dir is not None:
            if save_individual_history:
                # Save individual timestamped file immediately (for evaluation models)
                saved_path = self._save_training_history(
                    training_history_dir, model_name, dataset_id, experiment_id
                )
                self.logger.info(f"Individual training history saved to: {saved_path}")
            else:
                # Store directory for later consolidated save (for foundation models)
                self._pending_save_info = {
                    "training_history_dir": training_history_dir,
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                    "experiment_id": experiment_id,
                }

        return self.get_training_summary()

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
