"""
Standalone Model Trainer for HEP Foundation Pipeline.

This module provides training functionality for standalone DNN models
with advanced features like learning rate scheduling and gradient clipping.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from hep_foundation.config.logging_config import get_logger
from hep_foundation.standalone_models.standalone_config import StandaloneTrainingConfig


class StandaloneTrainer:
    """
    Trainer class for standalone DNN models with advanced training features.

    Includes support for:
    - Learning rate scheduling (ReduceLROnPlateau, ExponentialDecay, CosineDecay)
    - Gradient clipping
    - Early stopping
    - Comprehensive training monitoring
    """

    def _make_json_serializable(self, obj):
        """Convert numpy/tensorflow objects to JSON serializable types."""
        import numpy as np
        import tensorflow as tf

        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tf.TensorShape):
            return obj.as_list()
        elif hasattr(obj, "numpy"):  # TensorFlow tensor
            return obj.numpy().tolist()
        else:
            return obj

    def __init__(
        self,
        model: tf.keras.Model,
        training_config: StandaloneTrainingConfig,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: Optional[tf.keras.losses.Loss] = None,
        metrics: Optional[list[Union[str, tf.keras.metrics.Metric]]] = None,
    ):
        """
        Initialize StandaloneTrainer.

        Args:
            model: Keras model to train
            training_config: Training configuration
            optimizer: Keras optimizer (default: Adam)
            loss: Keras loss function (default: MeanSquaredError)
            metrics: List of metrics to track (default: ['mse', 'mae'])
        """
        self.logger = get_logger(__name__)
        self.model = model
        self.training_config = training_config

        # Set up optimizer with gradient clipping
        self.optimizer = self._create_optimizer(optimizer)

        # Set up loss and metrics
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        self.metrics = metrics or ["mse", "mae"]

        # Training history
        self.history = None
        self.metrics_history = {}

        self.logger.info("StandaloneTrainer initialized")
        self.logger.info(f"Model: {model.name}")
        self.logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        self.logger.info(f"Loss: {type(self.loss).__name__}")

    def _create_optimizer(
        self, optimizer: Optional[tf.keras.optimizers.Optimizer]
    ) -> tf.keras.optimizers.Optimizer:
        """Create optimizer with optional gradient clipping."""
        if optimizer is not None:
            return optimizer

        # Create Adam optimizer with optional gradient clipping
        optimizer_kwargs = {
            "learning_rate": self.training_config.learning_rate,
        }

        # Add gradient clipping if specified
        if self.training_config.gradient_clip_norm is not None:
            optimizer_kwargs["clipnorm"] = self.training_config.gradient_clip_norm
            self.logger.info(
                f"Gradient clipping enabled with norm={self.training_config.gradient_clip_norm}"
            )

        return tf.keras.optimizers.Adam(**optimizer_kwargs)

    def _create_lr_scheduler_callback(self) -> Optional[tf.keras.callbacks.Callback]:
        """
        Create learning rate scheduler callback based on configuration.

        Returns:
            Keras callback for learning rate scheduling or None if not configured
        """
        if not self.training_config.lr_scheduler:
            return None

        lr_config = self.training_config.lr_scheduler
        scheduler_type = lr_config.get("type", "reduce_on_plateau")

        if scheduler_type == "reduce_on_plateau":
            callback = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=lr_config.get("monitor", "val_loss"),
                factor=lr_config.get("factor", 0.5),
                patience=lr_config.get("patience", 10),
                min_lr=lr_config.get("min_lr", 1e-6),
                verbose=int(lr_config.get("verbose", True)),
                mode=lr_config.get("mode", "min"),
                min_delta=lr_config.get("min_delta", 1e-4),
                cooldown=lr_config.get("cooldown", 0),
            )
            self.logger.info(
                f"Created ReduceLROnPlateau scheduler: monitor={lr_config.get('monitor', 'val_loss')}, "
                f"factor={lr_config.get('factor', 0.5)}, patience={lr_config.get('patience', 10)}"
            )

        elif scheduler_type == "exponential_decay":
            # Create learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.training_config.learning_rate,
                decay_steps=lr_config.get("decay_steps", 1000),
                decay_rate=lr_config.get("decay_rate", 0.9),
                staircase=lr_config.get("staircase", False),
            )
            # Update optimizer with new schedule
            self.optimizer.learning_rate = lr_schedule
            self.logger.info(
                f"Created ExponentialDecay scheduler: decay_steps={lr_config.get('decay_steps', 1000)}, "
                f"decay_rate={lr_config.get('decay_rate', 0.9)}"
            )
            return None  # No callback needed for schedule-based LR

        elif scheduler_type == "cosine_decay":
            # Create learning rate schedule
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.training_config.learning_rate,
                decay_steps=lr_config.get("decay_steps", 10000),
                alpha=lr_config.get("alpha", 0.0),
            )
            # Update optimizer with new schedule
            self.optimizer.learning_rate = lr_schedule
            self.logger.info(
                f"Created CosineDecay scheduler: decay_steps={lr_config.get('decay_steps', 10000)}"
            )
            return None  # No callback needed for schedule-based LR

        else:
            self.logger.warning(
                f"Unknown scheduler type: {scheduler_type}. Skipping LR scheduling."
            )
            return None

        return callback

    def _create_callbacks(
        self,
        validation_data: Optional[tf.data.Dataset] = None,
        custom_callbacks: Optional[list[tf.keras.callbacks.Callback]] = None,
    ) -> list[tf.keras.callbacks.Callback]:
        """
        Create training callbacks including early stopping and learning rate scheduling.

        Args:
            validation_data: Validation dataset for monitoring
            custom_callbacks: Additional custom callbacks

        Returns:
            List of Keras callbacks
        """
        callbacks = custom_callbacks or []

        # Early stopping callback
        if validation_data is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.training_config.early_stopping_patience,
                min_delta=self.training_config.early_stopping_min_delta,
                restore_best_weights=True,
                verbose=1,
            )
            callbacks.append(early_stopping)
            self.logger.info(
                f"Added EarlyStopping: patience={self.training_config.early_stopping_patience}, "
                f"min_delta={self.training_config.early_stopping_min_delta}"
            )

        # Learning rate scheduler callback
        lr_callback = self._create_lr_scheduler_callback()
        if lr_callback is not None:
            callbacks.append(lr_callback)

        # Learning rate logging callback
        lr_logger = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr,  # Don't change LR, just log it
            verbose=0,
        )
        callbacks.append(lr_logger)

        return callbacks

    def train(
        self,
        dataset: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset] = None,
        callbacks: Optional[list[tf.keras.callbacks.Callback]] = None,
        training_history_dir: Optional[Path] = None,
        model_name: str = "standalone_dnn",
        dataset_id: str = "unknown",
        experiment_id: str = "standalone_regression",
        verbose: str = "auto",
        save_individual_history: bool = True,
    ) -> bool:
        """
        Train the standalone model.

        Args:
            dataset: Training dataset
            validation_data: Validation dataset
            callbacks: Additional callbacks
            training_history_dir: Directory to save training history
            model_name: Name of the model for logging
            dataset_id: Dataset identifier
            experiment_id: Experiment identifier
            verbose: Training verbosity
            save_individual_history: Whether to save individual training history

        Returns:
            True if training succeeded, False otherwise
        """
        self.logger.info(f"Starting training for {model_name}")
        self.logger.info(f"Training configuration: {self.training_config.to_dict()}")

        try:
            # Compile model
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=self.metrics,
            )

            # Create callbacks
            training_callbacks = self._create_callbacks(validation_data, callbacks)

            # Train model
            self.logger.info(f"Training for {self.training_config.epochs} epochs...")
            history = self.model.fit(
                dataset,
                validation_data=validation_data,
                epochs=self.training_config.epochs,
                callbacks=training_callbacks,
                verbose=1 if verbose == "auto" else verbose,
            )

            # Store training history
            self.history = history
            self._update_metrics_history(history)

            # Check for training issues
            self._validate_training_results()

            # Save training history if requested
            if save_individual_history and training_history_dir:
                self._save_training_history(
                    training_history_dir, model_name, dataset_id, experiment_id
                )

            self.logger.info(f"Training completed successfully for {model_name}")
            return True

        except Exception as e:
            self.logger.error(
                f"Training failed for {model_name}: {type(e).__name__}: {str(e)}"
            )
            return False

    def _update_metrics_history(self, history: tf.keras.callbacks.History) -> None:
        """Update internal metrics history with training results."""
        # Convert numpy values to Python floats for JSON serialization
        for metric, values in history.history.items():
            self.metrics_history[metric] = [float(v) for v in values]

    def _validate_training_results(self) -> None:
        """Check for invalid final metrics and log warnings."""
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
        else:
            # Log final metrics
            final_metrics_str = ", ".join(
                f"{metric}: {values[-1]:.6f}"
                for metric, values in self.metrics_history.items()
                if values
            )
            self.logger.info(f"Final metrics - {final_metrics_str}")

    def _save_training_history(
        self,
        history_dir: Path,
        model_name: str,
        dataset_id: str,
        experiment_id: str,
    ) -> None:
        """Save training history to JSON file."""
        try:
            history_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with model and dataset info
            filename = f"{model_name}_{dataset_id}.json"
            history_path = history_dir / filename

            # Prepare history data
            history_data = {
                "experiment_id": experiment_id,
                "model_name": model_name,
                "dataset_id": dataset_id,
                "timestamp": str(datetime.now()),
                "training_config": self.training_config.to_dict(),
                "history": self.metrics_history,
                "model_summary": {
                    "total_params": self.model.count_params(),
                    "trainable_params": sum(
                        [
                            tf.keras.backend.count_params(w)
                            for w in self.model.trainable_weights
                        ]
                    ),
                },
            }

            # Save to file
            with open(history_path, "w") as f:
                json.dump(self._make_json_serializable(history_data), f, indent=2)

            self.logger.info(f"Training history saved to: {history_path}")

        except Exception as e:
            self.logger.error(f"Failed to save training history: {e}")

    def evaluate(
        self,
        dataset: tf.data.Dataset,
        verbose: str = "auto",
    ) -> dict[str, float]:
        """
        Evaluate the trained model.

        Args:
            dataset: Dataset to evaluate on
            verbose: Evaluation verbosity

        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating model...")

        try:
            # Evaluate model
            results = self.model.evaluate(
                dataset,
                verbose=1 if verbose == "auto" else verbose,
                return_dict=True,
            )

            # Log results
            results_str = ", ".join(f"{k}: {v:.6f}" for k, v in results.items())
            self.logger.info(f"Evaluation results - {results_str}")

            return results

        except Exception as e:
            self.logger.error(f"Evaluation failed: {type(e).__name__}: {str(e)}")
            return {}

    def predict(
        self,
        dataset: tf.data.Dataset,
        verbose: str = "auto",
    ) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            dataset: Dataset to predict on
            verbose: Prediction verbosity

        Returns:
            Array of predictions
        """
        self.logger.info("Generating predictions...")

        try:
            predictions = self.model.predict(
                dataset,
                verbose=1 if verbose == "auto" else verbose,
            )

            self.logger.info(f"Generated predictions with shape: {predictions.shape}")
            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {type(e).__name__}: {str(e)}")
            return np.array([])

    def get_training_history(self) -> dict[str, list[float]]:
        """Get training history metrics."""
        return self.metrics_history.copy()

    def save_model(self, model_path: Path, save_format: str = "tf") -> bool:
        """
        Save the trained model.

        Args:
            model_path: Path to save the model
            save_format: Format to save ('tf' or 'h5')

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(model_path, save_format=save_format)
            self.logger.info(f"Model saved to: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
