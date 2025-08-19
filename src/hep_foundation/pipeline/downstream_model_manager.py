import copy
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import tensorflow as tf

from hep_foundation.config.logging_config import get_logger
from hep_foundation.models.base_model import CustomKerasModelWrapper
from hep_foundation.models.dnn_predictor import DNNPredictorConfig
from hep_foundation.models.model_factory import ModelFactory
from hep_foundation.training.model_trainer import ModelTrainer


class DownstreamModelType(Enum):
    """Types of downstream models that can be created."""

    FROM_SCRATCH = "from_scratch"
    FINE_TUNED = "fine_tuned"
    FIXED_ENCODER = "fixed_encoder"


class DownstreamModelManager:
    """
    Manages creation and training of downstream models for foundation model evaluation.

    This class provides shared functionality for both regression and signal classification
    evaluation tasks, reducing code duplication and improving maintainability.
    """

    def __init__(self, logger=None):
        """
        Initialize the downstream model manager.

        Args:
            logger: Logger instance (optional, will create one if not provided)
        """
        self.logger = logger or get_logger(__name__)

    def create_subset_dataset(
        self,
        dataset: tf.data.Dataset,
        num_events: int,
        batch_size: int,
        data_size_index: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Create a subset of the dataset with exactly num_events events.

        Args:
            dataset: The full dataset to subset
            num_events: Number of events to include in the subset
            batch_size: Batch size for the resulting dataset
            data_size_index: Index for seeding (ensures different subsets for different sizes)

        Returns:
            Subset dataset with the specified number of events
        """
        # Convert to unbatched dataset to count events precisely
        unbatched = dataset.unbatch()
        # Use different seed for each data size to ensure independent sampling
        # This prevents smaller datasets from being strict subsets of larger ones
        seed = 42 + (data_size_index or 0)
        shuffled = unbatched.shuffle(buffer_size=50000, seed=seed)
        subset = shuffled.take(num_events)
        # Rebatch with original batch size
        return subset.batch(batch_size)

    def build_model_head(
        self,
        dnn_model_config: DNNPredictorConfig,
        input_dim: int,
        output_shape: tuple[int, ...],
        output_activation: Optional[str] = None,
        name_suffix: str = "",
    ) -> tf.keras.Model:
        """
        Build a model head (regressor or classifier) with specified architecture.

        Args:
            dnn_model_config: Base DNN configuration to modify
            input_dim: Input dimension (latent space size)
            output_shape: Output shape for the head
            output_activation: Output activation function (e.g., 'sigmoid' for classification)
            name_suffix: Suffix to add to the model name

        Returns:
            Keras model for the head
        """
        # Create a copy of the DNN config with modified architecture
        head_config = copy.deepcopy(dnn_model_config)

        # Update the architecture for the head
        architecture_updates = {
            "input_shape": (input_dim,),
            "output_shape": output_shape,
            "name": f"{dnn_model_config.architecture.get('name', 'head')}_{name_suffix}",
        }

        if output_activation is not None:
            architecture_updates["output_activation"] = output_activation

        head_config.architecture.update(architecture_updates)

        head_model_wrapper = ModelFactory.create_model(
            model_type="dnn_predictor", config=head_config
        )
        head_model_wrapper.build()
        return head_model_wrapper.model

    def create_downstream_model(
        self,
        model_type: DownstreamModelType,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        pretrained_encoder: Optional[tf.keras.Model],
        encoder_hidden_layers: list[int],
        latent_dim: int,
        dnn_model_config: DNNPredictorConfig,
        encoder_activation: str = "relu",
        output_activation: Optional[str] = None,
    ) -> tf.keras.Model:
        """
        Create a downstream model of the specified type.

        Args:
            model_type: Type of model to create (FROM_SCRATCH, FINE_TUNED, or FIXED_ENCODER)
            input_shape: Input shape for the model
            output_shape: Output shape for the model
            pretrained_encoder: Pre-trained encoder (required for FINE_TUNED and FIXED_ENCODER)
            encoder_hidden_layers: List of hidden layer sizes for the encoder
            latent_dim: Latent space dimension
            dnn_model_config: Configuration for the DNN head
            encoder_activation: Activation function for encoder layers
            output_activation: Output activation function (e.g., 'sigmoid' for classification)

        Returns:
            Complete downstream model
        """
        model_inputs = tf.keras.Input(
            shape=input_shape, name=f"input_features_{model_type.value}"
        )

        if model_type == DownstreamModelType.FROM_SCRATCH:
            # Build encoder from scratch
            encoder_layers = []
            for units in encoder_hidden_layers:
                encoder_layers.append(
                    tf.keras.layers.Dense(units, activation=encoder_activation)
                )
            encoder_layers.append(
                tf.keras.layers.Dense(latent_dim, name="scratch_latent_space")
            )

            encoder_part = tf.keras.Sequential(encoder_layers, name="scratch_encoder")
            encoded = encoder_part(model_inputs)

        elif model_type == DownstreamModelType.FINE_TUNED:
            if pretrained_encoder is None:
                raise ValueError("Pretrained encoder is required for fine-tuned models")

            # Use pretrained encoder with training enabled
            encoded = pretrained_encoder(model_inputs)

            # Add dtype casting to ensure compatibility with QKeras layers
            encoded = tf.keras.layers.Lambda(
                lambda x: tf.cast(x, tf.float32), name=f"dtype_cast_{model_type.value}"
            )(encoded)

            # Create encoder wrapper to control trainability
            encoder_part = tf.keras.Model(
                inputs=model_inputs,
                outputs=encoded,
                name=f"{model_type.value}_pretrained_encoder",
            )
            encoder_part.trainable = True
            encoded = encoder_part(model_inputs)

        elif model_type == DownstreamModelType.FIXED_ENCODER:
            if pretrained_encoder is None:
                raise ValueError(
                    "Pretrained encoder is required for fixed encoder models"
                )

            # Use pretrained encoder with training disabled
            encoded = pretrained_encoder(model_inputs)

            # Add dtype casting to ensure compatibility with QKeras layers
            encoded = tf.keras.layers.Lambda(
                lambda x: tf.cast(x, tf.float32), name=f"dtype_cast_{model_type.value}"
            )(encoded)

            # Create encoder wrapper to control trainability
            encoder_part = tf.keras.Model(
                inputs=model_inputs,
                outputs=encoded,
                name=f"{model_type.value}_pretrained_encoder",
            )
            encoder_part.trainable = False
            encoded = encoder_part(model_inputs)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Build the task-specific head
        head_model = self.build_model_head(
            dnn_model_config=dnn_model_config,
            input_dim=latent_dim,
            output_shape=output_shape,
            output_activation=output_activation,
            name_suffix=model_type.value,
        )

        # Connect encoder and head
        predictions = head_model(encoded)

        # Create the complete model
        task_name = "Classifier" if output_activation == "sigmoid" else "Regressor"
        model_name = f"{task_name}_{model_type.value.replace('_', ' ').title().replace(' ', '_')}"

        complete_model = tf.keras.Model(
            inputs=model_inputs,
            outputs=predictions,
            name=model_name,
        )

        return complete_model

    def train_and_evaluate_model(
        self,
        model: tf.keras.Model,
        model_name: str,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        training_config: dict[str, Any],
        fixed_epochs: int,
        data_size: int,
        output_dir: Path,
        save_training_history: bool = False,
        verbose_training: str = "minimal",
        return_accuracy: bool = False,
    ) -> tuple[float, ...]:
        """
        Train and evaluate a downstream model.

        Args:
            model: Keras model to train
            model_name: Name for the model (used in logging and file naming)
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            training_config: Training configuration dictionary
            fixed_epochs: Number of epochs to train for
            data_size: Size of training data (for logging)
            output_dir: Directory to save training histories
            save_training_history: Whether to save training history
            verbose_training: Verbosity level for training
            return_accuracy: Whether to return accuracy metric (for classification)

        Returns:
            Tuple of (test_loss,) for regression or (test_loss, test_accuracy) for classification
        """
        self.logger.info(f"Training {model_name} model with {data_size} events...")

        # Wrap the Keras model with CustomKerasModelWrapper for ModelTrainer
        wrapped_model = CustomKerasModelWrapper(model, name=model_name)

        # Prepare training config with fixed epochs
        trainer_config = {
            "batch_size": training_config.get("batch_size", 32),
            "epochs": fixed_epochs,
            "learning_rate": training_config.get("learning_rate", 0.001),
            "early_stopping": {
                "patience": fixed_epochs + 1,  # Disable early stopping
                "min_delta": 0,
            },
        }

        trainer = ModelTrainer(model=wrapped_model, training_config=trainer_config)

        # Train the model
        _ = trainer.train(
            dataset=train_dataset,
            validation_data=val_dataset,
            callbacks=[],  # No callbacks for speed
            training_history_dir=output_dir / "training_histories"
            if save_training_history
            else None,
            model_name=model_name,
            dataset_id=f"downstream_eval_{data_size}",
            experiment_id="downstream_evaluation",
            verbose_training=verbose_training,
            save_individual_history=True,  # Save individual files for comparison plots
        )

        # Evaluate on test set
        test_metrics = trainer.evaluate(test_dataset)
        test_loss = test_metrics.get("test_loss", test_metrics.get("test_mse", 0.0))

        if return_accuracy:
            test_accuracy = test_metrics.get("test_binary_accuracy", 0.0)
            self.logger.info(
                f"{model_name} with {data_size} events - Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}"
            )
            return test_loss, test_accuracy
        else:
            self.logger.info(
                f"{model_name} with {data_size} events - Test Loss: {test_loss:.6f}"
            )
            return (test_loss,)
