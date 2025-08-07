"""
Standalone DNN Regressor Model for HEP Foundation Pipeline.

This module provides a standalone deep neural network model for regression tasks,
designed to work independently without requiring a foundation model.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from hep_foundation.config.logging_config import get_logger
from hep_foundation.models.base_model import BaseModel, ModelConfig


class StandaloneDNNConfig(ModelConfig):
    """
    Configuration class for Standalone DNN Regressor model.

    This model is designed for direct regression tasks without foundation model pretraining.
    """

    def validate(self) -> None:
        """
        Validate Standalone DNN configuration parameters

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required architecture parameters
        required_arch = [
            "input_shape",
            "output_shape",
            "hidden_layers",
            "activation",
            "output_activation",
        ]
        for param in required_arch:
            if param not in self.architecture:
                raise ValueError(f"Missing required architecture parameter: {param}")

        # Validate shapes
        if not isinstance(self.architecture["input_shape"], (tuple, list)):
            raise ValueError("input_shape must be a tuple or list")

        if not isinstance(self.architecture["output_shape"], (tuple, list)):
            raise ValueError("output_shape must be a tuple or list")

        # Validate hidden layers
        if not isinstance(self.architecture["hidden_layers"], list):
            raise ValueError("hidden_layers must be a list")

        for i, size in enumerate(self.architecture["hidden_layers"]):
            if not isinstance(size, int) or size < 1:
                raise ValueError(f"hidden_layers[{i}] must be a positive integer")

        # Validate activation functions
        valid_activations = ["relu", "tanh", "sigmoid", "elu", "selu", "linear"]

        if self.architecture["activation"] not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")

        if self.architecture["output_activation"] not in valid_activations:
            raise ValueError(f"output_activation must be one of {valid_activations}")

        # Validate hyperparameters
        if "dropout_rate" in self.hyperparameters:
            dropout_rate = self.hyperparameters["dropout_rate"]
            if not isinstance(dropout_rate, (int, float)) or not (
                0 <= dropout_rate < 1
            ):
                raise ValueError("dropout_rate must be a number between 0 and 1")

        if "l2_regularization" in self.hyperparameters:
            l2_reg = self.hyperparameters["l2_regularization"]
            if not isinstance(l2_reg, (int, float)) or l2_reg < 0:
                raise ValueError("l2_regularization must be a non-negative number")

        if "batch_normalization" in self.hyperparameters:
            if not isinstance(self.hyperparameters["batch_normalization"], bool):
                raise ValueError("batch_normalization must be a boolean")


class StandaloneDNNRegressor(BaseModel):
    """Standalone Deep Neural Network for regression tasks"""

    def __init__(self, config: StandaloneDNNConfig):
        """
        Initialize StandaloneDNNRegressor

        Args:
            config: StandaloneDNNConfig object containing model configuration
        """
        super().__init__()
        self.logger = get_logger(__name__)

        # Extract configuration parameters
        self.input_shape = config.architecture["input_shape"]
        self.output_shape = config.architecture["output_shape"]
        self.hidden_layers = config.architecture["hidden_layers"]
        self.activation = config.architecture.get("activation", "relu")
        self.output_activation = config.architecture.get("output_activation", "linear")
        self.name = config.architecture.get("name", "standalone_dnn")

        # Extract hyperparameters
        self.dropout_rate = config.hyperparameters.get("dropout_rate", 0.0)
        self.l2_regularization = config.hyperparameters.get("l2_regularization", 0.0)
        self.batch_normalization = config.hyperparameters.get(
            "batch_normalization", False
        )

        # Will be set during build
        self.model = None

    def build(self, input_shape: tuple = None) -> None:
        """Build the standalone DNN architecture"""
        if input_shape is None:
            input_shape = self.input_shape

        self.logger.info("Building standalone DNN regressor...")
        self.logger.info(f"Input shape: {input_shape}")
        self.logger.info(f"Hidden layers: {self.hidden_layers}")
        self.logger.info(f"Output shape: {self.output_shape}")

        # Input layer
        inputs = keras.Input(shape=input_shape, name="regressor_input")

        # Flatten input if needed
        x = keras.layers.Flatten(name="flatten")(inputs)

        # Add hidden layers
        for i, units in enumerate(self.hidden_layers):
            # Dense layer with optional L2 regularization
            if self.l2_regularization > 0:
                x = keras.layers.Dense(
                    units,
                    activation=None,  # Activation added separately
                    kernel_regularizer=keras.regularizers.l2(self.l2_regularization),
                    name=f"dense_{i}",
                )(x)
            else:
                x = keras.layers.Dense(units, activation=None, name=f"dense_{i}")(x)

            # Batch normalization (optional)
            if self.batch_normalization:
                x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)

            # Activation
            x = keras.layers.Activation(self.activation, name=f"activation_{i}")(x)

            # Dropout (optional)
            if self.dropout_rate > 0:
                x = keras.layers.Dropout(self.dropout_rate, name=f"dropout_{i}")(x)

        self.logger.info("Built hidden layers")

        # Output layer
        output_units = np.prod(self.output_shape)
        x = keras.layers.Dense(
            output_units, activation=self.output_activation, name="output_dense"
        )(x)

        # Reshape output to match expected shape
        if len(self.output_shape) > 1:
            outputs = keras.layers.Reshape(self.output_shape, name="output_reshape")(x)
        else:
            outputs = x

        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        self.logger.info("Completed standalone DNN architecture build")
        self.logger.info(f"Total parameters: {self.model.count_params():,}")

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            "model_type": "standalone_dnn_regressor",
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "dropout_rate": self.dropout_rate,
            "l2_regularization": self.l2_regularization,
            "batch_normalization": self.batch_normalization,
            "name": self.name,
        }

    def create_plots(
        self, plots_dir: Path, training_history_json_path: Optional[Path] = None
    ) -> None:
        """Create standalone DNN regressor-specific plots"""
        self.logger.info("Creating standalone DNN regressor plots...")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot model architecture
        try:
            tf.keras.utils.plot_model(
                self.model,
                to_file=str(plots_dir / "standalone_dnn_architecture.png"),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )
            self.logger.info("Created model architecture plot")
        except Exception as e:
            self.logger.warning(f"Failed to create architecture plot: {e}")

        # Training history plots will be handled by StandalonePlotManager
        self.logger.info(f"Standalone DNN plots saved to: {plots_dir}")

    def summary(self) -> None:
        """Print model summary"""
        if self.model is not None:
            self.model.summary()
        else:
            self.logger.warning("Model not built yet. Call build() first.")
