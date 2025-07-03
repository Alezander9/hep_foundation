from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from hep_foundation.config.logging_config import get_logger
from hep_foundation.models.base_model import BaseModel, ModelConfig


class DNNPredictorConfig(ModelConfig):
    """
    Configuration class for DNN Predictor model.

    This model is designed to predict specific labels from input features,
    with the label selection controlled by a label_index parameter.
    """

    def validate(self) -> None:
        """
        Validate DNN Predictor configuration parameters

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required architecture parameters
        required_arch = [
            "input_shape",
            "output_shape",
            "hidden_layers",
            "label_index",
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

        # Validate label index
        label_index = self.architecture["label_index"]
        if not isinstance(label_index, int) or label_index < 0:
            raise ValueError("label_index must be a non-negative integer")

        # Validate activation functions
        valid_activations = ["relu", "tanh", "sigmoid", "elu", "selu", "linear"]

        if self.architecture["activation"] not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")

        if self.architecture["output_activation"] not in valid_activations:
            raise ValueError(f"output_activation must be one of {valid_activations}")

        # Validate hyperparameters
        if "quant_bits" in self.hyperparameters:
            if not isinstance(self.hyperparameters["quant_bits"], (int, type(None))):
                raise ValueError("quant_bits must be an integer or None")
            if (
                isinstance(self.hyperparameters["quant_bits"], int)
                and self.hyperparameters["quant_bits"] < 1
            ):
                raise ValueError("quant_bits must be positive")

        # Optional hyperparameters validation
        if "dropout_rate" in self.hyperparameters:
            dropout_rate = self.hyperparameters["dropout_rate"]
            if not isinstance(dropout_rate, (float, type(None))) or (
                isinstance(dropout_rate, float) and not 0 <= dropout_rate <= 1
            ):
                raise ValueError(
                    "dropout_rate must be a float between 0 and 1, or None"
                )

        if "l2_regularization" in self.hyperparameters:
            l2_reg = self.hyperparameters["l2_regularization"]
            if not isinstance(l2_reg, (float, type(None))) or (
                isinstance(l2_reg, float) and l2_reg < 0
            ):
                raise ValueError(
                    "l2_regularization must be a non-negative float or None"
                )

    @staticmethod
    def get_template() -> dict:
        """
        Get a template configuration for the DNN Predictor

        Returns:
            Dictionary containing the template configuration
        """
        return {
            "model_type": "dnn_predictor",
            "architecture": {
                "input_shape": None,  # To be filled based on dataset
                "output_shape": None,  # To be filled based on label shape
                "hidden_layers": [128, 64, 32],
                "label_index": 0,  # Default to first label set
                "activation": "relu",
                "output_activation": "linear",
                "name": "dnn_predictor",
            },
            "hyperparameters": {
                "quant_bits": None,
                "dropout_rate": None,
                "l2_regularization": None,
            },
        }


class DNNPredictor(BaseModel):
    """Deep Neural Network for predicting specific labels from input features"""

    def __init__(self, config: DNNPredictorConfig):
        """
        Initialize DNNPredictor

        Args:
            config: DNNPredictorConfig object containing model configuration
        """
        super().__init__()
        self.logger = get_logger(__name__)

        # Extract configuration parameters
        self.input_shape = config.architecture["input_shape"]
        self.output_shape = config.architecture["output_shape"]
        self.hidden_layers = config.architecture["hidden_layers"]
        self.label_index = config.architecture["label_index"]
        self.activation = config.architecture["activation"]
        self.output_activation = config.architecture["output_activation"]
        self.name = config.architecture.get("name", "dnn_predictor")

        # Hyperparameters
        self.quant_bits = config.hyperparameters.get("quant_bits")
        self.dropout_rate = config.hyperparameters.get("dropout_rate")
        self.l2_regularization = config.hyperparameters.get("l2_regularization")

    def build(self, input_shape: tuple = None) -> None:
        """Build the DNN model"""
        if input_shape is None:
            input_shape = self.input_shape

        # Input layer
        inputs = keras.Input(shape=input_shape, name="input_layer")

        # Flatten the input to combine tracks and features
        x = keras.layers.Reshape((-1,))(inputs)

        # Add hidden layers
        for i, units in enumerate(self.hidden_layers):
            # Add dense layer
            x = keras.layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(self.l2_regularization)
                if self.l2_regularization
                else None,
                name=f"dense_{i}",
            )(x)

            # Add batch normalization
            x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)

            # Add dropout if specified
            if self.dropout_rate:
                x = keras.layers.Dropout(self.dropout_rate, name=f"dropout_{i}")(x)

        # Output layer
        x = keras.layers.Dense(
            np.prod(self.output_shape),
            activation=self.output_activation,
            name="output_dense",
        )(x)

        # Reshape output to match expected shape
        outputs = keras.layers.Reshape(self.output_shape, name="output_reshape")(x)

        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)

    def get_config(self) -> dict:
        return {
            "model_type": "dnn_predictor",
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "hidden_layers": self.hidden_layers,
            "label_index": self.label_index,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "quant_bits": self.quant_bits,
            "dropout_rate": self.dropout_rate,
            "l2_regularization": self.l2_regularization,
            "name": self.name,
        }

    def create_plots(self, plots_dir: Path) -> None:
        """Create predictor-specific visualization plots"""
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot model architecture
        tf.keras.utils.plot_model(
            self.model,
            to_file=str(plots_dir / "model_architecture.png"),
            show_shapes=True,
            show_layer_names=True,
        )

        # Add prediction vs. truth scatter plots if we have validation data
        if hasattr(self, "_validation_data"):
            # Implementation for prediction vs. truth plots
            pass
