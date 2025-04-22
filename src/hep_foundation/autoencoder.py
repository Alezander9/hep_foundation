import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from qkeras import QActivation, QDense, quantized_bits, quantized_relu
from tensorflow import keras

from hep_foundation.base_model import BaseModel, ModelConfig
from hep_foundation.logging_config import setup_logging


class AutoEncoderConfig(ModelConfig):
    """Configuration class for Autoencoder"""

    def validate(self) -> None:
        """
        Validate Autoencoder configuration parameters

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required architecture parameters
        required_arch = [
            "input_shape",
            "latent_dim",
            "encoder_layers",
            "decoder_layers",
        ]
        for param in required_arch:
            if param not in self.architecture:
                raise ValueError(f"Missing required architecture parameter: {param}")

        # Validate architecture parameter values
        if self.architecture["latent_dim"] < 1:
            raise ValueError("latent_dim must be positive")

        if (
            not isinstance(self.architecture["encoder_layers"], list)
            or not self.architecture["encoder_layers"]
        ):
            raise ValueError("encoder_layers must be a non-empty list")

        if (
            not isinstance(self.architecture["decoder_layers"], list)
            or not self.architecture["decoder_layers"]
        ):
            raise ValueError("decoder_layers must be a non-empty list")

        if not isinstance(self.architecture["input_shape"], (tuple, list)):
            raise ValueError("input_shape must be a tuple or list")

        # Validate layer sizes
        for i, size in enumerate(self.architecture["encoder_layers"]):
            if not isinstance(size, int) or size < 1:
                raise ValueError(f"encoder_layers[{i}] must be a positive integer")

        for i, size in enumerate(self.architecture["decoder_layers"]):
            if not isinstance(size, int) or size < 1:
                raise ValueError(f"decoder_layers[{i}] must be a positive integer")

        # Validate activation function
        if "activation" in self.architecture:
            valid_activations = ["relu", "tanh", "sigmoid", "elu", "selu"]
            if self.architecture["activation"] not in valid_activations:
                raise ValueError(f"activation must be one of {valid_activations}")

        # Validate normalize_latent
        if "normalize_latent" in self.architecture:
            if not isinstance(self.architecture["normalize_latent"], bool):
                raise ValueError("normalize_latent must be a boolean")

        # Validate hyperparameters
        if "quant_bits" in self.hyperparameters:
            if not isinstance(self.hyperparameters["quant_bits"], (int, type(None))):
                raise ValueError("quant_bits must be an integer or None")
            if (
                isinstance(self.hyperparameters["quant_bits"], int)
                and self.hyperparameters["quant_bits"] < 1
            ):
                raise ValueError("quant_bits must be positive")


class AutoEncoder(BaseModel):
    def __init__(self, config: AutoEncoderConfig = None, **kwargs):
        """
        Initialize AutoEncoder

        Args:
            config: AutoEncoderConfig object containing model configuration
            **kwargs: Alternative way to pass configuration parameters directly
        """
        super().__init__()
        setup_logging()

        self.input_shape = config.architecture["input_shape"]
        self.latent_dim = config.architecture["latent_dim"]
        self.encoder_layers = config.architecture["encoder_layers"]
        self.decoder_layers = config.architecture["decoder_layers"]
        self.activation = config.architecture.get("activation", "relu")
        self.normalize_latent = config.architecture.get("normalize_latent", False)
        self.quant_bits = config.hyperparameters.get("quant_bits")
        self.name = config.architecture.get("name", "track_autoencoder")

    def build(self, input_shape: tuple = None) -> None:
        """Build encoder and decoder networks"""

        # Input layer - now accepts 3D input (batch_size, n_tracks, n_features)
        inputs = keras.Input(shape=input_shape, name="input_layer")

        # Flatten the input to combine tracks and features
        x = keras.layers.Reshape((-1,))(
            inputs
        )  # Flatten to (batch_size, n_tracks * n_features)

        # Create encoder layers
        for i, units in enumerate(self.encoder_layers):
            x = self._add_dense_block(x, units, f"encoder_{i}")

        # Latent layer
        if self.quant_bits:
            latent = QDense(
                self.latent_dim,
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name="latent_layer",
            )(x)
        else:
            latent = keras.layers.Dense(self.latent_dim, name="latent_layer")(x)

        # Optionally normalize latent space
        if self.normalize_latent:
            latent = keras.layers.BatchNormalization(name="latent_normalization")(
                latent
            )

        # Decoder
        x = latent
        for i, units in enumerate(self.decoder_layers):
            x = self._add_dense_block(x, units, f"decoder_{i}")

        # Output layer - reshape back to original dimensions
        if self.quant_bits:
            x = QDense(
                np.prod(input_shape),  # Multiply dimensions to get total size
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name="output_dense",
            )(x)
        else:
            x = keras.layers.Dense(np.prod(input_shape), name="output_dense")(x)

        # Reshape back to 3D
        outputs = keras.layers.Reshape(input_shape, name="output_reshape")(x)

        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)

    def _add_dense_block(self, x, units: int, prefix: str):
        """Helper to add a dense block with activation and batch norm"""
        if self.quant_bits:
            x = QDense(
                units,
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name=f"{prefix}_dense",
            )(x)
            x = QActivation(
                quantized_relu(self.quant_bits), name=f"{prefix}_activation"
            )(x)
        else:
            x = keras.layers.Dense(units, name=f"{prefix}_dense")(x)
            x = keras.layers.Activation(self.activation, name=f"{prefix}_activation")(x)

        return keras.layers.BatchNormalization(name=f"{prefix}_bn")(x)

    def get_config(self) -> dict:
        return {
            "model_type": "autoencoder",
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "quant_bits": self.quant_bits,
            "activation": self.activation,
            "normalize_latent": self.normalize_latent,
            "name": self.name,
        }

    def create_plots(self, plots_dir: Path) -> None:
        """Create autoencoder-specific plots"""
        # For autoencoder, we might want to show:
        # 1. Latent space distributions
        # 2. Reconstruction examples
        # 3. Loss components if using custom loss

        logging.info("Creating autoencoder-specific plots...")

        # Example: Plot model architecture
        tf.keras.utils.plot_model(
            self.model,
            to_file=str(plots_dir / "model_architecture.pdf"),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
        )

        logging.info("Created model architecture plot")

        # Could add more autoencoder-specific visualizations:
        # - Latent space clustering
        # - Reconstruction quality examples
        # - Feature-wise reconstruction errors
