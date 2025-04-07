from typing import List, Optional
from tensorflow import keras
from qkeras import QDense, QActivation, quantized_bits, quantized_relu
import numpy as np
from pathlib import Path
import tensorflow as tf
import logging
from hep_foundation.logging_config import setup_logging
from hep_foundation.base_model import BaseModel

class AutoEncoder(BaseModel):
    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int,
        encoder_layers: List[int],
        decoder_layers: List[int],
        quant_bits: Optional[int] = None,
        activation: str = 'relu',
        normalize_latent: bool = False,
        name: str = 'track_autoencoder'
    ):
        super().__init__()
        # Setup logging
        setup_logging()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.quant_bits = quant_bits
        self.activation = activation
        self.normalize_latent = normalize_latent
        self.name = name
        
    def build(self, input_shape: tuple = None) -> None:
        """Build encoder and decoder networks"""
        if input_shape is None:
            input_shape = self.input_shape
            
        # Input layer - now accepts 3D input (batch_size, n_tracks, n_features)
        inputs = keras.Input(shape=input_shape, name='input_layer')
        
        # Flatten the input to combine tracks and features
        x = keras.layers.Reshape((-1,))(inputs)  # Flatten to (batch_size, n_tracks * n_features)
        
        # Create encoder layers
        for i, units in enumerate(self.encoder_layers):
            x = self._add_dense_block(x, units, f'encoder_{i}')
        
        # Latent layer
        if self.quant_bits:
            latent = QDense(
                self.latent_dim,
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name='latent_layer'
            )(x)
        else:
            latent = keras.layers.Dense(self.latent_dim, name='latent_layer')(x)
        
        # Optionally normalize latent space
        if self.normalize_latent:
            latent = keras.layers.BatchNormalization(name='latent_normalization')(latent)
        
        # Decoder
        x = latent
        for i, units in enumerate(self.decoder_layers):
            x = self._add_dense_block(x, units, f'decoder_{i}')
        
        # Output layer - reshape back to original dimensions
        if self.quant_bits:
            x = QDense(
                np.prod(input_shape),  # Multiply dimensions to get total size
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name='output_dense'
            )(x)
        else:
            x = keras.layers.Dense(np.prod(input_shape), name='output_dense')(x)
            
        # Reshape back to 3D
        outputs = keras.layers.Reshape(input_shape, name='output_reshape')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        logging.info("\nModel layer structure:")
        for layer in self.model.layers:
            logging.info(f"Layer: {layer.name}, Type: {type(layer)}")
        
    def _add_dense_block(self, x, units: int, prefix: str):
        """Helper to add a dense block with activation and batch norm"""
        if self.quant_bits:
            x = QDense(
                units,
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name=f'{prefix}_dense'
            )(x)
            x = QActivation(
                quantized_relu(self.quant_bits),
                name=f'{prefix}_activation'
            )(x)
        else:
            x = keras.layers.Dense(units, name=f'{prefix}_dense')(x)
            x = keras.layers.Activation(self.activation, name=f'{prefix}_activation')(x)
        
        return keras.layers.BatchNormalization(name=f'{prefix}_bn')(x)

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
            "name": self.name
        }

    def create_plots(self, plots_dir: Path) -> None:
        """Create autoencoder-specific plots"""
        # For autoencoder, we might want to show:
        # 1. Latent space distributions
        # 2. Reconstruction examples
        # 3. Loss components if using custom loss
        
        logging.info("\nCreating autoencoder-specific plots...")
        
        # Example: Plot model architecture
        tf.keras.utils.plot_model(
            self.model,
            to_file=str(plots_dir / 'model_architecture.pdf'),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True
        )

        logging.info("Created model architecture plot")
        
        # Could add more autoencoder-specific visualizations:
        # - Latent space clustering
        # - Reconstruction quality examples
        # - Feature-wise reconstruction errors