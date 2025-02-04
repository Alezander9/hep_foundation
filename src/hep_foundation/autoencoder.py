from typing import List, Optional
from tensorflow import keras
from qkeras import QDense, QActivation, quantized_bits, quantized_relu
import numpy as np

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
        name: str = 'track_autoencoder'
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.quant_bits = quant_bits
        self.activation = activation
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
            if self.quant_bits:
                # Dense layer
                x = QDense(
                    units,
                    kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    name=f'encoder_dense_{i}'
                )(x)
                
                # Activation layer
                x = QActivation(
                    quantized_relu(self.quant_bits),
                    name=f'encoder_activation_{i}'
                )(x)
                
                # Batch normalization
                x = keras.layers.BatchNormalization(
                    name=f'encoder_bn_{i}'
                )(x)
            else:
                x = keras.layers.Dense(units, name=f'encoder_dense_{i}')(x)
                x = keras.layers.Activation(self.activation, name=f'encoder_activation_{i}')(x)
                x = keras.layers.BatchNormalization(name=f'encoder_bn_{i}')(x)
        
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
        
        # Decoder
        x = latent
        for i, units in enumerate(self.decoder_layers):
            if self.quant_bits:
                x = QDense(
                    units,
                    kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    name=f'decoder_dense_{i}'
                )(x)
                x = QActivation(
                    quantized_relu(self.quant_bits),
                    name=f'decoder_activation_{i}'
                )(x)
                x = keras.layers.BatchNormalization(name=f'decoder_bn_{i}')(x)
            else:
                x = keras.layers.Dense(units, name=f'decoder_dense_{i}')(x)
                x = keras.layers.Activation(self.activation, name=f'decoder_activation_{i}')(x)
                x = keras.layers.BatchNormalization(name=f'decoder_bn_{i}')(x)
        
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
        
        print("\nModel layer structure:")
        for layer in self.model.layers:
            print(f"Layer: {layer.name}, Type: {type(layer)}")
        
    def get_config(self) -> dict:
        return {
            "model_type": "autoencoder",
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "quant_bits": self.quant_bits,
            "activation": self.activation,
            "name": self.name
        }