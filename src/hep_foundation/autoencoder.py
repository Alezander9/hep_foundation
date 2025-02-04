from typing import List, Optional
from tensorflow import keras
from qkeras import QDense, QActivation, quantized_bits, quantized_relu

from .base_model import BaseModel

class AutoEncoder(BaseModel):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: List[int],
        decoder_layers: List[int],
        quant_bits: Optional[int] = None,
        activation: str = 'relu',
        name: str = 'autoencoder'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.quant_bits = quant_bits
        self.activation = activation
        self.name = name
        
    def build(self, input_shape: tuple = None) -> None:
        """Build encoder and decoder networks"""
        if input_shape is None:
            input_shape = (self.input_dim,)
        
        # Input layer
        inputs = keras.Input(shape=input_shape, name='input_layer')
        
        # Create encoder layers
        encoder_layers = []
        for i, units in enumerate(self.encoder_layers):
            if self.quant_bits:
                # Dense layer
                dense = QDense(
                    units,
                    kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    name=f'encoder_dense_{i}'
                )
                encoder_layers.append(dense)
                
                # Activation layer
                activation = QActivation(
                    quantized_relu(self.quant_bits),
                    name=f'encoder_activation_{i}'
                )
                encoder_layers.append(activation)
                
                # Batch normalization
                batch_norm = keras.layers.BatchNormalization(
                    name=f'encoder_bn_{i}'
                )
                encoder_layers.append(batch_norm)
            else:
                dense = keras.layers.Dense(
                    units,
                    name=f'encoder_dense_{i}'
                )
                encoder_layers.append(dense)
                
                activation = keras.layers.Activation(
                    self.activation,
                    name=f'encoder_activation_{i}'
                )
                encoder_layers.append(activation)
                
                batch_norm = keras.layers.BatchNormalization(
                    name=f'encoder_bn_{i}'
                )
                encoder_layers.append(batch_norm)
        
        # Create latent layer
        if self.quant_bits:
            latent_layer = QDense(
                self.latent_dim,
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name='latent_layer'
            )
        else:
            latent_layer = keras.layers.Dense(
                self.latent_dim,
                name='latent_layer'
            )
        
        # Create decoder layers
        decoder_layers = []
        for i, units in enumerate(self.decoder_layers):
            if self.quant_bits:
                # Dense layer
                dense = QDense(
                    units,
                    kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    name=f'decoder_dense_{i}'
                )
                decoder_layers.append(dense)
                
                # Activation layer
                activation = QActivation(
                    quantized_relu(self.quant_bits),
                    name=f'decoder_activation_{i}'
                )
                decoder_layers.append(activation)
                
                # Batch normalization
                batch_norm = keras.layers.BatchNormalization(
                    name=f'decoder_bn_{i}'
                )
                decoder_layers.append(batch_norm)
            else:
                dense = keras.layers.Dense(
                    units,
                    name=f'decoder_dense_{i}'
                )
                decoder_layers.append(dense)
                
                activation = keras.layers.Activation(
                    self.activation,
                    name=f'decoder_activation_{i}'
                )
                decoder_layers.append(activation)
                
                batch_norm = keras.layers.BatchNormalization(
                    name=f'decoder_bn_{i}'
                )
                decoder_layers.append(batch_norm)
        
        # Create output layer
        if self.quant_bits:
            output_layer = QDense(
                self.input_dim,
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name='output_layer'
            )
        else:
            output_layer = keras.layers.Dense(
                self.input_dim,
                name='output_layer'
            )
        
        # Build the model by applying layers sequentially
        # Encoder
        x = inputs
        for layer in encoder_layers:
            x = layer(x)
        
        # Latent space
        latent = latent_layer(x)
        
        # Decoder
        x = latent
        for layer in decoder_layers:
            x = layer(x)
        
        # Output
        outputs = output_layer(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        print("\nModel layer structure:")
        for layer in self.model.layers:
            print(f"Layer: {layer.name}, Type: {type(layer)}")
        
    def get_config(self) -> dict:
        return {
            "model_type": "autoencoder",
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "quant_bits": self.quant_bits,
            "activation": self.activation,
            "name": self.name
        }