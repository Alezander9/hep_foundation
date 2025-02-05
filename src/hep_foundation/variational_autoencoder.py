from typing import List, Optional, Tuple
from tensorflow import keras
import tensorflow as tf
import numpy as np
from pathlib import Path
from qkeras import QDense, QActivation, quantized_bits, quantized_relu
import matplotlib.pyplot as plt
import seaborn as sns

from hep_foundation.base_model import BaseModel

class Sampling(keras.layers.Layer):
    """Reparameterization trick by sampling from a unit Gaussian"""
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAELayer(keras.layers.Layer):
    """Custom VAE layer combining encoder and decoder with loss tracking"""
    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        
        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Get latent space parameters and sample
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        
        # Flatten input and reconstruction for loss calculation
        flat_inputs = tf.reshape(inputs, [-1, tf.reduce_prod(inputs.shape[1:])])
        flat_reconstruction = tf.reshape(reconstruction, [-1, tf.reduce_prod(reconstruction.shape[1:])])
        
        # Calculate losses (reduce to scalar values)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(flat_inputs, flat_reconstruction)
        )
        
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        total_loss = reconstruction_loss + self.beta * kl_loss
        
        # Add metrics
        self.add_loss(total_loss)
        self.add_metric(reconstruction_loss, name="reconstruction_loss")
        self.add_metric(kl_loss, name="kl_loss")
        self.add_metric(total_loss, name="total_loss")
        
        return reconstruction

class BetaSchedule(keras.callbacks.Callback):
    """Callback for VAE beta parameter annealing"""
    def __init__(self, 
                 beta_start: float = 0.0,
                 beta_end: float = 1.0,
                 total_epochs: int = 100,
                 warmup_epochs: int = 50,
                 cycle_epochs: int = 20):
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cycle_epochs = cycle_epochs
        
    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        if epoch < self.warmup_epochs:
            beta = self.beta_start
        else:
            cycle_position = (epoch - self.warmup_epochs) % self.cycle_epochs
            cycle_ratio = cycle_position / self.cycle_epochs
            beta = self.beta_start + (self.beta_end - self.beta_start) * \
                   (np.sin(cycle_ratio * np.pi) + 1) / 2
            
        print(f"\nEpoch {epoch+1}: beta = {beta:.4f}")
        self.model.beta.assign(beta)
        self.model.get_layer('vae_layer').beta.assign(beta)

class VariationalAutoEncoder(BaseModel):
    """Variational Autoencoder implementation"""
    def __init__(
        self,
        input_shape: tuple,
        latent_dim: int,
        encoder_layers: List[int],
        decoder_layers: List[int],
        quant_bits: Optional[int] = None,
        activation: str = 'relu',
        beta_schedule: Optional[dict] = None,
        name: str = 'vae'
    ):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.quant_bits = quant_bits
        self.activation = activation
        self.beta_schedule = beta_schedule or {
            'start': 0.0,
            'end': 1.0,
            'warmup_epochs': 50,
            'cycle_epochs': 20
        }
        self.name = name
        
        # Will be set during build
        self.encoder = None
        self.decoder = None
        self.beta = None 

    def build(self, input_shape: tuple = None) -> None:
        """Build encoder and decoder networks with VAE architecture"""
        if input_shape is None:
            input_shape = self.input_shape
        
        # Build Encoder
        encoder_inputs = keras.Input(shape=input_shape, name='encoder_input')
        x = keras.layers.Reshape((-1,))(encoder_inputs)
        
        # Add encoder layers
        for i, units in enumerate(self.encoder_layers):
            if self.quant_bits:
                x = QDense(
                    units,
                    kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    name=f'encoder_{i}_dense'
                )(x)
                x = QActivation(
                    quantized_relu(self.quant_bits),
                    name=f'encoder_{i}_activation'
                )(x)
            else:
                x = keras.layers.Dense(units, name=f'encoder_{i}_dense')(x)
                x = keras.layers.Activation(self.activation, name=f'encoder_{i}_activation')(x)
            x = keras.layers.BatchNormalization(name=f'encoder_{i}_bn')(x)
        
        # VAE latent space parameters
        if self.quant_bits:
            z_mean = QDense(
                self.latent_dim,
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name='z_mean'
            )(x)
            z_log_var = QDense(
                self.latent_dim,
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name='z_log_var'
            )(x)
        else:
            z_mean = keras.layers.Dense(self.latent_dim, name='z_mean')(x)
            z_log_var = keras.layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        z = Sampling(name='sampling')([z_mean, z_log_var])
        
        # Create encoder model
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Build Decoder
        decoder_inputs = keras.Input(shape=(self.latent_dim,), name='decoder_input')
        x = decoder_inputs
        
        # Add decoder layers
        for i, units in enumerate(self.decoder_layers):
            if self.quant_bits:
                x = QDense(
                    units,
                    kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                    name=f'decoder_{i}_dense'
                )(x)
                x = QActivation(
                    quantized_relu(self.quant_bits),
                    name=f'decoder_{i}_activation'
                )(x)
            else:
                x = keras.layers.Dense(units, name=f'decoder_{i}_dense')(x)
                x = keras.layers.Activation(self.activation, name=f'decoder_{i}_activation')(x)
            x = keras.layers.BatchNormalization(name=f'decoder_{i}_bn')(x)
        
        # Output layer
        if self.quant_bits:
            x = QDense(
                np.prod(input_shape),
                kernel_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                bias_quantizer=quantized_bits(self.quant_bits, 1, alpha=1.0),
                name='decoder_output'
            )(x)
        else:
            x = keras.layers.Dense(np.prod(input_shape), name='decoder_output')(x)
        
        decoder_outputs = keras.layers.Reshape(input_shape)(x)
        
        # Create decoder model
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')
        
        # Create VAE model
        vae_inputs = keras.Input(shape=input_shape, name='vae_input')
        vae_outputs = VAELayer(self.encoder, self.decoder, name='vae_layer')(vae_inputs)
        self.model = keras.Model(vae_inputs, vae_outputs, name=self.name)
        
        # Add beta parameter
        self.beta = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.model.beta = self.beta

    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            "model_type": "variational_autoencoder",
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "quant_bits": self.quant_bits,
            "activation": self.activation,
            "beta_schedule": self.beta_schedule,
            "name": self.name
        }

    def create_plots(self, plots_dir: Path) -> None:
        """Create VAE-specific visualization plots"""
        print("\nCreating VAE-specific plots...")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Model Architecture
        tf.keras.utils.plot_model(
            self.model,
            to_file=str(plots_dir / 'vae_architecture.pdf'),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True
        )
        
        # 2. Latent Space Visualization (if we have encoded data)
        if hasattr(self, '_encoded_data'):
            plt.figure(figsize=(12, 5))
            
            # Plot z_mean distribution
            plt.subplot(121)
            sns.histplot(self._encoded_data[0].flatten(), bins=50)
            plt.title('Latent Space Mean Distribution')
            plt.xlabel('z_mean')
            
            # Plot z_log_var distribution
            plt.subplot(122)
            sns.histplot(self._encoded_data[1].flatten(), bins=50)
            plt.title('Latent Space Log Variance Distribution')
            plt.xlabel('z_log_var')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'latent_space_distributions.pdf')
            plt.close()
            
            # 3. 2D visualization of latent space (first two dimensions)
            if self.latent_dim >= 2:
                plt.figure(figsize=(10, 10))
                plt.scatter(
                    self._encoded_data[0][:, 0],
                    self._encoded_data[0][:, 1],
                    alpha=0.5,
                    s=2
                )
                plt.title('2D Latent Space Projection')
                plt.xlabel('First Latent Dimension')
                plt.ylabel('Second Latent Dimension')
                plt.savefig(plots_dir / 'latent_space_2d.pdf')
                plt.close()
        
        # 4. Beta Schedule Visualization
        epochs = range(self.beta_schedule['warmup_epochs'] + self.beta_schedule['cycle_epochs'])
        betas = []
        for epoch in epochs:
            if epoch < self.beta_schedule['warmup_epochs']:
                beta = self.beta_schedule['start']
            else:
                cycle_position = (epoch - self.beta_schedule['warmup_epochs']) % self.beta_schedule['cycle_epochs']
                cycle_ratio = cycle_position / self.beta_schedule['cycle_epochs']
                beta = self.beta_schedule['start'] + (self.beta_schedule['end'] - self.beta_schedule['start']) * \
                       (np.sin(cycle_ratio * np.pi) + 1) / 2
            betas.append(beta)
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, betas)
        plt.title('Beta Annealing Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Beta Value')
        plt.grid(True)
        plt.savefig(plots_dir / 'beta_schedule.pdf')
        plt.close() 