from typing import List, Optional, Tuple
from tensorflow import keras
import tensorflow as tf
import numpy as np
from pathlib import Path
from qkeras import QDense, QActivation, quantized_bits, quantized_relu
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from hep_foundation.base_model import BaseModel
from hep_foundation.plot_utils import (
    MARKER_SIZES
)

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
            
        logging.info(f"\nEpoch {epoch+1}: beta = {beta:.4f}")
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
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
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
        
        logging.info("\nBuilding VAE architecture...")
        
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
        
        logging.info("Built encoder layers")
        
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
        logging.info("Built encoder model")
        
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
        
        logging.info("Built decoder layers")
        
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
        logging.info("Built decoder model")
        
        # Create VAE model
        vae_inputs = keras.Input(shape=input_shape, name='vae_input')
        vae_outputs = VAELayer(self.encoder, self.decoder, name='vae_layer')(vae_inputs)
        self.model = keras.Model(vae_inputs, vae_outputs, name=self.name)
        
        # Add beta parameter
        self.beta = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.model.beta = self.beta
        
        logging.info("Completed VAE architecture build")

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
        logging.info("\nCreating VAE-specific plots...")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if hasattr(self.model, 'history') and self.model.history is not None:
                self._history = self.model.history.history
            else:
                logging.warning("No training history found in model")
                return

            logging.info(f"Available metrics: {list(self._history.keys())}")
            
            from hep_foundation.plot_utils import (
                set_science_style, get_figure_size, get_color_cycle,
                FONT_SIZES, LINE_WIDTHS
            )
            
            set_science_style(use_tex=False)
            
            if self._history:
                colors = get_color_cycle('high_contrast', 3)
                
                plt.figure(figsize=get_figure_size('single', ratio=1.2))
                ax1 = plt.gca()
                ax1.set_yscale('log')
                ax2 = ax1.twinx()
                
                epochs = range(1, len(self._history['reconstruction_loss']) + 1)
                
                ax1.plot(epochs, self._history['reconstruction_loss'], 
                        color=colors[0], label='Reconstruction Loss', 
                        linewidth=LINE_WIDTHS['thick'])
                ax1.plot(epochs, self._history['kl_loss'], 
                        color=colors[1], label='KL Loss',
                        linewidth=LINE_WIDTHS['thick'])
                
                betas = self._calculate_beta_schedule(len(epochs))
                ax2.plot(epochs, betas, color=colors[2], linestyle='--', 
                        label='Beta', linewidth=LINE_WIDTHS['thick'])
                
                ax1.set_xlabel('Epoch', fontsize=FONT_SIZES['large'])
                ax1.set_ylabel('Loss Components (log scale)', fontsize=FONT_SIZES['large'])
                ax2.set_ylabel('Beta (linear scale)', fontsize=FONT_SIZES['large'], color=colors[2])
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, 
                          loc='upper right', fontsize=FONT_SIZES['normal'])
                
                ax1.grid(True, alpha=0.3)
                plt.title('Training Losses and Annealing Schedule', 
                         fontsize=FONT_SIZES['xlarge'])
                
                plt.savefig(plots_dir / 'training_history.pdf', dpi=300, bbox_inches='tight')
                plt.close()
                logging.info("Created training history plot")
                
            else:
                logging.warning("No history data available for plotting")
            
        except Exception as e:
            logging.error(f"Error creating VAE plots: {str(e)}")
            import traceback
            traceback.print_exc()

        # 3. Latent Space Visualization
        if hasattr(self, '_encoded_data'):
            plt.figure(figsize=get_figure_size('double', ratio=2.0))
            
            # Plot z_mean distribution
            plt.subplot(121)
            sns.histplot(self._encoded_data[0].flatten(), bins=50)
            plt.title('Latent Space Mean Distribution', fontsize=FONT_SIZES['large'])
            plt.xlabel('Mean (z)', fontsize=FONT_SIZES['large'])
            plt.ylabel('Count', fontsize=FONT_SIZES['large'])
            
            # Plot z_log_var distribution
            plt.subplot(122)
            sns.histplot(self._encoded_data[1].flatten(), bins=50)
            plt.title('Latent Space Log Variance Distribution', fontsize=FONT_SIZES['large'])
            plt.xlabel('Log Variance (z)', fontsize=FONT_SIZES['large'])
            plt.ylabel('Count', fontsize=FONT_SIZES['large'])
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'vae_latent_space_distributions.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Created latent space distribution plots")
            
            # 4. 2D Latent Space Projection
            if self.latent_dim >= 2:
                plt.figure(figsize=get_figure_size('single', ratio=1.0))
                plt.scatter(
                    self._encoded_data[0][:, 0],
                    self._encoded_data[0][:, 1],
                    alpha=0.5,
                    s=MARKER_SIZES['tiny'],
                    c=get_color_cycle('aesthetic')[0]
                )
                plt.title('2D Latent Space Projection', fontsize=FONT_SIZES['xlarge'])
                plt.xlabel('z1', fontsize=FONT_SIZES['large'])
                plt.ylabel('z2', fontsize=FONT_SIZES['large'])
                plt.tight_layout()
                plt.savefig(plots_dir / 'vae_latent_space_2d.pdf', dpi=300, bbox_inches='tight')
                plt.close()
                logging.info("Created 2D latent space projection plot")

        logging.info(f"VAE plots saved to: {plots_dir}")

    def _calculate_beta_schedule(self, num_epochs):
        """
        Calculate beta values for each epoch based on the beta schedule configuration.
        
        Args:
            num_epochs: Number of epochs to calculate beta values for
            
        Returns:
            List of beta values for each epoch
        """
        if not hasattr(self, 'beta_schedule') or not self.beta_schedule:
            # Default constant beta if no schedule is provided
            return [0.0] * num_epochs
        
        start = self.beta_schedule.get('start', 0.0)
        end = self.beta_schedule.get('end', 1.0)
        warmup_epochs = self.beta_schedule.get('warmup_epochs', 0)
        cycle_epochs = self.beta_schedule.get('cycle_epochs', 0)
        
        betas = []
        for epoch in range(num_epochs):
            if cycle_epochs > 0 and epoch >= warmup_epochs:
                # Cyclic beta schedule after warmup
                cycle_position = (epoch - warmup_epochs) % cycle_epochs
                cycle_ratio = cycle_position / cycle_epochs
                # Use sine wave for smooth cycling (0 to 1 to 0)
                beta = start + (end - start) * (np.sin(cycle_ratio * np.pi) + 1) / 2
            elif warmup_epochs > 0 and epoch < warmup_epochs:
                # Linear warmup
                beta = start + (end - start) * (epoch / warmup_epochs)
            else:
                # Constant beta after warmup if no cycling
                beta = end
            
            betas.append(beta)
        
        return betas