from datetime import datetime
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from qkeras import QDense, QActivation, quantized_bits, quantized_relu
from sklearn.metrics import roc_curve, auc

from hep_foundation.base_model import BaseModel
from hep_foundation.plot_utils import (
    MARKER_SIZES, FONT_SIZES, LINE_WIDTHS,
    set_science_style, get_figure_size, get_color_cycle
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


class AnomalyDetectionEvaluator:
    """Class for evaluating trained models with various tests"""
    
    def __init__(
        self,
        model: BaseModel,
        test_dataset: tf.data.Dataset,
        signal_datasets: Optional[Dict[str, tf.data.Dataset]] = None,
        experiment_id: str = None,
        base_path: Path = Path("experiments")
    ):
        """
        Initialize the model tester
        
        Args:
            model: Trained model to evaluate
            test_dataset: Dataset of background events for testing
            signal_datasets: Dictionary of signal datasets for comparison
            experiment_id: ID of the experiment (e.g. '001_vae_test')
            base_path: Base path where experiments are stored
        """
        self.model = model
        self.test_dataset = test_dataset
        self.signal_datasets = signal_datasets or {}
        
        # Setup paths
        self.base_path = Path(base_path)
        if experiment_id is None:
            raise ValueError("experiment_id must be provided")
        
        self.experiment_path = self.base_path / experiment_id
        self.testing_path = self.experiment_path / "testing"
        self.testing_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Verify experiment data exists
        self.experiment_data_path = self.experiment_path / "experiment_data.json"
        if not self.experiment_data_path.exists():
            raise ValueError(f"No experiment data found at {self.experiment_data_path}")
            
        # Load existing experiment data
        with open(self.experiment_data_path, 'r') as f:
            self.experiment_data = json.load(f)
            
        # Initialize test results storage
        self.test_results = {}
        
    def _update_experiment_data(self, test_results: Dict) -> None:
        """Update experiment data with new test results"""
        # Add or update test results in experiment data
        if 'test_results' not in self.experiment_data:
            self.experiment_data['test_results'] = {}
        
        self.experiment_data['test_results'].update(test_results)
        
        # Save updated experiment data
        with open(self.experiment_data_path, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
            
    def _calculate_losses(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate reconstruction (MSE) and KL losses for a dataset
        
        Args:
            dataset: Dataset to evaluate
            
        Returns:
            Tuple of (reconstruction_losses, kl_losses) arrays
        """
        if not isinstance(self.model, VariationalAutoEncoder):
            raise ValueError("Model must be a VariationalAutoEncoder")
        
        reconstruction_losses = []
        kl_losses = []
        
        # Log dataset info
        total_batches = 0
        total_events = 0
        
        logging.info("\nCalculating losses for dataset...")
        for batch in dataset:
            total_batches += 1
            total_events += batch.shape[0]
            
            # Get encoder outputs
            z_mean, z_log_var, z = self.model.encoder(batch)
            
            # Get reconstructions
            reconstructions = self.model.decoder(z)
            
            # Flatten input and reconstruction for loss calculation
            flat_inputs = tf.reshape(batch, [-1, tf.reduce_prod(batch.shape[1:])])
            flat_reconstructions = tf.reshape(reconstructions, [-1, tf.reduce_prod(reconstructions.shape[1:])])
            
            # Calculate losses per event (not taking the mean)
            recon_losses_batch = tf.reduce_sum(
                tf.square(flat_inputs - flat_reconstructions),
                axis=1
            ).numpy()
            
            kl_losses_batch = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            ).numpy()
            
            # Append individual event losses
            reconstruction_losses.extend(recon_losses_batch.tolist())
            kl_losses.extend(kl_losses_batch.tolist())
        
        logging.info(f"Dataset stats:")
        logging.info(f"  Total batches: {total_batches}")
        logging.info(f"  Total events: {total_events}")
        logging.info(f"  Events per batch: {total_events/total_batches if total_batches > 0 else 0:.1f}")
        
        return np.array(reconstruction_losses), np.array(kl_losses)

    def _calculate_separation_metrics(
        self,
        background_losses: np.ndarray,
        signal_losses: np.ndarray,
        loss_type: str
    ) -> Dict:
        """
        Calculate metrics for separation between background and signal
        
        Args:
            background_losses: Array of losses for background events
            signal_losses: Array of losses for signal events
            loss_type: String identifier for the type of loss
            
        Returns:
            Dictionary of separation metrics
        """
        # Calculate basic statistics
        metrics = {
            'background_mean': float(np.mean(background_losses)),
            'background_std': float(np.std(background_losses)),
            'signal_mean': float(np.mean(signal_losses)),
            'signal_std': float(np.std(signal_losses)),
            'separation': float(
                abs(np.mean(signal_losses) - np.mean(background_losses)) /
                np.sqrt(np.std(signal_losses)**2 + np.std(background_losses)**2)
            )
        }
        
        # Add ROC curve metrics
        from sklearn.metrics import roc_curve, auc
        labels = np.concatenate([np.zeros(len(background_losses)), np.ones(len(signal_losses))])
        scores = np.concatenate([background_losses, signal_losses])
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        metrics['roc_auc'] = float(roc_auc)
        
        # Option 1: Don't store ROC curve data in JSON at all
        # Just store the AUC value which is what we primarily care about
        
        # Option 2: Downsample the ROC curve to reduce size
        # Only include a maximum of 20 points to keep the JSON file manageable
        if len(fpr) > 20:
            # Get indices for approximately 20 evenly spaced points
            indices = np.linspace(0, len(fpr) - 1, 20).astype(int)
            # Make sure to always include the endpoints
            if indices[0] != 0:
                indices[0] = 0
            if indices[-1] != len(fpr) - 1:
                indices[-1] = len(fpr) - 1
            
            metrics['roc_curve'] = {
                'fpr': fpr[indices].tolist(),
                'tpr': tpr[indices].tolist(),
                'thresholds': thresholds[indices].tolist() if len(thresholds) > 0 else []
            }
        else:
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        
        return metrics

    def run_anomaly_detection_test(self) -> Dict:
        """
        Evaluate model's anomaly detection capabilities
        
        Compares reconstruction error (MSE) and KL divergence distributions
        between background and signal datasets.
        
        Returns:
            Dictionary containing test metrics and results
        """
        logging.info("\nRunning anomaly detection test...")
        
        if not isinstance(self.model, VariationalAutoEncoder):
            raise ValueError("Anomaly detection test requires a VariationalAutoEncoder")
        
        # Create testing/anomaly_detection directory
        test_dir = self.testing_path / "anomaly_detection"
        test_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = test_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Log dataset info before testing
        logging.info("\nDataset information before testing:")
        for batch in self.test_dataset:
            logging.info(f"Background test dataset batch shape: {batch.shape}")
            break
        for signal_name, signal_dataset in self.signal_datasets.items():
            for batch in signal_dataset:
                logging.info(f"{signal_name} signal dataset batch shape: {batch.shape}")
                break
        
        # Calculate losses for background dataset
        logging.info("\nCalculating losses for background dataset...")
        bg_recon_losses, bg_kl_losses = self._calculate_losses(self.test_dataset)
        
        # Calculate losses for each signal dataset
        signal_results = {}
        for signal_name, signal_dataset in self.signal_datasets.items():
            logging.info(f"\nCalculating losses for signal dataset: {signal_name}")
            sig_recon_losses, sig_kl_losses = self._calculate_losses(signal_dataset)
            
            # Calculate separation metrics
            recon_metrics = self._calculate_separation_metrics(
                bg_recon_losses, sig_recon_losses, "reconstruction"
            )
            kl_metrics = self._calculate_separation_metrics(
                bg_kl_losses, sig_kl_losses, "kl_divergence"
            )
            
            signal_results[signal_name] = {
                'reconstruction_metrics': recon_metrics,
                'kl_divergence_metrics': kl_metrics,
                'n_events': len(sig_recon_losses)
            }
            
            # Create comparison plots
            self._plot_loss_distributions(
                bg_recon_losses, sig_recon_losses,
                bg_kl_losses, sig_kl_losses,
                signal_name, plots_dir
            )
        
        # Prepare test results
        test_results = {
            'anomaly_detection': {
                'timestamp': str(datetime.now()),
                'background_events': len(bg_recon_losses),
                'signal_results': signal_results,
                'plots_directory': str(plots_dir)
            }
        }
        
        # Update experiment data with test results
        self._update_experiment_data(test_results)
        
        return test_results

    def _plot_loss_distributions(
        self,
        bg_recon_losses: np.ndarray,
        sig_recon_losses: np.ndarray,
        bg_kl_losses: np.ndarray,
        sig_kl_losses: np.ndarray,
        signal_name: str,
        plots_dir: Path
    ) -> None:
        """Create plots comparing background and signal loss distributions"""
        # Set style
        set_science_style(use_tex=False)
        colors = get_color_cycle('high_contrast')
        
        # 1. Loss distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size('double'))
        
        # Calculate percentile-based limits for better visualization
        recon_upper_limit = np.percentile(np.concatenate([bg_recon_losses, sig_recon_losses]), 99)
        kl_upper_limit = np.percentile(np.concatenate([bg_kl_losses, sig_kl_losses]), 99)
        
        # Reconstruction loss
        ax1.hist(bg_recon_losses, bins=50, alpha=0.5, color=colors[0],
                 label='Background', density=True, range=(0, recon_upper_limit))
        ax1.hist(sig_recon_losses, bins=50, alpha=0.5, color=colors[1],
                 label=signal_name, density=True, range=(0, recon_upper_limit))
        ax1.set_xlabel('Reconstruction Loss')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # KL divergence
        ax2.hist(bg_kl_losses, bins=50, alpha=0.5, color=colors[0],
                 label='Background', density=True, range=(0, kl_upper_limit))
        ax2.hist(sig_kl_losses, bins=50, alpha=0.5, color=colors[1],
                 label=signal_name, density=True, range=(0, kl_upper_limit))
        ax2.set_xlabel('KL Divergence')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add percentile information to the plot titles
        ax1.set_title(f'Reconstruction Loss (99th percentile: {recon_upper_limit:.1f})')
        ax2.set_title(f'KL Divergence (99th percentile: {kl_upper_limit:.1f})')
        
        plt.suptitle(f'Loss Distributions: Background vs {signal_name}')
        plt.tight_layout()
        plt.savefig(plots_dir / f'loss_distributions_{signal_name}.pdf')
        
        
        # 2. ROC curves
        plt.figure(figsize=get_figure_size('single'))
        
        # ROC for reconstruction loss
        labels = np.concatenate([np.zeros(len(bg_recon_losses)), np.ones(len(sig_recon_losses))])
        scores = np.concatenate([bg_recon_losses, sig_recon_losses])
        fpr_recon, tpr_recon, _ = roc_curve(labels, scores)
        roc_auc_recon = auc(fpr_recon, tpr_recon)
        
        # ROC for KL loss
        scores_kl = np.concatenate([bg_kl_losses, sig_kl_losses])
        fpr_kl, tpr_kl, _ = roc_curve(labels, scores_kl)
        roc_auc_kl = auc(fpr_kl, tpr_kl)
        
        plt.plot(fpr_recon, tpr_recon, color=colors[0],
                 label=f'Reconstruction (AUC = {roc_auc_recon:.3f})')
        plt.plot(fpr_kl, tpr_kl, color=colors[1],
                 label=f'KL Divergence (AUC = {roc_auc_kl:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: Background vs {signal_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(plots_dir / f'roc_curves_{signal_name}.pdf')
        plt.close() 