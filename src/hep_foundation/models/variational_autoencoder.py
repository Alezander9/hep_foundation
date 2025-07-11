import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from tensorflow import keras

from hep_foundation.config.logging_config import get_logger
from hep_foundation.models.base_model import BaseModel, ModelConfig
from hep_foundation.utils.plot_utils import (
    MARKER_SIZES,
    get_color_cycle,
    get_figure_size,
    set_science_style,
)


class VAEConfig(ModelConfig):
    """Configuration class for Variational Autoencoder"""

    def validate(self) -> None:
        """
        Validate VAE configuration parameters

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

        # Validate activation function
        if "activation" in self.architecture:
            valid_activations = ["relu", "tanh", "sigmoid", "elu", "selu"]
            if self.architecture["activation"] not in valid_activations:
                raise ValueError(f"activation must be one of {valid_activations}")

        # Validate hyperparameters
        if "quant_bits" in self.hyperparameters:
            if not isinstance(self.hyperparameters["quant_bits"], (int, type(None))):
                raise ValueError("quant_bits must be an integer or None")
            if (
                isinstance(self.hyperparameters["quant_bits"], int)
                and self.hyperparameters["quant_bits"] < 1
            ):
                raise ValueError("quant_bits must be positive")

        # Validate beta schedule
        if "beta_schedule" not in self.hyperparameters:
            raise ValueError("beta_schedule is required for VAE")

        beta_schedule = self.hyperparameters["beta_schedule"]
        required_beta = ["start", "end", "warmup_epochs", "cycle_epochs"]
        for param in required_beta:
            if param not in beta_schedule:
                raise ValueError(f"Missing required beta_schedule parameter: {param}")

        # Validate beta schedule values
        if (
            not isinstance(beta_schedule["start"], (int, float))
            or beta_schedule["start"] < 0
        ):
            raise ValueError("beta_schedule.start must be a non-negative number")

        if (
            not isinstance(beta_schedule["end"], (int, float))
            or beta_schedule["end"] < beta_schedule["start"]
        ):
            raise ValueError("beta_schedule.end must be greater than or equal to start")

        if (
            not isinstance(beta_schedule["warmup_epochs"], int)
            or beta_schedule["warmup_epochs"] < 0
        ):
            raise ValueError(
                "beta_schedule.warmup_epochs must be a non-negative integer"
            )

        if (
            not isinstance(beta_schedule["cycle_epochs"], int)
            or beta_schedule["cycle_epochs"] < 0
        ):
            raise ValueError(
                "beta_schedule.cycle_epochs must be a non-negative integer"
            )


class Sampling(keras.layers.Layer):
    """Reparameterization trick by sampling from a unit Gaussian"""

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # Ensure epsilon has the same dtype as inputs (for mixed precision compatibility)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), dtype=z_mean.dtype)
        # Clip z_log_var to prevent overflow in exp operation
        z_log_var_clipped = tf.clip_by_value(z_log_var, -20.0, 20.0)
        return z_mean + tf.exp(0.5 * z_log_var_clipped) * epsilon


class VAELayer(keras.layers.Layer):
    """Custom VAE layer combining encoder and decoder with loss tracking"""

    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        # Handle both (features, _) and features-only cases
        if isinstance(inputs, tuple):
            features, _ = inputs
        else:
            features = inputs

        # Get latent space parameters and sample
        z_mean, z_log_var, z = self.encoder(features)
        reconstruction = self.decoder(z)

        # Use static shape to avoid retracing
        input_shape = tf.shape(features)
        flat_inputs = tf.reshape(features, [input_shape[0], -1])
        flat_reconstruction = tf.reshape(reconstruction, [input_shape[0], -1])

        # Calculate losses (reduce to scalar values)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(flat_inputs, flat_reconstruction)
        )

        # Clip z_log_var to prevent overflow in exp operation during training
        z_log_var_clipped = tf.clip_by_value(z_log_var, -20.0, 20.0)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var_clipped - tf.square(z_mean) - tf.exp(z_log_var_clipped)
        )

        # Ensure beta has the same dtype as losses for mixed precision compatibility
        beta_cast = tf.cast(self.beta, reconstruction_loss.dtype)
        total_loss = reconstruction_loss + beta_cast * kl_loss

        # Add metrics
        self.add_loss(total_loss)
        self.add_metric(reconstruction_loss, name="reconstruction_loss")
        self.add_metric(kl_loss, name="kl_loss")
        self.add_metric(total_loss, name="total_loss")

        return reconstruction


class BetaSchedule(keras.callbacks.Callback):
    """Callback for VAE beta parameter annealing"""

    def __init__(
        self,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        total_epochs: int = 100,
        warmup_epochs: int = 50,
        cycle_epochs: int = 20,
    ):
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cycle_epochs = cycle_epochs

        # Initialize logger
        self.logger = get_logger(__name__)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        if epoch < self.warmup_epochs:
            beta = self.beta_start
        else:
            cycle_position = (epoch - self.warmup_epochs) % self.cycle_epochs
            cycle_ratio = cycle_position / self.cycle_epochs
            beta = (
                self.beta_start
                + (self.beta_end - self.beta_start)
                * (np.sin(cycle_ratio * np.pi) + 1)
                / 2
            )

        self.logger.info(f"Epoch {epoch + 1}: beta = {beta:.4f}")
        self.model.beta.assign(beta)
        self.model.get_layer("vae_layer").beta.assign(beta)


class VariationalAutoEncoder(BaseModel):
    """Variational Autoencoder implementation"""

    def __init__(self, config: VAEConfig):
        """
        Initialize VariationalAutoEncoder

        Args:
            config: VAEConfig object containing model configuration
        """
        super().__init__()
        self.logger = get_logger(__name__)

        # Extract configuration parameters
        self.input_shape = config.architecture["input_shape"]
        self.latent_dim = config.architecture["latent_dim"]
        self.encoder_layers = config.architecture["encoder_layers"]
        self.decoder_layers = config.architecture["decoder_layers"]
        self.activation = config.architecture.get("activation", "relu")
        self.quant_bits = config.hyperparameters.get("quant_bits")
        self.beta_schedule = config.hyperparameters.get(
            "beta_schedule",
            {"start": 0.0, "end": 1.0, "warmup_epochs": 50, "cycle_epochs": 20},
        )
        self.name = config.architecture.get("name", "vae")

        # Will be set during build
        self.encoder = None
        self.deterministic_encoder = None
        self.decoder = None
        self.beta = None

    def build(self, input_shape: tuple = None) -> None:
        """Build encoder and decoder networks with VAE architecture"""
        if input_shape is None:
            input_shape = self.input_shape

        # Build Encoder
        encoder_inputs = keras.Input(shape=input_shape, name="encoder_input")
        x = keras.layers.Reshape((-1,))(encoder_inputs)

        # Add encoder layers
        for i, units in enumerate(self.encoder_layers):
            x = keras.layers.Dense(units, name=f"encoder_{i}_dense")(x)
            x = keras.layers.Activation(
                self.activation, name=f"encoder_{i}_activation"
            )(x)
            x = keras.layers.BatchNormalization(name=f"encoder_{i}_bn")(x)

        self.logger.info("Built encoder layers")

        # VAE latent space parameters
        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x)

        # Sampling layer
        z = Sampling(name="sampling")([z_mean, z_log_var])

        # Create encoder model
        self.encoder = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z], name="encoder"
        )
        self.logger.info("Built encoder model")

        # Create deterministic encoder (only z_mean output) for downstream tasks
        self.deterministic_encoder = keras.Model(
            encoder_inputs, z_mean, name="deterministic_encoder"
        )
        self.logger.info("Built deterministic encoder model")

        # Build Decoder
        decoder_inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")
        x = decoder_inputs

        # Add decoder layers
        for i, units in enumerate(self.decoder_layers):
            x = keras.layers.Dense(units, name=f"decoder_{i}_dense")(x)
            x = keras.layers.Activation(
                self.activation, name=f"decoder_{i}_activation"
            )(x)
            x = keras.layers.BatchNormalization(name=f"decoder_{i}_bn")(x)

        self.logger.info("Built decoder layers")

        # Output layer
        x = keras.layers.Dense(np.prod(input_shape), name="decoder_output")(x)

        decoder_outputs = keras.layers.Reshape(input_shape)(x)

        # Create decoder model
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        self.logger.info("Built decoder model")

        # Create VAE model
        vae_inputs = keras.Input(shape=input_shape, name="vae_input")
        vae_outputs = VAELayer(self.encoder, self.decoder, name="vae_layer")(vae_inputs)
        self.model = keras.Model(vae_inputs, vae_outputs, name=self.name)

        # Add beta parameter
        self.beta = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.model.beta = self.beta

        self.logger.info("Completed VAE architecture build")

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
            "name": self.name,
        }

    def create_plots(self, plots_dir: Path) -> None:
        """Create VAE-specific visualization plots"""
        self.logger.info("Creating VAE-specific plots...")
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            if hasattr(self.model, "history") and self.model.history is not None:
                self._history = self.model.history.history
            else:
                self.logger.warning("No training history found in model")
                return

            self.logger.info(f"Available metrics: {list(self._history.keys())}")

            from hep_foundation.utils.plot_utils import (
                FONT_SIZES,
                LINE_WIDTHS,
                get_color_cycle,
                get_figure_size,
                set_science_style,
            )

            set_science_style(use_tex=False)

            if self._history:
                colors = get_color_cycle("high_contrast", 3)

                plt.figure(figsize=get_figure_size("single", ratio=1.2))
                ax1 = plt.gca()
                ax1.set_yscale("log")
                ax2 = ax1.twinx()

                epochs = range(1, len(self._history["reconstruction_loss"]) + 1)

                ax1.plot(
                    epochs,
                    self._history["reconstruction_loss"],
                    color=colors[0],
                    label="Reconstruction Loss",
                    linewidth=LINE_WIDTHS["thick"],
                )
                ax1.plot(
                    epochs,
                    self._history["kl_loss"],
                    color=colors[1],
                    label="KL Loss",
                    linewidth=LINE_WIDTHS["thick"],
                )
                if "total_loss" in self._history:
                    # Fix: Use a color from the available high_contrast palette instead of invalid "bright" palette
                    # Get a fourth color by extending the high_contrast palette
                    extended_colors = get_color_cycle("high_contrast", 4)
                    ax1.plot(
                        epochs,
                        self._history["total_loss"],
                        color=extended_colors[3],
                        label="Total Loss",
                        linewidth=LINE_WIDTHS["thick"],
                    )

                # Debug: Print some loss values to help diagnose the overlap issue
                if len(self._history["reconstruction_loss"]) > 0:
                    self.logger.debug(
                        f"Sample reconstruction loss values: {self._history['reconstruction_loss'][:3]}..."
                    )
                    if "total_loss" in self._history:
                        self.logger.debug(
                            f"Sample total loss values: {self._history['total_loss'][:3]}..."
                        )
                    if "kl_loss" in self._history:
                        self.logger.debug(
                            f"Sample KL loss values: {self._history['kl_loss'][:3]}..."
                        )

                betas = self._calculate_beta_schedule(len(epochs))
                ax2.plot(
                    epochs,
                    betas,
                    color=colors[2],
                    linestyle="--",
                    label="Beta",
                    linewidth=LINE_WIDTHS["thick"],
                )

                # Debug: Print some beta values to help diagnose if beta is always 0
                self.logger.debug(f"Sample beta values: {betas[:5]}...")

                ax1.set_xlabel("Epoch", fontsize=FONT_SIZES["large"])
                ax1.set_ylabel(
                    "Loss Components (log scale)", fontsize=FONT_SIZES["large"]
                )
                ax2.set_ylabel(
                    "Beta (linear scale)", fontsize=FONT_SIZES["large"], color=colors[2]
                )

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(
                    lines1 + lines2,
                    labels1 + labels2,
                    loc="upper right",
                    fontsize=FONT_SIZES["normal"],
                )

                ax1.grid(True, alpha=0.3)
                plt.title(
                    "Training Losses and Annealing Schedule",
                    fontsize=FONT_SIZES["xlarge"],
                )

                plt.savefig(
                    plots_dir / "training_history.png", dpi=300, bbox_inches="tight"
                )
                plt.close()
                self.logger.info("Created training history plot")

            else:
                self.logger.warning("No history data available for plotting")

        except Exception as e:
            self.logger.error(f"Error creating VAE plots: {str(e)}")
            import traceback

            traceback.print_exc()

        # 3. Latent Space Visualization
        if hasattr(self, "_encoded_data"):
            plt.figure(figsize=get_figure_size("double", ratio=2.0))

            # Plot z_mean distribution
            plt.subplot(121)
            sns.histplot(self._encoded_data[0].flatten(), bins=50)
            plt.title("Latent Space Mean Distribution", fontsize=FONT_SIZES["large"])
            plt.xlabel("Mean (z)", fontsize=FONT_SIZES["large"])
            plt.ylabel("Count", fontsize=FONT_SIZES["large"])

            # Plot z_log_var distribution
            plt.subplot(122)
            sns.histplot(self._encoded_data[1].flatten(), bins=50)
            plt.title(
                "Latent Space Log Variance Distribution", fontsize=FONT_SIZES["large"]
            )
            plt.xlabel("Log Variance (z)", fontsize=FONT_SIZES["large"])
            plt.ylabel("Count", fontsize=FONT_SIZES["large"])

            plt.tight_layout()
            plt.savefig(
                plots_dir / "vae_latent_space_distributions.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            self.logger.info("Created latent space distribution plots")

            # 4. 2D Latent Space Projection
            if self.latent_dim >= 2:
                plt.figure(figsize=get_figure_size("single", ratio=1.0))
                plt.scatter(
                    self._encoded_data[0][:, 0],
                    self._encoded_data[0][:, 1],
                    alpha=0.5,
                    s=MARKER_SIZES["tiny"],
                    c=get_color_cycle("aesthetic")[0],
                )
                plt.title("2D Latent Space Projection", fontsize=FONT_SIZES["xlarge"])
                plt.xlabel("z1", fontsize=FONT_SIZES["large"])
                plt.ylabel("z2", fontsize=FONT_SIZES["large"])
                plt.tight_layout()
                plt.savefig(
                    plots_dir / "vae_latent_space_2d.png", dpi=300, bbox_inches="tight"
                )
                plt.close()
                self.logger.info("Created 2D latent space projection plot")

        self.logger.info(f"VAE plots saved to: {plots_dir}")

    def _calculate_beta_schedule(self, num_epochs):
        """
        Calculate beta values for each epoch based on the beta schedule configuration.

        Args:
            num_epochs: Number of epochs to calculate beta values for

        Returns:
            List of beta values for each epoch
        """
        if not hasattr(self, "beta_schedule") or not self.beta_schedule:
            # Default constant beta if no schedule is provided
            return [0.0] * num_epochs

        start = self.beta_schedule.get("start", 0.0)
        end = self.beta_schedule.get("end", 1.0)
        warmup_epochs = self.beta_schedule.get("warmup_epochs", 0)
        cycle_epochs = self.beta_schedule.get("cycle_epochs", 0)

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
        signal_datasets: Optional[dict[str, tf.data.Dataset]] = None,
        experiment_id: str = None,
        base_path: Path = Path("experiments"),
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

        # Setup self.logger
        self.logger = get_logger(__name__)

        # Load experiment info from the new file structure
        self.experiment_info_path = self.experiment_path / "_experiment_info.json"
        if not self.experiment_info_path.exists():
            raise ValueError(f"No experiment info found at {self.experiment_info_path}")

        # Load existing experiment info
        with open(self.experiment_info_path) as f:
            self.experiment_info = json.load(f)

    def _calculate_losses(
        self, dataset: tf.data.Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
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

        self.logger.info("Calculating losses for dataset...")
        for batch in dataset:
            total_batches += 1

            # Handle both (features, labels) and features-only cases
            if isinstance(batch, tuple):
                features, _ = batch  # We only need features for reconstruction
                total_events += features.shape[0]
                # Ensure input batch is float32
                features = tf.cast(features, tf.float32)
            else:
                features = batch
                total_events += features.shape[0]
                # Ensure input batch is float32
                features = tf.cast(features, tf.float32)

                # Get encoder outputs
            z_mean, z_log_var, z = self.model.encoder(features)

            # Get reconstructions
            reconstructions = self.model.decoder(z)

            # Use static shape to avoid retracing
            input_shape = tf.shape(features)
            flat_inputs = tf.reshape(features, [input_shape[0], -1])
            flat_reconstructions = tf.reshape(reconstructions, [input_shape[0], -1])

            # Ensure both tensors are the same type before subtraction
            flat_inputs = tf.cast(flat_inputs, tf.float32)
            flat_reconstructions = tf.cast(flat_reconstructions, tf.float32)

            # Calculate losses per event (not taking the mean) with bounds for signal data
            recon_losses_batch = tf.reduce_sum(
                tf.square(flat_inputs - flat_reconstructions), axis=1
            ).numpy()

            # Clip reconstruction losses to prevent extreme values from out-of-distribution signal data
            recon_losses_batch = np.clip(recon_losses_batch, 0.0, 1e8)

            # Calculate KL losses with enhanced overflow protection
            # More aggressive clipping for out-of-distribution signal data
            z_log_var_clipped = tf.clip_by_value(z_log_var, -20.0, 20.0)
            z_mean_clipped = tf.clip_by_value(z_mean, -10.0, 10.0)

            # Calculate KL divergence with additional safety
            kl_losses_batch = (
                -0.5
                * tf.reduce_sum(
                    1
                    + z_log_var_clipped
                    - tf.square(z_mean_clipped)
                    - tf.exp(z_log_var_clipped),
                    axis=1,
                ).numpy()
            )

            # Clip final KL losses to prevent extreme values from signal data
            kl_losses_batch = np.clip(kl_losses_batch, 0.0, 1e6)

            # Check for NaN/Inf in final loss calculations
            invalid_recon = np.sum(~np.isfinite(recon_losses_batch))
            invalid_kl = np.sum(~np.isfinite(kl_losses_batch))

            if invalid_recon > 0 or invalid_kl > 0:
                self.logger.warning(
                    f"Batch {total_batches}: Found extreme loss values (likely out-of-distribution signal data)"
                )
                self.logger.warning(
                    f"  Reconstruction loss - NaN: {np.sum(np.isnan(recon_losses_batch))}, Inf: {np.sum(np.isinf(recon_losses_batch))}"
                )
                self.logger.warning(
                    f"  KL loss - NaN: {np.sum(np.isnan(kl_losses_batch))}, Inf: {np.sum(np.isinf(kl_losses_batch))}"
                )

                # Log valid data ranges for context
                valid_recon = recon_losses_batch[np.isfinite(recon_losses_batch)]
                valid_kl = kl_losses_batch[np.isfinite(kl_losses_batch)]
                if len(valid_recon) > 0:
                    self.logger.warning(
                        f"  Valid reconstruction loss range: [{np.min(valid_recon):.3e}, {np.max(valid_recon):.3e}]"
                    )
                if len(valid_kl) > 0:
                    self.logger.warning(
                        f"  Valid KL loss range: [{np.min(valid_kl):.3e}, {np.max(valid_kl):.3e}]"
                    )

                self.logger.warning(
                    "  → This is expected for signal data that differs significantly from background training data"
                )
                self.logger.warning(
                    "  → Invalid values have been clipped and will be filtered in metrics calculation"
                )

            # Append individual event losses
            reconstruction_losses.extend(recon_losses_batch.tolist())
            kl_losses.extend(kl_losses_batch.tolist())

        self.logger.info("Dataset stats:")
        self.logger.info(f"  Total batches: {total_batches}")
        self.logger.info(f"  Total events: {total_events}")
        self.logger.info(
            f"  Events per batch: {total_events / total_batches if total_batches > 0 else 0:.1f}"
        )

        return np.array(reconstruction_losses), np.array(kl_losses)

    def _calculate_separation_metrics(
        self, background_losses: np.ndarray, signal_losses: np.ndarray, loss_type: str
    ) -> dict:
        """
        Calculate metrics for separation between background and signal

        Args:
            background_losses: Array of losses for background events
            signal_losses: Array of losses for signal events
            loss_type: String identifier for the type of loss

        Returns:
            Dictionary of separation metrics
        """
        # Check for NaN/Inf values and filter them out
        bg_nan_count = np.sum(np.isnan(background_losses))
        bg_inf_count = np.sum(np.isinf(background_losses))
        sig_nan_count = np.sum(np.isnan(signal_losses))
        sig_inf_count = np.sum(np.isinf(signal_losses))

        total_invalid = bg_nan_count + bg_inf_count + sig_nan_count + sig_inf_count

        if total_invalid > 0:
            self.logger.warning(
                f"Found {total_invalid} invalid values in {loss_type} losses!"
            )
            self.logger.warning(
                f"  Background: {bg_nan_count} NaN + {bg_inf_count} Inf = {bg_nan_count + bg_inf_count}"
            )
            self.logger.warning(
                f"  Signal: {sig_nan_count} NaN + {sig_inf_count} Inf = {sig_nan_count + sig_inf_count}"
            )

            # Filter out invalid values
            background_losses_clean = background_losses[np.isfinite(background_losses)]
            signal_losses_clean = signal_losses[np.isfinite(signal_losses)]

            self.logger.warning(
                f"After filtering: Background {len(background_losses_clean)}/{len(background_losses)}, Signal {len(signal_losses_clean)}/{len(signal_losses)}"
            )

            # Check if we have enough valid data left
            if len(background_losses_clean) == 0 or len(signal_losses_clean) == 0:
                self.logger.error(
                    f"No valid {loss_type} losses remaining after filtering NaN/Inf!"
                )
                # Return metrics with default values
                return {
                    "background_mean": 0.0,
                    "background_std": 0.0,
                    "signal_mean": 0.0,
                    "signal_std": 0.0,
                    "separation": 0.0,
                    "roc_auc": 0.5,  # Random performance
                    "roc_curve": {"fpr": [0, 1], "tpr": [0, 1], "thresholds": [1, 0]},
                    "data_quality_warning": f"All {loss_type} losses were NaN/Inf - using default values",
                }
        else:
            # No invalid values, use original arrays
            background_losses_clean = background_losses
            signal_losses_clean = signal_losses

        # Calculate basic statistics using cleaned data
        bg_std = float(np.std(background_losses_clean))
        sig_std = float(np.std(signal_losses_clean))

        # Calculate separation with protection against zero variance
        mean_diff = abs(np.mean(signal_losses_clean) - np.mean(background_losses_clean))
        std_sum_squared = sig_std**2 + bg_std**2

        if std_sum_squared == 0.0:
            # Both distributions have zero variance (all values identical)
            if mean_diff == 0.0:
                separation = 0.0  # Identical distributions
            else:
                separation = float("inf")  # Perfect separation with no overlap
            self.logger.warning(
                f"Zero variance detected in {loss_type} separation calculation: "
                f"bg_std={bg_std}, sig_std={sig_std}, mean_diff={mean_diff}. "
                f"Setting separation to {'0.0' if separation == 0.0 else 'inf'}."
            )
        else:
            separation = float(mean_diff / np.sqrt(std_sum_squared))

        metrics = {
            "background_mean": float(np.mean(background_losses_clean)),
            "background_std": bg_std,
            "signal_mean": float(np.mean(signal_losses_clean)),
            "signal_std": sig_std,
            "separation": separation,
        }

        # Add data quality info if filtering was needed
        if total_invalid > 0:
            metrics["data_quality_info"] = {
                "original_background_count": len(background_losses),
                "original_signal_count": len(signal_losses),
                "filtered_background_count": len(background_losses_clean),
                "filtered_signal_count": len(signal_losses_clean),
                "background_nan_count": int(bg_nan_count),
                "background_inf_count": int(bg_inf_count),
                "signal_nan_count": int(sig_nan_count),
                "signal_inf_count": int(sig_inf_count),
            }

        # Add ROC curve metrics using cleaned data
        from sklearn.metrics import auc, roc_curve

        labels = np.concatenate(
            [np.zeros(len(background_losses_clean)), np.ones(len(signal_losses_clean))]
        )
        scores = np.concatenate([background_losses_clean, signal_losses_clean])

        # Final safety check - this should not happen but just in case
        if not np.all(np.isfinite(scores)):
            self.logger.error("BUG: Still have NaN/Inf in scores after filtering!")
            valid_indices = np.isfinite(scores)
            scores = scores[valid_indices]
            labels = labels[valid_indices]

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        metrics["roc_auc"] = float(roc_auc)

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

            metrics["roc_curve"] = {
                "fpr": fpr[indices].tolist(),
                "tpr": tpr[indices].tolist(),
                "thresholds": thresholds[indices].tolist()
                if len(thresholds) > 0
                else [],
            }
        else:
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            }

        return metrics

    def _save_loss_distribution_data(
        self,
        losses: np.ndarray,
        loss_type: str,
        signal_name: str,
        plot_data_dir: Path,
        bin_edges: np.ndarray = None,
    ) -> Path:
        """
        Save loss distribution data as JSON for later combined plotting.

        Args:
            losses: Array of loss values
            loss_type: Type of loss ('reconstruction' or 'kl_divergence')
            signal_name: Name of the signal dataset ('background' for background data)
            plot_data_dir: Directory to save the JSON file
            bin_edges: Optional pre-calculated bin edges for coordinated binning

        Returns:
            Path to the saved JSON file
        """
        # Calculate histogram data
        if losses.size == 0:
            counts = []
            if bin_edges is not None:
                bin_edges_list = bin_edges.tolist()
            else:
                bin_edges_list = []
        else:
            if bin_edges is not None:
                # Use provided bin edges for coordinated binning
                counts, _ = np.histogram(losses, bins=bin_edges, density=True)
                bin_edges_list = bin_edges.tolist()
            else:
                # Fallback to individual percentile-based range
                p0_1, p99_9 = np.percentile(losses, [0.1, 99.9])
                if p0_1 == p99_9:
                    p0_1 -= 0.5
                    p99_9 += 0.5
                plot_range = (p0_1, p99_9)

                counts, calculated_bin_edges = np.histogram(
                    losses, bins=50, range=plot_range, density=True
                )
                bin_edges_list = calculated_bin_edges.tolist()

        # Create JSON data structure
        loss_data = {
            loss_type: {
                "counts": counts.tolist() if hasattr(counts, "tolist") else counts,
                "bin_edges": bin_edges_list,
                "n_events": len(losses),
                "mean": float(np.mean(losses)) if losses.size > 0 else 0.0,
                "std": float(np.std(losses)) if losses.size > 0 else 0.0,
            }
        }

        # Save to JSON file
        json_filename = f"loss_distributions_{signal_name}_{loss_type}_data.json"
        json_path = plot_data_dir / json_filename

        with open(json_path, "w") as f:
            json.dump(loss_data, f, indent=2)

        self.logger.info(
            f"Saved {loss_type} loss distribution data for {signal_name} to {json_path}"
        )
        return json_path

    def _save_roc_curve_data(
        self,
        bg_losses: np.ndarray,
        sig_losses: np.ndarray,
        loss_type: str,
        signal_name: str,
        plot_data_dir: Path,
    ) -> Path:
        """
        Save ROC curve data as JSON for later combined plotting.

        Args:
            bg_losses: Background loss values
            sig_losses: Signal loss values
            loss_type: Type of loss ('reconstruction' or 'kl_divergence')
            signal_name: Name of the signal dataset
            plot_data_dir: Directory to save the JSON file

        Returns:
            Path to the saved JSON file
        """
        from sklearn.metrics import auc, roc_curve

        # Create labels and scores for ROC calculation
        labels = np.concatenate([np.zeros(len(bg_losses)), np.ones(len(sig_losses))])
        scores = np.concatenate([bg_losses, sig_losses])

        # Filter out NaN and Inf values for ROC calculation
        valid_mask = np.isfinite(scores)
        if not np.any(valid_mask):
            self.logger.warning(
                f"All {loss_type} scores are NaN/Inf for {signal_name}! "
                f"Cannot calculate ROC curve, using default values."
            )
            # Return default ROC curve (diagonal line)
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            thresholds = np.array([np.inf, -np.inf])
            roc_auc = 0.5
        else:
            # Use only valid data points
            valid_labels = labels[valid_mask]
            valid_scores = scores[valid_mask]

            # Log if we filtered out any values
            n_invalid = np.sum(~valid_mask)
            if n_invalid > 0:
                self.logger.warning(
                    f"Filtered out {n_invalid} invalid values from {loss_type} scores for {signal_name} "
                    f"before ROC calculation (kept {np.sum(valid_mask)} valid values)"
                )

            # Calculate ROC curve with valid data
            fpr, tpr, thresholds = roc_curve(valid_labels, valid_scores)
            roc_auc = auc(fpr, tpr)

        # Downsample to reduce file size (keep max 50 points)
        if len(fpr) > 50:
            indices = np.linspace(0, len(fpr) - 1, 50).astype(int)
            # Always include endpoints
            if indices[0] != 0:
                indices[0] = 0
            if indices[-1] != len(fpr) - 1:
                indices[-1] = len(fpr) - 1
            fpr = fpr[indices]
            tpr = tpr[indices]
            thresholds = thresholds[indices] if len(thresholds) > 0 else []

        # Create JSON data structure
        roc_data = {
            loss_type: {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist() if len(thresholds) > 0 else [],
                "auc": float(roc_auc),
                "n_background": len(bg_losses),
                "n_signal": len(sig_losses),
            }
        }

        # Save to JSON file
        json_filename = f"roc_curves_{signal_name}_{loss_type}_data.json"
        json_path = plot_data_dir / json_filename

        with open(json_path, "w") as f:
            json.dump(roc_data, f, indent=2)

        self.logger.info(
            f"Saved {loss_type} ROC curve data for {signal_name} to {json_path}"
        )
        return json_path

    def _create_combined_loss_distribution_plots(
        self,
        bg_recon_losses: np.ndarray,
        bg_kl_losses: np.ndarray,
        signal_loss_data: dict,
        test_dir: Path,
    ) -> None:
        """
        Create combined loss distribution plots for all signals together.

        Args:
            bg_recon_losses: Background reconstruction losses
            bg_kl_losses: Background KL divergence losses
            signal_loss_data: Dictionary mapping signal names to their loss arrays
            test_dir: Base testing directory (contains plot_data/ and plots/ subdirectories)
        """
        self.logger.info("Creating combined loss distribution plots...")

        # Create plot_data and plots directories
        plot_data_dir = test_dir / "plot_data"
        plot_data_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = test_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Calculate global bin edges for coordinated binning
        self.logger.info(
            "Calculating global bin edges for coordinated loss distribution binning..."
        )

        # Collect all reconstruction losses
        all_recon_losses = [bg_recon_losses]
        for signal_name, (sig_recon_losses, _) in signal_loss_data.items():
            all_recon_losses.append(sig_recon_losses)
        combined_recon_losses = np.concatenate(all_recon_losses)

        # Collect all KL losses
        all_kl_losses = [bg_kl_losses]
        for signal_name, (_, sig_kl_losses) in signal_loss_data.items():
            all_kl_losses.append(sig_kl_losses)
        combined_kl_losses = np.concatenate(all_kl_losses)

        # Calculate global bin edges (0.1 to 99.9 percentiles)
        recon_p0_1, recon_p99_9 = np.percentile(combined_recon_losses, [0.1, 99.9])
        if recon_p0_1 == recon_p99_9:
            recon_p0_1 -= 0.5
            recon_p99_9 += 0.5
        recon_bin_edges = np.linspace(recon_p0_1, recon_p99_9, 51)  # 50 bins

        kl_p0_1, kl_p99_9 = np.percentile(combined_kl_losses, [0.1, 99.9])
        if kl_p0_1 == kl_p99_9:
            kl_p0_1 -= 0.5
            kl_p99_9 += 0.5
        kl_bin_edges = np.linspace(kl_p0_1, kl_p99_9, 51)  # 50 bins

        self.logger.info(
            f"Reconstruction loss bin range: [{recon_p0_1:.3f}, {recon_p99_9:.3f}] (covers 99.8% of all data)"
        )
        self.logger.info(
            f"KL divergence loss bin range: [{kl_p0_1:.3f}, {kl_p99_9:.3f}] (covers 99.8% of all data)"
        )
        self.logger.info(
            "Using coordinated bin edges to ensure perfect histogram alignment"
        )

        # Save bin edges metadata for reference
        bin_edges_metadata = {
            "reconstruction_loss": {
                "bin_edges": recon_bin_edges.tolist(),
                "percentile_range": [0.1, 99.9],
                "data_range": [float(recon_p0_1), float(recon_p99_9)],
                "n_bins": len(recon_bin_edges) - 1,
                "total_events": len(combined_recon_losses),
            },
            "kl_divergence": {
                "bin_edges": kl_bin_edges.tolist(),
                "percentile_range": [0.1, 99.9],
                "data_range": [float(kl_p0_1), float(kl_p99_9)],
                "n_bins": len(kl_bin_edges) - 1,
                "total_events": len(combined_kl_losses),
            },
            "timestamp": str(datetime.now()),
            "datasets": ["background"] + list(signal_loss_data.keys()),
        }

        bin_edges_metadata_path = plot_data_dir / "loss_bin_edges_metadata.json"
        with open(bin_edges_metadata_path, "w") as f:
            json.dump(bin_edges_metadata, f, indent=2)
        self.logger.info(f"Saved loss bin edges metadata to {bin_edges_metadata_path}")

        # Save background loss distribution data with coordinated bin edges
        bg_recon_json = self._save_loss_distribution_data(
            bg_recon_losses,
            "reconstruction",
            "background",
            plot_data_dir,
            recon_bin_edges,
        )
        bg_kl_json = self._save_loss_distribution_data(
            bg_kl_losses, "kl_divergence", "background", plot_data_dir, kl_bin_edges
        )

        # Save signal loss distribution data and collect paths
        signal_recon_jsons = []
        signal_kl_jsons = []
        signal_legend_labels = []

        # Save ROC curve data for each signal
        signal_recon_roc_jsons = []
        signal_kl_roc_jsons = []

        for signal_name, (sig_recon_losses, sig_kl_losses) in signal_loss_data.items():
            # Save loss distribution data with coordinated bin edges
            sig_recon_json = self._save_loss_distribution_data(
                sig_recon_losses,
                "reconstruction",
                signal_name,
                plot_data_dir,
                recon_bin_edges,
            )
            sig_kl_json = self._save_loss_distribution_data(
                sig_kl_losses, "kl_divergence", signal_name, plot_data_dir, kl_bin_edges
            )

            signal_recon_jsons.append(sig_recon_json)
            signal_kl_jsons.append(sig_kl_json)
            signal_legend_labels.append(signal_name)

            # Save ROC curve data
            sig_recon_roc_json = self._save_roc_curve_data(
                bg_recon_losses,
                sig_recon_losses,
                "reconstruction",
                signal_name,
                plot_data_dir,
            )
            sig_kl_roc_json = self._save_roc_curve_data(
                bg_kl_losses, sig_kl_losses, "kl_divergence", signal_name, plot_data_dir
            )

            signal_recon_roc_jsons.append(sig_recon_roc_json)
            signal_kl_roc_jsons.append(sig_kl_roc_json)

        # Create combined two-panel loss distribution plot
        try:
            # Import here to avoid circular imports
            from hep_foundation.data.dataset_visualizer import (
                create_combined_two_panel_loss_plot_from_json,
            )

            if signal_recon_jsons and signal_kl_jsons:
                recon_json_paths = [bg_recon_json] + signal_recon_jsons
                kl_json_paths = [bg_kl_json] + signal_kl_jsons
                legend_labels = ["Background"] + signal_legend_labels

                combined_plot_path = plots_dir / "combined_loss_distributions.png"
                create_combined_two_panel_loss_plot_from_json(
                    recon_json_paths=recon_json_paths,
                    kl_json_paths=kl_json_paths,
                    output_plot_path=str(combined_plot_path),
                    legend_labels=legend_labels,
                    title_prefix="Loss Distributions",
                )
                self.logger.info(
                    f"Saved combined two-panel loss distribution plot to {combined_plot_path}"
                )

        except ImportError:
            self.logger.error(
                "Failed to import create_combined_two_panel_loss_plot_from_json for combined loss plots"
            )
        except Exception as e:
            self.logger.error(f"Failed to create combined loss distribution plots: {e}")

        # Create combined ROC curves plot
        try:
            # Import here to avoid circular imports
            from hep_foundation.data.dataset_visualizer import (
                create_combined_roc_curves_plot_from_json,
            )

            if signal_recon_roc_jsons and signal_kl_roc_jsons:
                combined_roc_plot_path = plots_dir / "combined_roc_curves.png"
                create_combined_roc_curves_plot_from_json(
                    recon_roc_json_paths=signal_recon_roc_jsons,
                    kl_roc_json_paths=signal_kl_roc_jsons,
                    output_plot_path=str(combined_roc_plot_path),
                    legend_labels=signal_legend_labels,
                    title_prefix="ROC Curves",
                )
                self.logger.info(
                    f"Saved combined ROC curves plot to {combined_roc_plot_path}"
                )

        except ImportError:
            self.logger.error(
                "Failed to import create_combined_roc_curves_plot_from_json for ROC curves"
            )
        except Exception as e:
            self.logger.error(f"Failed to create combined ROC curves plots: {e}")

    def run_anomaly_detection_test(self) -> dict:
        """
        Evaluate model's anomaly detection capabilities

        Compares reconstruction error (MSE) and KL divergence distributions
        between background and signal datasets.

        Returns:
            Dictionary containing test metrics and results
        """
        self.logger.info("Running anomaly detection test...")

        if not isinstance(self.model, VariationalAutoEncoder):
            raise ValueError("Anomaly detection test requires a VariationalAutoEncoder")

        # Create testing/anomaly_detection directory
        test_dir = self.testing_path / "anomaly_detection"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Log dataset info before testing
        self.logger.info("Dataset information before testing:")
        for batch in self.test_dataset:
            if isinstance(batch, tuple):
                features, labels = batch
                self.logger.info("Background test dataset batch shapes:")
                self.logger.info(f"  Features: {features.shape}")
                self.logger.info(f"  Labels: {labels.shape}")
            else:
                self.logger.info(f"Background test dataset batch shape: {batch.shape}")
            break

        for signal_name, signal_dataset in self.signal_datasets.items():
            for batch in signal_dataset:
                if isinstance(batch, tuple):
                    features, labels = batch
                    self.logger.info(f"{signal_name} signal dataset batch shapes:")
                    self.logger.info(f"  Features: {features.shape}")
                    self.logger.info(f"  Labels: {labels.shape}")
                else:
                    self.logger.info(
                        f"{signal_name} signal dataset batch shape: {batch.shape}"
                    )
                break

        # Calculate losses for background dataset
        self.logger.info("Calculating losses for background dataset...")
        bg_recon_losses, bg_kl_losses = self._calculate_losses(self.test_dataset)

        # Calculate losses for each signal dataset and store for combined plotting
        signal_results = {}
        signal_loss_data = {}  # For combined plotting

        for signal_name, signal_dataset in self.signal_datasets.items():
            self.logger.info(f"Calculating losses for signal dataset: {signal_name}")
            sig_recon_losses, sig_kl_losses = self._calculate_losses(signal_dataset)

            # Store for combined plotting
            signal_loss_data[signal_name] = (sig_recon_losses, sig_kl_losses)

            # Calculate separation metrics
            recon_metrics = self._calculate_separation_metrics(
                bg_recon_losses, sig_recon_losses, "reconstruction"
            )
            kl_metrics = self._calculate_separation_metrics(
                bg_kl_losses, sig_kl_losses, "kl_divergence"
            )

            signal_results[signal_name] = {
                "reconstruction_metrics": recon_metrics,
                "kl_divergence_metrics": kl_metrics,
                "n_events": len(sig_recon_losses),
            }

        # Create combined loss distribution plots (replaces individual plots)
        if signal_loss_data:
            self._create_combined_loss_distribution_plots(
                bg_recon_losses,
                bg_kl_losses,
                signal_loss_data,
                test_dir,
            )

        # Prepare test results
        test_results = {
            "anomaly_detection": {
                "timestamp": str(datetime.now()),
                "background_events": len(bg_recon_losses),
                "signal_results": signal_results,
                "plots_directory": str(test_dir / "plots"),
                "data_directory": str(test_dir / "plot_data"),
            }
        }

        return test_results

    def _plot_loss_distributions(
        self,
        bg_recon_losses: np.ndarray,
        sig_recon_losses: np.ndarray,
        bg_kl_losses: np.ndarray,
        sig_kl_losses: np.ndarray,
        signal_name: str,
        plots_dir: Path,
    ) -> None:
        """Create plots comparing background and signal loss distributions"""
        # Set style
        set_science_style(use_tex=False)
        colors = get_color_cycle("high_contrast")

        # 1. Loss distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=get_figure_size("double"))

        # Calculate percentile-based limits for better visualization
        recon_upper_limit = np.percentile(
            np.concatenate([bg_recon_losses, sig_recon_losses]), 99
        )
        kl_upper_limit = np.percentile(
            np.concatenate([bg_kl_losses, sig_kl_losses]), 99
        )

        # Reconstruction loss
        ax1.hist(
            bg_recon_losses,
            bins=50,
            alpha=0.5,
            color=colors[0],
            label="Background",
            density=True,
            range=(0, recon_upper_limit),
        )
        ax1.hist(
            sig_recon_losses,
            bins=50,
            alpha=0.5,
            color=colors[1],
            label=signal_name,
            density=True,
            range=(0, recon_upper_limit),
        )
        ax1.set_xlabel("Reconstruction Loss")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # KL divergence
        ax2.hist(
            bg_kl_losses,
            bins=50,
            alpha=0.5,
            color=colors[0],
            label="Background",
            density=True,
            range=(0, kl_upper_limit),
        )
        ax2.hist(
            sig_kl_losses,
            bins=50,
            alpha=0.5,
            color=colors[1],
            label=signal_name,
            density=True,
            range=(0, kl_upper_limit),
        )
        ax2.set_xlabel("KL Divergence")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add percentile information to the plot titles
        ax1.set_title(f"Reconstruction Loss (99th percentile: {recon_upper_limit:.1f})")
        ax2.set_title(f"KL Divergence (99th percentile: {kl_upper_limit:.1f})")

        plt.suptitle(f"Loss Distributions: Background vs {signal_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"loss_distributions_{signal_name}.png")

        # 2. ROC curves
        plt.figure(figsize=get_figure_size("single"))

        # ROC for reconstruction loss
        labels = np.concatenate(
            [np.zeros(len(bg_recon_losses)), np.ones(len(sig_recon_losses))]
        )
        scores = np.concatenate([bg_recon_losses, sig_recon_losses])

        # Filter NaN/Inf values for reconstruction ROC
        valid_mask_recon = np.isfinite(scores)
        if np.any(valid_mask_recon):
            valid_labels_recon = labels[valid_mask_recon]
            valid_scores_recon = scores[valid_mask_recon]
            fpr_recon, tpr_recon, _ = roc_curve(valid_labels_recon, valid_scores_recon)
            roc_auc_recon = auc(fpr_recon, tpr_recon)
        else:
            # Use default values if all data is invalid
            fpr_recon, tpr_recon = np.array([0.0, 1.0]), np.array([0.0, 1.0])
            roc_auc_recon = 0.5

        # ROC for KL loss
        scores_kl = np.concatenate([bg_kl_losses, sig_kl_losses])

        # Filter NaN/Inf values for KL ROC
        valid_mask_kl = np.isfinite(scores_kl)
        if np.any(valid_mask_kl):
            valid_labels_kl = labels[valid_mask_kl]
            valid_scores_kl = scores_kl[valid_mask_kl]
            fpr_kl, tpr_kl, _ = roc_curve(valid_labels_kl, valid_scores_kl)
            roc_auc_kl = auc(fpr_kl, tpr_kl)
        else:
            # Use default values if all data is invalid
            fpr_kl, tpr_kl = np.array([0.0, 1.0]), np.array([0.0, 1.0])
            roc_auc_kl = 0.5

        plt.plot(
            fpr_recon,
            tpr_recon,
            color=colors[0],
            label=f"Reconstruction (AUC = {roc_auc_recon:.3f})",
        )
        plt.plot(
            fpr_kl,
            tpr_kl,
            color=colors[1],
            label=f"KL Divergence (AUC = {roc_auc_kl:.3f})",
        )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves: Background vs {signal_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.savefig(plots_dir / f"roc_curves_{signal_name}.png")
        plt.close()
