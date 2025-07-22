import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from hep_foundation.config.logging_config import get_logger
from hep_foundation.models.base_model import BaseModel, ModelConfig
from hep_foundation.plots.plot_utils import (
    MARKER_SIZES,
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
        required_beta = ["start", "warmup", "cycle_low", "cycle_high", "cycle_period"]
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
            not isinstance(beta_schedule["cycle_low"], (int, float))
            or beta_schedule["cycle_low"] < 0
        ):
            raise ValueError("beta_schedule.cycle_low must be a non-negative number")

        if (
            not isinstance(beta_schedule["cycle_high"], (int, float))
            or beta_schedule["cycle_high"] < beta_schedule["cycle_low"]
        ):
            raise ValueError(
                "beta_schedule.cycle_high must be greater than or equal to cycle_low"
            )

        if not isinstance(beta_schedule["warmup"], int) or beta_schedule["warmup"] < 0:
            raise ValueError("beta_schedule.warmup must be a non-negative integer")

        if (
            not isinstance(beta_schedule["cycle_period"], int)
            or beta_schedule["cycle_period"] <= 0
        ):
            raise ValueError("beta_schedule.cycle_period must be a positive integer")


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
        start: float = 0.0,
        warmup: int = 50,
        cycle_low: float = 0.0,
        cycle_high: float = 1.0,
        cycle_period: int = 20,
    ):
        super().__init__()
        self.start = start
        self.warmup = warmup
        self.cycle_low = cycle_low
        self.cycle_high = cycle_high
        self.cycle_period = cycle_period

        # Initialize logger
        self.logger = get_logger(__name__)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        if epoch < self.warmup:
            # Linear transition from start to cycle_low during warmup
            if self.warmup > 0:
                beta = self.start + (self.cycle_low - self.start) * (
                    epoch / self.warmup
                )
            else:
                beta = self.cycle_low
        else:
            # Sinusoidal oscillation between cycle_low and cycle_high
            cycle_position = (epoch - self.warmup) % self.cycle_period
            cycle_ratio = cycle_position / self.cycle_period
            beta = (
                self.cycle_low
                + (self.cycle_high - self.cycle_low)
                * (np.sin(cycle_ratio * 2 * np.pi) + 1)
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
            {
                "start": 0.0,
                "warmup": 50,
                "cycle_low": 0.0,
                "cycle_high": 1.0,
                "cycle_period": 20,
            },
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

    def create_plots(
        self, plots_dir: Path, training_history_json_path: Optional[Path] = None
    ) -> None:
        """Create VAE-specific visualization plots"""
        self.logger.info("Creating VAE-specific plots...")
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            # First try to load history from provided JSON file path
            if training_history_json_path and training_history_json_path.exists():
                self.logger.info(
                    f"Loading training history from: {training_history_json_path}"
                )
                with open(training_history_json_path) as f:
                    training_data = json.load(f)
                    self._history = training_data.get("history", {})
            # Fallback to searching for training history JSON files in the plots_dir parent
            elif training_history_json_path is None:
                # Look for training history JSON files in the training directory
                training_dir = plots_dir.parent
                if training_dir.name != "training":
                    training_dir = plots_dir.parent / "training"

                if training_dir.exists():
                    json_files = list(training_dir.glob("training_history_*.json"))
                    if json_files:
                        # Use the most recent training history file
                        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                        self.logger.info(
                            f"Found training history JSON file: {latest_json}"
                        )
                        with open(latest_json) as f:
                            training_data = json.load(f)
                            self._history = training_data.get("history", {})
                    else:
                        self.logger.warning(
                            f"No training history JSON files found in {training_dir}"
                        )
                        self._history = {}
                else:
                    self.logger.warning(f"Training directory not found: {training_dir}")
                    self._history = {}
            # Fallback to in-memory history (legacy support)
            elif hasattr(self.model, "history") and self.model.history is not None:
                self.logger.info("Using in-memory training history (legacy mode)")
                self._history = self.model.history.history
            elif hasattr(self, "_history") and self._history:
                self.logger.info("Using existing _history attribute")
                # _history is already set, use it as-is
            else:
                self.logger.warning(
                    "No training history found in JSON file, model, or _history attribute"
                )
                return

            if not self._history:
                self.logger.warning("Training history is empty")
                return

            self.logger.info(f"Available metrics: {list(self._history.keys())}")

            from hep_foundation.plots.plot_utils import (
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
        warmup = self.beta_schedule.get("warmup", 0)
        cycle_low = self.beta_schedule.get("cycle_low", 0.0)
        cycle_high = self.beta_schedule.get("cycle_high", 1.0)
        cycle_period = self.beta_schedule.get("cycle_period", 20)

        betas = []
        for epoch in range(num_epochs):
            if epoch < warmup:
                # Linear transition from start to cycle_low during warmup
                if warmup > 0:
                    beta = start + (cycle_low - start) * (epoch / warmup)
                else:
                    beta = cycle_low
            else:
                # Sinusoidal oscillation between cycle_low and cycle_high
                cycle_position = (epoch - warmup) % cycle_period
                cycle_ratio = cycle_position / cycle_period
                beta = (
                    cycle_low
                    + (cycle_high - cycle_low)
                    * (np.sin(cycle_ratio * 2 * np.pi) + 1)
                    / 2
                )

            betas.append(beta)

        return betas
