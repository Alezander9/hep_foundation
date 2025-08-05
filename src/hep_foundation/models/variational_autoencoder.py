import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import ScalarFormatter
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
        # Use softplus for numerical stability instead of exp
        variance = tf.nn.softplus(z_log_var) + 1e-6
        return z_mean + tf.sqrt(variance) * epsilon


class VAELayer(keras.layers.Layer):
    """Custom VAE layer combining encoder and decoder with loss tracking"""

    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(0.0, dtype=tf.float32, trainable=False)

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

        # Use softplus for numerical stability instead of exp
        variance = tf.nn.softplus(z_log_var) + 1e-6
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - variance)

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
            # Use cosine wave so that cycle_ratio=0 gives cycle_low, cycle_ratio=0.5 gives cycle_high
            beta = (
                self.cycle_low
                + (self.cycle_high - self.cycle_low)
                * (-np.cos(cycle_ratio * 2 * np.pi) + 1)
                / 2
            )

        self.logger.info(f"Epoch {epoch + 1}: beta = {beta:.4f}")
        self.model.beta.assign(beta)
        self.model.get_layer("vae_layer").beta.assign(beta)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Record the current beta value in the training logs"""
        if logs is not None:
            # Get the current beta value from the model
            current_beta = float(self.model.beta.numpy())
            logs["beta"] = current_beta


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

    def create_plots(
        self, plots_dir: Path, training_history_json_path: Optional[Path] = None
    ) -> None:
        """Create VAE-specific visualization plots"""
        self.logger.info("Creating VAE-specific plots...")
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Require training_history_json_path to be provided and valid
            if not training_history_json_path:
                self.logger.warning(
                    "training_history_json_path not provided. Cannot create training history plots."
                )
                return

            if not training_history_json_path.exists():
                self.logger.warning(
                    f"Training history file not found at: {training_history_json_path}. Cannot create training history plots."
                )
                return

            self.logger.info(
                f"Loading training history from: {training_history_json_path}"
            )
            with open(training_history_json_path) as f:
                training_data = json.load(f)
                self._history = training_data.get("history", {})

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

            # Create training history plot with 4 vertically stacked subplots
            colors = get_color_cycle("high_contrast", 4)

            # Create figure with 4 vertically stacked subplots
            fig, axes = plt.subplots(
                4, 1, figsize=get_figure_size("single", ratio=1), sharex=True
            )
            fig.suptitle(
                "Training Losses and Annealing Schedule", fontsize=FONT_SIZES["xlarge"]
            )

            epochs = range(1, len(self._history["reconstruction_loss"]) + 1)

            # Use recorded beta values from training history
            if "beta" in self._history:
                betas = self._history["beta"]
                self.logger.info("Using recorded beta values from training history")
            else:
                self.logger.error(
                    "Beta values not found in training history. This should not happen with current training setup."
                )
                return  # Skip plotting beta if not available

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

            # Subplot 1: Total Loss (if available)
            if "total_loss" in self._history:
                # Training data (solid line)
                axes[0].plot(
                    epochs,
                    self._history["total_loss"],
                    color=colors[0],
                    linewidth=LINE_WIDTHS["thick"],
                    linestyle="-",
                    label="Train",
                )
                # Validation data (dashed line)
                if "val_total_loss" in self._history:
                    axes[0].plot(
                        epochs,
                        self._history["val_total_loss"],
                        color=colors[0],
                        linewidth=LINE_WIDTHS["thick"],
                        linestyle="--",
                        label="Validation",
                    )
                # Test data (single marker)
                if "test_total_loss" in self._history:
                    test_value = self._history["test_total_loss"]
                    # Handle both single values and lists (in case test metric is a list)
                    if isinstance(test_value, list):
                        test_value = test_value[-1] if test_value else 0
                    axes[0].plot(
                        len(epochs),  # Plot at the last epoch
                        test_value,
                        color=colors[0],
                        marker="*",
                        markersize=MARKER_SIZES["large"],
                        linestyle="",
                        label="Test",
                    )
                axes[0].set_ylabel("Total Loss", fontsize=FONT_SIZES["normal"])
                axes[0].grid(True, alpha=0.3)
                axes[0].set_yscale("log")
                # Disable scientific notation for log scale when values are around O(1)
                formatter = ScalarFormatter()
                formatter.set_scientific(False)
                axes[0].yaxis.set_major_formatter(formatter)
                axes[0].legend(fontsize=FONT_SIZES["small"])
            else:
                # Hide this subplot if total_loss is not available
                axes[0].set_visible(False)

            # Subplot 2: Reconstruction Loss
            # Training data (solid line)
            axes[1].plot(
                epochs,
                self._history["reconstruction_loss"],
                color=colors[1],
                linewidth=LINE_WIDTHS["thick"],
                linestyle="-",
                label="Train",
            )
            # Validation data (dashed line)
            if "val_reconstruction_loss" in self._history:
                axes[1].plot(
                    epochs,
                    self._history["val_reconstruction_loss"],
                    color=colors[1],
                    linewidth=LINE_WIDTHS["thick"],
                    linestyle="--",
                    label="Validation",
                )
            # Test data (single marker)
            if "test_reconstruction_loss" in self._history:
                test_value = self._history["test_reconstruction_loss"]
                # Handle both single values and lists (in case test metric is a list)
                if isinstance(test_value, list):
                    test_value = test_value[-1] if test_value else 0
                axes[1].plot(
                    len(epochs),  # Plot at the last epoch
                    test_value,
                    color=colors[1],
                    marker="*",
                    markersize=MARKER_SIZES["large"],
                    linestyle="",
                    label="Test",
                )
            axes[1].set_ylabel("Recon. Loss", fontsize=FONT_SIZES["normal"])
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale("log")
            # Disable scientific notation for log scale when values are around O(1)
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            axes[1].yaxis.set_major_formatter(formatter)
            axes[1].legend(fontsize=FONT_SIZES["small"])

            # Subplot 3: KL Loss
            # Training data (solid line)
            axes[2].plot(
                epochs,
                self._history["kl_loss"],
                color=colors[2],
                linewidth=LINE_WIDTHS["thick"],
                linestyle="-",
                label="Train",
            )
            # Validation data (dashed line)
            if "val_kl_loss" in self._history:
                axes[2].plot(
                    epochs,
                    self._history["val_kl_loss"],
                    color=colors[2],
                    linewidth=LINE_WIDTHS["thick"],
                    linestyle="--",
                    label="Validation",
                )
            # Test data (single marker)
            if "test_kl_loss" in self._history:
                test_value = self._history["test_kl_loss"]
                # Handle both single values and lists (in case test metric is a list)
                if isinstance(test_value, list):
                    test_value = test_value[-1] if test_value else 0
                axes[2].plot(
                    len(epochs),  # Plot at the last epoch
                    test_value,
                    color=colors[2],
                    marker="*",
                    markersize=MARKER_SIZES["large"],
                    linestyle="",
                    label="Test",
                )
            axes[2].set_ylabel("KL Loss", fontsize=FONT_SIZES["normal"])
            axes[2].grid(True, alpha=0.3)
            axes[2].set_yscale("log")
            # Disable scientific notation for log scale when values are around O(1)
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            axes[2].yaxis.set_major_formatter(formatter)
            axes[2].legend(fontsize=FONT_SIZES["small"])

            # Subplot 4: Beta
            axes[3].plot(
                epochs,
                betas,
                color=colors[3],
                linestyle="-",
                linewidth=LINE_WIDTHS["thick"],
            )
            axes[3].set_ylabel("Beta", fontsize=FONT_SIZES["normal"])
            axes[3].set_xlabel("Epoch", fontsize=FONT_SIZES["large"])
            axes[3].grid(True, alpha=0.3)

            # Adjust subplot spacing with reduced vertical gaps
            plt.subplots_adjust(hspace=0.15, top=0.93, bottom=0.08)

            plt.savefig(
                plots_dir / "training_history.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            self.logger.info("Created training history plot")

        except Exception as e:
            self.logger.error(f"Error creating VAE plots: {str(e)}")
            import traceback

            traceback.print_exc()

        # # 3. Latent Space Visualization
        # if hasattr(self, "_encoded_data"):
        #     plt.figure(figsize=get_figure_size("double", ratio=2.0))

        #     # Plot z_mean distribution
        #     plt.subplot(121)
        #     sns.histplot(self._encoded_data[0].flatten(), bins=50)
        #     plt.title("Latent Space Mean Distribution", fontsize=FONT_SIZES["large"])
        #     plt.xlabel("Mean (z)", fontsize=FONT_SIZES["large"])
        #     plt.ylabel("Count", fontsize=FONT_SIZES["large"])

        #     # Plot z_log_var distribution
        #     plt.subplot(122)
        #     sns.histplot(self._encoded_data[1].flatten(), bins=50)
        #     plt.title(
        #         "Latent Space Log Variance Distribution", fontsize=FONT_SIZES["large"]
        #     )
        #     plt.xlabel("Log Variance (z)", fontsize=FONT_SIZES["large"])
        #     plt.ylabel("Count", fontsize=FONT_SIZES["large"])

        #     plt.tight_layout()
        #     plt.savefig(
        #         plots_dir / "vae_latent_space_distributions.png",
        #         dpi=300,
        #         bbox_inches="tight",
        #     )
        #     plt.close()
        #     self.logger.info("Created latent space distribution plots")

        #     # 4. 2D Latent Space Projection
        #     if self.latent_dim >= 2:
        #         plt.figure(figsize=get_figure_size("single", ratio=1.0))
        #         plt.scatter(
        #             self._encoded_data[0][:, 0],
        #             self._encoded_data[0][:, 1],
        #             alpha=0.5,
        #             s=MARKER_SIZES["tiny"],
        #             c=get_color_cycle("aesthetic")[0],
        #         )
        #         plt.title("2D Latent Space Projection", fontsize=FONT_SIZES["xlarge"])
        #         plt.xlabel("z1", fontsize=FONT_SIZES["large"])
        #         plt.ylabel("z2", fontsize=FONT_SIZES["large"])
        #         plt.tight_layout()
        #         plt.savefig(
        #             plots_dir / "vae_latent_space_2d.png", dpi=300, bbox_inches="tight"
        #         )
        #         plt.close()
        #         self.logger.info("Created 2D latent space projection plot")

        self.logger.info(f"VAE plots saved to: {plots_dir}")
