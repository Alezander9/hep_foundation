# vulture_whitelist.py

# Plot utils are for standardization of plots (including future plots) so is okay if not used
from src.hep_foundation.plots import plot_utils
plot_utils.get_model_line_style
plot_utils.get_available_line_styles

# Keras Layer methods that are called implicitly by the framework
from src.hep_foundation.models.variational_autoencoder import VAELayer
VAELayer.call
VAELayer.on_epoch_begin
VAELayer.on_epoch_end

# Model trainer methods are called implicitly by the framework
from src.hep_foundation.training import model_trainer
model_trainer.TrainingProgressCallback.on_train_begin
model_trainer.TrainingProgressCallback.on_epoch_begin
model_trainer.TrainingProgressCallback.on_epoch_end

# Keras model trainable attributes (essential for foundation model evaluation)
# These control whether model layers are trainable vs frozen - critical for comparing
# fine-tuned vs fixed encoder models in foundation model evaluation
# We need to simulate accessing the trainable attribute on model objects
import tensorflow as tf
# Use a layer instead of Model to avoid constructor issues
dummy_layer = tf.keras.layers.Dense(1)
dummy_layer.trainable  # Simulates usage of .trainable attribute

# YAML serialization functionality
from src.hep_foundation.utils.utils import ConfigSerializer
# This prevents YAML aliases which can cause config serialization issues
# We need to simulate accessing the ignore_aliases attribute on the class
ConfigSerializer.ignore_aliases

# Example format:
# from src.hep_foundation.models.your_model import YourCustomLayer
# YourCustomLayer.call
