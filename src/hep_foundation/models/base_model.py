from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from hep_foundation.config.logging_config import get_logger


@dataclass
class ModelConfig:
    """Base configuration for model architecture"""

    model_type: str
    architecture: dict[str, Any]  # Contains network architecture
    hyperparameters: dict[str, Any]  # Contains model hyperparameters

    def __init__(
        self,
        model_type: str,
        architecture: dict[str, Any],
        hyperparameters: dict[str, Any],
    ):
        self.model_type = model_type
        self.architecture = architecture
        self.hyperparameters = hyperparameters

    @abstractmethod
    def validate(self) -> None:
        """
        Validate model configuration parameters.
        To be implemented by specific model configurations.
        """
        pass

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary format.

        Returns:
            Dictionary containing model configuration
        """
        return {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "hyperparameters": self.hyperparameters,
        }


class BaseModel(ABC):
    """Base class for all models"""

    def __init__(self):
        self.model = None

    @abstractmethod
    def build(self, input_shape: tuple) -> None:
        """Build the model architecture"""
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """Get model configuration"""
        pass

    def compile(self, *args, **kwargs):
        """Compile the underlying Keras model"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """Train the underlying Keras model"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Evaluate the underlying Keras model"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.evaluate(*args, **kwargs)

    def save(self, path: str) -> None:
        """Save model weights and config"""
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load model weights"""
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.load_weights(path)

    @abstractmethod
    def create_plots(self, plots_dir: Path) -> None:
        """Create model-specific plots"""
        pass


class CustomKerasModelWrapper:
    """
    A wrapper for a pre-constructed tf.keras.Model to make it compatible
    with ModelTrainer and other parts of the hep_foundation framework
    that expect a model wrapper.
    """

    def __init__(self, keras_model: tf.keras.Model, name: str = "CustomKerasModel"):
        self.model = keras_model
        self.name = name if name else keras_model.name
        self.logger = get_logger(f"{__name__}.{self.name}")
        self.history = None
        self.config = {
            "name": self.name,
            "model_type": "custom_keras_wrapper",
        }  # Basic config

    def build(
        self, input_shape=None
    ):  # input_shape is for compatibility, Keras model is pre-built
        """
        Build method for compatibility. The Keras model is assumed to be
        already built or will be built upon first call/compilation.
        This method mainly ensures the logger knows the model is "built".
        """
        if not self.model.built:
            self.logger.info(
                f"Keras model '{self.name}' was not built, attempting to build with input_shape if provided."
            )
            if input_shape:
                try:
                    self.model.build(input_shape)
                    self.logger.info(
                        f"Keras model '{self.name}' built with input_shape {input_shape}."
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to build Keras model '{self.name}' with input_shape {input_shape}: {e}"
                    )
            else:
                self.logger.warning(
                    f"Keras model '{self.name}' is not built and no input_shape provided to build."
                )
        else:
            self.logger.info(
                f"Model '{self.name}' build called (Keras model already built)."
            )

    def compile(self, optimizer, loss, metrics):
        """Compiles the internal Keras model."""
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.logger.info(
            f"Model '{self.name}' compiled with optimizer, loss, and metrics."
        )
        self.history = None  # Reset history on re-compile

    def fit(
        self,
        x,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        """Fits the internal Keras model. Signature adapted for ModelTrainer."""
        self.logger.info(f"Fitting model '{self.name}' for {epochs} epochs.")
        # ModelTrainer typically passes a tf.data.Dataset as x
        if isinstance(x, tf.data.Dataset):
            self.history = self.model.fit(
                x,  # Dataset contains features and labels
                batch_size=batch_size,  # Usually None when x is a dataset
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data,
                shuffle=shuffle,  # Usually controlled by dataset shuffle
                class_weight=class_weight,
                sample_weight=sample_weight,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                validation_freq=validation_freq,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
            )
        else:  # For direct numpy array inputs, if ever used via this wrapper
            self.history = self.model.fit(
                x,
                y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_split=validation_split,
                validation_data=validation_data,
                shuffle=shuffle,
                class_weight=class_weight,
                sample_weight=sample_weight,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                validation_batch_size=validation_batch_size,
                validation_freq=validation_freq,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
            )
        return self.history

    def evaluate(
        self,
        x,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        return_dict=True,
    ):  # Default return_dict to True
        """Evaluates the internal Keras model. Signature adapted for ModelTrainer."""
        self.logger.info(f"Evaluating model '{self.name}'.")
        if isinstance(x, tf.data.Dataset):
            return self.model.evaluate(
                x,
                batch_size=batch_size,
                verbose=verbose,
                sample_weight=sample_weight,
                steps=steps,
                callbacks=callbacks,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                return_dict=return_dict,
            )
        else:  # For direct numpy array inputs
            return self.model.evaluate(
                x,
                y,
                batch_size=batch_size,
                verbose=verbose,
                sample_weight=sample_weight,
                steps=steps,
                callbacks=callbacks,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                return_dict=return_dict,
            )

    def predict(
        self,
        x,
        batch_size=None,
        verbose="auto",
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        """Predicts with the internal Keras model."""
        return self.model.predict(
            x,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )

    def get_metrics(self) -> dict:
        """Retrieves the final metrics from the training history."""
        if self.history and self.history.history:
            return {
                metric: values[-1]
                for metric, values in self.history.history.items()
                if values  # Ensure values is not empty
            }
        self.logger.warning(
            f"No history or empty history found for model '{self.name}' when calling get_metrics."
        )
        return {}

    def summary(self, print_fn=None):
        """Prints the summary of the internal Keras model."""
        if print_fn is None:
            print_fn = self.logger.info
        self.model.summary(print_fn=print_fn)

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.model.non_trainable_weights

    # Add any other attributes or methods ModelTrainer or other parts of the framework might expect
    # For example, if ModelRegistry needs to save/load this wrapper differently.
    # For now, this covers ModelTrainer needs.
