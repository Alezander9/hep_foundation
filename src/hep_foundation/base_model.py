from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Base configuration for model architecture"""

    model_type: str
    architecture: Dict[str, Any]  # Contains network architecture
    hyperparameters: Dict[str, Any]  # Contains model hyperparameters

    def __init__(
        self,
        model_type: str,
        architecture: Dict[str, Any],
        hyperparameters: Dict[str, Any],
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
