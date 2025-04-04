from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    model_type: str
    architecture: Dict[str, Any]  # Contains network architecture
    hyperparameters: Dict[str, Any]  # Contains model hyperparameters

    def __init__(
        self,
        model_type: str,
        latent_dim: int,
        encoder_layers: List[int],
        decoder_layers: List[int],
        quant_bits: Optional[int],
        activation: str,
        beta_schedule: Optional[Dict] = None
    ):
        self.model_type = model_type
        # Group architecture-related parameters
        self.architecture = {
            'latent_dim': latent_dim,
            'encoder_layers': encoder_layers,
            'decoder_layers': decoder_layers,
            'activation': activation
        }
        # Group hyperparameters
        self.hyperparameters = {
            'quant_bits': quant_bits
        }
        if beta_schedule:
            self.hyperparameters['beta_schedule'] = beta_schedule

    def validate(self) -> None:
        """Validate model configuration parameters"""
        valid_types = ["autoencoder", "variational_autoencoder"]
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")
        
        # Validate architecture
        if self.architecture['latent_dim'] < 1:
            raise ValueError("latent_dim must be positive")
            
        if not self.architecture['encoder_layers'] or not self.architecture['decoder_layers']:
            raise ValueError("encoder_layers and decoder_layers cannot be empty")
            
        # Validate VAE-specific parameters
        if self.model_type == "variational_autoencoder":
            if 'beta_schedule' not in self.hyperparameters:
                raise ValueError("beta_schedule required for VAE")
            required_beta_fields = ["start", "end", "warmup_epochs", "cycle_epochs"]
            missing = [f for f in required_beta_fields if f not in self.hyperparameters['beta_schedule']]
            if missing:
                raise ValueError(f"beta_schedule missing required fields: {missing}")


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