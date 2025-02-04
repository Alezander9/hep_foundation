from .base_model import BaseModel
from .autoencoder import AutoEncoder

class ModelFactory:
    """Factory for creating different model types"""
    @staticmethod
    def create_model(model_type: str, config: dict) -> BaseModel:
        if model_type == "autoencoder":
            config_copy = config.copy()
            config_copy.pop('model_type', None)  # Remove model_type key
            return AutoEncoder(
                input_dim=config['input_dim'],
                latent_dim=config['latent_dim'],
                encoder_layers=config['encoder_layers'],
                decoder_layers=config['decoder_layers'],
                quant_bits=config.get('quant_bits', None),
                activation=config.get('activation', 'relu'),
                name=config.get('name', 'autoencoder')
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    @staticmethod
    def from_config(config: dict) -> BaseModel:
        """Create model from config dictionary"""
        model_type = config.pop("model_type")
        return ModelFactory.create_model(model_type, config)
