from typing import Dict, Optional
from .base_model import BaseModel
from .autoencoder import AutoEncoder
from .variational_autoencoder import VariationalAutoEncoder

class ModelFactory:
    """Factory class for creating different types of models"""
    
    # Define required parameters for each model type
    REQUIRED_PARAMS = {
        "autoencoder": {
            "input_shape": "Shape of input data (n_tracks, n_features)",
            "latent_dim": "Dimension of latent space",
            "encoder_layers": "List of layer sizes for encoder",
            "decoder_layers": "List of layer sizes for decoder"
        },
        "variational_autoencoder": {
            "input_shape": "Shape of input data (n_tracks, n_features)",
            "latent_dim": "Dimension of latent space",
            "encoder_layers": "List of layer sizes for encoder",
            "decoder_layers": "List of layer sizes for decoder",
            "beta_schedule": "Dictionary containing beta annealing parameters"
        }
    }
    
    @staticmethod
    def validate_config(model_type: str, config: Dict) -> None:
        """
        Validate model configuration parameters
        
        Args:
            model_type: Type of model being created
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if model_type not in ModelFactory.REQUIRED_PARAMS:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported types are: {list(ModelFactory.REQUIRED_PARAMS.keys())}"
            )
        
        # Check for missing required parameters
        required = ModelFactory.REQUIRED_PARAMS[model_type]
        missing = [
            param for param in required 
            if param not in config
        ]
        
        if missing:
            raise ValueError(
                f"Model type '{model_type}' requires the following missing parameters:\n"
                + "\n".join(f"  - {param}: {required[param]}" for param in missing)
            )
        
        # Additional validation for specific parameters
        if model_type == "variational_autoencoder":
            beta_schedule = config["beta_schedule"]
            required_beta_params = ["start", "end", "warmup_epochs", "cycle_epochs"]
            missing_beta = [
                param for param in required_beta_params 
                if param not in beta_schedule
            ]
            
            if missing_beta:
                raise ValueError(
                    f"Beta schedule for VAE missing required parameters: {missing_beta}"
                )
    
    @staticmethod
    def create_model(model_type: str, config: Dict) -> BaseModel:
        """
        Create a model instance based on type and configuration
        
        Args:
            model_type: Type of model to create
            config: Model configuration dictionary
            
        Returns:
            Instance of specified model type
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        ModelFactory.validate_config(model_type, config)
        
        if model_type == "autoencoder":
            return AutoEncoder(
                input_shape=config['input_shape'],
                latent_dim=config['latent_dim'],
                encoder_layers=config['encoder_layers'],
                decoder_layers=config['decoder_layers'],
                quant_bits=config.get('quant_bits'),
                activation=config.get('activation', 'relu'),
                name=config.get('name', 'autoencoder')
            )
            
        elif model_type == "variational_autoencoder":
            return VariationalAutoEncoder(
                input_shape=config['input_shape'],
                latent_dim=config['latent_dim'],
                encoder_layers=config['encoder_layers'],
                decoder_layers=config['decoder_layers'],
                quant_bits=config.get('quant_bits'),
                activation=config.get('activation', 'relu'),
                beta_schedule=config.get('beta_schedule'),
                name=config.get('name', 'vae')
            )
            
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                "Supported types are: ['autoencoder', 'variational_autoencoder']"
            )
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict:
        """Get default configuration for specified model type"""
        base_config = {
            'input_shape': (20, 6),
            'latent_dim': 32,
            'encoder_layers': [256, 128, 64],
            'decoder_layers': [64, 128, 256],
            'quant_bits': 8,
            'activation': 'relu'
        }
        
        if model_type == "autoencoder":
            return base_config
            
        elif model_type == "variational_autoencoder":
            return {
                **base_config,
                'beta_schedule': {
                    'start': 0.0,
                    'end': 1.0,
                    'warmup_epochs': 50,
                    'cycle_epochs': 20
                }
            }
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
