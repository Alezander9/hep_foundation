from typing import Dict, Optional
from .base_model import BaseModel
from .autoencoder import AutoEncoder
from .variational_autoencoder import VariationalAutoEncoder

class ModelFactory:
    """Factory class for creating different types of models"""
    
    # Define required parameters for each model type
    REQUIRED_PARAMS = {
        "autoencoder": {
            "architecture": {
                "input_shape": "Shape of input data (n_tracks, n_features)",
                "latent_dim": "Dimension of latent space",
                "encoder_layers": "List of layer sizes for encoder",
                "decoder_layers": "List of layer sizes for decoder",
                "activation": "Activation function to use"
            },
            "hyperparameters": {
                "quant_bits": "Number of bits for quantization (optional)"
            }
        },
        "variational_autoencoder": {
            "architecture": {
                "input_shape": "Shape of input data (n_tracks, n_features)",
                "latent_dim": "Dimension of latent space",
                "encoder_layers": "List of layer sizes for encoder",
                "decoder_layers": "List of layer sizes for decoder",
                "activation": "Activation function to use"
            },
            "hyperparameters": {
                "quant_bits": "Number of bits for quantization (optional)",
                "beta_schedule": "Dictionary containing beta annealing parameters"
            }
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
        
        # Check for missing required groups
        required_groups = ["architecture", "hyperparameters"]
        missing_groups = [group for group in required_groups if group not in config]
        if missing_groups:
            raise ValueError(
                f"Configuration missing required groups: {missing_groups}"
            )
        
        # Check for missing required parameters in each group
        required = ModelFactory.REQUIRED_PARAMS[model_type]
        missing = []
        
        # Check architecture parameters
        for param in required["architecture"]:
            if param not in config["architecture"]:
                missing.append(f"architecture.{param}: {required['architecture'][param]}")
        
        # Check required hyperparameters (skip optional ones)
        if model_type == "variational_autoencoder":
            if "beta_schedule" not in config["hyperparameters"]:
                missing.append(f"hyperparameters.beta_schedule: {required['hyperparameters']['beta_schedule']}")
            elif "beta_schedule" in config["hyperparameters"]:
                beta_schedule = config["hyperparameters"]["beta_schedule"]
                required_beta_params = ["start", "end", "warmup_epochs", "cycle_epochs"]
                missing_beta = [param for param in required_beta_params if param not in beta_schedule]
                if missing_beta:
                    missing.append(f"beta_schedule missing parameters: {missing_beta}")
        
        if missing:
            raise ValueError(
                f"Model type '{model_type}' requires the following missing parameters:\n"
                + "\n".join(f"  - {param}" for param in missing)
            )
    
    @staticmethod
    def create_model(model_type: str, config: Dict) -> BaseModel:
        """
        Create a model instance based on type and configuration
        
        Args:
            model_type: Type of model to create
            config: Model configuration dictionary with architecture and hyperparameters groups
            
        Returns:
            Instance of specified model type
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        ModelFactory.validate_config(model_type, config)
        
        # Extract parameters from nested structure
        arch = config["architecture"]
        hyper = config["hyperparameters"]
        
        if model_type == "autoencoder":
            return AutoEncoder(
                input_shape=arch['input_shape'],
                latent_dim=arch['latent_dim'],
                encoder_layers=arch['encoder_layers'],
                decoder_layers=arch['decoder_layers'],
                quant_bits=hyper.get('quant_bits'),
                activation=arch.get('activation', 'relu'),
                name=config.get('name', 'autoencoder')
            )
            
        elif model_type == "variational_autoencoder":
            return VariationalAutoEncoder(
                input_shape=arch['input_shape'],
                latent_dim=arch['latent_dim'],
                encoder_layers=arch['encoder_layers'],
                decoder_layers=arch['decoder_layers'],
                quant_bits=hyper.get('quant_bits'),
                activation=arch.get('activation', 'relu'),
                beta_schedule=hyper['beta_schedule'],
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
            'architecture': {
                'input_shape': (20, 6),
                'latent_dim': 32,
                'encoder_layers': [256, 128, 64],
                'decoder_layers': [64, 128, 256],
                'activation': 'relu'
            },
            'hyperparameters': {
                'quant_bits': 8
            }
        }
        
        if model_type == "autoencoder":
            return base_config
            
        elif model_type == "variational_autoencoder":
            return {
                **base_config,
                'hyperparameters': {
                    **base_config['hyperparameters'],
                    'beta_schedule': {
                        'start': 0.0,
                        'end': 1.0,
                        'warmup_epochs': 50,
                        'cycle_epochs': 20
                    }
                }
            }
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
