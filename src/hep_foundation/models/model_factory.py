from typing import Union
from .autoencoder import AutoEncoder, AutoEncoderConfig
from .base_model import BaseModel, ModelConfig
from .dnn_predictor import DNNPredictor, DNNPredictorConfig
from .variational_autoencoder import VAEConfig, VariationalAutoEncoder


class ModelFactory:
    """Factory class for creating different types of models"""

    # Map model types to their respective config classes
    CONFIG_CLASSES = {
        "autoencoder": AutoEncoderConfig,
        "variational_autoencoder": VAEConfig,
        "dnn_predictor": DNNPredictorConfig,
    }

    # Map model types to their respective model classes
    MODEL_CLASSES = {
        "autoencoder": AutoEncoder,
        "variational_autoencoder": VariationalAutoEncoder,
        "dnn_predictor": DNNPredictor,
    }

    @staticmethod
    def create_model(model_type: str, config: Union[dict, ModelConfig]) -> BaseModel:
        """Create a model instance based on type and configuration."""
        # If it's already a config object, use it directly
        if isinstance(config, ModelConfig):
            model_config = config
            # Validate the config object
            model_config.validate()
        else:
            # Legacy support for dict configs
            # 1. Get the config class for this model type
            config_class = ModelFactory.CONFIG_CLASSES[model_type]
            if not config_class:
                raise ValueError(f"Unknown model type: {model_type}")

            # 2. Create and validate config object
            model_config = config_class(
                model_type=model_type,
                architecture=config["architecture"],
                hyperparameters=config["hyperparameters"],
            )

        # 3. Get the model class
        model_class = ModelFactory.MODEL_CLASSES[model_type]

        # 4. Create model with validated config object
        return model_class(config=model_config)

    @staticmethod
    def get_config_template(model_type: str) -> dict:
        """
        Get template configuration for a model type

        Args:
            model_type: Type of model to get configuration for

        Returns:
            Dictionary containing template configuration structure

        Raises:
            ValueError: If model type is unsupported
        """
        if model_type not in ModelFactory.CONFIG_CLASSES:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported types are: {list(ModelFactory.CONFIG_CLASSES.keys())}"
            )

        config_class = ModelFactory.CONFIG_CLASSES[model_type]
        return config_class.get_template()
