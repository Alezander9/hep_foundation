"""
Configuration classes for Standalone Regression Pipeline.

This module provides configuration classes specifically designed for
standalone regression tasks without foundation model dependencies.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from hep_foundation.config.logging_config import get_logger


@dataclass
class StandaloneTrainingConfig:
    """Configuration for standalone model training"""

    batch_size: int
    learning_rate: float
    epochs: int
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    plot_training: bool = True
    gradient_clip_norm: Optional[float] = None
    lr_scheduler: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        plot_training: bool = True,
        gradient_clip_norm: Optional[float] = None,
        lr_scheduler: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize standalone training configuration.

        Args:
            batch_size: Training batch size
            learning_rate: Initial learning rate
            epochs: Maximum number of training epochs
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Minimum delta for early stopping
            plot_training: Whether to generate training plots
            gradient_clip_norm: Gradient clipping norm (None to disable)
            lr_scheduler: Learning rate scheduler configuration
        """
        self.logger = get_logger(__name__)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.plot_training = plot_training
        self.gradient_clip_norm = gradient_clip_norm
        self.lr_scheduler = lr_scheduler or {}

    def validate(self) -> None:
        """Validate training configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")

        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")

        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive if specified")

        # Validate learning rate scheduler configuration
        if self.lr_scheduler:
            self._validate_lr_scheduler_config(self.lr_scheduler)

    def _validate_lr_scheduler_config(self, lr_config: Dict[str, Any]) -> None:
        """Validate learning rate scheduler configuration"""
        scheduler_type = lr_config.get("type", "reduce_on_plateau")

        if scheduler_type == "reduce_on_plateau":
            # Validate ReduceLROnPlateau parameters
            valid_monitors = ["loss", "val_loss", "mse", "val_mse", "mae", "val_mae"]
            monitor = lr_config.get("monitor", "val_loss")
            if monitor not in valid_monitors:
                raise ValueError(f"monitor must be one of {valid_monitors}")

            factor = lr_config.get("factor", 0.5)
            if not isinstance(factor, (int, float)) or not (0 < factor < 1):
                raise ValueError("factor must be a number between 0 and 1")

            patience = lr_config.get("patience", 10)
            if not isinstance(patience, int) or patience < 0:
                raise ValueError("patience must be a non-negative integer")

            min_lr = lr_config.get("min_lr", 1e-6)
            if not isinstance(min_lr, (int, float)) or min_lr < 0:
                raise ValueError("min_lr must be a non-negative number")

        elif scheduler_type == "exponential_decay":
            # Validate ExponentialDecay parameters
            decay_steps = lr_config.get("decay_steps", 1000)
            if not isinstance(decay_steps, int) or decay_steps <= 0:
                raise ValueError("decay_steps must be a positive integer")

            decay_rate = lr_config.get("decay_rate", 0.9)
            if not isinstance(decay_rate, (int, float)) or not (0 < decay_rate < 1):
                raise ValueError("decay_rate must be a number between 0 and 1")

        elif scheduler_type == "cosine_decay":
            # Validate CosineDecay parameters
            decay_steps = lr_config.get("decay_steps", 10000)
            if not isinstance(decay_steps, int) or decay_steps <= 0:
                raise ValueError("decay_steps must be a positive integer")

        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert StandaloneTrainingConfig to dictionary for serialization"""
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "plot_training": self.plot_training,
            "gradient_clip_norm": self.gradient_clip_norm,
            "lr_scheduler": self.lr_scheduler,
        }


@dataclass
class StandaloneEvaluationConfig:
    """Configuration for standalone model evaluation"""

    regression_data_sizes: list[int]
    fixed_epochs: int
    create_detailed_plots: bool = True
    include_feature_importance: bool = False
    error_analysis_bins: int = 50
    prediction_sample_size: int = 1000

    def __init__(
        self,
        regression_data_sizes: list[int],
        fixed_epochs: int = 50,
        create_detailed_plots: bool = True,
        include_feature_importance: bool = False,
        error_analysis_bins: int = 50,
        prediction_sample_size: int = 1000,
    ):
        """
        Initialize standalone evaluation configuration.

        Args:
            regression_data_sizes: List of training data sizes for evaluation
            fixed_epochs: Number of epochs for each evaluation run
            create_detailed_plots: Whether to generate detailed analysis plots
            include_feature_importance: Whether to compute feature importance
            error_analysis_bins: Number of bins for error analysis histograms
            prediction_sample_size: Sample size for prediction analysis
        """
        self.logger = get_logger(__name__)
        self.regression_data_sizes = regression_data_sizes
        self.fixed_epochs = fixed_epochs
        self.create_detailed_plots = create_detailed_plots
        self.include_feature_importance = include_feature_importance
        self.error_analysis_bins = error_analysis_bins
        self.prediction_sample_size = prediction_sample_size

    def validate(self) -> None:
        """Validate evaluation configuration parameters"""
        if not self.regression_data_sizes:
            raise ValueError("regression_data_sizes cannot be empty")

        if any(size <= 0 for size in self.regression_data_sizes):
            raise ValueError("All regression_data_sizes must be positive")

        if self.fixed_epochs <= 0:
            raise ValueError("fixed_epochs must be positive")

        if self.error_analysis_bins <= 0:
            raise ValueError("error_analysis_bins must be positive")

        if self.prediction_sample_size <= 0:
            raise ValueError("prediction_sample_size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert StandaloneEvaluationConfig to dictionary for serialization"""
        return {
            "regression_data_sizes": self.regression_data_sizes,
            "fixed_epochs": self.fixed_epochs,
            "create_detailed_plots": self.create_detailed_plots,
            "include_feature_importance": self.include_feature_importance,
            "error_analysis_bins": self.error_analysis_bins,
            "prediction_sample_size": self.prediction_sample_size,
        }


class StandaloneConfigLoader:
    """
    Configuration loader for standalone regression pipeline.

    Loads and processes YAML configuration files specifically for standalone tasks.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def load_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load standalone configuration from dictionary.

        Args:
            config_dict: Dictionary from loaded YAML config

        Returns:
            Dictionary containing typed config objects
        """
        self.logger.info("Creating standalone configuration objects...")

        # Extract standalone training config
        training_dict = config_dict.get("training", {}).get("standalone_dnn", {})
        training_config = self._create_training_config(training_dict)

        # Extract evaluation config
        eval_dict = config_dict.get("evaluation", {})
        evaluation_config = self._create_evaluation_config(eval_dict)

        # Extract model config (will be processed by the pipeline)
        model_dict = config_dict.get("models", {}).get("standalone_dnn", {})

        # Extract other settings
        pipeline_settings = config_dict.get("pipeline", {})
        dataset_settings = config_dict.get("dataset", {})
        task_settings = config_dict.get("task", {})

        return {
            "training_config": training_config,
            "evaluation_config": evaluation_config,
            "model_config_dict": model_dict,
            "pipeline_settings": pipeline_settings,
            "dataset_settings": dataset_settings,
            "task_settings": task_settings,
            "metadata": {
                "name": config_dict.get("name", "unnamed_standalone_experiment"),
                "description": config_dict.get("description", ""),
                "version": config_dict.get("version", "1.0"),
                "created_by": config_dict.get("created_by", "unknown"),
            },
        }

    def _create_training_config(
        self, training_dict: Dict[str, Any]
    ) -> StandaloneTrainingConfig:
        """Create StandaloneTrainingConfig from dictionary."""
        return StandaloneTrainingConfig(
            batch_size=training_dict.get("batch_size", 1024),
            learning_rate=training_dict.get("learning_rate", 0.001),
            epochs=training_dict.get("epochs", 100),
            early_stopping_patience=training_dict.get("early_stopping", {}).get(
                "patience", 15
            ),
            early_stopping_min_delta=training_dict.get("early_stopping", {}).get(
                "min_delta", 1e-4
            ),
            plot_training=training_dict.get("plot_training", True),
            gradient_clip_norm=training_dict.get("gradient_clip_norm"),
            lr_scheduler=training_dict.get("lr_scheduler"),
        )

    def _create_evaluation_config(
        self, eval_dict: Dict[str, Any]
    ) -> StandaloneEvaluationConfig:
        """Create StandaloneEvaluationConfig from dictionary."""
        return StandaloneEvaluationConfig(
            regression_data_sizes=eval_dict.get(
                "regression_data_sizes", [1000, 5000, 10000]
            ),
            fixed_epochs=eval_dict.get("fixed_epochs", 50),
            create_detailed_plots=eval_dict.get("create_detailed_plots", True),
            include_feature_importance=eval_dict.get(
                "include_feature_importance", False
            ),
            error_analysis_bins=eval_dict.get("error_analysis_bins", 50),
            prediction_sample_size=eval_dict.get("prediction_sample_size", 1000),
        )


def load_standalone_config(config_path: Union[str, Any]) -> Dict[str, Any]:
    """
    Load standalone configuration from YAML file or dictionary.

    Args:
        config_path: Path to YAML file or configuration dictionary

    Returns:
        Dictionary containing processed configuration objects
    """
    loader = StandaloneConfigLoader()

    if isinstance(config_path, (str, type(None))):
        # If it's a path, we would need to load YAML (not implemented here)
        # For now, assume it's a dictionary
        raise NotImplementedError("YAML file loading not implemented in this module")
    else:
        # Assume it's already a dictionary
        return loader.load_config(config_path)
