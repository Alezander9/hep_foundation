"""
Configuration for regression evaluation stage.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RegressionEvaluationConfig:
    """Configuration for regression evaluation stage."""

    run_stage: bool
    dataset: str
    model_type: str
    architecture: dict[str, Any]
    hyperparameters: dict[str, Any]
    data_sizes: list[int]
    batch_size: int
    learning_rate: float
    encoder_learning_rate: float
    epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    plot_training: bool
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "RegressionEvaluationConfig":
        """Create config from dictionary."""
        model_config = config_dict["model"]
        training_config = config_dict["training"]
        early_stopping = training_config.get("early_stopping", {})

        return cls(
            run_stage=config_dict["run_stage"],
            dataset=config_dict["dataset"],
            model_type=model_config["model_type"],
            architecture=model_config["architecture"],
            hyperparameters=model_config["hyperparameters"],
            data_sizes=config_dict["data_sizes"],
            batch_size=training_config["batch_size"],
            learning_rate=training_config["learning_rate"],
            encoder_learning_rate=training_config["encoder_learning_rate"],
            epochs=training_config["epochs"],
            early_stopping_patience=early_stopping.get("patience", 10),
            early_stopping_min_delta=early_stopping.get("min_delta", 1e-4),
            plot_training=training_config.get("plot_training", True),
            seed=config_dict.get("seed"),
        )
