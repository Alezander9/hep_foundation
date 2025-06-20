from dataclasses import dataclass
from typing import Any

from hep_foundation.config.logging_config import get_logger


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping: dict[str, Any]
    plot_training: bool

    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
        plot_training: bool,
    ):
        self.logger = get_logger(__name__)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping = {
            "patience": early_stopping_patience,
            "min_delta": early_stopping_min_delta,
        }
        self.plot_training = plot_training

    def validate(self) -> None:
        """Validate training configuration parameters"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.early_stopping["patience"] < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping["min_delta"] < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")

    def to_dict(self) -> dict:
        """
        Convert training configuration to dictionary format.
        
        Returns:
            Dictionary containing training configuration
        """
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "early_stopping": self.early_stopping,
            "plot_training": self.plot_training,
        }
