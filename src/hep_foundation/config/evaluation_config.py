"""
Evaluation configuration for HEP Foundation Pipeline.

This module provides configuration for evaluation tasks including
regression data efficiency studies and anomaly detection settings.
"""

from dataclasses import dataclass

from hep_foundation.config.logging_config import get_logger


@dataclass
class EvaluationConfig:
    """Configuration for evaluation tasks"""

    regression_data_sizes: list[int]
    signal_classification_data_sizes: list[int]
    fixed_epochs: int

    def __init__(
        self,
        regression_data_sizes: list[int],
        signal_classification_data_sizes: list[int] = None,
        fixed_epochs: int = 10,
    ):
        """
        Initialize evaluation configuration.

        Args:
            regression_data_sizes: List of training data sizes for regression evaluation
            signal_classification_data_sizes: List of training data sizes for signal classification evaluation
            fixed_epochs: Number of epochs to use for regression/classification models
        """
        self.logger = get_logger(__name__)
        self.regression_data_sizes = regression_data_sizes
        self.signal_classification_data_sizes = (
            signal_classification_data_sizes or regression_data_sizes
        )
        self.fixed_epochs = fixed_epochs

    def validate(self) -> None:
        """Validate evaluation configuration parameters"""
        if not self.regression_data_sizes:
            raise ValueError("regression_data_sizes cannot be empty")

        if any(size <= 0 for size in self.regression_data_sizes):
            raise ValueError("All regression_data_sizes must be positive")

        if not self.signal_classification_data_sizes:
            raise ValueError("signal_classification_data_sizes cannot be empty")

        if any(size <= 0 for size in self.signal_classification_data_sizes):
            raise ValueError("All signal_classification_data_sizes must be positive")

        if self.fixed_epochs <= 0:
            raise ValueError("fixed_epochs must be positive")

    def to_dict(self) -> dict:
        """Convert EvaluationConfig to dictionary for serialization"""
        return {
            "regression_data_sizes": self.regression_data_sizes,
            "signal_classification_data_sizes": self.signal_classification_data_sizes,
            "fixed_epochs": self.fixed_epochs,
        }
