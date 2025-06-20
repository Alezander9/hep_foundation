"""
Evaluation configuration for HEP Foundation Pipeline.

This module provides configuration for evaluation tasks including
regression data efficiency studies and anomaly detection settings.
"""

from dataclasses import dataclass
from typing import List, Optional

from hep_foundation.config.logging_config import get_logger


@dataclass
class EvaluationConfig:
    """Configuration for evaluation tasks"""

    regression_data_sizes: List[int]
    fixed_epochs: int
    anomaly_eval_batch_size: Optional[int] = 1024

    def __init__(
        self,
        regression_data_sizes: List[int],
        fixed_epochs: int,
        anomaly_eval_batch_size: int = 1024,
    ):
        """
        Initialize evaluation configuration.

        Args:
            regression_data_sizes: List of training data sizes for regression evaluation
            fixed_epochs: Number of epochs to use for regression models
            anomaly_eval_batch_size: Batch size for anomaly detection evaluation
        """
        self.logger = get_logger(__name__)
        self.regression_data_sizes = regression_data_sizes
        self.fixed_epochs = fixed_epochs
        self.anomaly_eval_batch_size = anomaly_eval_batch_size

    def validate(self) -> None:
        """Validate evaluation configuration parameters"""
        if not self.regression_data_sizes:
            raise ValueError("regression_data_sizes cannot be empty")
        
        if any(size <= 0 for size in self.regression_data_sizes):
            raise ValueError("All regression_data_sizes must be positive")
        
        if self.fixed_epochs <= 0:
            raise ValueError("fixed_epochs must be positive")
        
        if self.anomaly_eval_batch_size <= 0:
            raise ValueError("anomaly_eval_batch_size must be positive")

    def to_dict(self) -> dict:
        """Convert EvaluationConfig to dictionary for serialization"""
        return {
            "regression_data_sizes": self.regression_data_sizes,
            "fixed_epochs": self.fixed_epochs,
            "anomaly_eval_batch_size": self.anomaly_eval_batch_size,
        } 