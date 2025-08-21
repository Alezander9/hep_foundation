"""
Configuration for anomaly detection evaluation stage.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AnomalyDetectionEvaluationConfig:
    """Configuration for anomaly detection evaluation stage."""

    run_stage: bool
    sample_size: int

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any]
    ) -> "AnomalyDetectionEvaluationConfig":
        """Create config from dictionary."""
        return cls(
            run_stage=config_dict["run_stage"],
            sample_size=config_dict["sample_size"],
        )
