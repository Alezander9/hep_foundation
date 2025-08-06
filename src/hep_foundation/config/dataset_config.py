from dataclasses import dataclass
from typing import Optional

from hep_foundation.config.task_config import TaskConfig


@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""

    run_numbers: list[str]
    signal_keys: Optional[list[str]]
    catalog_limit: int
    validation_fraction: float
    test_fraction: float
    shuffle_buffer: int
    plot_distributions: bool
    save_raw_samples: bool = True  # Whether to save raw event samples
    event_limit: Optional[int] = None
    signal_event_limit: Optional[int] = None  # Separate event limit for signal data
    include_labels: bool = True
    hdf5_compression: bool = True  # Whether to compress HDF5 datasets (True for production, False for fast tests)
    task_config: Optional[TaskConfig] = None

    def validate(self) -> None:
        """Validate dataset configuration parameters"""
        if not self.run_numbers:
            raise ValueError("run_numbers cannot be empty")
        if self.catalog_limit < 1:
            raise ValueError("catalog_limit must be positive")
        if self.event_limit is not None and self.event_limit < 1:
            raise ValueError("event_limit must be positive when specified")
        if self.signal_event_limit is not None and self.signal_event_limit < 1:
            raise ValueError("signal_event_limit must be positive when specified")
        if not 0 <= self.validation_fraction + self.test_fraction < 1:
            raise ValueError("Sum of validation and test fractions must be less than 1")
        if self.task_config is None:
            raise ValueError("task_config must be provided")

    def to_dict(self) -> dict:
        """Convert DatasetConfig to dictionary for serialization"""
        return {
            "run_numbers": self.run_numbers,
            "signal_keys": self.signal_keys,
            "catalog_limit": self.catalog_limit,
            "event_limit": self.event_limit,
            "signal_event_limit": self.signal_event_limit,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "shuffle_buffer": self.shuffle_buffer,
            "plot_distributions": self.plot_distributions,
            "save_raw_samples": self.save_raw_samples,
            "include_labels": self.include_labels,
            "hdf5_compression": self.hdf5_compression,
            "task_config": self.task_config.to_dict() if self.task_config else None,
        }
