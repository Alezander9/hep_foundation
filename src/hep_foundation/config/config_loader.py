"""
Configuration loading system for HEP Foundation Pipeline.

This module provides a flexible configuration system that:
1. Uses YAML for human-readable, versionable configs
2. Supports Python extension points for dynamic values
3. Creates proper config objects for the pipeline
4. Eliminates desync between defined configs and registry storage
"""

import re
from pathlib import Path
from typing import Any, Union

import yaml

from hep_foundation.config.anomaly_detection_evaluation_config import (
    AnomalyDetectionEvaluationConfig,
)
from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.foundation_model_training_config import (
    FoundationModelTrainingConfig,
)
from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.regression_evaluation_config import (
    RegressionEvaluationConfig,
)
from hep_foundation.config.signal_classification_evaluation_config import (
    SignalClassificationEvaluationConfig,
)
from hep_foundation.config.task_config import TaskConfig
from hep_foundation.data.atlas_data import get_run_numbers


class PipelineConfigLoader:
    """
    Loads and processes pipeline configuration files.

    Features:
    - YAML-based configuration files
    - Python extension points for dynamic values
    - Automatic config object creation
    - Direct config file storage in registry (eliminates desync)
    """

    def __init__(self):
        self.logger = get_logger(__name__)

        # Register extension functions
        self.extension_functions = {
            "get_run_numbers": get_run_numbers,
        }

    def load_config(self, config_path: Union[str, Path]) -> dict[str, Any]:
        """
        Load configuration from YAML file with Python extension support.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing processed configuration
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Read raw YAML content
        with open(config_path) as f:
            content = f.read()

        # Process Python extensions
        processed_content = self._process_python_extensions(content)

        # Parse YAML
        config = yaml.safe_load(processed_content)

        # Store the source file path for registry
        config["_source_config_file"] = str(config_path.absolute())

        return config

    def _process_python_extensions(self, content: str) -> str:
        """
        Process !python tags in YAML content.

        Finds patterns like: !python get_run_numbers()[-3:]
        And replaces them with the evaluated result.
        """
        pattern = r"!python\s+([^\n]+)"

        def replace_extension(match):
            expression = match.group(1).strip()
            self.logger.debug(f"Processing Python extension: {expression}")

            try:
                # Create a safe namespace for evaluation
                namespace = {"__builtins__": {}, **self.extension_functions}

                # Evaluate the expression
                result = eval(expression, namespace)

                # Convert to YAML representation
                return yaml.dump(result, default_flow_style=True).strip()

            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate Python extension '{expression}': {e}"
                )
                raise ValueError(f"Invalid Python extension: {expression}")

        return re.sub(pattern, replace_extension, content)

    def create_config_objects(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Create proper config objects from the loaded configuration.

        Args:
            config_dict: Dictionary from loaded YAML config

        Returns:
            Dictionary containing typed config objects
        """
        self.logger.info("Creating configuration objects...")

        # Create TaskConfig
        task_config = self._create_task_config(config_dict["task"])

        # Create DatasetConfig
        dataset_config = self._create_dataset_config(
            config_dict["dataset"], task_config
        )

        # Create stage-specific configs
        foundation_model_training_config = FoundationModelTrainingConfig.from_dict(
            config_dict["foundation_model_training"]
        )

        anomaly_detection_evaluation_config = (
            AnomalyDetectionEvaluationConfig.from_dict(
                config_dict["anomaly_detection_evaluation"]
            )
        )

        regression_evaluation_config = RegressionEvaluationConfig.from_dict(
            config_dict["regression_evaluation"]
        )

        signal_classification_evaluation_config = (
            SignalClassificationEvaluationConfig.from_dict(
                config_dict["signal_classification_evaluation"]
            )
        )

        # Extract other settings
        pipeline_settings = config_dict.get("pipeline", {})

        return {
            "task_config": task_config,
            "dataset_config": dataset_config,
            "foundation_model_training_config": foundation_model_training_config,
            "anomaly_detection_evaluation_config": anomaly_detection_evaluation_config,
            "regression_evaluation_config": regression_evaluation_config,
            "signal_classification_evaluation_config": signal_classification_evaluation_config,
            "pipeline_settings": pipeline_settings,
            "metadata": {
                "name": config_dict.get("name", "unnamed_experiment"),
                "description": config_dict.get("description", ""),
                "version": config_dict.get("version", "1.0"),
                "created_by": config_dict.get("created_by", "unknown"),
            },
            "_source_config_file": config_dict.get("_source_config_file"),
        }

    def _create_task_config(self, task_dict: dict[str, Any]) -> TaskConfig:
        """Create TaskConfig from dictionary."""
        return TaskConfig.create_from_branch_names(
            event_filter_dict=task_dict.get("event_filters", {}),
            input_features=task_dict.get("input_features", []),
            input_array_aggregators=task_dict.get("input_array_aggregators", []),
            label_features=task_dict.get("label_features", []),
            label_array_aggregators=task_dict.get("label_array_aggregators", []),
        )

    def _create_dataset_config(
        self, dataset_dict: dict[str, Any], task_config: TaskConfig
    ) -> DatasetConfig:
        """Create DatasetConfig from dictionary."""
        return DatasetConfig(
            run_numbers=dataset_dict["run_numbers"],
            signal_keys=dataset_dict.get("signal_keys"),
            catalog_limit=dataset_dict["catalog_limit"],
            event_limit=dataset_dict.get("event_limit"),
            signal_event_limit=dataset_dict.get("signal_event_limit"),
            validation_fraction=dataset_dict["validation_fraction"],
            test_fraction=dataset_dict["test_fraction"],
            shuffle_buffer=dataset_dict["shuffle_buffer"],
            plot_distributions=dataset_dict.get("plot_distributions", True),
            save_raw_samples=dataset_dict.get("save_raw_samples", True),
            include_labels=dataset_dict.get("include_labels", True),
            hdf5_compression=dataset_dict.get(
                "hdf5_compression", True
            ),  # Default to True for backward compatibility
            task_config=task_config,
        )


def load_pipeline_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """
    Convenience function to load pipeline configuration.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing all config objects and settings
    """
    loader = PipelineConfigLoader()
    config_dict = loader.load_config(config_path)
    return loader.create_config_objects(config_dict)
