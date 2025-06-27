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

from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.evaluation_config import EvaluationConfig
from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import TaskConfig
from hep_foundation.config.training_config import TrainingConfig
from hep_foundation.data.atlas_data import get_run_numbers
from hep_foundation.models.dnn_predictor import DNNPredictorConfig
from hep_foundation.models.variational_autoencoder import VAEConfig


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

        self.logger.info(f"Loading configuration from: {config_path}")

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

        # Create TrainingConfigs
        vae_training_config = self._create_training_config(
            config_dict["training"]["vae"]
        )
        dnn_training_config = self._create_training_config(
            config_dict["training"]["dnn"]
        )

        # Create EvaluationConfig
        evaluation_config = self._create_evaluation_config(
            config_dict.get("evaluation", {})
        )

        # Create Model Configs
        vae_model_config = self._create_vae_model_config(config_dict["models"]["vae"])
        dnn_model_config = self._create_dnn_model_config(config_dict["models"]["dnn"])

        # Extract other settings
        pipeline_settings = config_dict.get("pipeline", {})

        return {
            "task_config": task_config,
            "dataset_config": dataset_config,
            "vae_model_config": vae_model_config,
            "dnn_model_config": dnn_model_config,
            "vae_training_config": vae_training_config,
            "dnn_training_config": dnn_training_config,
            "evaluation_config": evaluation_config,
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
            validation_fraction=dataset_dict["validation_fraction"],
            test_fraction=dataset_dict["test_fraction"],
            shuffle_buffer=dataset_dict["shuffle_buffer"],
            plot_distributions=dataset_dict.get("plot_distributions", True),
            include_labels=dataset_dict.get("include_labels", True),
            task_config=task_config,
        )

    def _create_training_config(self, training_dict: dict[str, Any]) -> TrainingConfig:
        """Create TrainingConfig from dictionary."""
        early_stopping = training_dict.get("early_stopping", {})

        return TrainingConfig(
            batch_size=training_dict["batch_size"],
            learning_rate=training_dict["learning_rate"],
            epochs=training_dict["epochs"],
            early_stopping_patience=early_stopping.get("patience", 10),
            early_stopping_min_delta=early_stopping.get("min_delta", 1e-4),
            plot_training=training_dict.get("plot_training", True),
        )

    def _create_evaluation_config(
        self, evaluation_dict: dict[str, Any]
    ) -> EvaluationConfig:
        """Create EvaluationConfig from dictionary."""
        regression_data_sizes = evaluation_dict.get(
            "regression_data_sizes", [1000, 2000, 5000]
        )
        return EvaluationConfig(
            regression_data_sizes=regression_data_sizes,
            signal_classification_data_sizes=evaluation_dict.get(
                "signal_classification_data_sizes", regression_data_sizes
            ),
            fixed_epochs=evaluation_dict.get("fixed_epochs", 10),
            anomaly_eval_batch_size=evaluation_dict.get(
                "anomaly_eval_batch_size", 1024
            ),
        )

    def _create_vae_model_config(self, vae_dict: dict[str, Any]) -> VAEConfig:
        """Create VAEConfig from dictionary."""
        return VAEConfig(
            model_type=vae_dict["model_type"],
            architecture=vae_dict["architecture"],
            hyperparameters=vae_dict["hyperparameters"],
        )

    def _create_dnn_model_config(self, dnn_dict: dict[str, Any]) -> DNNPredictorConfig:
        """Create DNNPredictorConfig from dictionary."""
        return DNNPredictorConfig(
            model_type=dnn_dict["model_type"],
            architecture=dnn_dict["architecture"],
            hyperparameters=dnn_dict["hyperparameters"],
        )

    def save_config_to_registry(
        self, config_file_path: Path, registry_dir: Path
    ) -> Path:
        """
        Copy the configuration file directly to the registry directory.

        This eliminates the desync problem by storing the exact config file
        used for the experiment.

        Args:
            config_file_path: Path to the original config file
            registry_dir: Directory in the registry where config should be stored

        Returns:
            Path to the saved config file in the registry
        """
        import shutil

        registry_dir.mkdir(parents=True, exist_ok=True)

        # Save with timestamp for uniqueness
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"pipeline_config_{timestamp}.yaml"

        registry_config_path = registry_dir / config_name
        shutil.copy2(config_file_path, registry_config_path)

        self.logger.info(f"Configuration saved to registry: {registry_config_path}")
        return registry_config_path


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
