import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import shutil

from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.atlas_file_manager import ATLASFileManager
from hep_foundation.data.physlite_feature_processor import PhysliteFeatureProcessor
from hep_foundation.data.task_config import TaskConfig
from hep_foundation.utils.utils import ConfigSerializer


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
    include_labels: bool = True
    task_config: Optional[TaskConfig] = None

    def validate(self) -> None:
        """Validate dataset configuration parameters"""
        if not self.run_numbers:
            raise ValueError("run_numbers cannot be empty")
        if self.catalog_limit < 1:
            raise ValueError("catalog_limit must be positive")
        if not 0 <= self.validation_fraction + self.test_fraction < 1:
            raise ValueError("Sum of validation and test fractions must be less than 1")
        if self.task_config is None:
            raise ValueError("task_config must be provided")


class DatasetManager:
    """Manages pre-processed ATLAS datasets with integrated processing capabilities"""

    def __init__(
        self,
        base_dir: str = "processed_datasets",
        atlas_manager: Optional[ATLASFileManager] = None,
    ):
        # Setup logging at INFO level
        self.logger = get_logger(__name__)

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.atlas_manager = atlas_manager or ATLASFileManager()

        # Add feature processor
        self.feature_processor = PhysliteFeatureProcessor()

        # Add state tracking
        self.current_dataset_id = None
        self.current_dataset_path = None
        self.current_dataset_info = None

    def get_dataset_dir(self, dataset_id: str) -> Path:
        """Get the directory path for a specific dataset.
        
        Args:
            dataset_id: The ID of the dataset
            
        Returns:
            Path to the dataset directory
        """
        return self.base_dir / dataset_id

    def get_dataset_paths(self, dataset_id: str) -> tuple[Path, Path]:
        """Get all relevant paths for a dataset.
        
        Args:
            dataset_id: The ID of the dataset
            
        Returns:
            Tuple containing:
            - Path to the dataset HDF5 file
            - Path to the config file
        """
        dataset_dir = self.get_dataset_dir(dataset_id)
        return (
            dataset_dir / "dataset.h5",  # dataset path
            dataset_dir / "config.yaml",  # config path
        )

    def get_current_dataset_id(self) -> str:
        """Get ID of currently loaded dataset"""
        if self.current_dataset_id is None:
            raise ValueError("No dataset currently loaded")
        return self.current_dataset_id

    def get_current_dataset_info(self) -> dict:
        """Get information about the currently loaded dataset"""
        if self.current_dataset_info is None:
            raise ValueError("No dataset currently loaded")
        return self.current_dataset_info

    def get_current_dataset_path(self) -> Path:
        """Get path of currently loaded dataset"""
        if self.current_dataset_path is None:
            raise ValueError("No dataset currently loaded")
        return self.current_dataset_path

    def generate_dataset_id(self, config: DatasetConfig) -> str:
        """Generate a human-readable dataset ID from a DatasetConfig object"""
        if not isinstance(config, DatasetConfig):
            raise ValueError("config must be a DatasetConfig object")

        config_dict = {
            "run_numbers": config.run_numbers,
            "signal_keys": config.signal_keys,
            "catalog_limit": config.catalog_limit,
            "task_config": config.task_config.to_dict() if config.task_config else None,
        }

        # Create a descriptive ID based on dataset type
        if config_dict.get("run_numbers"):
            sorted_run_numbers = sorted(config_dict["run_numbers"])
            run_str = f"{sorted_run_numbers[0]}-{len(sorted_run_numbers)}-{sorted_run_numbers[-1]}"
            id_components = ["dataset", f"runs_{run_str}"]
        elif config_dict.get("signal_keys"):
            signal_str = "_".join(sorted(config_dict["signal_keys"]))
            id_components = ["signal", f"types{signal_str}"]
        else:
            raise ValueError("No run numbers or signal keys provided")

        # Add a short hash for uniqueness
        config_hash = hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:8]

        id_components.append(config_hash)

        result = "_".join(id_components)
        self.logger.info(f"Generated dataset ID: {result}")
        return result

    def save_dataset_config(self, dataset_id: str, config: Union[dict, TaskConfig]):
        """Save full dataset configuration"""
        # Get paths
        dataset_dir = self.get_dataset_dir(dataset_id)
        config_path = dataset_dir / "config.yaml"
        
        # Ensure directory exists
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Convert TaskConfig to dict if needed
        if isinstance(config, TaskConfig):
            config = config.to_dict()

        # Prepare configuration
        full_config = {
            "dataset_id": dataset_id,
            "creation_date": str(datetime.now()),
            "config": config,
            "atlas_version": self.atlas_manager.get_version(),
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "tensorflow": tf.__version__,
                "h5py": h5py.__version__,
            },
        }

        # Save using our serializer
        ConfigSerializer.to_yaml(full_config, config_path)
        return config_path

    def get_dataset_info(self, dataset_id: str) -> dict:
        """Get full dataset information including recreation parameters"""
        config_path = self.get_dataset_dir(dataset_id) / "config.yaml"
        self.logger.info(f"Looking for config at: {config_path}")  # Debug print
        if not config_path.exists():
            self.logger.info(
                f"Available configs: {list(self.get_dataset_dir(dataset_id).glob('*.yaml'))}"
            )  # Debug print
            raise ValueError(f"No configuration found for dataset {dataset_id}")

        return ConfigSerializer.from_yaml(config_path)

    def _create_atlas_dataset(
        self,
        dataset_config: DatasetConfig,
        delete_catalogs: bool = True,
        plot_output: Optional[Path] = None,
    ) -> tuple[str, Path]:
        """
        Create new processed dataset from ATLAS data.

        Args:
            dataset_config: Configuration defining event filters, input features, and labels
            delete_catalogs: Whether to delete catalogs after processing
            plot_output: Optional path to directory for saving plots

        Returns:
            Tuple containing:
            - Dataset ID
            - Path to created dataset
        """
        self.logger.info("Creating new dataset")

        # Generate dataset ID and paths
        dataset_id = self.generate_dataset_id(dataset_config)
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_path, config_path = self.get_dataset_paths(dataset_id)
        
        # Create directory structure
        dataset_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save configuration first
            self.save_dataset_config(dataset_id, dataset_config)
            self.logger.info(f"Saved configuration to: {config_path}")

            # Process all runs
            all_inputs = []
            all_labels = []
            total_stats = {
                "total_events": 0,
                "processed_events": 0,
                "total_features": 0,
                "processing_time": 0,
            }

            first_event_logged = False
            for run_number in dataset_config.run_numbers:
                self.logger.info(f"Processing run {run_number}")
                try:
                    result = self.feature_processor._process_data(
                        task_config=dataset_config.task_config,
                        run_number=run_number,
                        catalog_limit=dataset_config.catalog_limit,
                        delete_catalogs=delete_catalogs,
                        plot_distributions=dataset_config.plot_distributions,
                        plot_output=plot_output,
                        first_event_logged=first_event_logged,
                    )
                    first_event_logged = True
                    inputs, labels, stats = result
                    all_inputs.extend(inputs)
                    all_labels.extend(labels)
                    total_stats["total_events"] += stats["total_events"]
                    total_stats["processed_events"] += stats["processed_events"]
                    total_stats["total_features"] += stats["total_features"]
                except Exception as e:
                    self.logger.error(f"Error unpacking process_data result: {str(e)}")
                    raise

            if not all_inputs:
                raise ValueError("No events passed selection criteria")

            # Create HDF5 dataset
            with h5py.File(dataset_path, "w") as f:
                # Create input features group
                features_group = f.create_group("features")

                # Save scalar features if any exist
                scalar_features = {
                    name: [] for name in all_inputs[0]["scalar_features"].keys()
                }
                for input_data in all_inputs:
                    for name, value in input_data["scalar_features"].items():
                        scalar_features[name].append(value)

                for name, values in scalar_features.items():
                    features_group.create_dataset(
                        f"scalar/{name}", data=np.array(values), compression="gzip"
                    )

                # Save aggregated features if any exist
                for agg_name, agg_data in inputs[0]["aggregated_features"].items():
                    stacked_data = np.stack(
                        [
                            input_data["aggregated_features"][agg_name]
                            for input_data in all_inputs
                        ]
                    )

                    features_group.create_dataset(
                        f"aggregated/{agg_name}", data=stacked_data, compression="gzip"
                    )

                # Create labels group if we have labels
                if all_labels and dataset_config.task_config.labels:
                    labels_group = f.create_group("labels")

                    self.logger.info(
                        f"Generating labels datasets for {len(dataset_config.task_config.labels)} labels"
                    )
                    # Process each label configuration
                    for label_idx, label_config in enumerate(
                        dataset_config.task_config.labels
                    ):
                        label_subgroup = labels_group.create_group(
                            f"config_{label_idx}"
                        )

                        # Get all label data for this configuration
                        label_data = [label_set[label_idx] for label_set in all_labels]

                        # Save scalar features
                        scalar_features = {
                            name: [] for name in label_data[0]["scalar_features"].keys()
                        }
                        for event_labels in label_data:
                            for name, value in event_labels["scalar_features"].items():
                                scalar_features[name].append(value)

                        for name, values in scalar_features.items():
                            label_subgroup.create_dataset(
                                f"scalar/{name}",
                                data=np.array(values),
                                compression="gzip",
                            )

                        # Save aggregated features if any exist
                        for agg_name, agg_data in label_data[0][
                            "aggregated_features"
                        ].items():
                            stacked_data = np.stack(
                                [
                                    event_labels["aggregated_features"][agg_name]
                                    for event_labels in label_data
                                ]
                            )

                            label_subgroup.create_dataset(
                                f"aggregated/{agg_name}",
                                data=stacked_data,
                                compression="gzip",
                            )

                # Compute and store normalization parameters
                norm_params = self.feature_processor._compute_dataset_normalization(
                    all_inputs, all_labels if all_labels else None
                )

                # Store attributes
                f.attrs.update(
                    {
                        "dataset_id": dataset_id,
                        "creation_date": str(datetime.now()),
                        "has_labels": bool(
                            all_labels and dataset_config.task_config.labels
                        ),
                        "normalization_params": json.dumps(norm_params),
                        "processing_stats": json.dumps(total_stats),
                    }
                )

            return dataset_id, dataset_path

        except Exception as e:
            # Clean up on failure
            if dataset_path.exists():
                dataset_path.unlink()
            if config_path.exists():
                config_path.unlink()
            if plot_output and plot_output.exists():
                shutil.rmtree(plot_output)
            if dataset_dir.exists() and not any(dataset_dir.iterdir()):
                dataset_dir.rmdir()
            raise Exception(f"Dataset creation failed: {str(e)}")

    def get_signal_dataset_paths(self, dataset_id: str) -> tuple[Path, Path]:
        """Get all relevant paths for a signal dataset.
        
        Args:
            dataset_id: The ID of the dataset
            
        Returns:
            Tuple containing:
            - Path to the dataset HDF5 file
            - Path to the config file
        """
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_path = dataset_dir / "signal_dataset.h5"
        config_path = dataset_dir / "config.yaml"
        
        self.logger.info(f"[Debug] Signal dataset directory: {dataset_dir}")
        self.logger.info(f"[Debug] Signal dataset path: {dataset_path}")
        self.logger.info(f"[Debug] Signal config path: {config_path}")
        
        return dataset_path, config_path

    def _create_signal_dataset(
        self, 
        dataset_config: DatasetConfig, 
    ) -> tuple[str, Path]:
        """
        Create new processed dataset from signal data.

        Args:
            dataset_config: Configuration defining event filters, input features, and labels

        Returns:
            Tuple containing:
            - Dataset ID
            - Path to created dataset
        """
        self.logger.info("[Debug] Creating new signal dataset")
        
        # Generate dataset ID and paths
        dataset_id = self.generate_dataset_id(dataset_config)
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_path, config_path = self.get_signal_dataset_paths(dataset_id)
        
        self.logger.info(f"[Debug] Generated dataset ID: {dataset_id}")
        self.logger.info(f"[Debug] Signal keys to process: {dataset_config.signal_keys}")
        self.logger.info(f"[Debug] Will create dataset at: {dataset_path}")
        
        try:

            with h5py.File(dataset_path, "w") as f:
                self.logger.info("[Debug] Created HDF5 file, processing signal types...")
                first_event_logged = False
                for signal_key in dataset_config.signal_keys:
                    self.logger.info(f"[Debug] Processing signal type: {signal_key}")

                    # Create plot directory if needed
                    if dataset_config.plot_distributions:
                        plots_dir = dataset_dir / "plots"
                        plots_dir.mkdir(parents=True, exist_ok=True)
                        plot_output = plots_dir / f"{signal_key}_dataset_features.png"
                        self.logger.info(f"[Debug] in signal dataset plots output: {plot_output}")
                    else:
                        plot_output = None

                    # Process data for this signal type
                    inputs, labels, stats = self.feature_processor._process_data(
                        task_config=dataset_config.task_config,
                        signal_key=signal_key,
                        delete_catalogs=False,  # Always keep signal catalogs
                        plot_distributions=dataset_config.plot_distributions,
                        plot_output=plot_output,
                        first_event_logged=first_event_logged,
                    )
                    first_event_logged = True
                    if not inputs:
                        self.logger.warning(f"No events passed selection for {signal_key}")
                        continue

                    # Create signal-specific group
                    signal_group = f.create_group(signal_key)

                    # Create features group
                    features_group = signal_group.create_group("features")

                    # Save scalar features
                    scalar_features = {
                        name: [] for name in inputs[0]["scalar_features"].keys()
                    }
                    for input_data in inputs:
                        for name, value in input_data["scalar_features"].items():
                            scalar_features[name].append(value)

                    for name, values in scalar_features.items():
                        features_group.create_dataset(
                            f"scalar/{name}", 
                            data=np.array(values), 
                            compression="gzip"
                        )

                    # Save aggregated features
                    for agg_name, agg_data in inputs[0]["aggregated_features"].items():
                        stacked_data = np.stack(
                            [
                                input_data["aggregated_features"][agg_name]
                                for input_data in inputs
                            ]
                        )
                        features_group.create_dataset(
                            f"aggregated/{agg_name}",
                            data=stacked_data,
                            compression="gzip",
                        )

                    # Create labels group if we have labels
                    if labels and dataset_config.task_config.labels:
                        labels_group = signal_group.create_group("labels")

                        # Process each label configuration
                        for label_idx, label_config in enumerate(
                            dataset_config.task_config.labels
                        ):
                            label_subgroup = labels_group.create_group(f"config_{label_idx}")

                            # Get all label data for this configuration
                            label_data = [label_set[label_idx] for label_set in labels]

                            # Save scalar features
                            scalar_features = {
                                name: [] for name in label_data[0]["scalar_features"].keys()
                            }
                            for event_labels in label_data:
                                for name, value in event_labels["scalar_features"].items():
                                    scalar_features[name].append(value)

                            for name, values in scalar_features.items():
                                label_subgroup.create_dataset(
                                    f"scalar/{name}",
                                    data=np.array(values),
                                    compression="gzip",
                                )

                            # Save aggregated features if any exist
                            for agg_name, agg_data in label_data[0]["aggregated_features"].items():
                                stacked_data = np.stack(
                                    [
                                        event_labels["aggregated_features"][agg_name]
                                        for event_labels in label_data
                                    ]
                                )
                                label_subgroup.create_dataset(
                                    f"aggregated/{agg_name}",
                                    data=stacked_data,
                                    compression="gzip",
                                )

                    # Compute and store normalization parameters
                    norm_params = self.feature_processor._compute_dataset_normalization(
                        inputs, labels if labels else None
                    )

                    # Store signal-specific attributes
                    signal_group.attrs.update(
                        {
                            "has_labels": bool(labels and dataset_config.task_config.labels),
                            "normalization_params": json.dumps(norm_params),
                            "processing_stats": json.dumps(stats),
                        }
                    )

                # Store global attributes
                f.attrs.update(
                    {"dataset_id": dataset_id, "creation_date": str(datetime.now())}
                )

            return dataset_id, dataset_path

        except Exception as e:
            # Clean up on failure
            if dataset_path.exists():
                dataset_path.unlink()
            raise Exception(f"Signal dataset creation failed: {str(e)}")

    def _get_catalog_paths(
        self,
        run_number: Optional[str] = None,
        signal_key: Optional[str] = None,
        catalog_limit: Optional[int] = None,
    ) -> list[Path]:
        """Get list of catalog paths for either ATLAS data or signal data"""
        if run_number is not None:
            # Get ATLAS data catalogs
            paths = []
            for catalog_idx in range(self.atlas_manager.get_catalog_count(run_number)):
                if catalog_limit and catalog_idx >= catalog_limit:
                    break
                catalog_path = self.atlas_manager.get_run_catalog_path(
                    run_number, catalog_idx
                )
                if not catalog_path.exists():
                    catalog_path = self.atlas_manager.download_run_catalog(
                        run_number, catalog_idx
                    )
                if catalog_path:
                    paths.append(catalog_path)
            self.logger.info(f"Found {len(paths)} catalogs for run {run_number}")
            return paths
        elif signal_key is not None:
            # Get signal data catalog
            catalog_path = self.atlas_manager.get_signal_catalog_path(signal_key, 0)
            if not catalog_path.exists():
                catalog_path = self.atlas_manager.download_signal_catalog(signal_key, 0)
            self.logger.info(f"Found signal catalog for {signal_key}")
            return [catalog_path] if catalog_path else []
        else:
            raise ValueError("Must provide either run_number or signal_key")

    def load_atlas_datasets(
        self,
        dataset_config: DatasetConfig,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        batch_size: int = 1000,
        shuffle_buffer: int = 10000,
        include_labels: bool = False,
        delete_catalogs: bool = True,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and split dataset into train/val/test."""
        self.logger.info("Loading datasets")
        try:
            # Generate dataset ID and load/create dataset file
            self.current_dataset_id = self.generate_dataset_id(dataset_config)
            self.current_dataset_path = (
                self.get_dataset_dir(self.current_dataset_id) / "dataset.h5"
            )
            plot_output = self.get_dataset_dir(self.current_dataset_id) / "plots/atlas_dataset_features.png"

            if not self.current_dataset_path.exists():
                self.current_dataset_id, self.current_dataset_path = (
                    self._create_atlas_dataset(
                        dataset_config,
                        delete_catalogs=delete_catalogs,
                        plot_output=plot_output,
                    )
                )

            # Load and process dataset
            with h5py.File(self.current_dataset_path, "r") as f:
                # Load features and labels
                features_dict = self.feature_processor._load_features_from_group(
                    f["features"]
                )
                labels_dict = (
                    self.feature_processor._load_labels_from_group(f["labels"])
                    if include_labels and "labels" in f
                    else None
                )

                # Get normalization parameters and number of events
                norm_params = json.loads(f.attrs["normalization_params"])
                n_events = next(iter(features_dict.values())).shape[0]

                # Create dataset
                dataset = self.feature_processor._create_normalized_dataset(
                    features_dict, norm_params, labels_dict
                )

                # Create splits
                train_size = n_events - int(
                    (validation_fraction + test_fraction) * n_events
                )
                val_size = int(validation_fraction * n_events)

                # Apply splits and batching
                train_dataset = (
                    dataset.take(train_size)
                    .shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )

                remaining = dataset.skip(train_size)
                val_dataset = (
                    remaining.take(val_size)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )
                test_dataset = (
                    remaining.skip(val_size)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )

                return train_dataset, val_dataset, test_dataset

        except Exception as e:
            raise Exception(f"Failed to load dataset: {str(e)}")

    def load_signal_datasets(
        self,
        dataset_config: DatasetConfig,
        batch_size: int = 1000,
        include_labels: bool = False,
    ) -> dict[str, tf.data.Dataset]:
        """Load signal datasets for evaluation."""
        self.logger.info("Loading signal datasets")
        try:
            # Generate dataset ID and load/create dataset file
            self.current_dataset_id = self.generate_dataset_id(dataset_config)
            self.current_dataset_path = (
                self.get_dataset_dir(self.current_dataset_id) / "signal_dataset.h5"
            )

            if not self.current_dataset_path.exists():
                self.current_dataset_id, self.current_dataset_path = (
                    self._create_signal_dataset(
                        dataset_config,
                    )
                )

            # Load datasets
            signal_datasets = {}
            with h5py.File(self.current_dataset_path, "r") as f:
                # Load and process each signal type
                for signal_key in f.keys():
                    signal_group = f[signal_key]

                    # Load features and labels using helper functions
                    features_dict = self.feature_processor._load_features_from_group(
                        signal_group["features"]
                    )
                    labels_dict = None
                    if (
                        include_labels
                        and "labels" in signal_group
                        and signal_group.attrs.get("has_labels", False)
                    ):
                        labels_dict = self.feature_processor._load_labels_from_group(
                            signal_group["labels"]
                        )

                    # Get normalization parameters
                    norm_params = json.loads(signal_group.attrs["normalization_params"])

                    # Create normalized dataset
                    dataset = self.feature_processor._create_normalized_dataset(
                        features_dict=features_dict,
                        norm_params=norm_params,
                        labels_dict=labels_dict,
                    )

                    # Batch and prefetch
                    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                    signal_datasets[signal_key] = dataset

                self.logger.info(f"Successfully loaded {len(signal_datasets)} signal datasets")
                return signal_datasets

        except Exception as e:
            raise Exception(f"Failed to load signal datasets: {str(e)}")

    def load_and_encode_atlas_datasets(
        self,
        dataset_config: DatasetConfig,
        encoder: tf.keras.Model,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        batch_size: int = 1000,
        shuffle_buffer: int = 10000,
        include_labels: bool = False,
        delete_catalogs: bool = True,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load and split dataset into train/val/test and encode the input features using the given encoder model.

        Args:
            dataset_config: Configuration for dataset processing
            encoder: Keras model to use for encoding the input features
            validation_fraction: Fraction of data to use for validation
            test_fraction: Fraction of data to use for testing
            batch_size: Size of batches in returned dataset
            shuffle_buffer: Size of buffer for shuffling training data
            include_labels: Whether to include labels in the dataset
            delete_catalogs: Whether to delete catalogs after processing

        Returns:
            Tuple containing:
            - Training dataset with encoded features
            - Validation dataset with encoded features
            - Test dataset with encoded features
        """
        self.logger.info("Loading and encoding datasets")
        try:
            # Generate dataset ID and load dataset file
            self.current_dataset_id = self.generate_dataset_id(dataset_config)
            self.current_dataset_path = (
                self.get_dataset_dir(self.current_dataset_id) / "dataset.h5"
            )

            if not self.current_dataset_path.exists():
                raise Exception(f"Dataset {self.current_dataset_id} does not exist")

            # Load and process dataset
            with h5py.File(self.current_dataset_path, "r") as f:
                # Load features and labels
                features_dict = self.feature_processor._load_features_from_group(
                    f["features"]
                )
                labels_dict = (
                    self.feature_processor._load_labels_from_group(f["labels"])
                    if include_labels and "labels" in f
                    else None
                )

                # Get normalization parameters and number of events
                norm_params = json.loads(f.attrs["normalization_params"])
                n_events = next(iter(features_dict.values())).shape[0]

                # Create normalized dataset
                dataset = self.feature_processor._create_normalized_dataset(
                    features_dict, norm_params, labels_dict
                )

                # Create splits
                train_size = n_events - int(
                    (validation_fraction + test_fraction) * n_events
                )
                val_size = int(validation_fraction * n_events)

                # Apply splits and batching
                train_dataset = (
                    dataset.take(train_size)
                    .shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )

                remaining = dataset.skip(train_size)
                val_dataset = (
                    remaining.take(val_size)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )
                test_dataset = (
                    remaining.skip(val_size)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )

                # Apply encoder to each dataset
                def encode_features(features, labels=None):
                    # Encode the features using the encoder model
                    encoded_features = encoder(features, training=False)

                    # Return encoded features with original labels if they exist
                    if labels is not None:
                        return encoded_features, labels
                    return encoded_features

                # Map the encoding function to each dataset
                encoded_train_dataset = train_dataset.map(
                    lambda x, y=None: encode_features(x, y),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

                encoded_val_dataset = val_dataset.map(
                    lambda x, y=None: encode_features(x, y),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

                encoded_test_dataset = test_dataset.map(
                    lambda x, y=None: encode_features(x, y),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

                return encoded_train_dataset, encoded_val_dataset, encoded_test_dataset

        except Exception as e:
            raise Exception(f"Failed to load and encode dataset: {str(e)}")