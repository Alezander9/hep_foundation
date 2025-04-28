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
        self.datasets_dir = self.base_dir / "datasets"
        self.configs_dir = self.base_dir / "configs"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.atlas_manager = atlas_manager or ATLASFileManager()

        # Add feature processor
        self.feature_processor = PhysliteFeatureProcessor()

        # Add state tracking
        self.current_dataset_id = None
        self.current_dataset_path = None
        self.current_dataset_info = None

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
        config_path = self.configs_dir / f"{dataset_id}_config.yaml"

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
        config_path = self.configs_dir / f"{dataset_id}_config.yaml"
        self.logger.info(f"Looking for config at: {config_path}")  # Debug print
        if not config_path.exists():
            self.logger.info(
                f"Available configs: {list(self.configs_dir.glob('*.yaml'))}"
            )  # Debug print
            raise ValueError(f"No configuration found for dataset {dataset_id}")

        return ConfigSerializer.from_yaml(config_path)

    def _create_atlas_dataset(
        self,
        dataset_config: DatasetConfig,
        plot_distributions: bool = False,
        delete_catalogs: bool = True,
    ) -> tuple[str, Path]:
        """
        Create new processed dataset from ATLAS data.

        Args:
            dataset_config: Configuration defining event filters, input features, and labels
            plot_distributions: Whether to generate distribution plots
            delete_catalogs: Whether to delete catalogs after processing

        Returns:
            Tuple containing:
            - Dataset ID
            - Path to created dataset
        """
        self.logger.info("Creating new dataset")

        # Generate dataset ID and paths
        dataset_id = self.generate_dataset_id(dataset_config)
        output_path = self.datasets_dir / f"{dataset_id}.h5"
        self.logger.info(f"Generated dataset ID: {dataset_id}")

        try:
            # Save configuration first
            config_path = self.save_dataset_config(dataset_id, dataset_config)
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

            for run_number in dataset_config.run_numbers:
                self.logger.info(f"Processing run {run_number}")
                try:
                    result = self.feature_processor._process_data(
                        task_config=dataset_config.task_config,
                        run_number=run_number,
                        catalog_limit=dataset_config.catalog_limit,
                        plot_distributions=plot_distributions,
                        delete_catalogs=delete_catalogs,
                    )
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
            with h5py.File(output_path, "w") as f:
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

            return dataset_id, output_path

        except Exception as e:
            # Clean up on failure
            if output_path.exists():
                output_path.unlink()
            if config_path.exists():
                config_path.unlink()
            raise Exception(f"Dataset creation failed: {str(e)}")

    def _create_signal_dataset(
        self, dataset_config: DatasetConfig, plot_distributions: bool = False
    ) -> tuple[str, Path]:
        """
        Create new processed dataset from signal data.

        Args:
            dataset_config: Configuration defining event filters, input features, and labels
            plot_distributions: Whether to generate distribution plots

        Returns:
            Tuple containing:
            - Dataset ID
            - Path to created dataset
        """
        self.logger.info("Creating new signal dataset")

        # Generate dataset ID and paths
        dataset_id = self.generate_dataset_id(dataset_config)
        output_path = self.datasets_dir / "signals" / f"{dataset_id}.h5"

        try:
            # Create signals directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save configuration
            config_path = self.save_dataset_config(dataset_id, dataset_config)

            # Process each signal type separately
            with h5py.File(output_path, "w") as f:
                # Create group for each signal type
                for signal_key in dataset_config.signal_keys:
                    self.logger.info(f"Processing signal type: {signal_key}")

                    # Process data for this signal type
                    inputs, labels, stats = self.feature_processor._process_data(
                        task_config=dataset_config.task_config,
                        signal_key=signal_key,
                        plot_distributions=plot_distributions,
                        delete_catalogs=False,  # Always keep signal catalogs
                    )

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
                            f"scalar/{name}", data=np.array(values), compression="gzip"
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
                            label_subgroup = labels_group.create_group(
                                f"config_{label_idx}"
                            )

                            # Get all label data for this configuration
                            label_data = [label_set[label_idx] for label_set in labels]

                            # Save scalar features
                            scalar_features = {
                                name: []
                                for name in label_data[0]["scalar_features"].keys()
                            }
                            for event_labels in label_data:
                                for name, value in event_labels[
                                    "scalar_features"
                                ].items():
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
                        inputs, labels if labels else None
                    )

                    # Store signal-specific attributes
                    signal_group.attrs.update(
                        {
                            "has_labels": bool(
                                labels and dataset_config.task_config.labels
                            ),
                            "normalization_params": json.dumps(norm_params),
                            "processing_stats": json.dumps(stats),
                        }
                    )

                # Store global attributes
                f.attrs.update(
                    {"dataset_id": dataset_id, "creation_date": str(datetime.now())}
                )

            return dataset_id, output_path

        except Exception as e:
            # Clean up on failure
            if output_path.exists():
                output_path.unlink()
            if config_path.exists():
                config_path.unlink()
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
                self.datasets_dir / f"{self.current_dataset_id}.h5"
            )

            if not self.current_dataset_path.exists():
                self.current_dataset_id, self.current_dataset_path = (
                    self._create_atlas_dataset(
                        dataset_config, delete_catalogs=delete_catalogs
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
        """
        Load signal datasets for evaluation.

        Args:
            dataset_config: Dataset configuration defining data selection and processing
            batch_size: Size of batches in returned dataset
            include_labels: Whether to include labels in the dataset

        Returns:
            Dictionary mapping signal_type to its corresponding TensorFlow dataset
        """
        self.logger.info("Loading signal datasets")
        try:
            # Generate dataset ID and paths
            dataset_id = self.generate_dataset_id(dataset_config)
            dataset_path = self.datasets_dir / "signals" / f"{dataset_id}.h5"

            # Create if doesn't exist
            if not dataset_path.exists():
                dataset_id, dataset_path = self._create_signal_dataset(
                    dataset_config=dataset_config
                )

            # Load datasets
            signal_datasets = {}
            with h5py.File(dataset_path, "r") as f:
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
                self.datasets_dir / f"{self.current_dataset_id}.h5"
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

    def _plot_distributions(
        self,
        pre_selection_stats: dict[str, list],
        post_selection_stats: dict[str, list],
        output_dir: Path,
    ):
        """Create distribution plots and print statistical summaries for track and event features"""
        self.logger.info(f"Generating plots in: {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Print track multiplicity statistics
        self.logger.info("=== Track Multiplicity Statistics ===")
        self.logger.info("Before Selection:")
        self.logger.info(
            f"  Total events: {len(pre_selection_stats['tracks_per_event']):,}"
        )
        self.logger.info(
            f"  Average tracks/event: {np.mean(pre_selection_stats['tracks_per_event']):.2f}"
        )
        self.logger.info(
            f"  Median tracks/event: {np.median(pre_selection_stats['tracks_per_event']):.2f}"
        )
        self.logger.info(f"  Min tracks: {min(pre_selection_stats['tracks_per_event'])}")
        self.logger.info(f"  Max tracks: {max(pre_selection_stats['tracks_per_event'])}")

        self.logger.info("After Selection:")
        self.logger.info(
            f"  Total events: {len(post_selection_stats['tracks_per_event']):,}"
        )
        self.logger.info(
            f"  Average tracks/event: {np.mean(post_selection_stats['tracks_per_event']):.2f}"
        )
        self.logger.info(
            f"  Median tracks/event: {np.median(post_selection_stats['tracks_per_event']):.2f}"
        )
        self.logger.info(f"  Min tracks: {min(post_selection_stats['tracks_per_event'])}")
        self.logger.info(f"  Max tracks: {max(post_selection_stats['tracks_per_event'])}")
        self.logger.info(
            f"  Selection efficiency: {100 * len(post_selection_stats['tracks_per_event']) / len(pre_selection_stats['tracks_per_event']):.1f}%"
        )

        # Use matplotlib style
        plt.style.use("seaborn-v0_8")

        self.logger.info("Creating track multiplicity plot...")
        plt.figure(figsize=(12, 6))

        # Calculate integer bin edges with percentile limits
        min_tracks = max(
            1, int(np.percentile(pre_selection_stats["tracks_per_event"], 1))
        )
        max_tracks = int(np.percentile(pre_selection_stats["tracks_per_event"], 99))

        # Create integer bins between these limits
        bins = np.arange(
            min_tracks - 0.5, max_tracks + 1.5, 1
        )  # +/- 0.5 centers bins on integers

        plt.hist(
            pre_selection_stats["tracks_per_event"],
            bins=bins,
            alpha=0.5,
            label="Before Selection",
            density=True,
        )
        plt.hist(
            post_selection_stats["tracks_per_event"],
            bins=bins,
            alpha=0.5,
            label="After Selection",
            density=True,
        )

        plt.xlabel("Number of Tracks per Event")
        plt.ylabel("Density")
        plt.title("Track Multiplicity Distribution")
        plt.legend()
        plt.grid(True)

        # Set x-axis limits to show the main distribution
        plt.xlim(min_tracks - 1, max_tracks + 1)

        plt.savefig(output_dir / "track_multiplicity.pdf")
        plt.close()

        self.logger.info("Creating track features plot...")
        # 2. Track features distributions (6x2 subplot grid)
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle("Track Feature Distributions (Before vs After Selection)")

        features = ["pt", "eta", "phi", "d0", "z0", "chi2_per_ndof"]
        for feature, ax in zip(features, axes.flat):
            if feature == "pt":
                ax.set_xlabel("pT [GeV]")
                ax.set_xscale("log")
                # Use log-spaced bins for pT
                log_bins = np.logspace(
                    np.log10(
                        max(0.1, np.percentile(pre_selection_stats[feature], 0.1))
                    ),  # min
                    np.log10(np.percentile(pre_selection_stats[feature], 99.9)),  # max
                    50,  # number of bins
                )
                ax.hist(
                    pre_selection_stats[feature],
                    bins=log_bins,
                    alpha=0.5,
                    label="Before Selection",
                    density=True,
                )
                ax.hist(
                    post_selection_stats[feature],
                    bins=log_bins,
                    alpha=0.5,
                    label="After Selection",
                    density=True,
                )
            else:
                # For other features, use percentile-based limits
                x_min = np.percentile(pre_selection_stats[feature], 0.1)
                x_max = np.percentile(pre_selection_stats[feature], 99.9)
                ax.set_xlim(x_min, x_max)
                ax.hist(
                    pre_selection_stats[feature],
                    bins=50,
                    alpha=0.5,
                    label="Before Selection",
                    density=True,
                    range=(x_min, x_max),
                )
                ax.hist(
                    post_selection_stats[feature],
                    bins=50,
                    alpha=0.5,
                    label="After Selection",
                    density=True,
                    range=(x_min, x_max),
                )
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True)

            # Add specific axis labels and ranges
            if feature == "pt":
                ax.set_xlabel("pT [GeV]")
                ax.set_xscale("log")
            elif feature == "eta":
                ax.set_xlabel("η")
            elif feature == "phi":
                ax.set_xlabel("φ")
                ax.set_xlim(-3.5, 3.5)
            elif feature == "d0":
                ax.set_xlabel("d0 [mm]")
            elif feature == "z0":
                ax.set_xlabel("z0 [mm]")

        plt.tight_layout()
        plt.savefig(output_dir / "track_features.pdf")
        plt.close()

        self.logger.info("=== Track Feature Statistics ===")
        features = ["pt", "eta", "phi", "d0", "z0", "chi2_per_ndof"]
        labels = {
            "pt": "pT [GeV]",
            "eta": "η",
            "phi": "φ",
            "d0": "d0 [mm]",
            "z0": "z0 [mm]",
            "chi2_per_ndof": "χ²/ndof",
        }

        for feature in features:
            self.logger.info(f"{labels[feature]}:")
            self.logger.info("  Before Selection:")
            self.logger.info(f"    Mean: {np.mean(pre_selection_stats[feature]):.3f}")
            self.logger.info(f"    Std:  {np.std(pre_selection_stats[feature]):.3f}")
            self.logger.info(f"    Min:  {np.min(pre_selection_stats[feature]):.3f}")
            self.logger.info(f"    Max:  {np.max(pre_selection_stats[feature]):.3f}")
            self.logger.info(f"    Tracks: {len(pre_selection_stats[feature]):,}")

            self.logger.info("  After Selection:")
            self.logger.info(f"    Mean: {np.mean(post_selection_stats[feature]):.3f}")
            self.logger.info(f"    Std:  {np.std(post_selection_stats[feature]):.3f}")
            self.logger.info(f"    Min:  {np.min(post_selection_stats[feature]):.3f}")
            self.logger.info(f"    Max:  {np.max(post_selection_stats[feature]):.3f}")
            self.logger.info(f"    Tracks: {len(post_selection_stats[feature]):,}")

        # Print correlation information
        self.logger.info("=== Feature Correlations ===")
        df = pd.DataFrame(
            {feature: post_selection_stats[feature] for feature in features}
        )
        corr_matrix = df.corr()

        self.logger.info("Correlation Matrix (after selection):")
        pd.set_option("display.float_format", "{:.3f}".format)
        self.logger.info(corr_matrix)

        self.logger.info("Creating correlation plot...")
        # 3. 2D correlation plots
        plt.figure(figsize=(12, 10))
        feature_data = {feature: post_selection_stats[feature] for feature in features}
        df = pd.DataFrame(feature_data)
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Track Feature Correlations (After Selection)")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_correlations.pdf")
        plt.close()

        self.logger.info("Plotting complete!")
