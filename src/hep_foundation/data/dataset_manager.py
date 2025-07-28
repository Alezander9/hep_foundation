import hashlib
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import tensorflow as tf

from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.atlas_file_manager import ATLASFileManager
from hep_foundation.data.physlite_feature_processor import PhysliteFeatureProcessor


class DatasetManager:
    """Manages pre-processed ATLAS datasets with integrated processing capabilities"""

    def __init__(
        self,
        base_dir: Union[str, Path] = "_processed_datasets",
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

    @staticmethod
    def _convert_numpy_types(obj):
        """
        Recursively convert NumPy scalar types to native Python types for JSON serialization.

        Args:
            obj: Object that might contain NumPy types

        Returns:
            Object with NumPy types converted to native Python types
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: DatasetManager._convert_numpy_types(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [DatasetManager._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _accumulate_histogram_data(self, accumulated_data: dict, new_data: dict):
        """
        Accumulate histogram data from multiple runs.

        Args:
            accumulated_data: Dictionary to accumulate data into
            new_data: New histogram data to add
        """
        if not new_data:
            self.logger.templog("[HISTOGRAM_ACCUM] No new data to accumulate")
            return

        # Log accumulation details
        new_sample_count = new_data.get("_metadata", {}).get(
            "total_sampled_events", "unknown"
        )
        data_type = new_data.get("_metadata", {}).get("data_type", "unknown")
        signal_key = new_data.get("_metadata", {}).get("signal_key", "unknown")
        self.logger.templog(
            f"[HISTOGRAM_ACCUM] Accumulating {data_type} data from {signal_key} with {new_sample_count} samples"
        )

        # Skip metadata for now - we'll handle it separately
        for feature_name, feature_data in new_data.items():
            if feature_name.startswith("_"):
                continue

            if not feature_data or not isinstance(feature_data, dict):
                continue

            counts = feature_data.get("counts", [])
            bin_edges = feature_data.get("bin_edges", [])

            if not counts or not bin_edges:
                continue

            if feature_name not in accumulated_data:
                # First time seeing this feature - use as reference
                accumulated_data[feature_name] = {
                    "counts": np.array(counts),
                    "bin_edges": np.array(bin_edges),
                    "total_samples": 1,
                }
            else:
                # Accumulate counts - handle potential bin edge mismatches
                new_counts = np.array(counts)
                existing_edges = accumulated_data[feature_name]["bin_edges"]
                new_edges = np.array(bin_edges)

                # Check if bin edges are compatible (same length and reasonably similar)
                if len(existing_edges) == len(new_edges) and np.allclose(
                    existing_edges, new_edges, rtol=1e-3
                ):
                    # Add the counts (they are already density-normalized per run)
                    accumulated_data[feature_name]["counts"] += new_counts
                    accumulated_data[feature_name]["total_samples"] += 1
                else:
                    # Bin edges don't match - this can happen when different runs have different data ranges
                    self.logger.info(
                        f"Bin edges mismatch for feature {feature_name} (shapes: {len(existing_edges)} vs {len(new_edges)}) - using first run's binning"
                    )

                    # For bin edge mismatches, we need to rebin the new data to match existing edges
                    # This ensures we don't lose data from subsequent runs
                    try:
                        # Simple rebinning: use existing bin edges but interpolate new counts
                        # This is approximate but preserves the data contribution
                        if len(new_counts) == len(existing_edges) - 1:
                            # Same number of bins, just slightly different edges - can add directly
                            accumulated_data[feature_name]["counts"] += new_counts
                            accumulated_data[feature_name]["total_samples"] += 1
                            self.logger.info(
                                f"Added rebinned histogram data for feature {feature_name}"
                            )
                        else:
                            # Different number of bins - skip this run but log the issue
                            self.logger.warning(
                                f"Cannot rebin feature {feature_name} - different bin count ({len(new_counts)} vs {len(accumulated_data[feature_name]['counts'])}). Skipping this run's data."
                            )
                            # Note: Don't increment total_samples if we're not adding the data
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to handle bin edge mismatch for feature {feature_name}: {e}"
                        )

        # Note: Do not normalize here to avoid bias - normalization happens once at the end

    def _save_accumulated_histogram_data(
        self,
        accumulated_data: dict,
        total_stats: dict,
        total_sampled_events: int,
        plot_output: Path,
        bin_edges_metadata_path: Optional[Path],
    ):
        """
        Save accumulated histogram data from all runs to a JSON file.

        Args:
            accumulated_data: Accumulated histogram data across all runs
            total_stats: Total statistics across all runs
            total_sampled_events: Total number of sampled events across all runs
            plot_output: Path for the plot output (used to determine JSON filename)
            bin_edges_metadata_path: Optional path to save bin edges metadata
        """
        try:
            # Create the final histogram data structure
            final_histogram_data = {}

            # Add metadata reflecting the full dataset
            final_histogram_data["_metadata"] = {
                "total_events": int(total_stats["total_events"]),
                "total_processed_events": int(total_stats["processed_events"]),
                "total_features": int(total_stats["total_features"]),
                "processing_time": float(total_stats["processing_time"]),
                "total_sampled_events": int(total_sampled_events),
                "signal_key": "background",
                "run_number": None,  # Multiple runs, so no single run number
            }

            # Add the accumulated feature data
            bin_edges_metadata = {}
            for feature_name, feature_data in accumulated_data.items():
                if feature_name.startswith("_"):
                    continue

                # Get the accumulated counts (sum of density-normalized counts from multiple runs)
                accumulated_counts = feature_data["counts"]

                # If we have multiple samples (runs), we need to renormalize properly
                if feature_data["total_samples"] > 1:
                    # Average the accumulated counts across runs
                    averaged_counts = accumulated_counts / feature_data["total_samples"]
                else:
                    averaged_counts = accumulated_counts

                # Ensure proper normalization: the counts should integrate to 1
                # For density histograms, integral = sum(counts) * bin_width
                bin_edges = feature_data["bin_edges"]
                if len(bin_edges) > 1 and len(averaged_counts) > 0:
                    # Calculate bin widths
                    bin_widths = np.diff(bin_edges)
                    # Calculate current integral
                    current_integral = np.sum(averaged_counts * bin_widths)

                    # Renormalize to ensure integral equals 1
                    if current_integral > 0:
                        normalized_counts = averaged_counts / current_integral
                    else:
                        normalized_counts = averaged_counts
                else:
                    normalized_counts = averaged_counts

                final_histogram_data[feature_name] = {
                    "counts": normalized_counts.tolist(),
                    "bin_edges": feature_data["bin_edges"].tolist(),
                }
                bin_edges_metadata[feature_name] = feature_data["bin_edges"].tolist()

            # Save the JSON file
            plot_data_dir = plot_output.parent.parent / "plot_data"
            plot_data_dir.mkdir(parents=True, exist_ok=True)

            data_file_path = plot_data_dir / (plot_output.stem + "_hist_data.json")
            with open(data_file_path, "w") as f:
                json.dump(final_histogram_data, f, indent=4)
            self.logger.info(
                f"Saved accumulated histogram data from all runs to {data_file_path}"
            )

            # Save bin edges metadata if path provided
            if bin_edges_metadata_path and bin_edges_metadata:
                try:
                    with open(bin_edges_metadata_path, "w") as f:
                        json.dump(bin_edges_metadata, f, indent=2)
                    self.logger.info(
                        f"Saved bin edges metadata to {bin_edges_metadata_path}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to save bin edges metadata: {e}")

        except Exception as e:
            self.logger.error(f"Failed to save accumulated histogram data: {e}")

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
            - Path to the dataset config file
        """
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_path = dataset_dir / "dataset.h5"
        config_path = dataset_dir / "_dataset_config.yaml"

        return dataset_path, config_path

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

        # Use the dynamic to_dict() method
        config_dict = config.to_dict()

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

    def save_dataset_config(
        self,
        dataset_id: str,
        config: "DatasetConfig",
        processing_stats: Optional[dict] = None,
    ):
        """Save dataset configuration as separate config and info files"""
        # Get paths
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_config_path = dataset_dir / "_dataset_config.yaml"
        dataset_info_path = dataset_dir / "_dataset_info.json"

        # Ensure directory exists
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Extract clean configuration using to_dict() methods (dynamic!)
        config_dict = config.to_dict()

        # Separate dataset config from task config
        task_config_dict = config_dict.pop("task_config")  # Remove and get task config
        dataset_config_dict = config_dict  # Everything else is dataset config

        # Create clean config YAML (like the source configs)
        clean_config = {"dataset": dataset_config_dict, "task": task_config_dict}

        # Save clean config as YAML
        import yaml

        with open(dataset_config_path, "w") as f:
            yaml.dump(clean_config, f, default_flow_style=False, indent=2)

        # Prepare metadata info
        dataset_info = {
            "dataset_id": dataset_id,
            "creation_date": str(datetime.now()),
            "atlas_version": self.atlas_manager.get_version(),
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "tensorflow": tf.__version__,
                "h5py": h5py.__version__,
            },
        }

        # Add processing stats if provided
        if processing_stats is not None:
            dataset_info["processing_stats"] = self._convert_numpy_types(
                processing_stats
            )

        # Save metadata as JSON
        import json

        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)

        self.logger.info(f"Saved dataset config to: {dataset_config_path}")
        self.logger.info(f"Saved dataset info to: {dataset_info_path}")

        return dataset_config_path, dataset_info_path

    def get_dataset_info(self, dataset_id: str) -> dict:
        """Get dataset information from separate config and info files"""
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_config_path = dataset_dir / "_dataset_config.yaml"
        dataset_info_path = dataset_dir / "_dataset_info.json"

        # Check for required files
        if not dataset_config_path.exists():
            raise ValueError(f"Dataset config file not found: {dataset_config_path}")

        if not dataset_info_path.exists():
            raise ValueError(f"Dataset info file not found: {dataset_info_path}")

        # Load both files and combine
        import json

        import yaml

        with open(dataset_config_path) as f:
            dataset_config = yaml.safe_load(f)
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)

        return {**dataset_info, "config": dataset_config}

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
            # Process all runs
            all_inputs = []
            all_labels = []
            total_stats = {
                "total_events": 0,
                "processed_events": 0,
                "total_features": 0,
                "processing_time": 0,
            }

            # Prepare bin edges metadata path for coordinated histogram binning
            bin_edges_metadata_path = None
            if dataset_config.plot_distributions and plot_output:
                # Create plot_data folder at dataset root level (not inside plots)
                plot_data_dir = dataset_dir / "plot_data"
                plot_data_dir.mkdir(parents=True, exist_ok=True)
                bin_edges_metadata_path = (
                    plot_data_dir / "background_bin_edges_metadata.json"
                )

            # Accumulate histogram data across all runs - separate for post-selection and zero-bias
            accumulated_histogram_data = {}
            accumulated_zero_bias_histogram_data = {}
            accumulated_sampled_events = 0
            accumulated_zero_bias_sampled_events = 0

            # Calculate total catalogs across all runs for proper sample targeting
            total_catalogs_all_runs = (
                len(dataset_config.run_numbers) * dataset_config.catalog_limit
            )
            self.logger.templog(
                f"[ATLAS_DATASET] Total catalogs across {len(dataset_config.run_numbers)} runs: {total_catalogs_all_runs}"
            )

            first_event_logged = False
            for run_number in dataset_config.run_numbers:
                self.logger.info(f"Processing run {run_number}")
                try:
                    result = self.feature_processor._process_data(
                        task_config=dataset_config.task_config,
                        run_number=run_number,
                        catalog_limit=dataset_config.catalog_limit,
                        event_limit=dataset_config.event_limit,
                        delete_catalogs=delete_catalogs,
                        plot_distributions=dataset_config.plot_distributions,
                        plot_output=plot_output,
                        first_event_logged=first_event_logged,
                        bin_edges_metadata_path=bin_edges_metadata_path,
                        return_histogram_data=dataset_config.plot_distributions,
                        total_catalogs_across_all_runs=total_catalogs_all_runs,
                    )
                    first_event_logged = True
                    inputs, labels, stats, histogram_data = result
                    all_inputs.extend(inputs)
                    all_labels.extend(labels)
                    total_stats["total_events"] += stats["total_events"]
                    total_stats["processed_events"] += stats["processed_events"]
                    total_stats["total_features"] += stats["total_features"]
                    total_stats["processing_time"] += stats["processing_time"]

                    # Accumulate histogram data if available (now handles both post-selection and zero-bias)
                    if histogram_data and dataset_config.plot_distributions:
                        # Handle the new structure with both post-selection and zero-bias data
                        if (
                            isinstance(histogram_data, dict)
                            and "post_selection" in histogram_data
                        ):
                            # New format with both datasets
                            post_selection_data = histogram_data.get("post_selection")
                            zero_bias_data = histogram_data.get("zero_bias")

                            if post_selection_data:
                                self._accumulate_histogram_data(
                                    accumulated_histogram_data, post_selection_data
                                )
                                if "_metadata" in post_selection_data:
                                    accumulated_sampled_events += post_selection_data[
                                        "_metadata"
                                    ].get("total_sampled_events", 0)

                            if zero_bias_data:
                                self._accumulate_histogram_data(
                                    accumulated_zero_bias_histogram_data, zero_bias_data
                                )
                                if "_metadata" in zero_bias_data:
                                    accumulated_zero_bias_sampled_events += (
                                        zero_bias_data["_metadata"].get(
                                            "total_sampled_events", 0
                                        )
                                    )
                        else:
                            # Legacy format (fallback for compatibility)
                            self._accumulate_histogram_data(
                                accumulated_histogram_data, histogram_data
                            )
                            if "_metadata" in histogram_data:
                                accumulated_sampled_events += histogram_data[
                                    "_metadata"
                                ].get("total_sampled_events", 0)

                except Exception as e:
                    self.logger.error(f"Error unpacking process_data result: {str(e)}")
                    raise

            if not all_inputs:
                raise ValueError("No events passed selection criteria")

            # Save accumulated histogram data if plotting was enabled (both post-selection and zero-bias)
            if dataset_config.plot_distributions and plot_output:
                # Save post-selection data
                if accumulated_histogram_data:
                    self._save_accumulated_histogram_data(
                        accumulated_histogram_data,
                        total_stats,
                        accumulated_sampled_events,
                        plot_output,
                        bin_edges_metadata_path,
                    )

                # Save zero-bias data
                if accumulated_zero_bias_histogram_data:
                    # Create zero-bias plot output path
                    zero_bias_plot_output = plot_output.parent / (
                        plot_output.stem + "_zero_bias" + plot_output.suffix
                    )
                    self._save_accumulated_histogram_data(
                        accumulated_zero_bias_histogram_data,
                        total_stats,
                        accumulated_zero_bias_sampled_events,
                        zero_bias_plot_output,
                        bin_edges_metadata_path,
                    )

            # Create HDF5 dataset
            with h5py.File(dataset_path, "w") as f:
                # Determine compression setting
                compression = "gzip" if dataset_config.hdf5_compression else None

                # Create input features group and save features using helper function
                features_group = f.create_group("features")
                self._save_features_to_hdf5_group(
                    features_group, all_inputs, compression
                )

                # Create labels group if we have labels
                if all_labels and dataset_config.task_config.labels:
                    labels_group = f.create_group("labels")
                    self._save_labels_to_hdf5_group(
                        labels_group,
                        all_labels,
                        dataset_config.task_config,
                        compression,
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
                        "processing_stats": json.dumps(
                            self._convert_numpy_types(total_stats)
                        ),
                    }
                )

            # Save configuration with processing stats
            dataset_config_path, dataset_info_path = self.save_dataset_config(
                dataset_id, dataset_config, total_stats
            )
            self.logger.info(
                f"Saved configuration with processing stats to: {dataset_config_path} and {dataset_info_path}"
            )

            return dataset_id, dataset_path

        except Exception as e:
            # Clean up on failure
            if dataset_path.exists():
                dataset_path.unlink()
            # Check if config paths were created before trying to clean them up
            dataset_config_path = dataset_dir / "_dataset_config.yaml"
            dataset_info_path = dataset_dir / "_dataset_info.json"
            if dataset_config_path.exists():
                dataset_config_path.unlink()
            if dataset_info_path.exists():
                dataset_info_path.unlink()
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
            - Path to the dataset config file
        """
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_path = dataset_dir / "signal_dataset.h5"
        config_path = dataset_dir / "_dataset_config.yaml"

        return dataset_path, config_path

    def _create_signal_dataset(
        self,
        dataset_config: DatasetConfig,
        background_hist_data_path: Optional[Path] = None,
    ) -> tuple[str, Path]:
        """
        Create new processed dataset from signal data.

        Args:
            dataset_config: Configuration defining event filters, input features, and labels
            background_hist_data_path: Optional path to background histogram data for comparison plot

        Returns:
            Tuple containing:
            - Dataset ID
            - Path to created dataset
        """
        self.logger.info("Creating new signal dataset")

        # Generate dataset ID and paths
        dataset_id = self.generate_dataset_id(dataset_config)
        dataset_dir = self.get_dataset_dir(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        dataset_path, config_path = self.get_signal_dataset_paths(dataset_id)

        self.logger.info(f"Signal keys to process: {dataset_config.signal_keys}")
        self.logger.info(f"Will create dataset at: {dataset_path}")

        collected_signal_hist_data_paths = []
        collected_signal_legend_labels = []

        try:
            with h5py.File(dataset_path, "w") as f:
                self.logger.info("Created HDF5 file, processing signal types...")
                # Determine compression setting
                compression = "gzip" if dataset_config.hdf5_compression else None

                first_event_logged = False
                for signal_key in dataset_config.signal_keys:
                    self.logger.info(f"Processing signal type: {signal_key}")

                    plot_output_for_signal_json = None
                    bin_edges_metadata_path = None
                    if dataset_config.plot_distributions:
                        plots_dir = dataset_dir / "plots"
                        plots_dir.mkdir(parents=True, exist_ok=True)
                        plot_data_dir = dataset_dir / "plot_data"
                        plot_data_dir.mkdir(parents=True, exist_ok=True)

                        # This plot_output is for _process_data to determine the JSON filename
                        # The actual JSON will be saved in plot_data folder
                        plot_output_for_signal_json = (
                            plots_dir / f"{signal_key}_dataset_features.png"
                        )
                        self.logger.info(
                            f"plot_output for signal JSON data: {plot_output_for_signal_json}"
                        )

                        # Use background bin edges metadata if available
                        if background_hist_data_path:
                            # Look for bin edges metadata in the plot_data folder at dataset root level
                            # background_hist_data_path is in dataset_dir/plot_data/filename.json
                            # so background_hist_data_path.parent is dataset_dir/plot_data
                            background_plot_data_dir = background_hist_data_path.parent
                            bin_edges_metadata_path = (
                                background_plot_data_dir
                                / "background_bin_edges_metadata.json"
                            )

                    # Process data for this signal type
                    # IMPORTANT: Ensure plot_distributions=dataset_config.plot_distributions
                    # so that _process_data generates the _hist_data.json files.
                    # Use signal_event_limit if specified, otherwise fall back to event_limit
                    signal_event_limit_to_use = (
                        dataset_config.signal_event_limit or dataset_config.event_limit
                    )
                    self.logger.info(
                        f"[SIGNAL_EVENT_LIMIT_DEBUG] signal_event_limit={dataset_config.signal_event_limit}, "
                        f"event_limit={dataset_config.event_limit}, using={signal_event_limit_to_use}"
                    )
                    inputs, labels, stats, _ = self.feature_processor._process_data(
                        task_config=dataset_config.task_config,
                        signal_key=signal_key,
                        event_limit=signal_event_limit_to_use,
                        delete_catalogs=False,  # Always keep signal catalogs
                        plot_distributions=dataset_config.plot_distributions,
                        plot_output=plot_output_for_signal_json,  # Pass path for JSON saving
                        first_event_logged=first_event_logged,
                        bin_edges_metadata_path=bin_edges_metadata_path,
                        total_catalogs_across_all_runs=1,  # Each signal has exactly 1 catalog
                    )
                    first_event_logged = True
                    if not inputs:
                        self.logger.warning(
                            f"No events passed selection for {signal_key}"
                        )
                        continue

                    # Create signal-specific group
                    signal_group = f.create_group(signal_key)

                    # Create features group and save features using helper function
                    features_group = signal_group.create_group("features")
                    self._save_features_to_hdf5_group(
                        features_group, inputs, compression
                    )

                    # Create labels group if we have labels
                    if labels and dataset_config.task_config.labels:
                        labels_group = signal_group.create_group("labels")
                        self._save_labels_to_hdf5_group(
                            labels_group,
                            labels,
                            dataset_config.task_config,
                            compression,
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
                            "processing_stats": json.dumps(
                                self._convert_numpy_types(stats)
                            ),
                        }
                    )

                    # Collect histogram data path for this signal for the combined plot
                    if (
                        dataset_config.plot_distributions
                        and plot_output_for_signal_json
                    ):
                        # Look for JSON file in plot_data folder
                        plot_data_dir = dataset_dir / "plot_data"
                        signal_json_path = plot_data_dir / (
                            plot_output_for_signal_json.stem + "_hist_data.json"
                        )
                        if signal_json_path.exists():
                            collected_signal_hist_data_paths.append(signal_json_path)
                            collected_signal_legend_labels.append(signal_key)
                        else:
                            self.logger.warning(
                                f"Signal hist_data.json for {signal_key} not found at {signal_json_path} after processing."
                            )

                # Store global attributes
                f.attrs.update(
                    {"dataset_id": dataset_id, "creation_date": str(datetime.now())}
                )

            # --- Create combined comparison plot AFTER all signals are processed ---
            if (
                dataset_config.plot_distributions
                and background_hist_data_path
                and background_hist_data_path.exists()
                and collected_signal_hist_data_paths
            ):
                self.logger.info(
                    "Creating combined input feature distribution comparison plot for background vs. signals."
                )
                hist_data_paths_for_plot = [
                    background_hist_data_path
                ] + collected_signal_hist_data_paths
                legend_labels_for_plot = [
                    "Background (ATLAS)"
                ] + collected_signal_legend_labels

                comparison_plot_output_dir = (
                    dataset_dir / "plots"
                )  # Ensure this dir exists
                comparison_plot_output_dir.mkdir(parents=True, exist_ok=True)
                comparison_plot_output_path = (
                    comparison_plot_output_dir
                    / "comparison_input_features_background_vs_signals.png"
                )

                try:
                    from hep_foundation.data.dataset_visualizer import (
                        create_plot_from_hist_data,
                    )

                    create_plot_from_hist_data(
                        hist_data_paths=hist_data_paths_for_plot,
                        output_plot_path=str(comparison_plot_output_path),
                        legend_labels=legend_labels_for_plot,
                        title_prefix="Input Features: Background vs Signals",
                    )
                    self.logger.info(
                        f"Saved combined comparison plot to {comparison_plot_output_path}"
                    )
                except ImportError:
                    self.logger.error(
                        "Failed to import create_plot_from_hist_data. Ensure dataset_visualizer is accessible."
                    )
                except Exception as e_comp_plot:
                    self.logger.error(
                        f"Failed to create combined comparison plot: {e_comp_plot}"
                    )

                # --- Create zero-bias comparison plot AFTER post-selection plot ---
                self.logger.info(
                    "Creating zero-bias input feature distribution comparison plot for background vs. signals."
                )

                # Find corresponding zero-bias histogram files
                background_zero_bias_hist_path = background_hist_data_path.parent / (
                    background_hist_data_path.stem.replace(
                        "_hist_data", "_zero_bias_hist_data"
                    )
                    + background_hist_data_path.suffix
                )

                zero_bias_signal_hist_paths = []
                zero_bias_legend_labels = []

                # Check if background zero-bias data exists
                if background_zero_bias_hist_path.exists():
                    # Collect zero-bias signal histogram paths
                    for i, signal_hist_path in enumerate(
                        collected_signal_hist_data_paths
                    ):
                        signal_zero_bias_path = signal_hist_path.parent / (
                            signal_hist_path.stem.replace(
                                "_hist_data", "_zero_bias_hist_data"
                            )
                            + signal_hist_path.suffix
                        )
                        if signal_zero_bias_path.exists():
                            zero_bias_signal_hist_paths.append(signal_zero_bias_path)
                            # Add "Zero-bias" to the original legend label
                            original_label = collected_signal_legend_labels[i]
                            zero_bias_legend_labels.append(
                                f"{original_label} (Zero-bias)"
                            )
                        else:
                            self.logger.warning(
                                f"Zero-bias histogram data not found for signal: {signal_zero_bias_path}"
                            )

                    # Create zero-bias plot if we have both background and at least one signal
                    if zero_bias_signal_hist_paths:
                        zero_bias_hist_data_paths_for_plot = [
                            background_zero_bias_hist_path
                        ] + zero_bias_signal_hist_paths
                        zero_bias_legend_labels_for_plot = [
                            "Background (ATLAS) Zero-bias"
                        ] + zero_bias_legend_labels

                        zero_bias_comparison_plot_path = (
                            comparison_plot_output_dir
                            / "comparison_input_features_zero_bias_background_vs_signals.png"
                        )

                        try:
                            create_plot_from_hist_data(
                                hist_data_paths=zero_bias_hist_data_paths_for_plot,
                                output_plot_path=str(zero_bias_comparison_plot_path),
                                legend_labels=zero_bias_legend_labels_for_plot,
                                title_prefix="Input Features Zero-bias: Background vs Signals",
                            )
                            self.logger.info(
                                f"Saved zero-bias comparison plot to {zero_bias_comparison_plot_path}"
                            )
                        except Exception as e_zero_bias_plot:
                            self.logger.error(
                                f"Failed to create zero-bias comparison plot: {e_zero_bias_plot}"
                            )
                    else:
                        self.logger.warning(
                            "No zero-bias signal histogram data found for zero-bias comparison plot."
                        )
                else:
                    self.logger.warning(
                        f"Background zero-bias histogram data not found at {background_zero_bias_hist_path}. Skipping zero-bias comparison plot."
                    )
                # --- End of zero-bias combined plot logic ---

            elif dataset_config.plot_distributions:
                if (
                    not background_hist_data_path
                    or not background_hist_data_path.exists()
                ):
                    self.logger.warning(
                        "Background histogram data not provided or not found for combined plot in _create_signal_dataset."
                    )
                if not collected_signal_hist_data_paths:
                    self.logger.warning(
                        "No signal histogram data found for combined plot in _create_signal_dataset."
                    )
            # --- End of combined plot logic ---

            return dataset_id, dataset_path

        except Exception as e:
            # Clean up on failure
            if dataset_path.exists():
                dataset_path.unlink()
            # Consider cleaning up config_path and plots_dir contents if necessary
            raise Exception(f"Signal dataset creation failed: {str(e)}")

    def _save_features_to_hdf5_group(
        self,
        features_group: h5py.Group,
        inputs: list[dict[str, np.ndarray]],
        compression: Optional[str] = None,
    ) -> None:
        """
        Helper function to save scalar and aggregated features to an HDF5 group.

        Args:
            features_group: HDF5 group to save features to
            inputs: List of input feature dictionaries
            compression: Optional compression setting for datasets
        """
        # Save scalar features if any exist
        if inputs and inputs[0]["scalar_features"]:
            scalar_features = {name: [] for name in inputs[0]["scalar_features"].keys()}
            for input_data in inputs:
                for name, value in input_data["scalar_features"].items():
                    scalar_features[name].append(value)

            for name, values in scalar_features.items():
                features_group.create_dataset(
                    f"scalar/{name}", data=np.array(values), compression=compression
                )

        # Save aggregated features if any exist
        if inputs and inputs[0]["aggregated_features"]:
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
                    compression=compression,
                )

    def _save_labels_to_hdf5_group(
        self,
        labels_group: h5py.Group,
        labels: list[list[dict[str, np.ndarray]]],
        task_config,
        compression: Optional[str] = None,
    ) -> None:
        """
        Helper function to save labels to an HDF5 group.

        Args:
            labels_group: HDF5 group to save labels to
            labels: List of label feature dictionaries
            task_config: Task configuration with label definitions
            compression: Optional compression setting for datasets
        """
        self.logger.info(
            f"Generating labels datasets for {len(task_config.labels)} labels"
        )
        # Process each label configuration
        for label_idx, label_config in enumerate(task_config.labels):
            label_subgroup = labels_group.create_group(f"config_{label_idx}")

            # Get all label data for this configuration
            label_data = [label_set[label_idx] for label_set in labels]

            # Save scalar features
            if label_data and label_data[0]["scalar_features"]:
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
                        compression=compression,
                    )

            # Save aggregated features if any exist
            if label_data and label_data[0]["aggregated_features"]:
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
                        compression=compression,
                    )

    def _load_and_normalize_dataset_from_group(
        self,
        features_group: h5py.Group,
        labels_group: Optional[h5py.Group] = None,
        include_labels: bool = False,
        batch_size: Optional[int] = 1000,
        apply_batching: bool = True,
    ) -> tf.data.Dataset:
        """
        Helper function to load features and labels from HDF5 groups and create normalized dataset.

        Args:
            features_group: HDF5 group containing features
            labels_group: Optional HDF5 group containing labels
            include_labels: Whether to include labels in the dataset
            batch_size: Batch size for the resulting dataset (ignored if apply_batching=False)
            apply_batching: Whether to apply batching and prefetching

        Returns:
            Normalized TensorFlow dataset
        """
        # Load features and labels
        features_dict = self.feature_processor._load_features_from_group(features_group)
        labels_dict = (
            self.feature_processor._load_labels_from_group(labels_group)
            if include_labels and labels_group is not None
            else None
        )

        # Get normalization parameters
        if (
            hasattr(features_group, "attrs")
            and "normalization_params" in features_group.attrs
        ):
            norm_params = json.loads(features_group.attrs["normalization_params"])
        elif (
            hasattr(features_group.parent, "attrs")
            and "normalization_params" in features_group.parent.attrs
        ):
            norm_params = json.loads(
                features_group.parent.attrs["normalization_params"]
            )
        else:
            raise ValueError("Normalization parameters not found in HDF5 file")

        # Create normalized dataset
        dataset = self.feature_processor._create_normalized_dataset(
            features_dict, norm_params, labels_dict
        )

        if apply_batching:
            return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            return dataset

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
            plot_output = (
                self.get_dataset_dir(self.current_dataset_id)
                / "plots/atlas_dataset_features.png"
            )

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

                # Create dataset (unbatched for splitting)
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
        background_hist_data_path: Optional[Path] = None,
    ) -> dict[str, tf.data.Dataset]:
        """Load signal datasets for evaluation."""
        self.logger.info("Loading signal datasets")
        try:
            # Generate dataset ID and load/create dataset file
            self.current_dataset_id = self.generate_dataset_id(dataset_config)
            dataset_dir = self.get_dataset_dir(self.current_dataset_id)
            self.current_dataset_path = dataset_dir / "signal_dataset.h5"

            if not self.current_dataset_path.exists():
                self.current_dataset_id, self.current_dataset_path = (
                    self._create_signal_dataset(
                        dataset_config=dataset_config,
                        background_hist_data_path=background_hist_data_path,
                    )
                )

            # Load datasets
            signal_datasets = {}

            with h5py.File(self.current_dataset_path, "r") as f:
                # Load and process each signal type
                for signal_key in f.keys():
                    signal_group = f[signal_key]

                    # Use helper function to load and normalize dataset
                    dataset = self._load_and_normalize_dataset_from_group(
                        features_group=signal_group["features"],
                        labels_group=signal_group.get("labels")
                        if (
                            include_labels
                            and "labels" in signal_group
                            and signal_group.attrs.get("has_labels", False)
                        )
                        else None,
                        include_labels=include_labels,
                        batch_size=batch_size,
                    )

                    signal_datasets[signal_key] = dataset

            self.logger.info(
                f"Successfully loaded {len(signal_datasets)} signal datasets"
            )
            return signal_datasets

        except Exception as e:
            raise Exception(f"Failed to load signal datasets: {str(e)}")
