import json  # Add json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import awkward as ak  # Add awkward import
import h5py
import numpy as np
import tensorflow as tf
import uproot

from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import (
    TaskConfig,
)
from hep_foundation.data.atlas_file_manager import ATLASFileManager
from hep_foundation.data.physlite_derived_features import (
    get_dependencies,
    get_derived_feature,
    is_derived_feature,
)
from hep_foundation.data.processed_event import ProcessedEvent
from hep_foundation.plots.histogram_manager import HistogramManager

# Import plotting utilities


class PhysliteCatalogProcessor:
    """
    Handles processing, filtering, and aggregation of ATLAS PhysLite features.

    This class is responsible for:
    - Applying event and feature filters
    - Processing and aggregating feature arrays
    - Computing normalization parameters
    - Creating normalized datasets
    - Loading features and labels from HDF5 files
    """

    def __init__(
        self,
        atlas_manager=None,
        custom_label_map_path: Optional[
            str
        ] = "src/hep_foundation/data/physlite_plot_labels.json",
    ):
        """
        Initialize the PhysliteCatalogProcessor.

        Args:
            atlas_manager: Optional ATLASFileManager instance
            custom_label_map_path: Optional path to a JSON file for custom plot labels.
        """
        self.logger = get_logger(__name__)

        # Could add configuration parameters here if needed in the future
        # For now, keeping it simple as the class is primarily stateless
        self.atlas_manager = atlas_manager or ATLASFileManager()

        self.custom_label_map = {}
        if custom_label_map_path:
            try:
                map_path = Path(custom_label_map_path)
                if map_path.exists():
                    with open(map_path) as f:
                        self.custom_label_map = json.load(f)
                    self.logger.info(
                        f"Loaded custom plot label map from {custom_label_map_path}"
                    )
                else:
                    self.logger.info(
                        f"Custom plot label map file not found: {custom_label_map_path}. Using default labels."
                    )
            except Exception as e:
                self.logger.error(
                    f"Error loading custom plot label map from {custom_label_map_path}: {e}"
                )

        if uproot is None:
            raise ImportError(
                "Uproot is required for PhysliteCatalogProcessor. Please install it."
            )
        if ak is None:  # Check awkward import
            raise ImportError(
                "Awkward Array is required for PhysliteCatalogProcessor. Please install it."
            )

        # Create event processor instance
        from hep_foundation.data.physlite_event_processor import PhysliteEventProcessor

        self.event_processor = PhysliteEventProcessor(
            atlas_manager=self.atlas_manager,
            custom_label_map_path=custom_label_map_path,
        )

        # Create histogram manager instance
        self.histogram_manager = HistogramManager()

    def _save_histogram_data_with_manager(
        self,
        scalar_features_dict: dict,
        aggregated_features_dict: dict,
        event_n_tracks_list: list,
        data_type_name: str,
        plot_output: Path,
        plot_data_dir: Optional[Path] = None,
    ) -> None:
        """
        Save histogram data using HistogramManager.

        Args:
            scalar_features_dict: Dictionary of scalar features with lists of values
            aggregated_features_dict: Dictionary of aggregated features
            event_n_tracks_list: List of track counts per event
            data_type_name: Either "post_selection" or "zero_bias"
            plot_output: Path for determining JSON file location
            plot_data_dir: Optional directory where plot data should be saved
        """
        if (
            not scalar_features_dict
            and not aggregated_features_dict
            and not event_n_tracks_list
        ):
            self.logger.warning(f"No histogram data to save for {data_type_name}")
            return

        # Prepare data in format expected by HistogramManager
        histogram_data = {}

        # Note: Track counts are now included directly as aggregator_X_valid_tracks in scalar_features_dict
        # No need to add separate N_Tracks_per_Event

        # Add all features from scalar_features_dict (now contains both scalar and flattened aggregated features)
        for feature_name, values_list in scalar_features_dict.items():
            if values_list and len(values_list) > 0:
                # Apply custom label mapping if available
                display_name = self.custom_label_map.get(feature_name, feature_name)

                # Convert to list if it's a numpy array
                if hasattr(values_list, "tolist"):
                    histogram_data[display_name] = values_list.tolist()
                else:
                    histogram_data[display_name] = list(values_list)

        if not histogram_data:
            self.logger.warning(f"No valid histogram data for {data_type_name}")
            return

        # Determine file path - use provided plot_data_dir or derive from plot_output
        if plot_data_dir is None:
            plot_data_dir = plot_output.parent.parent / "plot_data"
        plot_data_dir.mkdir(parents=True, exist_ok=True)

        if data_type_name == "zero_bias":
            json_file_path = plot_data_dir / (
                plot_output.stem + "_zero_bias_hist_data.json"
            )
        else:
            json_file_path = plot_data_dir / (plot_output.stem + "_hist_data.json")

        # Use HistogramManager to save with percentile coordination
        self.histogram_manager.save_to_hist_file(
            data=histogram_data,
            file_path=json_file_path,
            nbins=100,  # Use same default as original system
            use_percentile_file=True,
            update_percentile_file=True,
            use_percentile_cache=True,
        )

        self.logger.info(f"Saved {data_type_name} histogram data to {json_file_path}")

    def _collect_histogram_data(
        self,
        result: ProcessedEvent,
        data_type: str,
        scalar_features_dict: dict,
        aggregated_features_dict: dict,
        event_n_tracks_list: list,
        raw_samples_list: Optional[list] = None,
    ) -> Optional[int]:
        """
        Collect histogram data from a processed event result using the proper histogram data format.

        Args:
            result: The ProcessedEvent instance
            data_type: Either "zero_bias" or "post_selection"
            scalar_features_dict: Dictionary to accumulate scalar features (branch_name -> list of values)
            aggregated_features_dict: Dictionary to accumulate aggregated features (NOT USED - kept for compatibility)
            event_n_tracks_list: List to accumulate track counts
            raw_samples_list: Optional list to accumulate complete raw event samples

        Returns:
            Number of tracks for the first aggregator, or None if no aggregators
        """
        # Get histogram data from the new format
        hist_data = result.get_histogram_data(data_type)

        # Collect all histogram data
        for branch_name, branch_values in hist_data.items():
            # Flatten the branch values and extend the accumulated list
            if isinstance(branch_values, list):
                scalar_features_dict[branch_name].extend(branch_values)
            else:
                # Handle case where it's a single value
                scalar_features_dict[branch_name].append(branch_values)

        # Get track count from the new format
        num_tracks = result.get_num_tracks_for_plot()

        # Add track count for first aggregator if we have aggregators
        if result.input_selection_data.event_raw_aggregators_data:
            first_agg_idx = 0
            # Find first aggregator that passed filters
            for agg_idx in range(
                len(result.input_selection_data.event_raw_aggregators_data)
            ):
                if result.input_selection_data.event_aggregators_pass_filters[agg_idx]:
                    first_agg_idx = agg_idx
                    break

            # Add aggregator track count as a special histogram feature
            track_count_key = f"aggregator_{first_agg_idx}_valid_tracks"
            n_valid_elements = (
                result.input_selection_data.get_aggregator_filtered_count(first_agg_idx)
            )
            scalar_features_dict[track_count_key].append(n_valid_elements)

        # Add overall track count to the event tracks list
        if num_tracks is not None:
            event_n_tracks_list.append(num_tracks)

        # Collect complete raw sample if requested
        if raw_samples_list is not None:
            # Create a serializable copy of the result
            raw_sample = self._prepare_raw_sample_for_storage(result, data_type)
            raw_samples_list.append(raw_sample)

        return num_tracks

    def _prepare_raw_sample_for_storage(
        self, result: ProcessedEvent, data_type: str
    ) -> dict:
        """
        Prepare a processed event result for JSON serialization and storage.

        Uses the new format's built-in JSON serialization.

        Args:
            result: The ProcessedEvent instance
            data_type: Either "zero_bias" or "post_selection" to determine which histogram data to keep

        Returns:
            A serializable dictionary suitable for JSON storage with relevant histogram data
        """
        # Use the new format's built-in JSON serialization
        return result.to_json_dict(data_type)

    def _save_raw_samples(
        self,
        raw_samples_list: list,
        data_type_name: str,
        plot_output: Path,
        sample_data_dir: Optional[Path] = None,
    ) -> None:
        """
        Save raw event samples to JSON files for later reuse.

        Args:
            raw_samples_list: List of raw event samples
            data_type_name: Either "post_selection" or "zero_bias"
            plot_output: Path for determining JSON file location
            sample_data_dir: Optional directory where raw sample data should be saved
        """
        if not raw_samples_list:
            self.logger.warning(f"No raw samples to save for {data_type_name}")
            return

        # Determine file path - use provided sample_data_dir or derive from plot_output
        if sample_data_dir is None:
            sample_data_dir = plot_output.parent.parent / "sample_data"
        sample_data_dir.mkdir(parents=True, exist_ok=True)

        if data_type_name == "zero_bias":
            json_file_path = sample_data_dir / (
                plot_output.stem + "_zero_bias_raw_samples.json"
            )
        else:
            json_file_path = sample_data_dir / (plot_output.stem + "_raw_samples.json")

        # Save raw samples with metadata
        raw_samples_data = {
            "metadata": {
                "data_type": data_type_name,
                "num_samples": len(raw_samples_list),
                "creation_date": str(datetime.now()),
                "format_version": "1.0",
            },
            "samples": raw_samples_list,
        }

        try:
            with open(json_file_path, "w") as f:
                json.dump(raw_samples_data, f, indent=2)

            self.logger.info(
                f"Saved {len(raw_samples_list)} {data_type_name} raw samples to {json_file_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save raw samples to {json_file_path}: {e}")

    def process_catalogs(
        self,
        task_config: TaskConfig,
        catalog_paths: list[Path],
        data_type_label: str,
        event_limit: Optional[int] = None,
        plot_distributions: bool = False,
        save_raw_samples: bool = True,
        delete_catalogs: bool = True,
        plot_output: Optional[Path] = None,
        first_event_logged: bool = True,
        plot_data_dir: Optional[Path] = None,
        sample_data_dir: Optional[Path] = None,
    ) -> tuple[
        list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], dict, Optional[dict]
    ]:
        """
        Process ATLAS or signal data using task configuration.

        Args:
            task_config: Configuration defining event filters, input features, and labels
            catalog_paths: List of paths to catalog files to process
            data_type_label: Label describing the data type being processed (for logging)
            event_limit: Optional limit on number of events to process per catalog (only events that pass selection count)
            plot_distributions: Whether to generate distribution plots
            delete_catalogs: Whether to delete catalogs after processing
            plot_output: Optional complete path (including filename) for saving plots
            first_event_logged: Whether the first event has been logged
            plot_data_dir: Optional directory where plot data JSON files should be saved
            sample_data_dir: Optional directory where raw sample JSON files should be saved

        Returns:
            Tuple containing:
            - List of processed input features (each a dict with scalar and aggregated features)
            - List of processed label features (each a list of dicts for multiple label configs)
            - Processing statistics
            - Empty dict (preserved for backward compatibility)
        """
        self.logger.info(
            f"Processing {data_type_label} data from {len(catalog_paths)} catalogs"
        )

        if plot_distributions:
            if plot_output is None:
                self.logger.warning(
                    "Plot distributions enabled but no output path provided. Disabling plotting."
                )
                plot_distributions = False
            else:
                # Ensure parent directory exists
                plot_output.parent.mkdir(parents=True, exist_ok=True)

        # Initialize statistics
        stats = {
            "total_events": 0,
            "processed_events": 0,
            "total_features": 0,
            "processing_time": 0.0,
        }

        processed_inputs = []
        processed_labels = []

        # Collect all initially required branch names (including derived ones)
        initially_required_branches = self._get_required_branches(task_config)
        self.logger.info(
            f"Initially required branches (incl. derived): {initially_required_branches}"
        )

        # Separate derived features and identify their dependencies
        derived_features_requested = set()
        actual_branches_to_read = set()
        dependencies_to_add = set()

        for branch_name in initially_required_branches:
            if is_derived_feature(branch_name):
                derived_features_requested.add(branch_name)
                deps = get_dependencies(branch_name)
                if deps:
                    dependencies_to_add.update(deps)
                else:
                    # This case should be handled by definition checks, but log if it occurs
                    self.logger.warning(
                        f"Derived feature '{branch_name}' has no dependencies defined."
                    )
            else:
                # This is a real branch, add it directly
                actual_branches_to_read.add(branch_name)

        # Add the identified dependencies to the set of branches to read
        actual_branches_to_read.update(dependencies_to_add)

        if derived_features_requested:
            self.logger.info(
                f"Identified derived features: {derived_features_requested}"
            )
            self.logger.info(
                f"Added dependencies for derived features: {dependencies_to_add}"
            )
        self.logger.info(
            f"Actual branches to read from file: {actual_branches_to_read}"
        )

        # Check if we have anything to read
        if not actual_branches_to_read:
            self.logger.warning(
                "No actual branches identified to read from the file after processing dependencies. Check TaskConfig and derived feature definitions."
            )
            return [], [], stats, None  # Return empty results

        # Use provided catalog paths
        self.logger.info(f"Processing {len(catalog_paths)} catalog paths.")

        # --- Plotting Setup ---
        plotting_enabled = plot_distributions and plot_output is not None
        max_plot_samples_total = 5000  # Overall target for plotting samples

        # Separate counters for zero-bias vs post-selection samples
        total_plot_samples_count = 0  # Counter for post-selection samples
        total_zero_bias_samples_count = 0  # Counter for zero-bias samples
        samples_per_catalog = 0  # Target samples from each catalog

        if plotting_enabled and len(catalog_paths) > 0:
            samples_per_catalog = max(1, max_plot_samples_total // len(catalog_paths))
            self.logger.info(
                f"Plotting enabled. Aiming for ~{samples_per_catalog} samples per catalog for both zero-bias and post-selection (total target: {max_plot_samples_total} each)."
            )

        elif plotting_enabled:
            self.logger.warning("Plotting enabled, but no catalog paths found.")
            plotting_enabled = False  # Disable if no catalogs

        # Data accumulators for plotting (only if enabled) - DUAL SETS
        # Post-selection data
        sampled_scalar_features = defaultdict(list) if plotting_enabled else None
        sampled_aggregated_features = defaultdict(list) if plotting_enabled else None
        sampled_event_n_tracks_list = (
            [] if plotting_enabled else None
        )  # For track multiplicity
        sampled_raw_events = (
            [] if (plotting_enabled and save_raw_samples) else None
        )  # Raw samples for post-selection

        # Zero-bias data
        zero_bias_scalar_features = defaultdict(list) if plotting_enabled else None
        zero_bias_aggregated_features = defaultdict(list) if plotting_enabled else None
        zero_bias_event_n_tracks_list = (
            [] if plotting_enabled else None
        )  # For track multiplicity
        zero_bias_raw_events = (
            [] if (plotting_enabled and save_raw_samples) else None
        )  # Raw samples for zero-bias
        # --- End Plotting Setup ---

        # --- Main Processing Loop ---
        for catalog_idx, catalog_path in enumerate(catalog_paths):
            # Reset counters for this specific catalog
            current_catalog_samples_count = 0  # Counter for post-selection samples
            current_catalog_samples_zero_bias_count = 0  # Counter for zero-bias samples
            current_catalog_processed_events = 0  # Counter for event_limit
            catalog_event_limit_reached = False  # Flag to break out of batch loop

            self.logger.info(
                f"Processing catalog {catalog_idx + 1}/{len(catalog_paths)} with path: {catalog_path}"
            )

            try:
                catalog_start_time = datetime.now()
                catalog_stats = {"events": 0, "processed": 0}

                with uproot.open(catalog_path) as file:
                    tree = file["CollectionTree;1"]

                    # Check which required branches are actually available in the tree
                    available_branches = set(tree.keys())
                    branches_to_read_in_tree = actual_branches_to_read.intersection(
                        available_branches
                    )
                    missing_branches = (
                        actual_branches_to_read - branches_to_read_in_tree
                    )

                    if missing_branches:
                        self.logger.warning(
                            f"Branches required but not found in tree {catalog_path}: {missing_branches}"
                        )
                        # Decide if we should skip or continue with available ones. For now, let's try to continue.
                        if not branches_to_read_in_tree:
                            self.logger.error(
                                f"No required branches available in tree {catalog_path}. Skipping catalog."
                            )
                            continue  # Skip this catalog if none of the needed branches are present

                    # Read only the available required branches
                    for arrays in tree.iterate(
                        branches_to_read_in_tree, library="np", step_size=1000
                    ):
                        # Removed the stop_processing flag check here

                        try:
                            # Check if the returned dictionary itself is empty
                            if not arrays:
                                self.logger.debug("Skipping empty batch dictionary.")
                                continue

                            # Safely get the number of events in the batch
                            # Assumes uproot provides dicts with at least one key if not empty,
                            # and all arrays have the same length.
                            try:
                                first_key = next(iter(arrays.keys()))
                                num_events_in_batch = len(arrays[first_key])
                            except StopIteration:
                                # This handles the case where arrays is non-empty ({}) but has no keys/values
                                self.logger.warning(
                                    "Encountered batch dictionary with no arrays inside. Skipping."
                                )
                                continue
                            if num_events_in_batch == 0:
                                self.logger.debug("Skipping batch with 0 events.")
                                continue
                            catalog_stats["events"] += num_events_in_batch

                            # --- Event Loop ---
                            for evt_idx in range(num_events_in_batch):
                                # Extract data only for the branches read
                                raw_event_data = {
                                    branch_name: arrays[branch_name][evt_idx]
                                    for branch_name in branches_to_read_in_tree
                                    if branch_name in arrays  # Double check presence
                                }
                                # Skip if somehow event data is empty
                                if not raw_event_data:
                                    self.logger.warning(
                                        f"Skipping event {evt_idx} in batch {catalog_idx} because it has no data."
                                    )
                                    continue

                                # --- Calculate Derived Features ---
                                processed_event_data = self._calculate_derived_features(
                                    raw_event_data, derived_features_requested, evt_idx
                                )
                                if processed_event_data is None:
                                    continue  # Skip this event entirely if any derived feature calculation failed

                                # --- Determine if we need more zero-bias samples ---
                                need_more_zero_bias_samples = (
                                    plotting_enabled
                                    and current_catalog_samples_zero_bias_count
                                    < samples_per_catalog
                                )

                                # --- Process Event (using NEW format) ---

                                result, passed_filters = (
                                    self.event_processor.process_event(
                                        processed_event_data,
                                        task_config,
                                        plotting_enabled,
                                        need_more_zero_bias_samples,
                                    )
                                )

                                # --- Handle data collection based on result and filter status ---
                                if result is not None:
                                    # For zero-bias plotting: collect data from ALL events (filter-free) when we need zero-bias samples
                                    if (
                                        plotting_enabled
                                        and current_catalog_samples_zero_bias_count
                                        < samples_per_catalog
                                    ):
                                        # Collect zero-bias histogram data using extracted method
                                        self._collect_histogram_data(
                                            result,
                                            "zero_bias",
                                            zero_bias_scalar_features,
                                            zero_bias_aggregated_features,
                                            zero_bias_event_n_tracks_list,
                                            zero_bias_raw_events,
                                        )

                                        current_catalog_samples_zero_bias_count += 1
                                        total_zero_bias_samples_count += 1

                                    # For filtered data: collect for dataset creation when filters passed
                                    if passed_filters:
                                        # Store the result for dataset creation
                                        # Extract only the arrays for the dataset using ProcessedEvent helper method
                                        input_features_for_dataset = (
                                            result.get_dataset_features()
                                        )

                                        # Check if we actually have useful aggregated features for the dataset
                                        # If all aggregators failed min_length, we shouldn't count this as passed
                                        has_useful_aggregated_features = bool(
                                            input_features_for_dataset[
                                                "aggregated_features"
                                            ]
                                        )

                                        if not has_useful_aggregated_features:
                                            # Event passed scalar filters but failed all aggregator min_length requirements
                                            # Don't include in dataset, but continue for potential zero-bias data collection
                                            continue
                                        # Process labels similarly for the dataset
                                        label_features_for_dataset = (
                                            result.get_label_features()
                                        )

                                        processed_inputs.append(
                                            input_features_for_dataset
                                        )
                                        processed_labels.append(
                                            label_features_for_dataset
                                        )

                                        # Log processed data only for the very first event and set the flag
                                        if not first_event_logged:
                                            self.logger.info(
                                                f"First event raw data (Catalog {catalog_idx + 1}, Batch Event {evt_idx}): {raw_event_data}"
                                            )
                                            self.logger.info(
                                                f"First event processed input_features_for_dataset: {input_features_for_dataset}"
                                            )
                                            self.logger.info(
                                                f"First event processed label_features_for_dataset: {label_features_for_dataset}"
                                            )
                                            first_event_logged = True  # Set flag after logging the first processed event

                                        catalog_stats["processed"] += 1
                                        current_catalog_processed_events += 1

                                        # --- Accumulate post-selection data for plotting ---
                                        if (
                                            plotting_enabled
                                            and current_catalog_samples_count
                                            < samples_per_catalog
                                        ):
                                            # Collect post-selection histogram data using extracted method
                                            self._collect_histogram_data(
                                                result,
                                                "post_selection",
                                                sampled_scalar_features,
                                                sampled_aggregated_features,
                                                sampled_event_n_tracks_list,
                                                sampled_raw_events,
                                            )

                                            current_catalog_samples_count += 1
                                            total_plot_samples_count += 1

                                        # Update feature statistics
                                        # Note: This might need adjustment if derived features impact how stats are counted
                                        if input_features_for_dataset[
                                            "aggregated_features"
                                        ]:
                                            first_aggregator_key = next(
                                                iter(
                                                    input_features_for_dataset[
                                                        "aggregated_features"
                                                    ].keys()
                                                ),
                                                None,
                                            )
                                            if first_aggregator_key:
                                                first_aggregator = (
                                                    input_features_for_dataset[
                                                        "aggregated_features"
                                                    ][first_aggregator_key]
                                                )
                                                # Ensure the aggregator output is not empty before checking shape
                                                if (
                                                    first_aggregator is not None
                                                    and first_aggregator.size > 0
                                                ):
                                                    try:
                                                        # Check for non-zero elements along the feature axis
                                                        stats["total_features"] += (
                                                            np.sum(
                                                                np.any(
                                                                    first_aggregator
                                                                    != 0,
                                                                    axis=1,
                                                                )
                                                            )
                                                        )
                                                    except np.AxisError:
                                                        # Handle potential scalar case if aggregator somehow returns it
                                                        stats["total_features"] += (
                                                            np.sum(
                                                                first_aggregator != 0
                                                            )
                                                        )

                                        # Check event limit for this catalog
                                        if (
                                            event_limit is not None
                                            and current_catalog_processed_events
                                            >= event_limit
                                        ):
                                            self.logger.info(
                                                f"Reached event limit of {event_limit} for catalog {catalog_idx + 1}. Stopping processing of this catalog."
                                            )
                                            catalog_event_limit_reached = True
                                            break  # Exit the event loop for this catalog

                        except Exception as e:
                            # General error handling for the batch processing
                            self.logger.error(
                                f"Error processing batch in catalog {catalog_path}: {str(e)}",
                                exc_info=True,
                            )
                            continue  # Try to continue with the next batch

                        # Check if event limit was reached during this batch
                        if catalog_event_limit_reached:
                            break  # Exit the batch iteration loop

                # Update statistics
                catalog_duration = (datetime.now() - catalog_start_time).total_seconds()
                stats["processing_time"] += catalog_duration
                stats["total_events"] += catalog_stats["events"]
                stats["processed_events"] += catalog_stats["processed"]

                # Print catalog summary
                if catalog_duration > 0:
                    rate = catalog_stats["events"] / catalog_duration
                else:
                    rate = float(
                        "inf"
                    )  # Avoid division by zero if processing was instant

                self.logger.info(
                    f"Catalog {catalog_idx}: {catalog_stats['events']} events read, "
                    f"{catalog_stats['processed']} passed selection, "
                    f"{catalog_duration:.2f}s ({rate:.1f} events/s)"
                )

            except FileNotFoundError:
                self.logger.error(f"Catalog file not found: {catalog_path}. Skipping.")
                continue
            except Exception as e:
                self.logger.error(
                    f"Critical error processing catalog {catalog_path}: {str(e)}",
                    exc_info=True,
                )  # Add traceback
                # Depending on error, might want to stop or continue
                continue  # Try to continue with the next catalog

            finally:
                # Optional: Consider if deleting catalogs is still desired if errors occurred
                if delete_catalogs and Path(catalog_path).exists():
                    try:
                        os.remove(catalog_path)
                        self.logger.info(f"Deleted catalog file: {catalog_path}")
                    except OSError as delete_e:
                        self.logger.error(
                            f"Error deleting catalog file {catalog_path}: {delete_e}"
                        )

        # Add final sample collection summary
        if plotting_enabled:
            self.logger.info(
                f"{data_type_label} - final totals: "
                f"Zero-bias samples: {total_zero_bias_samples_count}, "
                f"Post-selection samples: {total_plot_samples_count}"
            )

        # --- Generate and Save Histogram Data using HistogramManager ---

        # Save post-selection histogram data
        if plotting_enabled and total_plot_samples_count > 0 and plot_output:
            self.logger.info(
                f"Saving post-selection histogram data from {total_plot_samples_count} sampled events"
            )
            self._save_histogram_data_with_manager(
                sampled_scalar_features,
                sampled_aggregated_features,
                sampled_event_n_tracks_list,
                "post_selection",
                plot_output,
                plot_data_dir,
            )

        # Save zero-bias histogram data
        if plotting_enabled and total_zero_bias_samples_count > 0 and plot_output:
            self.logger.info(
                f"Saving zero-bias histogram data from {total_zero_bias_samples_count} sampled events"
            )
            self._save_histogram_data_with_manager(
                zero_bias_scalar_features,
                zero_bias_aggregated_features,
                zero_bias_event_n_tracks_list,
                "zero_bias",
                plot_output,
                plot_data_dir,
            )

        # Save raw samples alongside histogram data
        if plotting_enabled and save_raw_samples and plot_output:
            # Save post-selection raw samples
            if total_plot_samples_count > 0 and sampled_raw_events:
                self.logger.info(
                    f"Saving post-selection raw samples from {len(sampled_raw_events)} events"
                )
                self._save_raw_samples(
                    sampled_raw_events,
                    "post_selection",
                    plot_output,
                    sample_data_dir,
                )

            # Save zero-bias raw samples
            if total_zero_bias_samples_count > 0 and zero_bias_raw_events:
                self.logger.info(
                    f"Saving zero-bias raw samples from {len(zero_bias_raw_events)} events"
                )
                self._save_raw_samples(
                    zero_bias_raw_events,
                    "zero_bias",
                    plot_output,
                    sample_data_dir,
                )

        # --- End of Histogram Data Generation ---

        # Convert stats to native Python types
        stats = {
            "total_events": int(stats["total_events"]),
            "processed_events": int(stats["processed_events"]),
            "total_features": int(stats["total_features"]),
            "processing_time": float(stats["processing_time"]),
        }

        return (
            processed_inputs,
            processed_labels,
            stats,
        )

    def compute_dataset_normalization(
        self,
        inputs: list[dict[str, dict[str, np.ndarray]]],
        labels: Optional[list[list[dict[str, dict[str, np.ndarray]]]]] = None,
    ) -> dict:
        """
        Compute normalization parameters for all features and labels.

        Args:
            inputs: List of input feature dictionaries
            labels: Optional list of label feature dictionaries

        Returns:
            Dictionary containing normalization parameters for all features and labels
        """
        norm_params = {"features": {}}

        # Compute for scalar features
        if inputs[0]["scalar_features"]:
            scalar_features = {name: [] for name in inputs[0]["scalar_features"].keys()}
            for input_data in inputs:
                for name, value in input_data["scalar_features"].items():
                    scalar_features[name].append(value)

            norm_params["features"]["scalar"] = {
                name: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values) or 1.0),
                }
                for name, values in scalar_features.items()
            }

            # Compute for aggregated features
        if inputs[0]["aggregated_features"]:
            norm_params["features"]["aggregated"] = {}
            for agg_name in inputs[0]["aggregated_features"].keys():
                # Stack all events for this aggregator
                stacked_data = np.stack(
                    [
                        input_data["aggregated_features"][agg_name]
                        for input_data in inputs
                    ]
                )

                # Create mask for zero-padded values
                mask = np.any(stacked_data != 0, axis=-1, keepdims=True)
                mask = np.broadcast_to(mask, stacked_data.shape)

                # Create masked array
                masked_data = np.ma.array(stacked_data, mask=~mask)

                # Compute stats along event and element dimensions
                means = np.ma.mean(masked_data, axis=(0, 1)).data
                stds = np.ma.std(masked_data, axis=(0, 1)).data

                # Ensure no zero standard deviations
                stds = np.maximum(stds, 1e-6)

                norm_params["features"]["aggregated"][agg_name] = {
                    "means": means.tolist(),
                    "stds": stds.tolist(),
                }

        # Compute for labels if present
        if labels and labels[0]:
            norm_params["labels"] = []
            for label_idx in range(len(labels[0])):
                label_norm = {}

                # Get all label data for this configuration
                label_data = [label_set[label_idx] for label_set in labels]

                # Compute for scalar features
                if label_data[0]["scalar_features"]:
                    scalar_features = {
                        name: [] for name in label_data[0]["scalar_features"].keys()
                    }
                    for event_labels in label_data:
                        for name, value in event_labels["scalar_features"].items():
                            scalar_features[name].append(value)

                    label_norm["scalar"] = {
                        name: {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values) or 1.0),
                        }
                        for name, values in scalar_features.items()
                    }

                    # Compute for aggregated features
                if label_data[0]["aggregated_features"]:
                    label_norm["aggregated"] = {}
                    for agg_name in label_data[0]["aggregated_features"].keys():
                        stacked_data = np.stack(
                            [
                                event_labels["aggregated_features"][agg_name]
                                for event_labels in label_data
                            ]
                        )

                        mask = np.any(stacked_data != 0, axis=-1, keepdims=True)
                        mask = np.broadcast_to(mask, stacked_data.shape)
                        masked_data = np.ma.array(stacked_data, mask=~mask)

                        means = np.ma.mean(masked_data, axis=(0, 1)).data
                        stds = np.ma.std(masked_data, axis=(0, 1)).data
                        stds = np.maximum(stds, 1e-6)

                        label_norm["aggregated"][agg_name] = {
                            "means": means.tolist(),
                            "stds": stds.tolist(),
                        }

                norm_params["labels"].append(label_norm)

        return norm_params

    def create_normalized_dataset(
        self,
        features_dict: dict[str, np.ndarray],
        norm_params: dict,
        labels_dict: Optional[dict[str, dict[str, np.ndarray]]] = None,
    ) -> tf.data.Dataset:
        """Create a normalized TensorFlow dataset from processed features"""
        n_events = next(iter(features_dict.values())).shape[0]
        normalized_features = []

        # Validate and clean input features before normalization
        cleaned_features_dict = {}
        for name, feature_array in features_dict.items():
            # Check for inf/nan values
            invalid_mask = ~np.isfinite(feature_array)
            if np.any(invalid_mask):
                self.logger.warning(
                    f"Found {np.sum(invalid_mask)} invalid values in feature '{name}', replacing with median"
                )
                # Replace inf/nan with median of valid values
                valid_values = feature_array[np.isfinite(feature_array)]
                if len(valid_values) > 0:
                    replacement_value = np.median(valid_values)
                else:
                    replacement_value = 0.0
                feature_array = feature_array.copy()
                feature_array[invalid_mask] = replacement_value
            cleaned_features_dict[name] = feature_array

        for name in sorted(cleaned_features_dict.keys()):
            feature_array = cleaned_features_dict[name]
            if name.startswith("scalar/"):
                params = norm_params["features"]["scalar"][name.split("/", 1)[1]]
                # Add epsilon to prevent division by zero
                std_safe = max(params["std"], 1e-8)
                normalized = (feature_array - params["mean"]) / std_safe
            else:  # aggregated features
                params = norm_params["features"]["aggregated"][name.split("/", 1)[1]]
                # Add epsilon to prevent division by zero
                stds_safe = np.maximum(np.array(params["stds"]), 1e-8)
                normalized = (feature_array - np.array(params["means"])) / stds_safe

            # Final safety check - clip extreme values
            normalized = np.clip(normalized, -10.0, 10.0)
            normalized_features.append(normalized.reshape(n_events, -1))

        # Concatenate normalized features
        all_features = np.concatenate(normalized_features, axis=1)

        # Final validation
        if not np.all(np.isfinite(all_features)):
            self.logger.error(
                "Still have invalid values after cleaning - this should not happen!"
            )
            # Emergency cleanup
            all_features = np.nan_to_num(
                all_features, nan=0.0, posinf=10.0, neginf=-10.0
            )

        if labels_dict:
            # Create labels organized by configuration (not concatenated)
            # This allows ModelTrainer to select correct label using label_index
            all_label_configs = []

            # Process each label configuration separately
            for config_name in sorted(labels_dict.keys()):
                config_features = labels_dict[config_name]
                normalized_config_labels = []

                for name in sorted(config_features.keys()):
                    label_array = config_features[name]

                    # Clean labels too
                    invalid_mask = ~np.isfinite(label_array)
                    if np.any(invalid_mask):
                        self.logger.warning(
                            f"Found invalid values in label '{name}', replacing with median"
                        )
                        valid_values = label_array[np.isfinite(label_array)]
                        replacement_value = (
                            np.median(valid_values) if len(valid_values) > 0 else 0.0
                        )
                        label_array = label_array.copy()
                        label_array[invalid_mask] = replacement_value

                    if name.startswith("scalar/"):
                        params = norm_params["labels"][int(config_name.split("_")[1])][
                            "scalar"
                        ][name.split("/", 1)[1]]
                        std_safe = max(params["std"], 1e-8)
                        normalized = (label_array - params["mean"]) / std_safe
                    else:  # aggregated features
                        params = norm_params["labels"][int(config_name.split("_")[1])][
                            "aggregated"
                        ][name.split("/", 1)[1]]
                        stds_safe = np.maximum(np.array(params["stds"]), 1e-8)
                        normalized = (
                            label_array - np.array(params["means"])
                        ) / stds_safe

                    # Clip extreme label values
                    normalized = np.clip(normalized, -10.0, 10.0)
                    normalized_config_labels.append(normalized.reshape(n_events, -1))

                # Concatenate features within this config
                config_labels = np.concatenate(normalized_config_labels, axis=1)
                all_label_configs.append(config_labels)

            # Convert to tuple so ModelTrainer can index by label_index
            labels_tuple = tuple(all_label_configs)
            return tf.data.Dataset.from_tensor_slices((all_features, labels_tuple))
        else:
            return tf.data.Dataset.from_tensor_slices(all_features)

    def load_features_from_group(
        self, features_group: h5py.Group
    ) -> dict[str, np.ndarray]:
        """Load features from an HDF5 group."""
        features_dict = {}

        # Load scalar features
        if "scalar" in features_group:
            for name, dataset in features_group["scalar"].items():
                features_dict[f"scalar/{name}"] = dataset[:]

        # Load aggregated features
        if "aggregated" in features_group:
            for name, dataset in features_group["aggregated"].items():
                features_dict[f"aggregated/{name}"] = dataset[:]

        return features_dict

    def load_labels_from_group(
        self, labels_group: h5py.Group
    ) -> dict[str, dict[str, np.ndarray]]:
        """Load labels from an HDF5 group."""
        labels_dict = {}

        for config_name, label_group in labels_group.items():
            config_dict = {}

            # Load scalar features
            if "scalar" in label_group:
                for name, dataset in label_group["scalar"].items():
                    config_dict[f"scalar/{name}"] = dataset[:]

            # Load aggregated features
            if "aggregated" in label_group:
                for name, dataset in label_group["aggregated"].items():
                    config_dict[f"aggregated/{name}"] = dataset[:]

            labels_dict[config_name] = config_dict

        return labels_dict

    def _get_required_branches(self, task_config: TaskConfig) -> set:
        """
        Get set of all required branch names for a given task configuration.
        This includes both real and potentially derived branches at this stage.

        Args:
            task_config: TaskConfig object containing event filters, input features, and labels

        Returns:
            set: Set of branch names required for processing (including derived ones initially)
        """
        required_branches = set()

        # Add event filter branches
        for filter_item in task_config.event_filters:
            required_branches.add(filter_item.branch.name)

        # Process all selection configs (input and labels)
        for selection_config in [task_config.input, *task_config.labels]:
            # Add feature selector branches
            for selector in selection_config.feature_selectors:
                required_branches.add(selector.branch.name)

            # Add aggregator branches
            for aggregator in selection_config.feature_array_aggregators:
                self.logger.info(
                    f"Adding aggregator branches: {aggregator.input_branches}"
                )
                # Add input branches
                for selector in aggregator.input_branches:
                    required_branches.add(selector.branch.name)
                # Add filter branches
                for filter_item in aggregator.filter_branches:
                    required_branches.add(filter_item.branch.name)
                # Add sort branch if present
                if aggregator.sort_by_branch:
                    required_branches.add(aggregator.sort_by_branch.branch.name)

        return required_branches

    def _calculate_derived_features(
        self,
        raw_event_data: dict[str, np.ndarray],
        derived_features_requested: set[str],
        evt_idx: int,
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Calculate derived features for an event.

        Args:
            raw_event_data: Dictionary mapping branch names to their raw values
            derived_features_requested: Set of derived feature names to calculate
            evt_idx: Event index (for logging)

        Returns:
            Dictionary with both raw and derived features, or None if calculation failed
        """
        processed_event_data = raw_event_data.copy()  # Start with real data

        for derived_name in derived_features_requested:
            derived_feature_def = get_derived_feature(derived_name)
            if not derived_feature_def:
                continue  # Should not happen

            # Check if all dependencies were read successfully for this event
            dependencies_present = all(
                dep in raw_event_data for dep in derived_feature_def.dependencies
            )

            if dependencies_present:
                try:
                    # Prepare dependency data for calculation
                    dependency_values = {
                        dep: raw_event_data[dep]
                        for dep in derived_feature_def.dependencies
                    }
                    # Calculate and add to the processed data
                    calculated_value = derived_feature_def.calculate(dependency_values)
                    processed_event_data[derived_name] = calculated_value
                except Exception as calc_e:
                    self.logger.error(
                        f"Error calculating derived feature '{derived_name}' for event {evt_idx} in batch: {calc_e}"
                    )
                    return None  # Signal calculation failure
            else:
                missing_deps = [
                    dep
                    for dep in derived_feature_def.dependencies
                    if dep not in raw_event_data
                ]
                self.logger.warning(
                    f"Cannot calculate derived feature '{derived_name}' due to missing dependencies: {missing_deps}. Skipping for this event."
                )
                return None  # Signal calculation failure

        return processed_event_data
