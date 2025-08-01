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

    def process_catalogs(
        self,
        task_config: TaskConfig,
        catalog_paths: list[Path],
        data_type_label: str,
        event_limit: Optional[int] = None,
        plot_distributions: bool = False,
        delete_catalogs: bool = True,
        plot_output: Optional[Path] = None,
        first_event_logged: bool = True,
        bin_edges_metadata_path: Optional[Path] = None,
        return_histogram_data: bool = False,
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
            bin_edges_metadata_path: Optional path to save/load bin edges metadata for coordinated histogram binning
            return_histogram_data: If True, return histogram data instead of saving it to file

        Returns:
            Tuple containing:
            - List of processed input features (each a dict with scalar and aggregated features)
            - List of processed label features (each a list of dicts for multiple label configs)
            - Processing statistics
            - Optional histogram data dictionary (if return_histogram_data=True)
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
        initially_required_branches = self.event_processor.get_required_branches(
            task_config
        )
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
        overall_plot_samples_count = 0  # Counter for post-selection samples
        zero_bias_samples_count = 0  # Counter for zero-bias samples
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
        # Post-selection data (existing)
        sampled_scalar_features = defaultdict(list) if plotting_enabled else None
        sampled_aggregated_features = defaultdict(list) if plotting_enabled else None
        sampled_event_n_tracks_list = (
            [] if plotting_enabled else None
        )  # For track multiplicity
        raw_histogram_data_for_file = (
            {} if plotting_enabled else None
        )  # To store hist data for saving

        # Zero-bias data (new)
        zero_bias_scalar_features = defaultdict(list) if plotting_enabled else None
        zero_bias_aggregated_features = defaultdict(list) if plotting_enabled else None
        zero_bias_event_n_tracks_list = (
            [] if plotting_enabled else None
        )  # For track multiplicity
        zero_bias_histogram_data_for_file = (
            {} if plotting_enabled else None
        )  # To store hist data for saving
        # --- End Plotting Setup ---

        # --- Main Processing Loop ---
        for catalog_idx, catalog_path in enumerate(catalog_paths):
            # Reset counters for this specific catalog
            current_catalog_samples_count = 0
            current_catalog_zero_bias_count = 0
            processed_events_in_catalog = 0  # Counter for event_limit
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
                                    and current_catalog_zero_bias_count
                                    < samples_per_catalog
                                )

                                # --- Process Event (unified method handles both filtered and zero-bias data) ---
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
                                    # For zero-bias plotting: collect data when filters failed but we need zero-bias samples
                                    if (
                                        not passed_filters
                                        and plotting_enabled
                                        and current_catalog_zero_bias_count
                                        < samples_per_catalog
                                    ):
                                        # Accumulate zero-bias data
                                        for name, value in result[
                                            "scalar_features"
                                        ].items():
                                            zero_bias_scalar_features[name].append(
                                                value
                                            )

                                        for agg_key, agg_data_with_plot_names in result[
                                            "aggregated_features"
                                        ].items():
                                            zero_bias_aggregated_features[
                                                agg_key
                                            ].append(agg_data_with_plot_names)

                                        # Get first aggregator's track count for zero-bias
                                        first_agg_key = next(
                                            iter(result["aggregated_features"].keys()),
                                            None,
                                        )
                                        if first_agg_key:
                                            num_tracks = result["aggregated_features"][
                                                first_agg_key
                                            ].get("n_valid_elements")
                                        if num_tracks is not None:
                                            zero_bias_event_n_tracks_list.append(
                                                num_tracks
                                            )

                                        current_catalog_zero_bias_count += 1
                                        zero_bias_samples_count += 1

                                    # For filtered data: collect for dataset creation when filters passed
                                    if passed_filters:
                                        # Store the result for dataset creation
                                        # Correctly extract only the arrays for the dataset
                                        input_features_for_dataset = {
                                            "scalar_features": result[
                                                "scalar_features"
                                            ],
                                            "aggregated_features": {
                                                agg_key: agg_data[
                                                    "array"
                                                ]  # Ensure only array is stored for dataset
                                                for agg_key, agg_data in result[
                                                    "aggregated_features"
                                                ].items()
                                            },
                                        }
                                    # Process labels similarly for the dataset
                                    label_features_for_dataset = []
                                    for label_set in result["label_features"]:
                                        label_set_for_dataset = {
                                            "scalar_features": label_set[
                                                "scalar_features"
                                            ],
                                            "aggregated_features": {
                                                agg_key: agg_data[
                                                    "array"
                                                ]  # Ensure only array is stored for dataset
                                                for agg_key, agg_data in label_set[
                                                    "aggregated_features"
                                                ].items()
                                            },
                                        }
                                        label_features_for_dataset.append(
                                            label_set_for_dataset
                                        )

                                    processed_inputs.append(input_features_for_dataset)
                                    processed_labels.append(label_features_for_dataset)

                                    # Log processed data only for the very first event and set the flag
                                    if not first_event_logged and evt_idx == 0:
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
                                    processed_events_in_catalog += 1

                                    # --- Accumulate post-selection data for plotting ---
                                    if (
                                        plotting_enabled
                                        and current_catalog_samples_count
                                        < samples_per_catalog
                                    ):
                                        # Append scalar features
                                        for name, value in result[
                                            "scalar_features"
                                        ].items():
                                            sampled_scalar_features[name].append(value)

                                        # Append aggregated features using clipped data (after clipping, before padding)
                                        first_agg_key = None
                                        for agg_key, agg_data_with_plot_names in result[
                                            "aggregated_features"
                                        ].items():
                                            if first_agg_key is None:
                                                first_agg_key = agg_key  # Capture first aggregator key

                                        # Use clipped histogram data if available, otherwise fall back to original
                                        if (
                                            "clipped_histogram_data"
                                            in agg_data_with_plot_names
                                        ):
                                            # Extract clipped features array and flatten track-by-track for histogram
                                            clipped_hist_data = (
                                                agg_data_with_plot_names[
                                                    "clipped_histogram_data"
                                                ]
                                            )
                                            clipped_array = clipped_hist_data[
                                                "clipped_features"
                                            ]  # Shape: (n_tracks, n_features)
                                            plot_feature_names = clipped_hist_data[
                                                "plot_feature_names"
                                            ]

                                            # Create a modified aggregator data structure that matches the original format
                                            # but with individual track features flattened for histogram collection
                                            modified_agg_data = {
                                                "array": clipped_array,  # This will be handled differently in histogram generation
                                                "plot_feature_names": plot_feature_names,
                                                "n_valid_elements": clipped_hist_data[
                                                    "clipped_num_tracks"
                                                ],
                                            }
                                            sampled_aggregated_features[agg_key].append(
                                                modified_agg_data
                                            )

                                            # Use clipped track count for plotting from first aggregator only
                                            if agg_key == first_agg_key:
                                                sampled_event_n_tracks_list.append(
                                                    clipped_hist_data[
                                                        "clipped_num_tracks"
                                                    ]
                                                )
                                        else:
                                            # Fallback to original behavior if clipped data not available
                                            sampled_aggregated_features[agg_key].append(
                                                agg_data_with_plot_names
                                            )

                                            # Append num_tracks_for_plot if available (fallback) from first aggregator only
                                            if agg_key == first_agg_key:
                                                num_tracks = result.get(
                                                    "num_tracks_for_plot"
                                                )
                                                if num_tracks is not None:
                                                    sampled_event_n_tracks_list.append(
                                                        num_tracks
                                                    )

                                    current_catalog_samples_count += 1
                                    overall_plot_samples_count += 1

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
                                            if first_aggregator.size > 0:
                                                try:
                                                    # Check for non-zero elements along the feature axis
                                                    stats["total_features"] += np.sum(
                                                        np.any(
                                                            first_aggregator != 0,
                                                            axis=1,
                                                        )
                                                    )
                                                except np.AxisError:
                                                    # Handle potential scalar case if aggregator somehow returns it
                                                    stats["total_features"] += np.sum(
                                                        first_aggregator != 0
                                                    )

                                    # Check event limit for this catalog
                                    if (
                                        event_limit is not None
                                        and processed_events_in_catalog >= event_limit
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
                f"Zero-bias samples: {zero_bias_samples_count}, "
                f"Post-selection samples: {overall_plot_samples_count}"
            )

        # --- Generate and Save Dual Histogram Data (Zero-bias + Post-selection) ---

        # Generate post-selection histogram data (existing behavior)
        if (
            plotting_enabled
            and overall_plot_samples_count > 0
            and raw_histogram_data_for_file is not None
        ):
            self.logger.info(
                f"Preparing post-selection histogram data (clipped tracks, before padding) from {overall_plot_samples_count} sampled events for {plot_output}"
            )

            raw_histogram_data_for_file = self._generate_histogram_data(
                sampled_scalar_features,
                sampled_aggregated_features,
                sampled_event_n_tracks_list,
                overall_plot_samples_count,
                "post_selection",
                bin_edges_metadata_path,
                stats,
                data_type_label,
            )

            # Collect new bin edges that will be saved
            new_bin_edges_metadata = {}

            # Save the post-selection histogram data to plot_data folder or return it
            if (
                raw_histogram_data_for_file
                and plot_output
                and not return_histogram_data
            ):
                data_file_path = None
                try:
                    # Create plot_data folder alongside the plots folder
                    plot_data_dir = plot_output.parent.parent / "plot_data"
                    plot_data_dir.mkdir(parents=True, exist_ok=True)

                    # Save JSON to plot_data folder
                    data_file_path = plot_data_dir / (
                        plot_output.stem + "_hist_data.json"
                    )
                    with open(data_file_path, "w") as f_json:
                        json.dump(raw_histogram_data_for_file, f_json, indent=4)
                    self.logger.info(
                        f"Saved post-selection histogram data to {data_file_path}"
                    )

                except Exception as e_save:
                    self.logger.error(
                        f"Failed to save post-selection histogram data: {e_save}"
                    )

        # Generate zero-bias histogram data (new behavior)
        if (
            plotting_enabled
            and zero_bias_samples_count > 0
            and zero_bias_histogram_data_for_file is not None
        ):
            self.logger.info(
                f"Preparing zero-bias histogram data (truly unfiltered detector data) from {zero_bias_samples_count} sampled events for {plot_output}"
            )

            zero_bias_histogram_data_for_file = self._generate_histogram_data(
                zero_bias_scalar_features,
                zero_bias_aggregated_features,
                zero_bias_event_n_tracks_list,
                zero_bias_samples_count,
                "zero_bias",
                bin_edges_metadata_path,
                stats,
                data_type_label,
            )

            # Save the zero-bias histogram data to plot_data folder or return it
            if (
                zero_bias_histogram_data_for_file
                and plot_output
                and not return_histogram_data
            ):
                data_file_path = None
                try:
                    # Create plot_data folder alongside the plots folder
                    plot_data_dir = plot_output.parent.parent / "plot_data"
                    plot_data_dir.mkdir(parents=True, exist_ok=True)

                    # Save JSON to plot_data folder with zero_bias prefix
                    data_file_path = plot_data_dir / (
                        plot_output.stem + "_zero_bias_hist_data.json"
                    )
                    with open(data_file_path, "w") as f_json:
                        json.dump(zero_bias_histogram_data_for_file, f_json, indent=4)
                    self.logger.info(
                        f"Saved zero-bias histogram data to {data_file_path}"
                    )

                except Exception as e_save:
                    self.logger.error(
                        f"Failed to save zero-bias histogram data: {e_save}"
                    )

            # Save bin edges metadata if we have new bin edges to save
            if bin_edges_metadata_path:
                try:
                    # Extract bin edges from the zero-bias data for saving (since it's processed first)
                    new_bin_edges_metadata = {}
                    for (
                        feature_name,
                        feature_data,
                    ) in zero_bias_histogram_data_for_file.items():
                        if (
                            not feature_name.startswith("_")
                            and "bin_edges" in feature_data
                        ):
                            new_bin_edges_metadata[feature_name] = feature_data[
                                "bin_edges"
                            ]

                    if new_bin_edges_metadata:
                        self._save_bin_edges_metadata(
                            new_bin_edges_metadata, bin_edges_metadata_path
                        )
                except Exception as e_save_bin_edges:
                    self.logger.error(
                        f"Failed to save bin edges metadata: {e_save_bin_edges}"
                    )

        # --- End of Dual Histogram Data Generation ---

        # Convert stats to native Python types
        stats = {
            "total_events": int(stats["total_events"]),
            "processed_events": int(stats["processed_events"]),
            "total_features": int(stats["total_features"]),
            "processing_time": float(stats["processing_time"]),
        }

        # Return both post-selection and zero-bias histogram data if requested
        histogram_data_to_return = None
        if return_histogram_data:
            histogram_data_to_return = {
                "post_selection": raw_histogram_data_for_file,
                "zero_bias": zero_bias_histogram_data_for_file,
            }

        return (
            processed_inputs,
            processed_labels,
            stats,
            histogram_data_to_return,
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
            for agg_name, agg_data in inputs[0]["aggregated_features"].items():
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
                    for agg_name, agg_data in label_data[0][
                        "aggregated_features"
                    ].items():
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

    def _save_bin_edges_metadata(
        self, bin_edges_data: dict, metadata_path: Path
    ) -> None:
        """
        Save bin edges metadata to a JSON file for coordinated histogram binning.

        Args:
            bin_edges_data: Dictionary mapping feature names to their bin edges
            metadata_path: Path to save the metadata file
        """
        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(bin_edges_data, f, indent=2)
            self.logger.info(f"Saved bin edges metadata to {metadata_path}")
        except Exception as e:
            self.logger.error(
                f"Failed to save bin edges metadata to {metadata_path}: {e}"
            )

    def _load_bin_edges_metadata(self, metadata_path: Path) -> Optional[dict]:
        """
        Load bin edges metadata from a JSON file for coordinated histogram binning.

        Args:
            metadata_path: Path to the metadata file

        Returns:
            Dictionary mapping feature names to their bin edges, or None if loading fails
        """
        try:
            if not metadata_path.exists():
                return None
            with open(metadata_path) as f:
                bin_edges_data = json.load(f)
            self.logger.info(f"Loaded bin edges metadata from {metadata_path}")
            return bin_edges_data
        except Exception as e:
            self.logger.error(
                f"Failed to load bin edges metadata from {metadata_path}: {e}"
            )
            return None

    def _check_data_range_compatibility(
        self, values: np.ndarray, existing_bin_edges: np.ndarray, feature_name: str
    ) -> tuple[np.ndarray, int, int]:
        """
        Check if data fits within existing bin edges and filter out-of-range values.

        Args:
            values: New data values
            existing_bin_edges: Existing bin edges
            feature_name: Name of the feature (for logging)

        Returns:
            Tuple of (filtered_values, n_excluded_low, n_excluded_high)
        """
        if values.size == 0:
            return values, 0, 0

        edges_min, edges_max = existing_bin_edges[0], existing_bin_edges[-1]

        # Count out-of-range values
        n_below = np.sum(values < edges_min)
        n_above = np.sum(values > edges_max)

        if n_below > 0 or n_above > 0:
            total_values = len(values)
            pct_below = 100 * n_below / total_values
            pct_above = 100 * n_above / total_values
            self.logger.info(
                f"Feature '{feature_name}': {n_below} values ({pct_below:.1f}%) below bin range, "
                f"{n_above} values ({pct_above:.1f}%) above bin range. Excluding from histogram to maintain consistent binning."
            )

        # Filter out values outside the bin range (don't clip to avoid edge pileup)
        mask = (values >= edges_min) & (values <= edges_max)
        filtered_values = values[mask]

        return filtered_values

    def _generate_histogram_data(
        self,
        scalar_features_dict,
        aggregated_features_dict,
        event_n_tracks_list,
        overall_samples_count,
        data_type_name,
        bin_edges_metadata_path,
        stats,
        data_type_label,
    ):
        if overall_samples_count == 0:
            return None

        histogram_data = {}

        # Load existing bin edges metadata if available (for coordinated binning)
        existing_bin_edges = None
        if bin_edges_metadata_path:
            existing_bin_edges = self._load_bin_edges_metadata(bin_edges_metadata_path)

        # Add metadata including event count for legend display
        histogram_data["_metadata"] = {
            "total_events": int(stats["total_events"]),
            "total_processed_events": int(stats["processed_events"]),
            "total_features": int(stats["total_features"]),
            "processing_time": float(stats["processing_time"]),
            "total_sampled_events": int(overall_samples_count),
            "data_source": data_type_label,
            "data_type": data_type_name,  # "zero_bias" (raw detector) or "post_selection" (clipped, before padding)
        }

        # Details for N_Tracks_per_Event
        if event_n_tracks_list:
            counts_arr = np.array(event_n_tracks_list)
            if counts_arr.size > 0:
                if existing_bin_edges and "N_Tracks_per_Event" in existing_bin_edges:
                    # Use existing bin edges, filter out-of-range data
                    stored_edges = np.array(existing_bin_edges["N_Tracks_per_Event"])
                    filtered_data = self._check_data_range_compatibility(
                        counts_arr, stored_edges, "N_Tracks_per_Event"
                    )
                    counts, bin_edges = np.histogram(
                        filtered_data, bins=stored_edges, density=True
                    )
                else:
                    # Create new bin edges
                    min_val = int(np.min(counts_arr))
                    max_val = int(np.max(counts_arr))
                    if min_val == max_val:
                        bin_edges = np.array([min_val - 0.5, min_val + 0.5])
                    else:
                        bin_edges = np.arange(min_val - 0.5, max_val + 1.5, 1)
                    counts, _ = np.histogram(counts_arr, bins=bin_edges, density=True)
                histogram_data["N_Tracks_per_Event"] = {
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }
            else:
                histogram_data["N_Tracks_per_Event"] = {
                    "counts": [],
                    "bin_edges": [],
                }

        # Details for Scalar Features
        for name, values_list in scalar_features_dict.items():
            values_arr = np.array(values_list)
            if values_arr.size > 0:
                if existing_bin_edges and name in existing_bin_edges:
                    # Use existing bin edges, filter out-of-range data
                    stored_edges = np.array(existing_bin_edges[name])
                    filtered_data = self._check_data_range_compatibility(
                        values_arr, stored_edges, name
                    )
                    counts, bin_edges = np.histogram(
                        filtered_data, bins=stored_edges, density=True
                    )
                else:
                    # Create new bin edges using wider percentiles to reduce harsh cutoffs
                    p0_1, p99_9 = np.percentile(values_arr, [0.1, 99.9])
                    if p0_1 == p99_9:
                        p0_1 -= 0.5
                        p99_9 += 0.5
                    plot_range = (p0_1, p99_9)
                    counts, bin_edges = np.histogram(
                        values_arr, bins=100, range=plot_range, density=True
                    )

                histogram_data[name] = {
                    "counts": counts.tolist(),
                    "bin_edges": bin_edges.tolist(),
                }
            else:
                histogram_data[name] = {"counts": [], "bin_edges": []}

        # Details for Aggregated Features
        for agg_key, arrays_data_list in aggregated_features_dict.items():
            if arrays_data_list:
                plot_feature_names = arrays_data_list[0].get("plot_feature_names", [])
                actual_arrays_to_stack = [
                    item["array"]
                    for item in arrays_data_list
                    if "array" in item and item["array"] is not None
                ]

                if not actual_arrays_to_stack:
                    # If no arrays, store empty for all expected features
                    for feature_name_val in plot_feature_names:
                        histogram_data[feature_name_val] = {
                            "counts": [],
                            "bin_edges": [],
                        }
                    continue

                # NEW APPROACH: Handle variable-shaped arrays by collecting track features directly
                try:
                    # Check if all arrays have the same shape for the original stacking approach
                    first_shape = actual_arrays_to_stack[0].shape
                    shapes_consistent = all(
                        a.shape == first_shape for a in actual_arrays_to_stack
                    )

                    if shapes_consistent:
                        # Original approach: all arrays have same shape, can stack normally
                        stacked_array = np.stack(actual_arrays_to_stack, axis=0)
                        n_features_in_agg = stacked_array.shape[
                            2
                        ]  # Features along 3rd axis

                        # Ensure plot_feature_names matches n_features_in_agg
                        if len(plot_feature_names) != n_features_in_agg:
                            self.logger.warning(
                                f"Mismatch plot_feature_names for agg '{agg_key}' ({len(plot_feature_names)}) vs actual features ({n_features_in_agg}). Using generic."
                            )
                            effective_plot_names = [
                                f"{agg_key}_Feature_{k}"
                                for k in range(n_features_in_agg)
                            ]
                        else:
                            effective_plot_names = plot_feature_names

                        for k in range(n_features_in_agg):
                            feature_name = effective_plot_names[k]
                            data_slice = stacked_array[
                                :, :, k
                            ]  # All events, all tracks for feature k
                            mask = data_slice != 0  # Considering 0 as padding
                            valid_data = data_slice[mask]

                            if valid_data.size > 0:
                                if (
                                    existing_bin_edges
                                    and feature_name in existing_bin_edges
                                ):
                                    # Use existing bin edges, filter out-of-range data
                                    stored_edges = np.array(
                                        existing_bin_edges[feature_name]
                                    )
                                    filtered_data = (
                                        self._check_data_range_compatibility(
                                            valid_data, stored_edges, feature_name
                                        )
                                    )
                                    counts, bin_edges = np.histogram(
                                        filtered_data,
                                        bins=stored_edges,
                                        density=True,
                                    )
                                else:
                                    # Create new bin edges using wider percentiles to reduce harsh cutoffs
                                    p0_1, p99_9 = np.percentile(valid_data, [0.1, 99.9])
                                    if p0_1 == p99_9:
                                        p0_1 -= 0.5
                                        p99_9 += 0.5
                                    plot_range = (p0_1, p99_9)
                                    counts, bin_edges = np.histogram(
                                        valid_data,
                                        bins=100,
                                        range=plot_range,
                                        density=True,
                                    )

                                histogram_data[feature_name] = {
                                    "counts": counts.tolist(),
                                    "bin_edges": bin_edges.tolist(),
                                }
                            else:
                                histogram_data[feature_name] = {
                                    "counts": [],
                                    "bin_edges": [],
                                }
                    else:
                        # NEW APPROACH: Variable shapes - collect track features directly
                        self.logger.debug(
                            f"Variable array shapes for aggregator '{agg_key}', using track-by-track collection"
                        )

                        # Determine number of features from the first array
                        if (
                            len(actual_arrays_to_stack) > 0
                            and actual_arrays_to_stack[0].size > 0
                        ):
                            n_features_in_agg = (
                                actual_arrays_to_stack[0].shape[1]
                                if actual_arrays_to_stack[0].ndim > 1
                                else 1
                            )
                        else:
                            n_features_in_agg = (
                                len(plot_feature_names) if plot_feature_names else 0
                            )

                        # Ensure plot_feature_names matches n_features_in_agg
                        if len(plot_feature_names) != n_features_in_agg:
                            self.logger.warning(
                                f"Mismatch plot_feature_names for agg '{agg_key}' ({len(plot_feature_names)}) vs actual features ({n_features_in_agg}). Using generic."
                            )
                            effective_plot_names = [
                                f"{agg_key}_Feature_{k}"
                                for k in range(n_features_in_agg)
                            ]
                        else:
                            effective_plot_names = plot_feature_names

                        # Collect all track features by concatenating across events
                        for k in range(n_features_in_agg):
                            feature_name = effective_plot_names[k]
                            all_track_values = []

                            for array in actual_arrays_to_stack:
                                if array.size > 0 and array.ndim >= 2:
                                    # Extract feature k from all tracks in this event
                                    feature_values = array[:, k]
                                    # Remove padding (zeros) and collect valid track values
                                    valid_tracks = feature_values[feature_values != 0]
                                    all_track_values.extend(valid_tracks)
                                elif array.size > 0 and array.ndim == 1 and k == 0:
                                    # Handle 1D case for single feature
                                    valid_tracks = array[array != 0]
                                    all_track_values.extend(valid_tracks)

                            if all_track_values:
                                valid_data = np.array(all_track_values)

                                if (
                                    existing_bin_edges
                                    and feature_name in existing_bin_edges
                                ):
                                    # Use existing bin edges, filter out-of-range data
                                    stored_edges = np.array(
                                        existing_bin_edges[feature_name]
                                    )
                                    filtered_data = (
                                        self._check_data_range_compatibility(
                                            valid_data, stored_edges, feature_name
                                        )
                                    )
                                    counts, bin_edges = np.histogram(
                                        filtered_data,
                                        bins=stored_edges,
                                        density=True,
                                    )
                                else:
                                    # Create new bin edges using wider percentiles to reduce harsh cutoffs
                                    p0_1, p99_9 = np.percentile(valid_data, [0.1, 99.9])
                                    if p0_1 == p99_9:
                                        p0_1 -= 0.5
                                        p99_9 += 0.5
                                    plot_range = (p0_1, p99_9)
                                    counts, bin_edges = np.histogram(
                                        valid_data,
                                        bins=100,
                                        range=plot_range,
                                        density=True,
                                    )

                                histogram_data[feature_name] = {
                                    "counts": counts.tolist(),
                                    "bin_edges": bin_edges.tolist(),
                                }
                            else:
                                histogram_data[feature_name] = {
                                    "counts": [],
                                    "bin_edges": [],
                                }

                except Exception as e_agg_hist:
                    self.logger.error(
                        f"Error generating histogram data for aggregated feature in {agg_key}: {e_agg_hist}"
                    )
                    # Store empty for all features of this problematic aggregator
                    for feature_name_val in plot_feature_names:
                        histogram_data[feature_name_val] = {
                            "counts": [],
                            "bin_edges": [],
                        }

        return histogram_data
