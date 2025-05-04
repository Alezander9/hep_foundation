import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict
import math # Import math for ceil

import h5py
import numpy as np
import tensorflow as tf
import uproot
import matplotlib.pyplot as plt

# Import plotting utilities
from hep_foundation.utils import plot_utils

from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.atlas_file_manager import ATLASFileManager
from hep_foundation.data.physlite_derived_features import (
    is_derived_feature,
    get_derived_feature,
    get_dependencies,
)
from hep_foundation.data.task_config import (
    PhysliteFeatureArrayAggregator,
    PhysliteFeatureArrayFilter,
    PhysliteFeatureFilter,
    PhysliteFeatureSelector,
    PhysliteSelectionConfig,
    TaskConfig,
)


class PhysliteFeatureProcessor:
    """
    Handles processing, filtering, and aggregation of ATLAS PhysLite features.

    This class is responsible for:
    - Applying event and feature filters
    - Processing and aggregating feature arrays
    - Computing normalization parameters
    - Creating normalized datasets
    - Loading features and labels from HDF5 files
    """

    def __init__(self, atlas_manager=None):
        """
        Initialize the PhysliteFeatureProcessor.

        Args:
            atlas_manager: Optional ATLASFileManager instance
        """
        self.logger = get_logger(__name__)

        # Could add configuration parameters here if needed in the future
        # For now, keeping it simple as the class is primarily stateless
        self.atlas_manager = atlas_manager or ATLASFileManager()

    def get_required_branches(self, task_config: TaskConfig) -> set:
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

    # functions brought over from dataset_manager.py

    def _apply_event_filters(
        self,
        event_data: dict[str, np.ndarray],
        event_filters: list[PhysliteFeatureFilter],
    ) -> bool:
        """
        Apply event-level filters to determine if an event should be processed.

        Args:
            event_data: Dictionary mapping branch names to their values
            event_filters: List of event-level feature filters to apply

        Returns:
            bool: True if event passes all filters, False otherwise
        """
        if not event_filters:
            return True

        for filter in event_filters:
            value = event_data.get(filter.branch.name)
            if value is None:
                return False

            # For event filters, we expect scalar values
            if isinstance(value, np.ndarray) and value.size > 1:
                return False

            # Convert to scalar if needed
            value = value.item() if isinstance(value, np.ndarray) else value

            # Apply min/max filters
            if filter.min_value is not None and value < filter.min_value:
                return False
            if filter.max_value is not None and value > filter.max_value:
                return False

        return True

    def _apply_feature_filters(
        self,
        feature_values: dict[str, np.ndarray],
        filters: list[PhysliteFeatureFilter],
    ) -> bool:
        """
        Apply scalar feature filters to event-level features.

        Args:
            feature_values: Dictionary mapping feature names to their values
            filters: List of PhysliteFeatureFilter objects to apply

        Returns:
            bool: True if all filters pass, False otherwise
        """
        if not filters:
            return True

        for filter in filters:
            # Get the feature value using the branch name
            value = feature_values.get(filter.branch.name)
            if value is None:
                return False

            # Apply min/max filters if they exist
            if filter.min_value is not None and value < filter.min_value:
                return False
            if filter.max_value is not None and value > filter.max_value:
                return False

        return True

    def _apply_feature_array_filters(
        self,
        feature_arrays: dict[str, np.ndarray],
        filters: list[PhysliteFeatureArrayFilter],
    ) -> np.ndarray:
        """
        Apply array feature filters to track-level features.
        Assumes filters list is not empty when called.

        Args:
            feature_arrays: Dictionary mapping feature names (for filtering) to their array values.
                            Must contain arrays corresponding to filters.
            filters: List of PhysliteFeatureArrayFilter objects to apply (non-empty).

        Returns:
            np.ndarray: Boolean mask indicating which array elements pass all filters.
        """
        # This function now assumes 'filters' is not empty.
        # The case of no filters is handled by the caller (_process_selection_config).

        # Determine the length from the first filter array provided
        # This assumes all arrays in feature_arrays for filtering have the same length
        if not feature_arrays:
             # This case should ideally not be reached if filters are present and data was fetched
             self.logger.warning("_apply_feature_array_filters called with empty feature_arrays dictionary despite non-empty filters. Returning empty mask.")
             return np.array([], dtype=bool) # Return empty mask if feature_arrays is empty

        try:
            first_array_key = filters[0].branch.name # Use first filter's branch name
            initial_length = len(feature_arrays[first_array_key])
        except KeyError:
             self.logger.error(f"Filter branch '{filters[0].branch.name}' not found in feature_arrays passed to _apply_feature_array_filters. Returning empty mask.")
             return np.array([], dtype=bool) # Return empty if the first filter's array is missing
        except Exception as e:
             self.logger.error(f"Error determining length in _apply_feature_array_filters: {e}. Returning empty mask.")
             return np.array([], dtype=bool)


        # Start with all True mask of the correct length
        mask = np.ones(initial_length, dtype=bool)

        for filter_item in filters:
            # Get the feature array using the branch name
            values = feature_arrays.get(filter_item.branch.name)
            if values is None:
                # If a required filter feature is missing, reject all elements
                self.logger.warning(f"Filter branch '{filter_item.branch.name}' missing in _apply_feature_array_filters. Rejecting all elements.")
                mask[:] = False
                break
            if len(values) != initial_length:
                 # If lengths mismatch, something is wrong
                 self.logger.warning(f"Length mismatch for filter branch '{filter_item.branch.name}' ({len(values)}) vs expected ({initial_length}). Rejecting all elements.")
                 mask[:] = False
                 break


            # Apply min filter if it exists
            if filter_item.min_value is not None:
                mask &= values >= filter_item.min_value

            # Apply max filter if it exists
            if filter_item.max_value is not None:
                mask &= values <= filter_item.max_value

        return mask

    def _apply_feature_array_aggregator(
        self,
        feature_arrays: dict[str, np.ndarray],
        aggregator: PhysliteFeatureArrayAggregator,
        valid_mask: np.ndarray,
        sort_indices: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Apply feature array aggregator to combine and sort multiple array features.

        Args:
            feature_arrays: Dictionary mapping feature names to their array values
            aggregator: PhysliteFeatureArrayAggregator configuration
            valid_mask: Boolean mask indicating which array elements passed filters
            sort_indices: Indices to use for sorting the filtered arrays

        Returns:
            np.ndarray: Aggregated and sorted array of shape (max_length, n_features)
                       or None if requirements not met
        """
        # Get the number of valid elements after filtering
        n_valid = np.sum(valid_mask)

        # Check if we meet minimum length requirement
        if n_valid < aggregator.min_length:
            return None

        # Extract arrays for each input branch
        feature_list = []
        for selector in aggregator.input_branches:
            values = feature_arrays.get(selector.branch.name)
            if values is None:
                return None
            # Apply mask and reshape to column
            filtered_values = values[valid_mask].reshape(-1, 1)
            feature_list.append(filtered_values)

        # Stack features horizontally
        features = np.hstack(feature_list)  # Shape: (n_valid, n_features)

        # Apply sorting using provided indices
        sorted_features = features[sort_indices]

        # Handle length requirements
        if len(sorted_features) > aggregator.max_length:
            # Truncate
            final_features = sorted_features[: aggregator.max_length]
        else:
            # Pad with zeros
            padding = np.zeros(
                (aggregator.max_length - len(sorted_features), features.shape[1])
            )
            final_features = np.vstack([sorted_features, padding])

        return final_features

    def _extract_selected_features(
        self,
        event_data: dict[str, np.ndarray],
        feature_selectors: list[PhysliteFeatureSelector],
    ) -> dict[str, np.ndarray]:
        """
        Extract selected features from event data.

        Args:
            event_data: Dictionary mapping branch names to their values
            feature_selectors: List of feature selectors defining what to extract

        Returns:
            Dictionary mapping branch names to their selected values
        """
        selected_features = {}

        for selector in feature_selectors:
            value = event_data.get(selector.branch.name)
            if value is None:
                continue

            # For scalar features, ensure we get a single value
            if selector.branch.is_feature:
                if isinstance(value, np.ndarray):
                    if value.size == 0:
                        continue
                    value = value.item()

            selected_features[selector.branch.name] = value

        return selected_features

    def _process_event(
        self, event_data: dict[str, np.ndarray], task_config: TaskConfig
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Process a single event using the task configuration.
        (Now returns the structure including original branch names for plotting)

        Args:
            event_data: Dictionary mapping branch names (real and derived) to their raw values
            task_config: TaskConfig object containing event filters, input features, and labels

        Returns:
            Dictionary of processed features or None if event rejected. Structure:
            {
                "scalar_features": {branch_name: value, ...},
                "aggregated_features": {
                     aggregator_key: { # e.g., "aggregator_0"
                          "array": aggregated_array,
                          "branch_names": [original_branch_name1, ...]
                     }, ...
                },
                "label_features": [ # List for multiple label sets
                     {
                          "scalar_features": {branch_name: value, ...},
                          "aggregated_features": {agg_key: {"array": ..., "branch_names": [...]}}
                     }, ...
                ]
            }
        """
        # Apply event filters first
        if not self._apply_event_filters(event_data, task_config.event_filters):
            return None

        # Process input selection config
        input_result = self._process_selection_config(event_data, task_config.input)
        if input_result is None:
            return None

        # Process each label config if present
        label_results = []
        for label_config in task_config.labels:
            label_result = self._process_selection_config(event_data, label_config)
            if label_result is None:
                # If any label processing fails for a supervised task, reject the event
                # Modify this if partial labels are acceptable
                return None
            label_results.append(label_result)

        # Construct the final result dictionary, adding branch names to aggregators
        final_result = {
            "scalar_features": input_result["scalar_features"],
            "aggregated_features": {},
            "label_features": []
        }

        # Add input aggregators with branch names
        for agg_key, agg_array in input_result["aggregated_features"].items():
             agg_index = int(agg_key.split("_")[1])
             original_branches = [
                 sel.branch.name for sel in task_config.input.feature_array_aggregators[agg_index].input_branches
             ]
             final_result["aggregated_features"][agg_key] = {
                 "array": agg_array,
                 "branch_names": original_branches
             }

        # Add label results with branch names
        for i, label_res in enumerate(label_results):
             processed_label = {
                 "scalar_features": label_res["scalar_features"],
                 "aggregated_features": {}
             }
             for agg_key, agg_array in label_res["aggregated_features"].items():
                  # Assuming label aggregators follow the same index structure
                  agg_index = int(agg_key.split("_")[1])
                  # Ensure the label config exists and has aggregators
                  if i < len(task_config.labels) and agg_index < len(task_config.labels[i].feature_array_aggregators):
                      original_branches = [
                          sel.branch.name for sel in task_config.labels[i].feature_array_aggregators[agg_index].input_branches
                      ]
                      processed_label["aggregated_features"][agg_key] = {
                           "array": agg_array,
                           "branch_names": original_branches
                      }
                  else:
                       self.logger.warning(f"Could not map label aggregator key '{agg_key}' back to TaskConfig definition for label set {i}.")
                       # Fallback: store without branch names
                       processed_label["aggregated_features"][agg_key] = {
                           "array": agg_array,
                           "branch_names": []
                       }

             final_result["label_features"].append(processed_label)


        return final_result

    def _process_selection_config(
        self,
        event_data: dict[str, np.ndarray],
        selection_config: PhysliteSelectionConfig,
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Process a single selection configuration to extract and aggregate features.
        (Modified to return structure compatible with _process_event needs)

        Args:
            event_data: Dictionary mapping branch names to their values
            selection_config: Configuration defining what features to select and how to aggregate arrays

        Returns:
            Dictionary containing:
                - 'scalar_features': Dict of selected scalar features {branch_name: value}
                - 'aggregated_features': Dict of aggregated array features {aggregator_key: array}
            Returns None if any required features are missing or aggregation fails
        """
        # Extract scalar features
        scalar_features = {}
        if selection_config.feature_selectors:
             scalar_features = self._extract_selected_features(
                  event_data, selection_config.feature_selectors
             )

        # Process aggregators
        aggregated_features = {}
        if selection_config.feature_array_aggregators:
            for idx, aggregator in enumerate(
                selection_config.feature_array_aggregators
            ):
                # --- Check Requirements and Get Input Arrays ---
                array_features = {} # Stores input arrays for aggregation
                required_for_agg = set(s.branch.name for s in aggregator.input_branches)
                filter_branch_names = set(f.branch.name for f in aggregator.filter_branches)
                required_for_agg.update(filter_branch_names)
                if aggregator.sort_by_branch:
                    required_for_agg.add(aggregator.sort_by_branch.branch.name)

                if not all(branch_name in event_data for branch_name in required_for_agg):
                     missing = required_for_agg - set(event_data.keys())
                     self.logger.debug(f"Skipping aggregator {idx} due to missing branches in event data: {missing}")
                     return None

                initial_length = -1
                for selector in aggregator.input_branches:
                    branch_name = selector.branch.name
                    current_array = event_data[branch_name]
                    array_features[branch_name] = current_array
                    current_length = len(current_array)
                    if current_length == 0:
                        self.logger.debug(f"Skipping aggregator {idx} because required input branch '{branch_name}' is empty.")
                        return None
                    if initial_length == -1:
                        initial_length = current_length
                    elif initial_length != current_length:
                         self.logger.warning(f"Input array length mismatch in aggregator {idx} ('{branch_name}' has {current_length}, expected {initial_length}). Skipping event.")
                         return None # Lengths of input arrays must be consistent

                if initial_length == -1:
                     # This happens if there were no input branches, should not occur if TaskConfig validation is correct
                     self.logger.warning(f"Aggregator {idx} has no input branches defined. Skipping.")
                     return None


                # --- Prepare and Apply Filters ---
                valid_mask = np.ones(initial_length, dtype=bool) # Start with all true mask
                if aggregator.filter_branches:
                     filter_arrays = {}
                     for filter_branch_name in filter_branch_names:
                         filter_array = event_data[filter_branch_name]
                         if len(filter_array) != initial_length:
                              self.logger.warning(f"Filter array length mismatch in aggregator {idx} ('{filter_branch_name}' has {len(filter_array)}, expected {initial_length}). Skipping event.")
                              return None
                         filter_arrays[filter_branch_name] = filter_array

                     # Only call if filters exist and filter_arrays is populated
                     if filter_arrays:
                          filter_mask = self._apply_feature_array_filters(
                               filter_arrays, aggregator.filter_branches
                          )
                          if len(filter_mask) != initial_length:
                               self.logger.warning(f"Filter mask length ({len(filter_mask)}) mismatch vs expected ({initial_length}) for aggregator {idx}. Skipping event.")
                               return None
                          valid_mask &= filter_mask


                # --- Prepare and Apply Sorting ---
                if aggregator.sort_by_branch:
                    sort_branch_name = aggregator.sort_by_branch.branch.name
                    sort_value = event_data[sort_branch_name]
                    if len(sort_value) != initial_length:
                        self.logger.warning(f"Sort array length mismatch in aggregator {idx} ('{sort_branch_name}' has {len(sort_value)}, expected {initial_length}). Skipping event.")
                        return None

                    masked_sort_value = sort_value[valid_mask]
                    if len(masked_sort_value) == 0:
                         sort_indices = np.array([], dtype=int)
                    else:
                         sort_indices = np.argsort(masked_sort_value)[::-1]
                else:
                    sort_indices = np.arange(np.sum(valid_mask)) # Indices relative to masked array


                # --- Apply Aggregator ---
                result = self._apply_feature_array_aggregator(
                    array_features, aggregator, valid_mask, sort_indices
                )
                if result is None:
                    return None

                aggregated_features[f"aggregator_{idx}"] = result


        # Return results if we have anything
        if not scalar_features and not aggregated_features:
             self.logger.debug("No scalar features found and no successful aggregations for this selection config.")
             return None

        return {
            "scalar_features": scalar_features,
            "aggregated_features": aggregated_features,
        }

    def _process_data(
        self,
        task_config: TaskConfig,
        run_number: Optional[str] = None,
        signal_key: Optional[str] = None,
        catalog_limit: Optional[int] = None,
        plot_distributions: bool = False,
        delete_catalogs: bool = True,
        plot_output: Optional[Path] = None,
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], dict]:
        """
        Process either ATLAS or signal data using task configuration.

        Args:
            task_config: Configuration defining event filters, input features, and labels
            run_number: Optional run number for ATLAS data
            signal_key: Optional signal type for signal data
            catalog_limit: Optional limit on number of catalogs to process
            plot_distributions: Whether to generate distribution plots
            delete_catalogs: Whether to delete catalogs after processing
            plot_output: Optional complete path (including filename) for saving plots

        Returns:
            Tuple containing:
            - List of processed input features (each a dict with scalar and aggregated features)
            - List of processed label features (each a list of dicts for multiple label configs)
            - Processing statistics
        """
        if signal_key:
            self.logger.info(f"Processing signal data for {signal_key}")
        elif run_number:
            self.logger.info(f"Processing ATLAS data for run {run_number}")
        else:
            raise ValueError("Must provide either run_number or signal_key")

        if plot_distributions:
            if plot_output is None:
                self.logger.warning("Plot distributions enabled but no output path provided. Disabling plotting.")
                plot_distributions = False
            else:
                # Ensure parent directory exists
                plot_output.parent.mkdir(parents=True, exist_ok=True)
                self.logger.warning("Plotting distributions is enabled. This will significantly slow down processing.")

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
        initially_required_branches = self.get_required_branches(task_config)
        self.logger.debug(f"Initially required branches (incl. derived): {initially_required_branches}")

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
                    self.logger.warning(f"Derived feature '{branch_name}' has no dependencies defined.")
            else:
                # This is a real branch, add it directly
                actual_branches_to_read.add(branch_name)

        # Add the identified dependencies to the set of branches to read
        actual_branches_to_read.update(dependencies_to_add)

        if derived_features_requested:
             self.logger.info(f"Identified derived features: {derived_features_requested}")
             self.logger.info(f"Added dependencies for derived features: {dependencies_to_add}")
        self.logger.info(f"Actual branches to read from file: {actual_branches_to_read}")

        # Check if we have anything to read
        if not actual_branches_to_read:
             self.logger.warning("No actual branches identified to read from the file after processing dependencies. Check TaskConfig and derived feature definitions.")
             return [], [], stats # Return empty results


        # Get catalog paths
        self.logger.info(
            f"Getting catalog paths for run {run_number} and signal {signal_key}"
        )
        catalog_paths = self._get_catalog_paths(run_number, signal_key, catalog_limit)
        self.logger.info(f"Found {len(catalog_paths)} catalog paths to process.")

        # --- Plotting Setup ---
        plotting_enabled = plot_distributions and plot_output is not None
        max_plot_samples_total = 5000 # Overall target for plotting samples
        overall_plot_samples_count = 0 # Counter for total samples accumulated
        samples_per_catalog = 0       # Target samples from each catalog
        num_catalogs_to_process = len(catalog_paths)

        if plotting_enabled and num_catalogs_to_process > 0:
            # Calculate target samples per catalog, ensuring at least 1
            samples_per_catalog = max(1, max_plot_samples_total // num_catalogs_to_process)
            self.logger.info(f"Plotting enabled. Aiming for ~{samples_per_catalog} samples per catalog (total target: {max_plot_samples_total}).")
        elif plotting_enabled:
            self.logger.warning("Plotting enabled, but no catalog paths found.")
            plotting_enabled = False # Disable if no catalogs

        # Data accumulators for plotting (only if enabled)
        sampled_scalar_features = defaultdict(list) if plotting_enabled else None
        sampled_aggregated_features = defaultdict(list) if plotting_enabled else None
        # --- End Plotting Setup ---

        # --- Main Processing Loop ---
        for catalog_idx, catalog_path in enumerate(catalog_paths):
            # Reset counter for this specific catalog
            current_catalog_samples_count = 0

            self.logger.info(f"Processing catalog {catalog_idx+1}/{num_catalogs_to_process} with path: {catalog_path}")

            try:
                catalog_start_time = datetime.now()
                catalog_stats = {"events": 0, "processed": 0}

                with uproot.open(catalog_path) as file:
                    tree = file["CollectionTree;1"]

                    # Check which required branches are actually available in the tree
                    available_branches = set(tree.keys())
                    branches_to_read_in_tree = actual_branches_to_read.intersection(available_branches)
                    missing_branches = actual_branches_to_read - branches_to_read_in_tree

                    if missing_branches:
                        self.logger.warning(f"Branches required but not found in tree {catalog_path}: {missing_branches}")
                        # Decide if we should skip or continue with available ones. For now, let's try to continue.
                        if not branches_to_read_in_tree:
                             self.logger.error(f"No required branches available in tree {catalog_path}. Skipping catalog.")
                             continue # Skip this catalog if none of the needed branches are present

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
                                self.logger.warning("Encountered batch dictionary with no arrays inside. Skipping.")
                                continue
                            except KeyError:
                                # Should not happen if StopIteration doesn't, but defensive
                                self.logger.error("Internal error accessing first key in non-empty batch. Skipping.")
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
                                    if branch_name in arrays # Double check presence
                                }
                                # Skip if somehow event data is empty (shouldn't happen with checks above)
                                if not raw_event_data:
                                    continue

                                # --- Calculate Derived Features ---
                                processed_event_data = raw_event_data.copy() # Start with real data
                                calculation_failed = False
                                for derived_name in derived_features_requested:
                                    derived_feature_def = get_derived_feature(derived_name)
                                    if not derived_feature_def: continue # Should not happen

                                    # Check if all dependencies were read successfully for this event
                                    dependencies_present = all(dep in raw_event_data for dep in derived_feature_def.dependencies)

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
                                            self.logger.error(f"Error calculating derived feature '{derived_name}' for event {evt_idx} in batch: {calc_e}")
                                            calculation_failed = True
                                            break # Stop processing derived features for this event
                                    else:
                                        missing_deps = [dep for dep in derived_feature_def.dependencies if dep not in raw_event_data]
                                        self.logger.warning(f"Cannot calculate derived feature '{derived_name}' due to missing dependencies: {missing_deps}. Skipping for this event.")
                                        calculation_failed = True
                                        break # Stop processing derived features for this event

                                if calculation_failed:
                                    continue # Skip this event entirely if any derived feature calculation failed

                                # --- Process Event (using data with derived features added) ---
                                result = self._process_event(processed_event_data, task_config)
                                if result is not None:
                                    # Store the result for dataset creation
                                    # We only store the arrays for the final dataset, not the metadata like branch names
                                    input_features_for_dataset = {
                                         "scalar_features": result["scalar_features"],
                                         "aggregated_features": {
                                              agg_key: agg_data["array"]
                                              for agg_key, agg_data in result["aggregated_features"].items()
                                         }
                                    }
                                    # Process labels similarly for the dataset
                                    label_features_for_dataset = []
                                    for label_set in result["label_features"]:
                                         label_set_for_dataset = {
                                              "scalar_features": label_set["scalar_features"],
                                              "aggregated_features": {
                                                   agg_key: agg_data["array"]
                                                   for agg_key, agg_data in label_set["aggregated_features"].items()
                                              }
                                         }
                                         label_features_for_dataset.append(label_set_for_dataset)


                                    processed_inputs.append(input_features_for_dataset)
                                    processed_labels.append(label_features_for_dataset)
                                    catalog_stats["processed"] += 1


                                    # --- Accumulate data for plotting (conditional on per-catalog count) ---
                                    if plotting_enabled and current_catalog_samples_count < samples_per_catalog:
                                        # Append scalar features
                                        for name, value in result["scalar_features"].items():
                                            sampled_scalar_features[name].append(value)
                                        # Append aggregated features (dict with array + names)
                                        for agg_key, agg_data in result["aggregated_features"].items():
                                            sampled_aggregated_features[agg_key].append(agg_data)

                                        current_catalog_samples_count += 1 # Increment count for THIS catalog
                                        overall_plot_samples_count += 1   # Increment overall count

                                    # Update feature statistics
                                    # Note: This might need adjustment if derived features impact how stats are counted
                                    if input_features_for_dataset["aggregated_features"]:
                                         first_aggregator_key = next(iter(input_features_for_dataset["aggregated_features"].keys()), None)
                                         if first_aggregator_key:
                                             first_aggregator = input_features_for_dataset["aggregated_features"][first_aggregator_key]
                                             # Ensure the aggregator output is not empty before checking shape
                                             if first_aggregator.size > 0:
                                                  try:
                                                       # Check for non-zero elements along the feature axis
                                                       stats["total_features"] += np.sum(
                                                            np.any(first_aggregator != 0, axis=1)
                                                       )
                                                  except np.AxisError:
                                                       # Handle potential scalar case if aggregator somehow returns it
                                                       stats["total_features"] += np.sum(first_aggregator != 0)


                        except Exception as e:
                            # General error handling for the batch processing
                            self.logger.error(f"Error processing batch in catalog {catalog_path}: {str(e)}", exc_info=True)
                            continue # Try to continue with the next batch

                # Update statistics
                catalog_duration = (datetime.now() - catalog_start_time).total_seconds()
                stats["processing_time"] += catalog_duration
                stats["total_events"] += catalog_stats["events"]
                stats["processed_events"] += catalog_stats["processed"]

                # Print catalog summary
                if catalog_duration > 0:
                     rate = catalog_stats['events'] / catalog_duration
                else:
                     rate = float('inf') # Avoid division by zero if processing was instant

                self.logger.info(f"Catalog {catalog_idx} summary:")
                self.logger.info(f"  Total events read: {catalog_stats['events']}")
                self.logger.info(
                    f"  Events passing selection (incl. derived calc): {catalog_stats['processed']}"
                )
                self.logger.info(f"  Processing time: {catalog_duration:.2f}s")
                self.logger.info(f"  Rate: {rate:.1f} events/s")


                if catalog_limit and catalog_idx >= catalog_limit - 1:
                    break

            except FileNotFoundError:
                 self.logger.error(f"Catalog file not found: {catalog_path}. Skipping.")
                 continue
            except Exception as e:
                self.logger.error(f"Critical error processing catalog {catalog_path}: {str(e)}", exc_info=True) # Add traceback
                # Depending on error, might want to stop or continue
                continue # Try to continue with the next catalog

            finally:
                # Optional: Consider if deleting catalogs is still desired if errors occurred
                if delete_catalogs and Path(catalog_path).exists():
                   try:
                       os.remove(catalog_path)
                       self.logger.info(f"Deleted catalog file: {catalog_path}")
                   except OSError as delete_e:
                       self.logger.error(f"Error deleting catalog file {catalog_path}: {delete_e}")

        # --- Plotting Section (Uses sampled data) ---
        if plotting_enabled and overall_plot_samples_count > 0:
            self.logger.info(f"Plotting feature distributions from {overall_plot_samples_count} sampled events (distributed across catalogs) to {plot_output}")

            plot_utils.set_science_style(use_tex=False)

            # --- Prepare Subplots ---
            num_scalar_plots = len(sampled_scalar_features)
            num_agg_features = 0
            agg_feature_details = [] # Store details: (agg_key, feature_index, feature_name)

            for agg_key, arrays_data_list in sampled_aggregated_features.items():
                if arrays_data_list:
                    branch_names = arrays_data_list[0].get("branch_names", [])
                    if "array" in arrays_data_list[0] and arrays_data_list[0]["array"] is not None:
                         try:
                              n_features_in_agg = arrays_data_list[0]["array"].shape[1]
                              if len(branch_names) != n_features_in_agg:
                                   plot_branch_names = [f"Feature_{k}" for k in range(n_features_in_agg)]
                              else:
                                   plot_branch_names = branch_names
                              num_agg_features += n_features_in_agg
                              for k in range(n_features_in_agg):
                                   agg_feature_details.append((agg_key, k, plot_branch_names[k]))
                         except IndexError:
                              self.logger.warning(f"Aggregator '{agg_key}' data seems malformed (expected 2D array). Skipping its plots.")
                         except Exception as e:
                              self.logger.error(f"Error processing aggregator '{agg_key}' details: {e}. Skipping its plots.")
                    else:
                         self.logger.warning(f"Aggregator '{agg_key}' data list item missing 'array' key or array is None. Skipping its plots.")

            total_plots = num_scalar_plots + num_agg_features
            if total_plots == 0:
                 self.logger.warning("No features found to plot.")
            else:
                 ncols = max(1, int(math.ceil(math.sqrt(total_plots))))
                 nrows = max(1, int(math.ceil(total_plots / ncols)))

                 # Get figure size with a fixed 16:9 aspect ratio
                 target_ratio = 16 / 9
                 fig_width, fig_height = plot_utils.get_figure_size(
                     width="double",  # Use wider figure for subplots
                     ratio=target_ratio # Set fixed aspect ratio
                 )
                 # Adjust height slightly based on rows to prevent crowding if many rows
                 # Heuristic: increase height slightly for > 3 rows
                 if nrows > 3:
                      fig_height *= (nrows / 3.0)**0.5 # Increase height gently with more rows


                 fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
                 axes = axes.flatten()
                 plot_idx = 0
                 colors = plot_utils.get_color_cycle(palette="high_contrast", n=total_plots)


                 # --- Plot Scalar Features ---
                 self.logger.info("Plotting sampled scalar features...")
                 for name, values_list in sampled_scalar_features.items():
                     if plot_idx >= len(axes): break
                     ax = axes[plot_idx]
                     try:
                         values_arr = np.array(values_list)
                         if values_arr.size == 0:
                              ax.set_title(f"{name}\n(No Data)", fontsize=plot_utils.FONT_SIZES["small"])
                              ax.tick_params(axis='both', which='major', labelsize=plot_utils.FONT_SIZES['tiny'])
                              ax.grid(True, linestyle='--', alpha=0.6)
                              plot_idx += 1
                              continue
                         p1, p99 = np.percentile(values_arr, [1, 99])
                         if p1 == p99:
                             p1 -= 0.5
                             p99 += 0.5
                         plot_range = (p1, p99)
                         ax.hist(values_arr, bins='auto', range=plot_range, histtype='stepfilled', density=True, color=colors[plot_idx % len(colors)], alpha=0.7)
                         ax.set_ylabel("Density")
                         ax.set_title(f"{name}", fontsize=plot_utils.FONT_SIZES["small"])
                         ax.grid(True, linestyle='--', alpha=0.6)
                         ax.tick_params(axis='both', which='major', labelsize=plot_utils.FONT_SIZES['tiny'])
                         plot_idx += 1
                     except Exception as plot_e:
                          self.logger.error(f"Failed to plot scalar feature '{name}': {plot_e}", exc_info=True)
                          ax.set_title(f"Error plotting {name}")
                          plot_idx += 1


                 # --- Plot Aggregated Features ---
                 self.logger.info("Plotting sampled aggregated features...")
                 agg_data_cache = {}
                 for agg_key, k, feature_name in agg_feature_details:
                      if plot_idx >= len(axes): break
                      ax = axes[plot_idx]
                      try:
                           if agg_key not in agg_data_cache:
                                arrays_data_list = sampled_aggregated_features.get(agg_key, [])
                                if not arrays_data_list: continue
                                arrays_list = [item["array"] for item in arrays_data_list if item.get("array") is not None]
                                if not arrays_list: continue
                                first_shape = arrays_list[0].shape
                                if not all(a.shape == first_shape for a in arrays_list):
                                     self.logger.warning(f"Inconsistent array shapes found within aggregator '{agg_key}'. Skipping plot for this aggregator.")
                                     agg_data_cache[agg_key] = None
                                     ax.set_title(f"Error: Inconsistent shapes in {agg_key}")
                                     plot_idx += (first_shape[1] - k)
                                     continue
                                stacked_array = np.stack(arrays_list, axis=0)
                                agg_data_cache[agg_key] = stacked_array
                           else:
                                stacked_array = agg_data_cache[agg_key]
                                if stacked_array is None:
                                     ax.set_title(f"Error: Skipped due to shape inconsistency")
                                     plot_idx += 1
                                     continue
                           if stacked_array.size == 0: continue
                           if k >= stacked_array.shape[2]:
                                self.logger.warning(f"Feature index {k} out of bounds for {agg_key}")
                                continue
                           data_slice = stacked_array[:, :, k]
                           mask = data_slice != 0
                           valid_data = data_slice[mask]
                           if valid_data.size == 0:
                                self.logger.info(f"No non-zero data for aggregated feature '{feature_name}' in {agg_key}. Skipping plot.")
                                ax.set_title(f"{feature_name}\n(No Data)", fontsize=plot_utils.FONT_SIZES["small"])
                                ax.tick_params(axis='both', which='major', labelsize=plot_utils.FONT_SIZES['tiny'])
                                ax.grid(True, linestyle='--', alpha=0.6)
                                plot_idx += 1
                                continue
                           p1, p99 = np.percentile(valid_data, [1, 99])
                           if p1 == p99:
                               p1 -= 0.5
                               p99 += 0.5
                           plot_range = (p1, p99)
                           ax.hist(valid_data, bins='auto', range=plot_range, histtype='stepfilled', density=True, color=colors[plot_idx % len(colors)], alpha=0.7)
                           ax.set_ylabel("Density")
                           ax.set_title(f"{feature_name}", fontsize=plot_utils.FONT_SIZES["small"])
                           ax.grid(True, linestyle='--', alpha=0.6)
                           ax.tick_params(axis='both', which='major', labelsize=plot_utils.FONT_SIZES['tiny'])
                           plot_idx += 1
                      except Exception as outer_plot_e:
                           self.logger.error(f"Failed to process or plot feature '{feature_name}' for aggregator '{agg_key}': {outer_plot_e}", exc_info=True)
                           ax.set_title(f"Error plotting {feature_name}")
                           plot_idx += 1


                 # --- Finalize Figure ---
                 for i in range(plot_idx, len(axes)):
                     axes[i].axis('off')
                 fig.suptitle(f"Sampled Feature Distributions ({overall_plot_samples_count} Events)", fontsize=plot_utils.FONT_SIZES['large'])
                 plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                 fig.savefig(plot_output, dpi=300, bbox_inches='tight')
                 plt.close(fig)
                 self.logger.info(f"Saved combined distribution plot to {plot_output}")


        # Convert stats to native Python types
        stats = {
            "total_events": int(stats["total_events"]),
            "processed_events": int(stats["processed_events"]),
            "total_features": int(stats["total_features"]), # Note: feature counting might need adjustment for derived
            "processing_time": float(stats["processing_time"]),
        }

        return processed_inputs, processed_labels, stats

    def _compute_dataset_normalization(
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

    def _create_normalized_dataset(
        self,
        features_dict: dict[str, np.ndarray],
        norm_params: dict,
        labels_dict: Optional[dict[str, dict[str, np.ndarray]]] = None,
    ) -> tf.data.Dataset:
        """
        Create a normalized TensorFlow dataset from features and optional labels.

        Args:
            features_dict: Dictionary mapping feature names to their array values
            norm_params: Dictionary containing normalization parameters
            labels_dict: Optional dictionary mapping label config names to their feature dictionaries

        Returns:
            tf.data.Dataset containing normalized features (and labels if provided)
        """
        # Get number of events from first feature
        n_events = next(iter(features_dict.values())).shape[0]

        # Normalize and reshape features
        normalized_features = []
        for name in sorted(features_dict.keys()):
            feature_array = features_dict[name]
            if name.startswith("scalar/"):
                params = norm_params["features"]["scalar"][name.split("/", 1)[1]]
                normalized = (feature_array - params["mean"]) / params["std"]
            else:  # aggregated features
                params = norm_params["features"]["aggregated"][name.split("/", 1)[1]]
                normalized = (feature_array - np.array(params["means"])) / np.array(
                    params["stds"]
                )
            normalized_features.append(normalized.reshape(n_events, -1))

        # Concatenate normalized features
        all_features = np.concatenate(normalized_features, axis=1)

        if labels_dict:
            # Normalize and reshape labels
            normalized_labels = []
            for config_name in sorted(labels_dict.keys()):
                config_features = labels_dict[config_name]
                for name in sorted(config_features.keys()):
                    label_array = config_features[name]
                    if name.startswith("scalar/"):
                        params = norm_params["labels"][int(config_name.split("_")[1])][
                            "scalar"
                        ][name.split("/", 1)[1]]
                        normalized = (label_array - params["mean"]) / params["std"]
                    else:  # aggregated features
                        params = norm_params["labels"][int(config_name.split("_")[1])][
                            "aggregated"
                        ][name.split("/", 1)[1]]
                        normalized = (
                            label_array - np.array(params["means"])
                        ) / np.array(params["stds"])
                    normalized_labels.append(normalized.reshape(n_events, -1))

            # Concatenate normalized labels
            all_labels = np.concatenate(normalized_labels, axis=1)
            return tf.data.Dataset.from_tensor_slices((all_features, all_labels))
        else:
            return tf.data.Dataset.from_tensor_slices(all_features)

    def _load_features_from_group(
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

    def _load_labels_from_group(
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

    def _get_catalog_paths(
        self,
        run_number: Optional[str] = None,
        signal_key: Optional[str] = None,
        catalog_limit: Optional[int] = None,
    ) -> list[Path]:
        """
        Get list of catalog paths for either ATLAS data or signal data

        Args:
            run_number: Optional run number for ATLAS data
            signal_key: Optional signal type for signal data
            catalog_limit: Optional limit on number of catalogs to process

        Returns:
            List of paths to catalog files
        """
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
            return paths
        elif signal_key is not None:
            # Get signal data catalog
            catalog_path = self.atlas_manager.get_signal_catalog_path(signal_key, 0)
            if not catalog_path.exists():
                catalog_path = self.atlas_manager.download_signal_catalog(signal_key, 0)
            return [catalog_path] if catalog_path else []
        else:
            raise ValueError("Must provide either run_number or signal_key")
