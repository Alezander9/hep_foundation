import json  # Add json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import awkward as ak  # Add awkward import
import h5py
import numpy as np
import tensorflow as tf
import uproot

from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import (
    PhysliteFeatureArrayAggregator,
    PhysliteFeatureFilter,
    PhysliteFeatureSelector,
    PhysliteSelectionConfig,
    TaskConfig,
)
from hep_foundation.data.atlas_file_manager import ATLASFileManager
from hep_foundation.data.physlite_derived_features import (
    get_dependencies,
    get_derived_feature,
    is_derived_feature,
)

# Import plotting utilities


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

    def __init__(
        self,
        atlas_manager=None,
        custom_label_map_path: Optional[
            str
        ] = "src/hep_foundation/data/physlite_plot_labels.json",
    ):
        """
        Initialize the PhysliteFeatureProcessor.

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
                "Uproot is required for PhysliteFeatureProcessor. Please install it."
            )
        if ak is None:  # Check awkward import
            raise ImportError(
                "Awkward Array is required for PhysliteFeatureProcessor. Please install it."
            )

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
        self,
        event_data: dict[str, np.ndarray],
        task_config: TaskConfig,
        plotting_enabled: bool = False,
        need_more_zero_bias_samples: bool = False,
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Process a single event using the task configuration.
        (Now returns the structure including original branch names for plotting)

        Args:
            event_data: Dictionary mapping branch names (real and derived) to their raw values
            task_config: TaskConfig object containing event filters, input features, and labels
            plotting_enabled: Whether plotting is enabled
            need_more_zero_bias_samples: Whether we still need more zero-bias samples for plotting

        Returns:
            Dictionary of processed features or None if event rejected. Structure:
            {
                "scalar_features": {branch_name: value, ...},
                "aggregated_features": {
                     aggregator_key: { # e.g., "aggregator_0"
                          "array": aggregated_array,
                          "plot_feature_names": [list_of_final_plot_names_for_each_column],
                          "n_valid_elements": int # Add this for track counting
                     }, ...
                },
                "label_features": [ # List for multiple label sets
                     {
                          "scalar_features": {branch_name: value, ...},
                          "aggregated_features": {agg_key: {"array": ..., "plot_feature_names": [...]}}
                     }, ...
                ],
                "passed_filters": bool  # Add this to track filter status
            }
        """
        # Apply event filters first and track the result
        passed_filters = self._apply_event_filters(
            event_data, task_config.event_filters
        )

        # If event doesn't pass filters, only continue if we need zero-bias samples for plotting
        if not passed_filters and not (
            plotting_enabled and need_more_zero_bias_samples
        ):
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
            "label_features": [],
            "num_tracks_for_plot": None,  # Initialize
            "passed_filters": passed_filters,  # Track filter status
        }

        # Add input aggregators with branch names and n_valid_elements
        first_input_agg_key = None
        for agg_key, agg_data_dict in input_result["aggregated_features"].items():
            if (
                first_input_agg_key is None
            ):  # Capture the key of the first input aggregator
                first_input_agg_key = agg_key

            agg_array = agg_data_dict["array"]
            input_branch_details = agg_data_dict[
                "input_branch_details"
            ]  # name, num_output_cols
            n_valid_elements = agg_data_dict.get(
                "n_valid_elements"
            )  # Get from selection_config processing
            hist_data = agg_data_dict.get("hist_data")  # Get histogram data

            plot_feature_names = []
            for detail in input_branch_details:
                branch_name = detail["name"]
                num_output_cols = detail["num_output_cols"]
                custom_titles = self.custom_label_map.get(branch_name)

                if (
                    isinstance(custom_titles, list)
                    and len(custom_titles) == num_output_cols
                ):
                    plot_feature_names.extend(custom_titles)
                elif isinstance(custom_titles, str) and num_output_cols == 1:
                    plot_feature_names.append(custom_titles)
                else:  # Fallback
                    if num_output_cols == 1:
                        plot_feature_names.append(branch_name)
                    else:
                        plot_feature_names.extend(
                            [f"{branch_name}_comp{j}" for j in range(num_output_cols)]
                        )

            aggregator_data = {
                "array": agg_array,
                "plot_feature_names": plot_feature_names,
                "n_valid_elements": n_valid_elements,
            }

            # Add clipped histogram data if available (for post-selection plotting)
            if hist_data and plotting_enabled:
                aggregator_data["clipped_histogram_data"] = {
                    "clipped_features": hist_data["clipped_features"],
                    "clipped_num_tracks": hist_data["clipped_num_tracks"],
                    "plot_feature_names": plot_feature_names,
                }

            final_result["aggregated_features"][agg_key] = aggregator_data

        # Set num_tracks_for_plot from the first input aggregator if available
        if (
            first_input_agg_key
            and first_input_agg_key in final_result["aggregated_features"]
        ):
            final_result["num_tracks_for_plot"] = final_result["aggregated_features"][
                first_input_agg_key
            ].get("n_valid_elements")

        # Add label results with branch names
        for i, label_res in enumerate(label_results):
            processed_label = {
                "scalar_features": label_res["scalar_features"],
                "aggregated_features": {},
            }
            for agg_key, agg_data_dict in label_res["aggregated_features"].items():
                agg_array = agg_data_dict["array"]
                input_branch_details = agg_data_dict["input_branch_details"]
                n_valid_elements = agg_data_dict.get("n_valid_elements")
                hist_data = agg_data_dict.get("hist_data")  # Get histogram data

                plot_feature_names = []
                # Ensure the label config exists and has aggregators for details
                # agg_index = int(agg_key.split("_")[1]) # Not needed if details are passed up

                for detail in input_branch_details:
                    branch_name = detail["name"]
                    num_output_cols = detail["num_output_cols"]
                    custom_titles = self.custom_label_map.get(branch_name)

                    if (
                        isinstance(custom_titles, list)
                        and len(custom_titles) == num_output_cols
                    ):
                        plot_feature_names.extend(custom_titles)
                    elif isinstance(custom_titles, str) and num_output_cols == 1:
                        plot_feature_names.append(custom_titles)
                    else:  # Fallback
                        if num_output_cols == 1:
                            plot_feature_names.append(branch_name)
                        else:
                            plot_feature_names.extend(
                                [
                                    f"{branch_name}_comp{j}"
                                    for j in range(num_output_cols)
                                ]
                            )

                label_aggregator_data = {
                    "array": agg_array,
                    "plot_feature_names": plot_feature_names,
                    "n_valid_elements": n_valid_elements,
                }

                # Add clipped histogram data if available (for post-selection plotting)
                if hist_data and plotting_enabled:
                    label_aggregator_data["clipped_histogram_data"] = {
                        "clipped_features": hist_data["clipped_features"],
                        "clipped_num_tracks": hist_data["clipped_num_tracks"],
                        "plot_feature_names": plot_feature_names,
                    }

                processed_label["aggregated_features"][agg_key] = label_aggregator_data
            final_result["label_features"].append(processed_label)

        # Only return the result for dataset creation if the event passed filters
        if passed_filters:
            return final_result
        else:
            # Event failed filters but was processed for zero-bias plotting - don't include in dataset
            return None

    def _convert_stl_vector_array(self, branch_name: str, data: Any) -> np.ndarray:
        """
        Converts input data (NumPy array potentially containing STLVector objects,
        or a single STLVector object) into a numerical NumPy array using awkward-array,
        preserving dimensions where possible.

        Args:
            branch_name: Name of the branch (for logging)
            data: Input data (np.ndarray or potentially STLVector object)

        Returns:
            Converted numerical NumPy array (e.g., float32) or the original data if no conversion needed/possible.
            Returns an empty float32 array if the input represents an empty structure.
        """
        try:
            is_numpy = isinstance(data, np.ndarray)

            # --- Handle NumPy Array Input ---
            if is_numpy:
                if data.ndim == 0:
                    # Handle 0-dimensional array (likely containing a single object)
                    item = data.item()
                    if "STLVector" in str(type(item)):
                        self.logger.debug(
                            f"Converting 0-D array item for '{branch_name}'..."
                        )
                        # Convert the single item
                        ak_array = ak.Array([item])  # Wrap item for awkward conversion
                        np_array = ak.to_numpy(ak_array)  # REMOVED .flatten()
                    else:
                        # 0-D array but not STLVector, return as is (or wrap if needed?)
                        # If downstream expects at least 1D, wrap it.
                        return np.atleast_1d(data)
                elif data.dtype == object:
                    # Handle N-dimensional object array
                    if data.size == 0:
                        return np.array(
                            [], dtype=np.float32
                        )  # Handle empty object array
                    # Check first element to guess if it contains STL Vectors
                    first_element = data.flat[0]
                    if "STLVector" in str(type(first_element)):
                        self.logger.debug(
                            f"Converting N-D object array for '{branch_name}'..."
                        )
                        # Use awkward-array for robust conversion
                        ak_array = ak.from_iter(data)
                        np_array = ak.to_numpy(ak_array)  # REMOVED .flatten()
                    else:
                        # N-D object array, but doesn't seem to contain STL Vectors
                        self.logger.warning(
                            f"Branch '{branch_name}' is N-D object array but doesn't appear to be STLVector. Returning as is."
                        )
                        return data  # Return original object array
                else:
                    # Standard numerical NumPy array, return as is
                    return data

            # --- Handle Non-NumPy Input (e.g., direct STLVector) ---
            elif "STLVector" in str(type(data)):
                self.logger.debug(
                    f"Converting direct STLVector object for '{branch_name}'..."
                )
                # Convert the single STLVector object
                ak_array = ak.Array([data])  # Wrap object for awkward conversion
                np_array = ak.to_numpy(ak_array)[
                    0
                ]  # Convert and get the single resulting array

            # --- Handle Other Non-NumPy Input Types ---
            else:
                # E.g., could be a standard Python scalar (int, float) - ensure it's a numpy array
                if isinstance(data, (int, float, bool)):
                    return np.array([data])  # Return as 1-element array
                else:
                    # Unexpected type
                    self.logger.warning(
                        f"Branch '{branch_name}' has unexpected type '{type(data)}'. Returning as is."
                    )
                    return data  # Return original data if type is unexpected

            # --- Post-Conversion Processing (applies if conversion happened) ---
            # Check if the result of conversion is numeric and cast
            if np.issubdtype(np_array.dtype, np.number):
                converted_array = np_array.astype(np.float32)
                self.logger.debug(
                    f"Successfully converted '{branch_name}' from object dtype to {converted_array.dtype}, shape {converted_array.shape}"
                )  # Updated log
                return converted_array
            else:
                self.logger.warning(
                    f"Branch '{branch_name}' converted, but result is not numeric ({np_array.dtype}). Returning non-flattened array."
                )  # Updated log
                return np_array

        except Exception as e:
            self.logger.error(
                f"Failed to convert data for branch '{branch_name}' (type: {type(data)}): {e}",
                exc_info=True,
            )
            # Return original array or raise error depending on desired handling
            raise  # Re-raise the exception to stop processing if conversion fails

    def _process_selection_config(
        self,
        event_data: dict[str, np.ndarray],
        selection_config: PhysliteSelectionConfig,
    ) -> Optional[dict[str, Any]]:
        """
        Process an event according to a selection configuration.

        New approach:
        - Each aggregator ALWAYS returns correctly shaped arrays (max_length x num_features)
        - If aggregator fails, return zeros with correct shape
        - If any aggregator fails a filter, reject entire event
        - Concatenate all aggregator results at the end
        - Skip event if all aggregators are zero
        """
        # Extract scalar features
        scalar_features = {}
        if selection_config.feature_selectors:
            scalar_features = self._extract_selected_features(
                event_data, selection_config.feature_selectors
            )

        # Process aggregators with new approach
        aggregated_features = {}
        is_passing_filters = True  # Track if event passes all filters

        if selection_config.feature_array_aggregators:
            for idx, aggregator in enumerate(
                selection_config.feature_array_aggregators
            ):
                agg_key = f"aggregator_{idx}"

                # Try to process this aggregator
                try:
                    result_array, input_branch_details, n_valid_elements, hist_data = (
                        self._process_single_aggregator(event_data, aggregator, idx)
                    )

                    # Check if this aggregator failed a filter (indicated by None return)
                    if result_array is None:
                        # Filter failed - reject entire event
                        is_passing_filters = False
                        break

                    # Store successful aggregator result
                    aggregated_features[agg_key] = {
                        "array": result_array,
                        "input_branch_details": input_branch_details,
                        "n_valid_elements": n_valid_elements,
                        "hist_data": hist_data,
                    }

                except Exception as e:
                    self.logger.warning(f"Error processing aggregator {idx}: {e}")
                    # On error, create zero-filled array with correct shape
                    zero_array, zero_branch_details, zero_n_valid, zero_hist_data = (
                        self._create_zero_aggregator(event_data, aggregator, idx)
                    )
                    aggregated_features[agg_key] = {
                        "array": zero_array,
                        "input_branch_details": zero_branch_details,
                        "n_valid_elements": zero_n_valid,
                        "hist_data": zero_hist_data,
                    }

        # If any aggregator failed filters, reject the entire event
        if not is_passing_filters:
            self.logger.debug(
                "Event rejected due to filter failure in one or more aggregators"
            )
            return None

        # Check if all aggregators are zero (no useful data)
        if aggregated_features:
            all_zero = True
            for agg_data in aggregated_features.values():
                if agg_data["n_valid_elements"] > 0:
                    all_zero = False
                    break

            if all_zero:
                self.logger.debug("Event rejected: all aggregators returned zero data")
                return None

        # Return results if we have anything
        if not scalar_features and not aggregated_features:
            self.logger.debug("No scalar features and no aggregated features found")
            return None

        return {
            "scalar_features": scalar_features,
            "aggregated_features": aggregated_features,
        }

    def _process_single_aggregator(
        self,
        event_data: dict[str, np.ndarray],
        aggregator: PhysliteFeatureArrayAggregator,
        idx: int,
    ) -> tuple[Optional[np.ndarray], list[dict], int, Optional[dict]]:
        """
        Process a single aggregator, always returning correctly shaped arrays.

        Returns:
            - result_array: Always (max_length, num_features) or None if filter failed
            - input_branch_details: List of branch detail dicts
            - n_valid_elements: Number of valid elements before padding
            - hist_data: Histogram data or None
        """
        # --- Check Requirements and Get Input Arrays ---
        array_features = {}  # Stores input arrays for aggregation
        required_for_agg = set(s.branch.name for s in aggregator.input_branches)
        filter_branch_names = set(f.branch.name for f in aggregator.filter_branches)
        required_for_agg.update(filter_branch_names)
        if aggregator.sort_by_branch:
            required_for_agg.add(aggregator.sort_by_branch.branch.name)

        # Check if all required branches exist
        if not all(branch_name in event_data for branch_name in required_for_agg):
            missing = required_for_agg - set(event_data.keys())
            self.logger.debug(
                f"Missing required branches for aggregator {idx}: {missing}"
            )
            # Return zero-filled array with correct shape
            return self._create_zero_aggregator(event_data, aggregator, idx)

        # --- Check length consistency on RAW arrays ---
        initial_length = -1
        raw_array_features = {}
        for branch_name in required_for_agg:
            current_array = event_data[branch_name]
            raw_array_features[branch_name] = current_array

            try:
                current_length = len(current_array)
            except TypeError:
                self.logger.warning(
                    f"Could not get length of raw data for branch '{branch_name}' in aggregator {idx}"
                )
                return self._create_zero_aggregator(event_data, aggregator, idx)

            if initial_length == -1:
                initial_length = current_length
            elif initial_length != current_length:
                self.logger.warning(f"RAW array length mismatch in aggregator {idx}")
                return self._create_zero_aggregator(event_data, aggregator, idx)

        # If no length found or zero length
        if initial_length <= 0:
            self.logger.debug(
                f"Initial length is {initial_length} for aggregator {idx}"
            )
            return self._create_zero_aggregator(event_data, aggregator, idx)

        # --- Convert STL Vectors and prepare arrays ---
        array_features = {}
        filter_arrays = {}
        sort_value = None
        post_conversion_length = -1

        for branch_name in required_for_agg:
            converted_array = self._convert_stl_vector_array(
                branch_name, raw_array_features[branch_name]
            )

            is_input_branch = branch_name in (
                s.branch.name for s in aggregator.input_branches
            )
            is_filter_branch = branch_name in filter_branch_names
            is_sort_branch = (
                aggregator.sort_by_branch
                and branch_name == aggregator.sort_by_branch.branch.name
            )

            if is_input_branch:
                array_features[branch_name] = converted_array
            if is_filter_branch:
                filter_arrays[branch_name] = converted_array
            if is_sort_branch:
                sort_value = converted_array

            if post_conversion_length == -1:
                if is_sort_branch or is_filter_branch or is_input_branch:
                    post_conversion_length = len(converted_array)

        # --- Apply Filters ---
        valid_mask = np.ones(post_conversion_length, dtype=bool)
        for filter_config in aggregator.filter_branches:
            filter_name = filter_config.branch.name
            filter_array = filter_arrays.get(filter_name)
            if filter_array is None:
                continue

            # Apply filter
            if hasattr(filter_config, "min") and filter_config.min is not None:
                valid_mask &= filter_array >= filter_config.min
            if hasattr(filter_config, "max") and filter_config.max is not None:
                valid_mask &= filter_array <= filter_config.max

        # Check if we meet minimum length requirement after filtering
        n_valid = np.sum(valid_mask)
        if n_valid < aggregator.min_length:
            self.logger.debug(
                f"Aggregator {idx}: n_valid ({n_valid}) < min_length ({aggregator.min_length})"
            )
            return self._create_zero_aggregator(event_data, aggregator, idx)

        # --- Apply Sorting ---
        if aggregator.sort_by_branch and sort_value is not None:
            if len(sort_value) != post_conversion_length:
                self.logger.warning(f"Sort array length mismatch in aggregator {idx}")
                return self._create_zero_aggregator(event_data, aggregator, idx)

            masked_sort_value = sort_value[valid_mask]
            if len(masked_sort_value) == 0:
                sort_indices = np.array([], dtype=int)
            else:
                sort_indices = np.argsort(masked_sort_value)[::-1]
        else:
            sort_indices = np.arange(n_valid)

        # --- Apply Aggregation ---
        processed_feature_segments = []
        for selector in aggregator.input_branches:
            branch_name = selector.branch.name
            values = array_features.get(branch_name)
            if values is None:
                self.logger.warning(
                    f"Input branch '{branch_name}' not found for aggregator {idx}"
                )
                return self._create_zero_aggregator(event_data, aggregator, idx)

            # Apply mask and reshape
            filtered_values = values[valid_mask]
            if filtered_values.shape[0] != n_valid:
                self.logger.warning(
                    f"Filtered array shape mismatch for '{branch_name}' in aggregator {idx}"
                )
                return self._create_zero_aggregator(event_data, aggregator, idx)

            if filtered_values.ndim == 1:
                filtered_values = filtered_values.reshape(-1, 1)
            elif filtered_values.ndim != 2:
                self.logger.warning(
                    f"Unexpected array dimension for '{branch_name}' in aggregator {idx}"
                )
                return self._create_zero_aggregator(event_data, aggregator, idx)

            processed_feature_segments.append(filtered_values)

        if not processed_feature_segments:
            self.logger.warning(f"No feature segments collected for aggregator {idx}")
            return self._create_zero_aggregator(event_data, aggregator, idx)

        # Get num_cols per segment
        num_cols_per_segment = [
            seg.shape[1] if seg.ndim == 2 else 1 for seg in processed_feature_segments
        ]

        # Stack features horizontally
        try:
            features = np.hstack(processed_feature_segments)
        except ValueError as e:
            self.logger.error(f"Error during np.hstack in aggregator {idx}: {e}")
            return self._create_zero_aggregator(event_data, aggregator, idx)

        # Apply sorting
        if len(sort_indices) != n_valid:
            self.logger.warning(f"Sort indices length mismatch in aggregator {idx}")
            return self._create_zero_aggregator(event_data, aggregator, idx)

        try:
            sort_indices = np.asarray(sort_indices, dtype=int)
            sorted_features = features[sort_indices]
        except (IndexError, Exception) as e:
            self.logger.error(f"Error applying sort_indices in aggregator {idx}: {e}")
            return self._create_zero_aggregator(event_data, aggregator, idx)

        # Handle length requirements (padding/truncation)
        num_selected_tracks = sorted_features.shape[0]
        num_features_per_track = sorted_features.shape[1]

        # Capture clipped features for histogram
        if num_selected_tracks > aggregator.max_length:
            clipped_features_for_hist = sorted_features[: aggregator.max_length, :]
            clipped_num_tracks = aggregator.max_length
            final_features = clipped_features_for_hist.copy()
        elif num_selected_tracks < aggregator.max_length:
            clipped_features_for_hist = sorted_features.copy()
            clipped_num_tracks = num_selected_tracks
            # Pad with zeros
            num_padding_tracks = aggregator.max_length - num_selected_tracks
            padding = np.zeros(
                (num_padding_tracks, num_features_per_track),
                dtype=sorted_features.dtype,
            )
            final_features = np.vstack([sorted_features, padding])
        else:
            clipped_features_for_hist = sorted_features.copy()
            clipped_num_tracks = num_selected_tracks
            final_features = sorted_features

        # Build input_branch_details
        input_branch_details = []
        for i, selector in enumerate(aggregator.input_branches):
            input_branch_details.append(
                {
                    "name": selector.branch.name,
                    "num_output_cols": num_cols_per_segment[i],
                }
            )

        # Prepare histogram data
        hist_data = {
            "clipped_features": clipped_features_for_hist,
            "clipped_num_tracks": clipped_num_tracks,
            "num_cols_per_segment": num_cols_per_segment,
        }

        return final_features, input_branch_details, n_valid, hist_data

    def _create_zero_aggregator(
        self,
        event_data: dict[str, np.ndarray],
        aggregator: PhysliteFeatureArrayAggregator,
        idx: int,
    ) -> tuple[np.ndarray, list[dict], int, None]:
        """
        Create a zero-filled aggregator result with correct shape.

        Returns:
            - zero_array: (max_length, total_features) filled with zeros
            - input_branch_details: List of branch detail dicts
            - n_valid_elements: 0 (no valid elements for failed aggregator)
            - hist_data: None (no histogram data for failed aggregator)
        """
        total_features = 0
        input_branch_details = []

        for selector in aggregator.input_branches:
            branch_name = selector.branch.name

            # Determine actual column count by conversion
            try:
                sample_data = event_data.get(branch_name)
                if sample_data is not None:
                    converted_sample = self._convert_stl_vector_array(
                        branch_name, sample_data
                    )
                    if converted_sample.ndim == 1:
                        num_cols = 1
                    elif converted_sample.ndim == 2:
                        num_cols = converted_sample.shape[1]
                    else:
                        num_cols = 1
                else:
                    num_cols = 1
            except Exception:
                num_cols = 1

            total_features += num_cols
            input_branch_details.append(
                {
                    "name": selector.branch.name,
                    "num_output_cols": num_cols,
                }
            )

        # Create zero-filled array with correct shape
        zero_array = np.zeros((aggregator.max_length, total_features), dtype=np.float32)

        return zero_array, input_branch_details, 0, None

    def _process_data(
        self,
        task_config: TaskConfig,
        run_number: Optional[str] = None,
        signal_key: Optional[str] = None,
        catalog_limit: Optional[int] = None,
        event_limit: Optional[int] = None,
        plot_distributions: bool = False,
        delete_catalogs: bool = True,
        plot_output: Optional[Path] = None,
        first_event_logged: bool = True,
        bin_edges_metadata_path: Optional[Path] = None,
        return_histogram_data: bool = False,
        total_catalogs_across_all_runs: Optional[int] = None,
    ) -> tuple[
        list[dict[str, np.ndarray]], list[dict[str, np.ndarray]], dict, Optional[dict]
    ]:
        """
        Process either ATLAS or signal data using task configuration.

        Args:
            task_config: Configuration defining event filters, input features, and labels
            run_number: Optional run number for ATLAS data
            signal_key: Optional signal type for signal data
            catalog_limit: Optional limit on number of catalogs to process
            event_limit: Optional limit on number of events to process per catalog (only events that pass selection count)
            plot_distributions: Whether to generate distribution plots
            delete_catalogs: Whether to delete catalogs after processing
            plot_output: Optional complete path (including filename) for saving plots
            first_event_logged: Whether the first event has been logged
            bin_edges_metadata_path: Optional path to save/load bin edges metadata for coordinated histogram binning
            return_histogram_data: If True, return histogram data instead of saving it to file
            total_catalogs_across_all_runs: Total number of catalogs across all runs for proper sample calculation

        Returns:
            Tuple containing:
            - List of processed input features (each a dict with scalar and aggregated features)
            - List of processed label features (each a list of dicts for multiple label configs)
            - Processing statistics
            - Optional histogram data dictionary (if return_histogram_data=True)
        """
        # Add logging to identify data type being processed
        data_type_label = (
            f"BACKGROUND(run={run_number})"
            if run_number
            else f"SIGNAL(key={signal_key})"
        )

        if signal_key:
            self.logger.info(f"Processing signal data for {signal_key}")
        elif run_number:
            self.logger.info(f"Processing ATLAS data for run {run_number}")
        else:
            raise ValueError("Must provide either run_number or signal_key")

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
        initially_required_branches = self.get_required_branches(task_config)
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

        # Get catalog paths
        self.logger.info(
            f"Getting catalog paths for run {run_number} and signal {signal_key}"
        )
        catalog_paths = self._get_catalog_paths(run_number, signal_key, catalog_limit)
        self.logger.info(f"Found {len(catalog_paths)} catalog paths to process.")

        # --- Plotting Setup ---
        plotting_enabled = plot_distributions and plot_output is not None
        max_plot_samples_total = 5000  # Overall target for plotting samples

        # Separate counters for zero-bias vs post-selection samples
        overall_plot_samples_count = 0  # Counter for post-selection samples
        zero_bias_samples_count = 0  # Counter for zero-bias samples
        samples_per_catalog = 0  # Target samples from each catalog
        num_catalogs_to_process = len(catalog_paths)

        if plotting_enabled and num_catalogs_to_process > 0:
            # Calculate target samples per catalog, ensuring at least 1
            # Use total catalogs across all runs if provided, otherwise use catalogs for this run only
            total_catalogs_for_calculation = (
                total_catalogs_across_all_runs or num_catalogs_to_process
            )
            samples_per_catalog = max(
                1, max_plot_samples_total // total_catalogs_for_calculation
            )
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
                f"Processing catalog {catalog_idx + 1}/{num_catalogs_to_process} with path: {catalog_path}"
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
                            except KeyError:
                                # Should not happen if StopIteration doesn't, but defensive
                                self.logger.error(
                                    "Internal error accessing first key in non-empty batch. Skipping."
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
                                # Skip if somehow event data is empty (shouldn't happen with checks above)
                                if not raw_event_data:
                                    continue

                                # --- Calculate Derived Features ---
                                processed_event_data = (
                                    raw_event_data.copy()
                                )  # Start with real data
                                calculation_failed = False
                                for derived_name in derived_features_requested:
                                    derived_feature_def = get_derived_feature(
                                        derived_name
                                    )
                                    if not derived_feature_def:
                                        continue  # Should not happen

                                    # Check if all dependencies were read successfully for this event
                                    dependencies_present = all(
                                        dep in raw_event_data
                                        for dep in derived_feature_def.dependencies
                                    )

                                    if dependencies_present:
                                        try:
                                            # Prepare dependency data for calculation
                                            dependency_values = {
                                                dep: raw_event_data[dep]
                                                for dep in derived_feature_def.dependencies
                                            }
                                            # Calculate and add to the processed data
                                            calculated_value = (
                                                derived_feature_def.calculate(
                                                    dependency_values
                                                )
                                            )
                                            processed_event_data[derived_name] = (
                                                calculated_value
                                            )
                                        except Exception as calc_e:
                                            self.logger.error(
                                                f"Error calculating derived feature '{derived_name}' for event {evt_idx} in batch: {calc_e}"
                                            )
                                            calculation_failed = True
                                            break  # Stop processing derived features for this event
                                    else:
                                        missing_deps = [
                                            dep
                                            for dep in derived_feature_def.dependencies
                                            if dep not in raw_event_data
                                        ]
                                        self.logger.warning(
                                            f"Cannot calculate derived feature '{derived_name}' due to missing dependencies: {missing_deps}. Skipping for this event."
                                        )
                                        calculation_failed = True
                                        break  # Stop processing derived features for this event

                                if calculation_failed:
                                    continue  # Skip this event entirely if any derived feature calculation failed

                                # --- Determine if we need more zero-bias samples ---
                                need_more_zero_bias_samples = (
                                    plotting_enabled
                                    and current_catalog_zero_bias_count
                                    < samples_per_catalog
                                )

                                # --- Process Event (using data with derived features added) ---
                                result = self._process_event(
                                    processed_event_data,
                                    task_config,
                                    plotting_enabled,
                                    need_more_zero_bias_samples,
                                )

                                # --- Handle zero-bias data collection (before filter check) ---
                                if (
                                    plotting_enabled
                                    and current_catalog_zero_bias_count
                                    < samples_per_catalog
                                ):
                                    # Collect TRUE zero-bias data using dedicated method that bypasses ALL filtering
                                    zero_bias_result = self._process_event_zero_bias(
                                        processed_event_data,
                                        task_config,
                                    )

                                    if zero_bias_result is not None:
                                        # Accumulate zero-bias data
                                        for name, value in zero_bias_result[
                                            "scalar_features"
                                        ].items():
                                            zero_bias_scalar_features[name].append(
                                                value
                                            )

                                        for (
                                            agg_key,
                                            agg_data_with_plot_names,
                                        ) in zero_bias_result[
                                            "aggregated_features"
                                        ].items():
                                            zero_bias_aggregated_features[
                                                agg_key
                                            ].append(agg_data_with_plot_names)

                                        # Append num_tracks_for_plot for zero-bias
                                        num_tracks = zero_bias_result.get(
                                            "num_tracks_for_plot"
                                        )
                                        if num_tracks is not None:
                                            zero_bias_event_n_tracks_list.append(
                                                num_tracks
                                            )

                                        current_catalog_zero_bias_count += 1
                                        zero_bias_samples_count += 1

                                # --- Handle post-selection data collection (only if event passed filters) ---
                                if result is not None:
                                    # Store the result for dataset creation
                                    # Correctly extract only the arrays for the dataset
                                    input_features_for_dataset = {
                                        "scalar_features": result["scalar_features"],
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

                # Add sample collection tracking per catalog
                if plotting_enabled:
                    self.logger.templog(
                        f"[SAMPLE_TRACKING] {data_type_label} - Catalog {catalog_idx + 1}: "
                        f"Zero-bias samples: {current_catalog_zero_bias_count}/{samples_per_catalog}, "
                        f"Post-selection samples: {current_catalog_samples_count}/{samples_per_catalog}"
                    )

                if catalog_limit and catalog_idx >= catalog_limit - 1:
                    break

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
            self.logger.templog(
                f"[SAMPLE_TRACKING] {data_type_label} - FINAL TOTALS: "
                f"Zero-bias samples: {zero_bias_samples_count}, "
                f"Post-selection samples: {overall_plot_samples_count}"
            )

        # --- Generate and Save Dual Histogram Data (Zero-bias + Post-selection) ---
        # Helper function to generate histogram data for a dataset
        def generate_histogram_data(
            scalar_features_dict,
            aggregated_features_dict,
            event_n_tracks_list,
            overall_samples_count,
            data_type_name,
        ):
            if overall_samples_count == 0:
                return None

            histogram_data = {}

            # Load existing bin edges metadata if available (for coordinated binning)
            existing_bin_edges = None
            if bin_edges_metadata_path:
                existing_bin_edges = self._load_bin_edges_metadata(
                    bin_edges_metadata_path
                )

            # Add metadata including event count for legend display
            histogram_data["_metadata"] = {
                "total_events": int(stats["total_events"]),
                "total_processed_events": int(stats["processed_events"]),
                "total_features": int(stats["total_features"]),
                "processing_time": float(stats["processing_time"]),
                "total_sampled_events": int(overall_samples_count),
                "signal_key": signal_key if signal_key else "background",
                "run_number": run_number
                if run_number and not return_histogram_data
                else None,
                "data_type": data_type_name,  # "zero_bias" (raw detector) or "post_selection" (clipped, before padding)
            }

            # Details for N_Tracks_per_Event
            if event_n_tracks_list:
                counts_arr = np.array(event_n_tracks_list)
                if counts_arr.size > 0:
                    if (
                        existing_bin_edges
                        and "N_Tracks_per_Event" in existing_bin_edges
                    ):
                        # Use existing bin edges, filter out-of-range data
                        stored_edges = np.array(
                            existing_bin_edges["N_Tracks_per_Event"]
                        )
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
                        counts, _ = np.histogram(
                            counts_arr, bins=bin_edges, density=True
                        )
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
                    plot_feature_names = arrays_data_list[0].get(
                        "plot_feature_names", []
                    )
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
                                        p0_1, p99_9 = np.percentile(
                                            valid_data, [0.1, 99.9]
                                        )
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
                                        valid_tracks = feature_values[
                                            feature_values != 0
                                        ]
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
                                        p0_1, p99_9 = np.percentile(
                                            valid_data, [0.1, 99.9]
                                        )
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

        # Generate post-selection histogram data (existing behavior)
        if (
            plotting_enabled
            and overall_plot_samples_count > 0
            and raw_histogram_data_for_file is not None
        ):
            self.logger.info(
                f"Preparing post-selection histogram data (clipped tracks, before padding) from {overall_plot_samples_count} sampled events for {plot_output}"
            )

            raw_histogram_data_for_file = generate_histogram_data(
                sampled_scalar_features,
                sampled_aggregated_features,
                sampled_event_n_tracks_list,
                overall_plot_samples_count,
                "post_selection",
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

            zero_bias_histogram_data_for_file = generate_histogram_data(
                zero_bias_scalar_features,
                zero_bias_aggregated_features,
                zero_bias_event_n_tracks_list,
                zero_bias_samples_count,
                "zero_bias",
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

    def _process_event_zero_bias(
        self,
        event_data: dict[str, np.ndarray],
        task_config: TaskConfig,
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Process a single event for zero-bias plotting - bypasses ALL filtering.
        This shows the true raw detector response before any selections.

        Args:
            event_data: Dictionary mapping branch names (real and derived) to their raw values
            task_config: TaskConfig object containing input feature definitions (filters ignored)

        Returns:
            Dictionary of raw features or None if no valid tracks exist
        """
        # Extract scalar features without any filtering
        scalar_features = {}
        if task_config.input.feature_selectors:
            scalar_features = self._extract_selected_features(
                event_data, task_config.input.feature_selectors
            )

        # Process aggregators without any filtering - collect ALL tracks
        aggregated_features = {}
        if task_config.input.feature_array_aggregators:
            for idx, aggregator in enumerate(
                task_config.input.feature_array_aggregators
            ):
                # Get all required branches for this aggregator
                required_for_agg = set(s.branch.name for s in aggregator.input_branches)

                # Check if all required branches exist
                if not all(
                    branch_name in event_data for branch_name in required_for_agg
                ):
                    missing = required_for_agg - set(event_data.keys())
                    self.logger.debug(
                        f"Missing branches for zero-bias aggregator {idx}: {missing}"
                    )
                    continue

                # Get raw arrays without any length checks or filtering
                raw_array_features = {}
                for branch_name in required_for_agg:
                    raw_array_features[branch_name] = event_data[branch_name]

                # Convert STL vectors
                array_features = {}
                for branch_name in required_for_agg:
                    converted_array = self._convert_stl_vector_array(
                        branch_name, raw_array_features[branch_name]
                    )
                    array_features[branch_name] = converted_array

                # Get the track count from the first array (no length requirements)
                first_branch = list(required_for_agg)[0]
                if first_branch in array_features:
                    raw_track_count = len(array_features[first_branch])

                    if raw_track_count == 0:
                        continue  # Skip if no tracks at all

                    # Stack features horizontally without any filtering
                    processed_feature_segments = []
                    for selector in aggregator.input_branches:
                        branch_name = selector.branch.name
                        values = array_features.get(branch_name)
                        if values is None:
                            continue

                        # Reshape 1D to 2D if needed
                        if values.ndim == 1:
                            values = values.reshape(-1, 1)
                        processed_feature_segments.append(values)

                    if not processed_feature_segments:
                        continue

                    # Stack horizontally
                    try:
                        raw_features = np.hstack(processed_feature_segments)

                        # Apply sorting if specified (but no track count limits)
                        if aggregator.sort_by_branch:
                            sort_branch_name = aggregator.sort_by_branch.branch.name
                            if sort_branch_name in array_features:
                                sort_values = array_features[sort_branch_name]
                                if len(sort_values) == raw_track_count:
                                    sort_indices = np.argsort(sort_values)[::-1]
                                    raw_features = raw_features[sort_indices]

                        # Create branch details for plotting
                        num_cols_per_segment = [
                            seg.shape[1] if seg.ndim == 2 else 1
                            for seg in processed_feature_segments
                        ]
                        input_branch_details = []
                        for i, selector in enumerate(aggregator.input_branches):
                            input_branch_details.append(
                                {
                                    "name": selector.branch.name,
                                    "num_output_cols": num_cols_per_segment[i],
                                }
                            )

                        # Create plot feature names
                        plot_feature_names = []
                        for detail in input_branch_details:
                            branch_name = detail["name"]
                            num_output_cols = detail["num_output_cols"]
                            custom_titles = self.custom_label_map.get(branch_name)

                            if (
                                isinstance(custom_titles, list)
                                and len(custom_titles) == num_output_cols
                            ):
                                plot_feature_names.extend(custom_titles)
                            elif (
                                isinstance(custom_titles, str) and num_output_cols == 1
                            ):
                                plot_feature_names.append(custom_titles)
                            else:  # Fallback
                                if num_output_cols == 1:
                                    plot_feature_names.append(branch_name)
                                else:
                                    plot_feature_names.extend(
                                        [
                                            f"{branch_name}_comp{j}"
                                            for j in range(num_output_cols)
                                        ]
                                    )

                        aggregated_features[f"aggregator_{idx}"] = {
                            "array": raw_features,  # All tracks, no padding/clipping
                            "plot_feature_names": plot_feature_names,
                            "n_valid_elements": raw_track_count,
                        }

                    except Exception as e:
                        self.logger.debug(
                            f"Error processing zero-bias aggregator {idx}: {e}"
                        )
                        continue

        # Return results if we have anything
        if not scalar_features and not aggregated_features:
            return None

        # Set num_tracks_for_plot from first aggregator
        num_tracks_for_plot = None
        if aggregated_features:
            first_agg_key = next(iter(aggregated_features.keys()))
            num_tracks_for_plot = aggregated_features[first_agg_key].get(
                "n_valid_elements"
            )

        return {
            "scalar_features": scalar_features,
            "aggregated_features": aggregated_features,
            "label_features": [],  # No labels for zero-bias
            "num_tracks_for_plot": num_tracks_for_plot,
            "passed_filters": False,  # Zero-bias data didn't pass filters by definition
        }
