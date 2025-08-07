import json
from pathlib import Path
from typing import Any, Optional

import awkward as ak
import numpy as np

from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import (
    PhysliteFeatureArrayAggregator,
    PhysliteFeatureFilter,
    PhysliteFeatureSelector,
    PhysliteSelectionConfig,
    TaskConfig,
)
from hep_foundation.data.atlas_file_manager import ATLASFileManager


class PhysliteEventProcessor:
    """
    Handles processing and filtering of individual ATLAS PhysLite events.

    This class is responsible for:
    - Processing single events through task configurations
    - Applying event and feature filters
    - Processing and aggregating feature arrays
    - Converting STL vectors to numpy arrays
    - Zero-bias event processing
    """

    def __init__(
        self,
        atlas_manager=None,
        custom_label_map_path: Optional[
            str
        ] = "src/hep_foundation/data/physlite_plot_labels.json",
    ):
        """
        Initialize the PhysliteEventProcessor.

        Args:
            atlas_manager: Optional ATLASFileManager instance
            custom_label_map_path: Optional path to a JSON file for custom plot labels.
        """
        self.logger = get_logger(__name__)
        self.atlas_manager = atlas_manager or ATLASFileManager()

        # Load custom label mapping for plot names
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
        need_more_zero_bias_samples: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Process an event according to a selection configuration.

        Approach:
        - For dataset creation: aggregators must pass min_length filter or entire event is rejected
        - For zero-bias sampling: process histogram data even if min_length fails
        - Return None if aggregators fail for dataset creation, but may still collect histogram data
        """
        # Extract scalar features
        scalar_features = {}
        if selection_config.feature_selectors:
            scalar_features = self._extract_selected_features(
                event_data, selection_config.feature_selectors
            )

        # Process aggregators
        aggregated_features = {}
        all_aggregators_passed_filters = True  # Track if ALL aggregators pass filters

        if selection_config.feature_array_aggregators:
            for idx, aggregator in enumerate(
                selection_config.feature_array_aggregators
            ):
                agg_key = f"aggregator_{idx}"

                # Process this aggregator
                try:
                    (
                        aggregator_result_array,
                        n_valid_elements,
                        post_selection_hist_data,
                        zero_bias_hist_data,
                        aggregator_passed_filters,
                    ) = self._process_single_aggregator(
                        event_data,
                        aggregator,
                        idx,
                        need_more_zero_bias_samples,
                    )

                    # Check if this aggregator passed filters for dataset creation
                    if not aggregator_passed_filters:
                        all_aggregators_passed_filters = False

                    # Check for fatal errors (None return for array)
                    if (
                        aggregator_result_array is None
                        and not need_more_zero_bias_samples
                    ):
                        # Fatal error and we don't need zero-bias data, exit immediately
                        self.logger.debug(
                            f"Fatal error in aggregator {idx}, early exit"
                        )
                        return None
                    elif (
                        aggregator_result_array is None and need_more_zero_bias_samples
                    ):
                        # Aggregator failed min_length but we need zero-bias data
                        # Store the histogram data even though there's no training array
                        all_aggregators_passed_filters = False
                        aggregated_features[agg_key] = {
                            "array": None,  # No training array since min_length failed
                            "n_valid_elements": n_valid_elements,
                            "post_selection_hist_data": post_selection_hist_data,  # Should be None since filters failed
                            "zero_bias_hist_data": zero_bias_hist_data,  # Should contain actual data
                        }
                        continue

                    # Store aggregator result (for successful aggregators)
                    if aggregator_result_array is not None:
                        aggregated_features[agg_key] = {
                            "array": aggregator_result_array,
                            "n_valid_elements": n_valid_elements,
                            "post_selection_hist_data": post_selection_hist_data,
                            "zero_bias_hist_data": zero_bias_hist_data,
                        }

                except Exception as e:
                    self.logger.warning(f"Error processing aggregator {idx}: {e}")
                    # Fatal error - mark as failed and potentially stop
                    all_aggregators_passed_filters = False
                    if not need_more_zero_bias_samples:
                        return None
                    # If we need zero-bias data, continue with other aggregators

        # Decision logic:
        # 1. If this is for dataset creation (not need_more_zero_bias_samples) and any aggregator failed min_length, reject event
        # 2. If this is for zero-bias sampling (need_more_zero_bias_samples), always return data for histogram collection

        # For dataset creation (when filters must pass)
        if not all_aggregators_passed_filters and not need_more_zero_bias_samples:
            self.logger.debug(
                "Event rejected: one or more aggregators failed min_length requirement"
            )
            return None

        # For zero-bias sampling: always return data even if no valid aggregators for dataset
        # This ensures histogram data can be collected from failed events
        if need_more_zero_bias_samples:
            # Return whatever we have, even if aggregated_features is empty
            # The histogram data is stored within each aggregator's data
            return {
                "scalar_features": scalar_features,
                "aggregated_features": aggregated_features,
            }

        # For normal dataset creation: check if we have valid data
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
        need_more_zero_bias_samples: bool = False,
    ) -> tuple[Optional[np.ndarray], int, Optional[dict], Optional[dict], bool]:
        """
        Process a single aggregator.

        Returns:
            - aggregator_result_array: (max_length, num_features) array for dataset, or None if fatal error/early termination
            - n_valid_elements: Number of valid elements before padding
            - post_selection_hist_data: Dict of {branch_name: array} for post-selection plotting (only if filters pass)
            - zero_bias_hist_data: Dict of {branch_name: array} for zero-bias plotting (if needed)
            - passed_filters: Boolean indicating whether this aggregator passed min_length requirements
        """
        # --- Check Requirements and Get Input Arrays ---
        array_features = {}  # Stores input arrays for aggregation
        required_for_agg = set(s.branch.name for s in aggregator.input_branches)
        filter_branch_names = set(f.branch.name for f in aggregator.filter_branches)
        required_for_agg.update(filter_branch_names)
        if aggregator.sort_by_branch:
            required_for_agg.add(aggregator.sort_by_branch.branch.name)

        # Check if all required branches exist - fatal error
        if not all(branch_name in event_data for branch_name in required_for_agg):
            missing = required_for_agg - set(event_data.keys())
            self.logger.debug(
                f"Missing required branches for aggregator {idx}: {missing}"
            )
            return None, 0, None, None, False

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
                return None, 0, None, None, False

            if initial_length == -1:
                initial_length = current_length
            elif initial_length != current_length:
                self.logger.warning(f"RAW array length mismatch in aggregator {idx}")
                return None, 0, None, None, False

        # If no length found or zero length - fatal error
        if initial_length <= 0:
            self.logger.debug(
                f"Initial length is {initial_length} for aggregator {idx}"
            )
            return None, 0, None, None, False

        # --- Convert STL Vectors and prepare arrays ---
        array_features = {}
        filter_arrays = {}
        sort_value = None
        post_conversion_length = -1

        try:
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
        except Exception as e:
            self.logger.error(f"Error during STL conversion in aggregator {idx}: {e}")
            return None, 0, None, None, False

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
        passed_aggregator_filters = n_valid >= aggregator.min_length

        # Early exit logic: if filters fail and we don't need zero-bias data, return None immediately
        if not passed_aggregator_filters and not need_more_zero_bias_samples:
            self.logger.debug(
                f"Aggregator {idx}: n_valid ({n_valid}) < min_length ({aggregator.min_length}) - early exit"
            )
            return None, n_valid, None, None, False

        # Initialize histogram data containers
        post_selection_hist_data = None
        zero_bias_hist_data = None

        # --- Process data for histogram collection (only if we need zero-bias or if filters pass) ---
        if passed_aggregator_filters or need_more_zero_bias_samples:
            # Initialize histogram data dictionaries based on what we need
            if passed_aggregator_filters:
                post_selection_hist_data = {}
            if need_more_zero_bias_samples:
                zero_bias_hist_data = {}

            # Process histogram data for each input branch
            for selector in aggregator.input_branches:
                branch_name = selector.branch.name
                values = array_features.get(branch_name)
                if values is None:
                    self.logger.warning(
                        f"Input branch '{branch_name}' not found for aggregator {idx}"
                    )
                    return None, n_valid, None, None, False

                # Generate histogram data for post-selection (filtered values)
                if passed_aggregator_filters and post_selection_hist_data is not None:
                    filtered_values = values[valid_mask]
                    # Store as flat list for proper histogram format
                    post_selection_hist_data[branch_name] = (
                        filtered_values.flatten().tolist()
                    )

                # Generate histogram data for zero-bias (all values, no filters)
                if need_more_zero_bias_samples:
                    unfiltered_values = values.copy()
                    # Store as flat list for proper histogram format
                    zero_bias_hist_data[branch_name] = (
                        unfiltered_values.flatten().tolist()
                    )

            # No need to sort histogram data - histograms are order-independent
            # Sorting is only needed for training arrays where truncation matters

        # --- Create model training data (only if filters pass) ---
        aggregator_result_array = None

        if passed_aggregator_filters:
            try:
                processed_feature_segments = []

                for selector in aggregator.input_branches:
                    branch_name = selector.branch.name
                    values = array_features.get(branch_name)
                    if values is None:
                        self.logger.warning(
                            f"Input branch '{branch_name}' not found for aggregator {idx}"
                        )
                        return None, n_valid, None, None, False

                    # Apply mask and reshape for model training
                    filtered_values = values[valid_mask]
                    if filtered_values.shape[0] != n_valid:
                        self.logger.warning(
                            f"Filtered array shape mismatch for '{branch_name}' in aggregator {idx}"
                        )
                        return None, n_valid, None, None, False

                    if filtered_values.ndim == 1:
                        filtered_values = filtered_values.reshape(-1, 1)
                    elif filtered_values.ndim != 2:
                        self.logger.warning(
                            f"Unexpected array dimension for '{branch_name}' in aggregator {idx}"
                        )
                        return None, n_valid, None, None, False

                    processed_feature_segments.append(filtered_values)

                if not processed_feature_segments:
                    self.logger.warning(
                        f"No feature segments collected for aggregator {idx}"
                    )
                    return None, n_valid, None, None, False

                # Stack features horizontally
                features = np.hstack(processed_feature_segments)

                # Determine sorting indices for model training data
                if aggregator.sort_by_branch and sort_value is not None:
                    if len(sort_value) != post_conversion_length:
                        self.logger.warning(
                            f"Sort array length mismatch in aggregator {idx}"
                        )
                        return None, n_valid, None, None, False

                    masked_sort_value = sort_value[valid_mask]
                    if len(masked_sort_value) == 0:
                        sort_indices = np.array([], dtype=int)
                    else:
                        sort_indices = np.argsort(masked_sort_value)[::-1]
                else:
                    sort_indices = np.arange(n_valid)

                # Apply sorting
                if len(sort_indices) != n_valid:
                    self.logger.warning(
                        f"Sort indices length mismatch in aggregator {idx}"
                    )
                    return None, n_valid, None, None, False

                sort_indices = np.asarray(sort_indices, dtype=int)
                sorted_features = features[sort_indices]

                # Apply length requirements for model training (padding/truncation)
                num_selected_tracks = sorted_features.shape[0]
                num_features_per_track = sorted_features.shape[1]

                # Always apply max_length limits and padding for model training data
                if num_selected_tracks > aggregator.max_length:
                    aggregator_result_array = sorted_features[
                        : aggregator.max_length, :
                    ]
                elif num_selected_tracks < aggregator.max_length:
                    # Pad with zeros for model training
                    num_padding_tracks = aggregator.max_length - num_selected_tracks
                    padding = np.zeros(
                        (num_padding_tracks, num_features_per_track),
                        dtype=sorted_features.dtype,
                    )
                    aggregator_result_array = np.vstack([sorted_features, padding])
                else:
                    aggregator_result_array = sorted_features

            except Exception as e:
                self.logger.error(
                    f"Error creating model training data in aggregator {idx}: {e}"
                )
                return None, n_valid, None, None, False

        return (
            aggregator_result_array,
            n_valid,
            post_selection_hist_data,
            zero_bias_hist_data,
            passed_aggregator_filters,
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

    def process_event(
        self,
        event_data: dict[str, np.ndarray],
        task_config: TaskConfig,
        plotting_enabled: bool = False,
        need_more_zero_bias_samples: bool = False,
    ) -> tuple[Optional[dict[str, np.ndarray]], bool]:
        """
        Process a single event using the task configuration.

        Args:
            event_data: Dictionary mapping branch names (real and derived) to their raw values
            task_config: TaskConfig object containing event filters, input features, and labels
            plotting_enabled: Whether plotting is enabled
            need_more_zero_bias_samples: Whether we still need more zero-bias samples for plotting

        Returns:
            Tuple of (processed_features, passed_filters):
            - processed_features: Dictionary of processed features or None if processing failed
            - passed_filters: Boolean indicating whether the event passed all filters

            processed_features structure:
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
            }
        """
        # Apply event filters first and track the result
        passed_filters = self._apply_event_filters(
            event_data, task_config.event_filters
        )

        # Determine if we should process this event
        # For zero-bias sampling, we process regardless of filter status
        # For filtered data, we only process if filters passed
        process_for_zero_bias = plotting_enabled and need_more_zero_bias_samples
        should_process = passed_filters or process_for_zero_bias

        if not should_process:
            return None, passed_filters

        # Process input selection config
        input_result = self._process_selection_config(
            event_data,
            task_config.input,
            need_more_zero_bias_samples,
        )

        # For zero-bias sampling, we always want to return data even if aggregators failed min_length
        # For normal dataset creation, we reject if input_result is None
        if input_result is None:
            if need_more_zero_bias_samples:
                # For zero-bias, create empty result but continue to potentially collect histogram data
                # from individual aggregator processing
                input_result = {"scalar_features": {}, "aggregated_features": {}}
            else:
                # For dataset creation, None means we should reject the event
                return None, passed_filters

        # Process each label config if present (only for filtered events going to dataset)
        label_results = []
        if passed_filters:  # Only process labels for events going into the dataset
            for label_config in task_config.labels:
                label_result = self._process_selection_config(
                    event_data,
                    label_config,
                    False,  # Labels: no zero-bias sampling needed
                )
                if label_result is None:
                    # If any label processing fails for a supervised task, reject the event
                    return None, passed_filters
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
            n_valid_elements = agg_data_dict.get("n_valid_elements")

            aggregator_data_object = {
                "array": agg_array,
                "n_valid_elements": n_valid_elements,
                "post_selection_hist_data": agg_data_dict.get(
                    "post_selection_hist_data"
                ),
                "zero_bias_hist_data": agg_data_dict.get("zero_bias_hist_data"),
            }

            final_result["aggregated_features"][agg_key] = aggregator_data_object

        # Set num_tracks_for_plot from the first input aggregator if available
        # This should reflect the actual number of tracks used for plotting (respecting max_length)
        if (
            first_input_agg_key
            and first_input_agg_key in final_result["aggregated_features"]
        ):
            n_valid = final_result["aggregated_features"][first_input_agg_key].get(
                "n_valid_elements", 0
            )
            # Get max_length from the first aggregator in task config
            max_length = None
            if task_config.input.feature_array_aggregators:
                max_length = task_config.input.feature_array_aggregators[0].max_length

            # Apply max_length constraint to num_tracks_for_plot
            if max_length is not None and n_valid > max_length:
                final_result["num_tracks_for_plot"] = max_length
            else:
                final_result["num_tracks_for_plot"] = n_valid

        # Add label results with branch names
        for i, label_res in enumerate(label_results):
            processed_label = {
                "scalar_features": label_res["scalar_features"],
                "aggregated_features": {},
            }
            for agg_key, agg_data_dict in label_res["aggregated_features"].items():
                agg_array = agg_data_dict["array"]
                n_valid_elements = agg_data_dict.get("n_valid_elements")

                label_aggregator_data_object = {
                    "array": agg_array,
                    "n_valid_elements": n_valid_elements,
                }

                processed_label["aggregated_features"][agg_key] = (
                    label_aggregator_data_object
                )
            final_result["label_features"].append(processed_label)

        # Return both the processed result and filter status
        return final_result, passed_filters
