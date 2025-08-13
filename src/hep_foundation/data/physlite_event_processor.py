import json
from pathlib import Path
from typing import Optional

import numpy as np

from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.new_processed_event import (
    EventSelectionData,
    NewProcessedEvent,
)
from hep_foundation.config.task_config import (
    PhysliteFeatureArrayAggregator,
    PhysliteFeatureFilter,
    PhysliteFeatureSelector,
    PhysliteSelectionConfig,
    TaskConfig,
)
from hep_foundation.data.atlas_file_manager import ATLASFileManager
from hep_foundation.data.physlite_utilities import (
    convert_stl_vector_array,
)


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

    def process_event_new_format(
        self,
        event_data: dict[str, np.ndarray],
        task_config: TaskConfig,
        plotting_enabled: bool = False,
        need_more_zero_bias_samples: bool = False,
    ) -> tuple[Optional[NewProcessedEvent], bool]:
        """
        Process a single event using the new data format.

        Args:
            event_data: Dictionary mapping branch names (real and derived) to their raw values
            task_config: TaskConfig object containing event filters, input features, and labels
            plotting_enabled: Whether plotting is enabled
            need_more_zero_bias_samples: Whether we still need more zero-bias samples for plotting

        Returns:
            Tuple of (new_processed_event, passed_filters):
            - new_processed_event: NewProcessedEvent instance or None if processing failed
            - passed_filters: Boolean indicating whether the event passed all filters
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

        # Process input selection config using new format
        input_selection_data = self._process_selection_config_new_format(
            event_data,
            task_config.input,
            need_more_zero_bias_samples,
        )

        # For zero-bias sampling, we always want to return data even if aggregators failed min_length
        # For normal dataset creation, we reject if input_selection_data is None
        if input_selection_data is None:
            if need_more_zero_bias_samples:
                # For zero-bias, create empty selection data
                input_selection_data = EventSelectionData(
                    selection_config_name=task_config.input.name,
                    event_raw_selector_data={},
                    event_raw_aggregators_data=[],
                    event_aggregators_sort_filter_indices=[],
                    event_aggregators_pass_filters=[],
                    aggregator_configs=[],
                )
            else:
                # For dataset creation, None means we should reject the event
                return None, passed_filters

        # Process each label config if present (only for filtered events going to dataset)
        label_selection_data = []
        if passed_filters:  # Only process labels for events going into the dataset
            for label_config in task_config.labels:
                label_data = self._process_selection_config_new_format(
                    event_data,
                    label_config,
                    False,  # Labels: no zero-bias sampling needed
                )
                if label_data is None:
                    # If any label processing fails for a supervised task, reject the event
                    return None, passed_filters
                label_selection_data.append(label_data)

        # Create and return NewProcessedEvent object
        new_processed_event = NewProcessedEvent(
            passed_filters=passed_filters,
            input_selection_data=input_selection_data,
            label_selection_data=label_selection_data,
        )

        return new_processed_event, passed_filters

    def _process_selection_config_new_format(
        self,
        event_data: dict[str, np.ndarray],
        selection_config: PhysliteSelectionConfig,
        need_more_zero_bias_samples: bool = False,
    ) -> Optional[EventSelectionData]:
        """
        Process an event according to a selection configuration using new format.

        Args:
            event_data: Dictionary mapping branch names to their raw values
            selection_config: Selection configuration to process
            need_more_zero_bias_samples: Whether we need zero-bias data even if filters fail

        Returns:
            EventSelectionData instance or None if processing failed and zero-bias not needed
        """
        # Extract scalar features (raw data, no processing needed)
        event_raw_selector_data = {}
        if selection_config.feature_selectors:
            event_raw_selector_data = self._extract_selected_features(
                event_data, selection_config.feature_selectors
            )

        # Process aggregators to get raw data and indices
        event_raw_aggregators_data = []
        event_aggregators_sort_filter_indices = []
        event_aggregators_pass_filters = []
        aggregator_configs = []

        all_aggregators_passed_filters = True  # Track if ALL aggregators pass filters

        if selection_config.feature_array_aggregators:
            for idx, aggregator in enumerate(
                selection_config.feature_array_aggregators
            ):
                try:
                    (
                        raw_aggregator_data,
                        sort_filter_indices,
                        aggregator_passed_filters,
                    ) = self._process_single_aggregator_new_format(
                        event_data,
                        aggregator,
                        idx,
                        need_more_zero_bias_samples,
                    )

                    # Check if this aggregator passed filters for dataset creation
                    if not aggregator_passed_filters:
                        all_aggregators_passed_filters = False

                    # Store aggregator results (even if filters failed, for zero-bias data)
                    event_raw_aggregators_data.append(raw_aggregator_data)
                    event_aggregators_sort_filter_indices.append(sort_filter_indices)
                    event_aggregators_pass_filters.append(aggregator_passed_filters)
                    aggregator_configs.append(aggregator)

                except Exception as e:
                    self.logger.warning(f"Error processing aggregator {idx}: {e}")
                    # Fatal error - mark as failed and potentially stop
                    all_aggregators_passed_filters = False
                    if not need_more_zero_bias_samples:
                        return None
                    # If we need zero-bias data, continue with empty data for this aggregator
                    event_raw_aggregators_data.append({})
                    event_aggregators_sort_filter_indices.append([])
                    event_aggregators_pass_filters.append(False)
                    aggregator_configs.append(aggregator)

        # Decision logic:
        # 1. If this is for dataset creation (not need_more_zero_bias_samples) and any aggregator failed min_length, reject event
        # 2. If this is for zero-bias sampling (need_more_zero_bias_samples), always return data for histogram collection

        # For dataset creation (when filters must pass)
        if not all_aggregators_passed_filters and not need_more_zero_bias_samples:
            self.logger.debug(
                "Event rejected: one or more aggregators failed min_length requirement"
            )
            return None

        # For normal dataset creation: check if we have valid data
        if (
            not need_more_zero_bias_samples
            and not event_raw_selector_data
            and not any(event_aggregators_pass_filters)
        ):
            self.logger.debug(
                "No scalar features and no valid aggregated features found"
            )
            return None

        # Create and return EventSelectionData
        return EventSelectionData(
            selection_config_name=selection_config.name,
            event_raw_selector_data=event_raw_selector_data,
            event_raw_aggregators_data=event_raw_aggregators_data,
            event_aggregators_sort_filter_indices=event_aggregators_sort_filter_indices,
            event_aggregators_pass_filters=event_aggregators_pass_filters,
            aggregator_configs=aggregator_configs,
        )

    def _process_single_aggregator_new_format(
        self,
        event_data: dict[str, np.ndarray],
        aggregator: PhysliteFeatureArrayAggregator,
        idx: int,
        need_more_zero_bias_samples: bool = False,
    ) -> tuple[dict[str, list[float]], list[int], bool]:
        """
        Process a single aggregator using new format.

        Returns:
            - raw_aggregator_data: Dict of {branch_name: [values]} for this aggregator
            - sort_filter_indices: List of indices after sorting and filtering
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
            return {}, [], False

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
                return {}, [], False

            if initial_length == -1:
                initial_length = current_length
            elif initial_length != current_length:
                self.logger.warning(f"RAW array length mismatch in aggregator {idx}")
                return {}, [], False

        # If no length found or zero length - fatal error
        if initial_length <= 0:
            self.logger.debug(
                f"Initial length is {initial_length} for aggregator {idx}"
            )
            return {}, [], False

        # --- Convert STL Vectors and prepare arrays ---
        array_features = {}
        filter_arrays = {}
        sort_value = None
        post_conversion_length = -1

        try:
            for branch_name in required_for_agg:
                converted_array = convert_stl_vector_array(
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
            return {}, [], False

        # Store raw data (before filtering) in the format needed for EventSelectionData
        raw_aggregator_data = {}
        for branch_name in (s.branch.name for s in aggregator.input_branches):
            if branch_name in array_features:
                # Convert to list of floats for consistent format
                raw_aggregator_data[branch_name] = array_features[branch_name].tolist()

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

        # Early exit logic: if filters fail and we don't need zero-bias data, return empty
        if not passed_aggregator_filters and not need_more_zero_bias_samples:
            self.logger.debug(
                f"Aggregator {idx}: n_valid ({n_valid}) < min_length ({aggregator.min_length}) - early exit"
            )
            return raw_aggregator_data, [], False

        # --- Generate sort and filter indices ---
        sort_filter_indices = []

        if passed_aggregator_filters:
            # Get indices of valid elements
            valid_indices = np.where(valid_mask)[0]

            # Determine sorting indices
            if aggregator.sort_by_branch and sort_value is not None:
                if len(sort_value) != post_conversion_length:
                    self.logger.warning(
                        f"Sort array length mismatch in aggregator {idx}"
                    )
                    return raw_aggregator_data, [], False

                # Sort the valid indices by their sort values (descending order)
                masked_sort_value = sort_value[valid_indices]
                if len(masked_sort_value) == 0:
                    sort_indices_relative = np.array([], dtype=int)
                else:
                    sort_indices_relative = np.argsort(masked_sort_value)[::-1]

                # Convert relative indices back to original array indices
                sort_filter_indices = valid_indices[sort_indices_relative].tolist()
            else:
                # No sorting, just use valid indices in original order
                sort_filter_indices = valid_indices.tolist()

        return raw_aggregator_data, sort_filter_indices, passed_aggregator_filters
