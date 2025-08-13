"""
New event processing data structures that mirror TaskConfig organization.

This module provides a cleaner separation between raw event data and transformed outputs,
eliminating redundant data transformations and providing clear APIs for different use cases.
"""

from typing import Any, Optional

import numpy as np


class EventSelectionData:
    """
    Contains processed data for a single selection configuration (input or label).

    This class stores raw event data once and provides different views through helper methods.
    It mirrors the structure of PhysliteSelectionConfig from the task configuration.
    """

    def __init__(
        self,
        selection_config_name: str,
        event_raw_selector_data: dict[str, float],
        event_raw_aggregators_data: list[dict[str, list[float]]],
        event_aggregators_sort_filter_indices: list[list[int]],
        event_aggregators_pass_filters: list[bool],
        aggregator_configs: list[
            Any
        ],  # References to original aggregator configs for metadata
    ):
        """
        Initialize event selection data.

        Args:
            selection_config_name: Name of the selection config this data represents
            event_raw_selector_data: Raw scalar feature data {branch_name: value}
            event_raw_aggregators_data: Raw aggregator data [{branch_name: [values]}, ...] per aggregator
            event_aggregators_sort_filter_indices: Sorted and filtered indices [aggregator_idx][track_indices]
            event_aggregators_pass_filters: Whether each aggregator passed its filters
            aggregator_configs: Original aggregator configurations for metadata access
        """
        self.selection_config_name = selection_config_name
        self.event_raw_selector_data = event_raw_selector_data
        self.event_raw_aggregators_data = event_raw_aggregators_data
        self.event_aggregators_sort_filter_indices = (
            event_aggregators_sort_filter_indices
        )
        self.event_aggregators_pass_filters = event_aggregators_pass_filters
        self.aggregator_configs = aggregator_configs

        # Validate data consistency
        self._validate_data_consistency()

    def _validate_data_consistency(self) -> None:
        """Validate that data structures are consistent."""
        num_aggregators = len(self.event_raw_aggregators_data)

        if len(self.event_aggregators_sort_filter_indices) != num_aggregators:
            raise ValueError("Mismatch between aggregator data and sort/filter indices")

        if len(self.event_aggregators_pass_filters) != num_aggregators:
            raise ValueError("Mismatch between aggregator data and pass filters")

        if len(self.aggregator_configs) != num_aggregators:
            raise ValueError("Mismatch between aggregator data and configs")

    def get_aggregator_filtered_count(self, aggregator_idx: int) -> int:
        """Get number of valid elements for a specific aggregator after filtering."""
        if aggregator_idx >= len(self.event_aggregators_sort_filter_indices):
            return 0

        return len(self.event_aggregators_sort_filter_indices[aggregator_idx])

    def get_zero_bias_hist_data(self) -> dict[str, list[float]]:
        """
        Get raw unfiltered data in histogram format (key -> list of values).

        Returns:
            Dictionary mapping branch names to lists of values for histogramming
        """
        hist_data = {}

        # Add scalar features (each becomes a single-item list)
        for branch_name, value in self.event_raw_selector_data.items():
            hist_data[branch_name] = [value] if not isinstance(value, list) else value

        # Add aggregated features (flatten all aggregators into branch -> list format)
        for aggregator_idx, aggregator_data in enumerate(
            self.event_raw_aggregators_data
        ):
            for branch_name, values_list in aggregator_data.items():
                # Flatten and extend - each track contributes its value(s)
                if branch_name in hist_data:
                    hist_data[branch_name].extend(values_list)
                else:
                    hist_data[branch_name] = list(values_list)

        return hist_data

    def get_post_selection_hist_data(self) -> dict[str, list[float]]:
        """
        Get filtered and sorted data in histogram format (key -> list of values).

        Returns:
            Dictionary mapping branch names to lists of values for histogramming,
            or empty dict if no aggregators passed filters
        """
        hist_data = {}

        # Add scalar features (same as zero bias since they're not filtered)
        for branch_name, value in self.event_raw_selector_data.items():
            hist_data[branch_name] = [value] if not isinstance(value, list) else value

        # Add aggregated features (apply filtering and sorting)
        for aggregator_idx, aggregator_data in enumerate(
            self.event_raw_aggregators_data
        ):
            # Skip aggregators that didn't pass filters
            if not self.event_aggregators_pass_filters[aggregator_idx]:
                continue

            # Get the filtered indices for this aggregator
            filtered_indices = self.event_aggregators_sort_filter_indices[
                aggregator_idx
            ]

            # Apply filtering to each branch in this aggregator
            for branch_name, values_list in aggregator_data.items():
                # Convert to numpy for indexing, then back to list
                values_array = np.array(values_list)
                filtered_values = values_array[filtered_indices].tolist()

                # Add to histogram data
                if branch_name in hist_data:
                    hist_data[branch_name].extend(filtered_values)
                else:
                    hist_data[branch_name] = filtered_values

        return hist_data

    def get_ML_training_data(self) -> dict[str, np.ndarray]:
        """
        Get data formatted for ML training (scalar + padded/truncated aggregated features).

        Returns:
            Dictionary with:
            - 'scalar_features': dict mapping branch names to scalar values
            - 'aggregated_features': dict mapping aggregator keys to padded numpy arrays
        """
        ml_data = {
            "scalar_features": self.event_raw_selector_data.copy(),
            "aggregated_features": {},
        }

        # Process each aggregator
        for aggregator_idx, aggregator_data in enumerate(
            self.event_raw_aggregators_data
        ):
            # Skip aggregators that didn't pass filters
            if not self.event_aggregators_pass_filters[aggregator_idx]:
                continue

            # Get the filtered indices and config for this aggregator
            filtered_indices = self.event_aggregators_sort_filter_indices[
                aggregator_idx
            ]
            aggregator_config = self.aggregator_configs[aggregator_idx]

            # Stack features for this aggregator
            feature_arrays = []
            for branch_name in sorted(
                aggregator_data.keys()
            ):  # Ensure consistent ordering
                values_list = aggregator_data[branch_name]
                values_array = np.array(values_list)

                # Apply filtering and sorting
                filtered_values = values_array[filtered_indices]

                # Reshape to ensure 2D (n_tracks, n_features_per_track)
                if filtered_values.ndim == 1:
                    filtered_values = filtered_values.reshape(-1, 1)

                feature_arrays.append(filtered_values)

            if feature_arrays:
                # Stack horizontally to get (n_tracks, total_features)
                stacked_features = np.hstack(feature_arrays)

                # Apply padding/truncation to match max_length
                max_length = aggregator_config.max_length
                n_tracks, n_features = stacked_features.shape

                if n_tracks > max_length:
                    # Truncate
                    final_array = stacked_features[:max_length, :]
                elif n_tracks < max_length:
                    # Pad with zeros
                    padding_shape = (max_length - n_tracks, n_features)
                    padding = np.zeros(padding_shape, dtype=stacked_features.dtype)
                    final_array = np.vstack([stacked_features, padding])
                else:
                    final_array = stacked_features

                # Store with aggregator key
                aggregator_key = f"aggregator_{aggregator_idx}"
                ml_data["aggregated_features"][aggregator_key] = final_array

        return ml_data

    def to_json_dict(self, data_type: str = "post_selection") -> dict[str, Any]:
        """
        Convert to JSON-serializable format for raw sample storage.

        Args:
            data_type: Either "post_selection" or "zero_bias" to determine which data to include

        Returns:
            JSON-serializable dictionary
        """

        def convert_numpy_types(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Get the appropriate histogram data
        if data_type == "zero_bias":
            hist_data = self.get_zero_bias_hist_data()
        else:
            hist_data = self.get_post_selection_hist_data()

        json_data = {
            "selection_config_name": self.selection_config_name,
            "histogram_data": hist_data,
            "aggregator_pass_filters": self.event_aggregators_pass_filters,
            "aggregator_filtered_counts": [
                self.get_aggregator_filtered_count(i)
                for i in range(len(self.event_raw_aggregators_data))
            ],
        }

        return convert_numpy_types(json_data)


class NewProcessedEvent:
    """
    New standardized result from processing a single ATLAS PhysLite event.

    This class provides a cleaner separation between raw data and transformed outputs,
    mirroring the TaskConfig structure directly.
    """

    def __init__(
        self,
        passed_filters: bool,
        input_selection_data: EventSelectionData,
        label_selection_data: list[EventSelectionData],
    ):
        """
        Initialize a processed event.

        Args:
            passed_filters: Whether the event passed all event-level filters
            input_selection_data: Processed input selection data
            label_selection_data: List of processed label selection data (one per label config)
        """
        self.passed_filters = passed_filters
        self.input_selection_data = input_selection_data
        self.label_selection_data = label_selection_data

    def get_num_tracks_for_plot(self) -> Optional[int]:
        """
        Get number of tracks for plotting from the first input aggregator.

        Returns:
            Number of tracks after filtering/truncation, or None if no aggregators
        """
        if not self.input_selection_data.event_raw_aggregators_data:
            return None

        # Get count from first aggregator that passed filters
        for aggregator_idx in range(
            len(self.input_selection_data.event_raw_aggregators_data)
        ):
            if self.input_selection_data.event_aggregators_pass_filters[aggregator_idx]:
                filtered_count = (
                    self.input_selection_data.get_aggregator_filtered_count(
                        aggregator_idx
                    )
                )

                # Apply max_length truncation for plotting
                aggregator_config = self.input_selection_data.aggregator_configs[
                    aggregator_idx
                ]
                max_length = aggregator_config.max_length

                return min(filtered_count, max_length)

        return None

    def get_dataset_features(self) -> dict[str, Any]:
        """
        Extract features in the format needed for HDF5 dataset storage.

        Returns:
            Dictionary with input features formatted for ML training
        """
        return self.input_selection_data.get_ML_training_data()

    def get_label_features(self) -> list[dict[str, Any]]:
        """
        Extract label features in the format needed for HDF5 dataset storage.

        Returns:
            List of label features (one dict per label config)
        """
        return [
            label_data.get_ML_training_data()
            for label_data in self.label_selection_data
        ]

    def get_histogram_data(self, data_type: str) -> dict[str, list[float]]:
        """
        Extract histogram data for plotting.

        Args:
            data_type: Either "post_selection" or "zero_bias"

        Returns:
            Dictionary mapping branch names to lists of values for histogramming
        """
        if data_type == "zero_bias":
            return self.input_selection_data.get_zero_bias_hist_data()
        else:
            return self.input_selection_data.get_post_selection_hist_data()

    def to_json_dict(self, data_type: str = "post_selection") -> dict[str, Any]:
        """
        Convert to JSON-serializable format for raw sample storage.

        Args:
            data_type: Either "post_selection" or "zero_bias"

        Returns:
            JSON-serializable dictionary
        """
        return {
            "passed_filters": self.passed_filters,
            "input_selection_data": self.input_selection_data.to_json_dict(data_type),
            "label_selection_data": [
                label_data.to_json_dict(data_type)
                for label_data in self.label_selection_data
            ],
            "num_tracks_for_plot": self.get_num_tracks_for_plot(),
        }
