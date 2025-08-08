"""
Data structures for processed ATLAS PhysLite events.

This module defines the standardized output format for the event processing pipeline,
providing clear types and helper methods for different downstream use cases.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class AggregatedFeatureData:
    """
    Container for aggregated feature data from a single aggregator.

    Attributes:
        array: (max_length, num_features) array for model training, or None if min_length failed
        n_valid_elements: Number of valid elements before padding/truncation
        post_selection_hist_data: Histogram data after applying filters (for post-selection plots)
        zero_bias_hist_data: Histogram data without filters (for zero-bias plots)
    """

    array: Optional[np.ndarray]
    n_valid_elements: int
    post_selection_hist_data: Optional[dict[str, list]] = None
    zero_bias_hist_data: Optional[dict[str, list]] = None


@dataclass
class ProcessedEvent:
    """
    Standardized result from processing a single ATLAS PhysLite event.

    This class encapsulates all the data produced by the event processing pipeline,
    providing clear access patterns for different downstream consumers.

    Attributes:
        scalar_features: Dictionary mapping branch names to scalar values
        aggregated_features: Dictionary mapping aggregator keys to AggregatedFeatureData
        label_features: List of label configurations (for supervised tasks)
        num_tracks_for_plot: Number of tracks for plotting (respects max_length truncation)
        passed_filters: Whether the event passed all event-level filters
    """

    scalar_features: dict[str, Any]
    aggregated_features: dict[str, AggregatedFeatureData]
    label_features: list[dict[str, Any]]
    num_tracks_for_plot: Optional[int]
    passed_filters: bool

    def get_dataset_features(self) -> dict[str, Any]:
        """
        Extract features in the format needed for HDF5 dataset storage.

        Returns:
            Dictionary with scalar_features and aggregated_features (arrays only)
        """
        return {
            "scalar_features": self.scalar_features,
            "aggregated_features": {
                agg_key: agg_data.array
                for agg_key, agg_data in self.aggregated_features.items()
                if agg_data.array is not None
            },
        }

    # def get_histogram_data(self, data_type: str) -> Dict[str, List]:
    #     """
    #     Extract histogram data for plotting.

    #     Args:
    #         data_type: Either "post_selection" or "zero_bias"

    #     Returns:
    #         Dictionary mapping branch names to lists of values for histogramming
    #     """
    #     hist_data = {}

    #     # Add scalar features (they go to both types)
    #     for name, value in self.scalar_features.items():
    #         hist_data[name] = [value] if not isinstance(value, list) else value

    #     # Add aggregated features based on data type
    #     for agg_key, agg_data in self.aggregated_features.items():
    #         if data_type == "post_selection" and agg_data.post_selection_hist_data:
    #             hist_data.update(agg_data.post_selection_hist_data)
    #         elif data_type == "zero_bias" and agg_data.zero_bias_hist_data:
    #             hist_data.update(agg_data.zero_bias_hist_data)

    #     return hist_data

    # def has_valid_training_data(self) -> bool:
    #     """
    #     Check if this event has valid data for model training.

    #     Returns:
    #         True if event has at least one aggregator with a valid array
    #     """
    #     return any(
    #         agg_data.array is not None for agg_data in self.aggregated_features.values()
    #     )

    # def get_track_count_for_aggregator(self, aggregator_key: str) -> int:
    #     """
    #     Get the number of valid tracks for a specific aggregator.

    #     Args:
    #         aggregator_key: Key identifying the aggregator (e.g., "aggregator_0")

    #     Returns:
    #         Number of valid tracks, or 0 if aggregator not found
    #     """
    #     agg_data = self.aggregated_features.get(aggregator_key)
    #     return agg_data.n_valid_elements if agg_data else 0
