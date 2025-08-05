"""
Centralized histogram data management for coordinating distributions across
runs, pipelines, and different data sources.

This module provides functions to manage histogram binning, accumulation,
and normalization with coordinated bin edges for direct comparison.
"""

import json
import logging
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cache for the loaded physlite percentiles data
_physlite_percentiles_data = None


class HistogramManager:
    """Manages histogram data with coordinated bin edges and persistent percentile tracking."""

    def __init__(self):
        """Initialize the histogram manager and load physlite percentiles index."""
        self.logger = logging.getLogger(__name__)
        self._file_percentiles_data = self._load_physlite_percentiles()
        self._session_cache = {}  # Session-level cache for percentiles

    def _load_physlite_percentiles(self) -> dict:
        """Load the physlite percentiles data from the JSON file, creating if not exists."""
        global _physlite_percentiles_data

        if _physlite_percentiles_data is not None:
            return _physlite_percentiles_data

        try:
            with resources.open_text(
                "hep_foundation.data", "physlite_percentiles.json"
            ) as f:
                _physlite_percentiles_data = json.load(f)
        except FileNotFoundError:
            # Create empty structure if file doesn't exist
            self.logger.info("Creating new physlite percentiles data structure")
            _physlite_percentiles_data = {}
            self._save_physlite_percentiles(_physlite_percentiles_data)
        except Exception as e:
            self.logger.error(f"Failed to load physlite percentiles data: {e}")
            raise

        return _physlite_percentiles_data

    def _load_current_file_percentiles(self) -> dict:
        """Load current percentiles data directly from file (bypassing cache)."""
        file_path = Path(__file__).parent.parent / "data" / "physlite_percentiles.json"

        try:
            if file_path.exists():
                with open(file_path) as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load current percentiles from file: {e}")
            return {}

    def _save_physlite_percentiles(self, data: dict) -> None:
        """Save the physlite percentiles data to the JSON file."""
        file_path = Path(__file__).parent.parent / "data" / "physlite_percentiles.json"

        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Saved physlite percentiles to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save physlite percentiles data: {e}")
            raise

    def save_to_hist_file(
        self,
        data: dict[str, list[float]],
        file_path: Path,
        nbins: int = 50,
        use_percentile_file: bool = False,
        update_percentile_file: bool = False,
        use_percentile_cache: bool = True,
    ) -> None:
        """Save histogram data to JSON file with optional percentile coordination.

        Args:
            data: Dictionary mapping feature names to lists of values
            file_path: Path where to save the JSON histogram data
            nbins: Number of bins for histograms
            use_percentile_file: Whether to initialize cache from file-stored percentiles
            update_percentile_file: Whether to update file when percentiles exceed range
            use_percentile_cache: Whether to use cached percentiles for bin coordination
        """
        hist_data = {}

        for key, values in data.items():
            if not values:
                continue

            values_array = np.array(values)

            # Compute current data percentiles
            current_p01 = np.percentile(values_array, 0.1)
            current_p999 = np.percentile(values_array, 99.9)

            # Determine bin edges based on the new parameter logic
            if use_percentile_cache:
                # Check session cache first
                cached_percentiles = self._session_cache.get(key)

                if cached_percentiles:
                    # Use cached percentiles
                    bin_min = cached_percentiles["0.1"]
                    bin_max = cached_percentiles["99.9"]

                elif use_percentile_file:
                    # Try to get from file percentiles
                    file_percentiles = self._file_percentiles_data.get(key)
                    if file_percentiles:
                        bin_min = file_percentiles["0.1"]
                        bin_max = file_percentiles["99.9"]

                        # Store in session cache for future use
                        self._session_cache[key] = {
                            "0.1": bin_min,
                            "99.9": bin_max,
                            "timestamp": datetime.now().isoformat(),
                        }

                    else:
                        # No file percentiles, use current data and cache it
                        bin_min = current_p01
                        bin_max = current_p999

                        self._session_cache[key] = {
                            "0.1": bin_min,
                            "99.9": bin_max,
                            "timestamp": datetime.now().isoformat(),
                        }

                else:
                    # Not using file, calculate from current data and cache it
                    bin_min = current_p01
                    bin_max = current_p999

                    self._session_cache[key] = {
                        "0.1": bin_min,
                        "99.9": bin_max,
                        "timestamp": datetime.now().isoformat(),
                    }

                # Calculate what percentage of data falls outside range
                below_min = np.sum(values_array < bin_min) / len(values_array) * 100
                above_max = np.sum(values_array > bin_max) / len(values_array) * 100

                # Update file if requested and current data exceeds range
                if update_percentile_file and (
                    current_p01 < bin_min or current_p999 > bin_max
                ):
                    new_min = min(current_p01, bin_min)
                    new_max = max(current_p999, bin_max)

                    # Update file (but not session cache to avoid invalidating current session)
                    # Reload current file contents to avoid race conditions
                    current_file_percentiles = self._load_current_file_percentiles()
                    current_file_percentiles[key] = {
                        "0.1": new_min,
                        "99.9": new_max,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Save to disk and update global cache
                    self._save_physlite_percentiles(current_file_percentiles)

                    self.logger.info(
                        f"Updated FILE percentiles for {key}: "
                        f"0.1%={new_min:.3f}, 99.9%={new_max:.3f}"
                    )

                self.logger.info(
                    f"For {key} using cached percentiles 0.1: {bin_min:.3f}, "
                    f"99.9: {bin_max:.3f}, {below_min:.1f}% of data is below 0.1% "
                    f"and {above_max:.1f}% of data is above 99.9%"
                )
            else:
                # Not using cache at all, calculate fresh each time
                bin_min = current_p01
                bin_max = current_p999

            # Create histogram
            bin_edges = np.linspace(bin_min, bin_max, nbins + 1)
            counts, _ = np.histogram(values_array, bins=bin_edges, density=True)

            # Store histogram data
            hist_data[key] = {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            }

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        with open(file_path, "w") as f:
            json.dump(hist_data, f, indent=4)

        self.logger.info(f"Saved histogram data to {file_path}")

    def load_hist_file(
        self,
        file_path: Path,
        normalized: bool = False,
    ) -> tuple[dict[str, dict[str, list[float]]], Optional[dict]]:
        """Load histogram data from JSON file with optional normalization.

        Args:
            file_path: Path to the JSON histogram data file
            normalized: Whether to normalize the histogram counts

        Returns:
            Tuple containing:
            - Dictionary mapping feature names to histogram data (counts, bin_edges)
            - Metadata dictionary (if present) or None
        """
        try:
            with open(file_path) as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load histogram data from {file_path}: {e}")
            raise

        # Extract metadata if present
        metadata = data.get("_metadata", None)

        # Extract histogram data (exclude metadata keys)
        hist_data = {}
        for key, value in data.items():
            if key.startswith("_"):
                continue

            if isinstance(value, dict) and "counts" in value and "bin_edges" in value:
                if normalized:
                    # Normalize counts to ensure integral equals 1
                    counts = np.array(value["counts"])
                    bin_edges = np.array(value["bin_edges"])
                    bin_widths = bin_edges[1:] - bin_edges[:-1]

                    # Normalize so that integral = sum(counts * bin_widths) = 1
                    integral = np.sum(counts * bin_widths)
                    if integral > 0:
                        counts = counts / integral

                    hist_data[key] = {
                        "counts": counts.tolist(),
                        "bin_edges": value["bin_edges"],
                    }
                else:
                    hist_data[key] = value

        return hist_data, metadata
