"""
Utilities for accessing ATLAS data from the atlas_index.json file.

This module provides functions to load and access ATLAS run numbers,
catalog counts, and signal catalog information from a central JSON file.
"""

import json
import logging
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache for the loaded atlas index data
_atlas_index_data = None


def _load_atlas_index() -> dict:
    """Load the ATLAS index data from the JSON file."""
    global _atlas_index_data

    if _atlas_index_data is not None:
        return _atlas_index_data

    try:
        # Try to load from package resources first
        try:
            with (
                resources.files("hep_foundation.data")
                .joinpath("atlas_index.json")
                .open() as f
            ):
                _atlas_index_data = json.load(f)
        except (ImportError, FileNotFoundError):
            # Fallback to direct file path if package resources fail
            file_path = Path(__file__).parent / "atlas_index.json"
            with open(file_path) as f:
                _atlas_index_data = json.load(f)

        return _atlas_index_data
    except Exception as e:
        logger.error(f"Failed to load ATLAS index data: {e}")
        raise


def get_run_numbers() -> list[str]:
    """Get the list of ATLAS run numbers.

    Returns:
        list[str]: A list of ATLAS run numbers.
    """
    atlas_data = _load_atlas_index()
    # Get the keys from the catalog_counts dictionary as run numbers
    return list(atlas_data["catalog_counts"].keys())


def get_catalog_count(run_number: str) -> int:
    """Get the number of catalogs for a run number.

    Args:
        run_number (str): The ATLAS run number.

    Returns:
        int: The number of catalogs for the run.

    Raises:
        ValueError: If the run number is not found.
    """
    atlas_data = _load_atlas_index()
    if run_number not in atlas_data["catalog_counts"]:
        raise ValueError(f"No catalog count found for run number: {run_number}")
    return atlas_data["catalog_counts"][run_number]


def get_signal_catalog(signal_key: str) -> str:
    """Get the catalog file name for a signal key.

    Args:
        signal_key (str): The signal key (e.g., 'zprime_tt', 'wprime_qq').

    Returns:
        str: The catalog file name.

    Raises:
        ValueError: If the signal key is not found.
    """
    atlas_data = _load_atlas_index()
    if signal_key not in atlas_data["signal_catalogs"]:
        raise ValueError(f"No signal catalog found for key: {signal_key}")
    return atlas_data["signal_catalogs"][signal_key]


def get_signal_catalog_keys() -> list[str]:
    """Get all available signal catalog keys.

    Returns:
        list[str]: A list of signal catalog keys.
    """
    atlas_data = _load_atlas_index()
    return list(atlas_data["signal_catalogs"].keys())
