from pathlib import Path
from typing import Any, Optional, Union
import json

import uproot


def get_branch_info(
    file_path: Union[str, Path], max_entries: int = 100
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Extract branch names, shapes, and types from a PhysLite ROOT file.

    Args:
        file_path: Path to ROOT file
        max_entries: Maximum number of entries to read for shape inference

    Returns:
        Dictionary of branch information organized by category
    """
    print(f"Reading branch information from {file_path}")
    branch_info = {}

    with uproot.open(file_path) as file:
        tree = file["CollectionTree;1"]
        all_branch_names = tree.keys()

        # Initialize categories and branches
        for branch in all_branch_names:
            # Skip internal uproot keys
            if branch.startswith("_"):
                continue

            # Extract category (prefix before the first dot)
            if "." in branch:
                category, feature = branch.split(".", 1)
                if category not in branch_info:
                    branch_info[category] = {}
                branch_info[category][feature] = {
                    "shape": None,
                    "dtype": None,
                    "status": "unread",
                }
            else:
                # Handle branches without dots
                if "Other" not in branch_info:
                    branch_info["Other"] = {}
                branch_info["Other"][branch] = {
                    "shape": None,
                    "dtype": None,
                    "status": "unread",
                }

        # Process branches in batches by category to handle errors gracefully
        for category, features in branch_info.items():
            print(f"Processing category: {category} with {len(features)} branches")

            # For 'Other' category, we use branch names directly
            if category == "Other":
                branch_list = list(features.keys())
            else:
                # For normal categories, we prefix with category name
                branch_list = [f"{category}.{feature}" for feature in features]

            # Try to read each branch individually
            for full_branch in branch_list:
                if "." in full_branch:
                    cat, feat = full_branch.split(".", 1)
                else:
                    cat, feat = "Other", full_branch

                try:
                    # Read just this branch with a small number of entries
                    arrays = tree.arrays(
                        [full_branch], library="np", entry_stop=max_entries
                    )

                    if full_branch in arrays:
                        # Get array for first entry to determine shape and type
                        sample = arrays[full_branch]

                        # Check if we have at least one entry
                        if len(sample) > 0:
                            first_entry = sample[0]

                            # Determine shape
                            if hasattr(first_entry, "shape"):
                                # For numpy arrays
                                shape = first_entry.shape
                                item_type = str(first_entry.dtype)
                            elif isinstance(first_entry, (list, tuple)):
                                # For Python sequences
                                shape = (len(first_entry),)
                                item_type = (
                                    type(first_entry[0]).__name__
                                    if first_entry
                                    else "unknown"
                                )
                            else:
                                # For scalars
                                shape = ()
                                item_type = type(first_entry).__name__

                            # Record shape information
                            branch_info[cat][feat]["shape"] = shape
                            branch_info[cat][feat]["dtype"] = item_type
                            branch_info[cat][feat]["status"] = "success"
                        else:
                            branch_info[cat][feat]["status"] = "empty"
                            print(f"Branch {full_branch} has no entries")

                except Exception as e:
                    branch_info[cat][feat]["status"] = f"error: {str(e)}"
                    print(f"Error reading branch {full_branch}: {str(e)}")
                    continue

        # Calculate success rate
        total_branches = sum(len(features) for features in branch_info.values())
        successful_branches = sum(
            1
            for cat in branch_info
            for feat in branch_info[cat]
            if branch_info[cat][feat]["status"] == "success"
        )
        success_rate = successful_branches / total_branches if total_branches > 0 else 0
        print(
            f"Branch info extraction success rate: {success_rate:.1%} ({successful_branches}/{total_branches})"
        )

        return branch_info


def analyze_branch_sample(
    file_path: Union[str, Path], category: str, feature: str, max_entries: int = 10
) -> dict[str, Any]:
    """
    Analyze a sample of values from a specific branch to get detailed information.

    Args:
        file_path: Path to ROOT file
        category: Branch category
        feature: Feature name
        max_entries: Maximum number of entries to examine

    Returns:
        Dictionary with detailed branch information
    """
    full_branch = f"{category}.{feature}" if category != "Other" else feature

    info = {
        "name": full_branch,
        "shape": None,
        "dtype": None,
        "sample_values": [],
        "min_length": None,
        "max_length": None,
        "has_nulls": False,
        "status": "unread",
    }

    try:
        with uproot.open(file_path) as file:
            tree = file["CollectionTree;1"]
            arrays = tree.arrays([full_branch], library="np", entry_stop=max_entries)

            if full_branch in arrays:
                samples = arrays[full_branch]

                if len(samples) > 0:
                    # Collect sample information
                    lengths = []

                    for i, sample in enumerate(samples):
                        if i >= max_entries:
                            break

                        # Track array lengths
                        if hasattr(sample, "shape"):
                            lengths.append(
                                sample.shape[0] if len(sample.shape) > 0 else 1
                            )
                        elif isinstance(sample, (list, tuple)):
                            lengths.append(len(sample))
                        else:
                            lengths.append(1)

                        # Track sample values (limited)
                        if i < 3:  # Only store a few samples
                            info["sample_values"].append(
                                str(sample)[:100] + "..."
                                if len(str(sample)) > 100
                                else str(sample)
                            )

                    # First entry for shape and type
                    first_entry = samples[0]

                    # Determine shape and type
                    if hasattr(first_entry, "shape"):
                        info["shape"] = first_entry.shape
                        info["dtype"] = str(first_entry.dtype)
                    elif isinstance(first_entry, (list, tuple)):
                        info["shape"] = (len(first_entry),)
                        info["dtype"] = (
                            type(first_entry[0]).__name__ if first_entry else "unknown"
                        )
                    else:
                        info["shape"] = ()
                        info["dtype"] = type(first_entry).__name__

                    # Calculate length stats
                    if lengths:
                        info["min_length"] = min(lengths)
                        info["max_length"] = max(lengths)

                    info["status"] = "success"
                else:
                    info["status"] = "empty"

    except Exception as e:
        info["status"] = f"error: {str(e)}"

    return info


def save_branch_dictionary(
    run_number: str,
    catalog_index: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
    download_if_missing: bool = True,
) -> Optional[Path]:
    """
    Save the branch dictionary with shape information for a specific ATLAS run
    and catalog to a JSON file within the specified output directory.

    Args:
        run_number: ATLAS run number
        catalog_index: Catalog index to use
        output_dir: Path where to save the output JSON file
                    (default: src/hep_foundation/)
        download_if_missing: Whether to download the catalog if it doesn't exist

    Returns:
        Path to the saved JSON file or None if failed
    """
    try:
        # This is a simple way to get access to the atlas_manager
        # Ensure this import works relative to where the script is run
        # Or adjust sys.path if needed when running the script
        try:
            from hep_foundation.atlas_file_manager import ATLASFileManager
        except ImportError:
            # Simple fallback if running script directly from scripts/
            import sys
            script_dir = Path(__file__).parent.parent
            sys.path.append(str(script_dir))
            from hep_foundation.atlas_file_manager import ATLASFileManager


        atlas_manager = ATLASFileManager()
        catalog_path = atlas_manager.get_run_catalog_path(run_number, catalog_index)

        if not catalog_path.exists() and download_if_missing:
            catalog_path = atlas_manager.download_run_catalog(run_number, catalog_index)

        if not catalog_path or not catalog_path.exists():
            print(
                f"Could not find or download catalog {catalog_index} for run {run_number}"
            )
            return None

        print(f"Extracting branch information from {catalog_path}...")

        # Get detailed information for ALL branches
        branch_info = get_branch_info(catalog_path)

        # Prepare output file path
        if output_dir is None:
            # Default to src/hep_foundation relative to project root
            # Assumes script is run from project root or hep_foundation/scripts
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "src" / "hep_foundation"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        output_path = output_dir / "physlite_branch_index.json"


        # Count success rate (based on successful shape/dtype extraction)
        total_branches = sum(len(features) for features in branch_info.values())
        successful_branches = sum(
            1
            for cat in branch_info
            for feat in branch_info[cat]
            if branch_info[cat][feat]["status"] == "success"
        )
        success_rate = successful_branches / total_branches if total_branches > 0 else 0

        # Prepare data for JSON serialization (ensure basic types)
        serializable_branch_info = {}
        for category, features in branch_info.items():
            serializable_branch_info[category] = {}
            for feature, info in features.items():
                serializable_branch_info[category][feature] = {
                    # Convert shape tuple to list for JSON
                    'shape': list(info.get("shape")) if info.get("shape") is not None else None,
                    'dtype': info.get("dtype", "unknown"),
                    'status': info.get("status", "unknown")
                }


        # Save as a JSON file
        print(f"Saving branch information dictionary to {output_path}...")
        metadata = {
            "generation_info": {
                "run_number": run_number,
                "catalog_index": catalog_index,
                "success_rate": f"{successful_branches}/{total_branches} ({success_rate:.1%})",
            },
            "physlite_branches": serializable_branch_info
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=4) # Use indent for readability

        print(f"Branch information dictionary saved to {output_path}")
        print(
            f"Success rate: {successful_branches}/{total_branches} branches ({success_rate:.1%})"
        )
        return output_path

    except Exception as e:
        print(f"Error in save_branch_dictionary: {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None


if __name__ == "__main__":
    # Example usage: Save to default location (src/hep_foundation/)
    saved_path = save_branch_dictionary("00311481", 1)

    # Example: Save to a specific directory
    # script_dir = Path(__file__).parent
    # saved_path = save_branch_dictionary("00311481", 1, output_dir=script_dir / "output_data")

    if saved_path:
        print(f"Successfully generated: {saved_path}")
    else:
        print("Failed to generate branch index.")
