# This file was used to generate the physlite_branch_index.json file.
# However it is no longer needed as the data does not change.
# The script is kept here for reference.
# There may be some errors that need to get fixed

import json
from pathlib import Path
from typing import Any, Optional, Union

import uproot
import uproot.containers


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

                # --- DEBUG: Add verbose printing for specific category ---
                # debug_category = "InDetTrackParticlesAuxDyn"
                # do_debug_print = cat == debug_category
                # if do_debug_print:
                #     print(f"\\n---> Processing Branch: {full_branch}")
                # --- END DEBUG ---
                do_debug_print = False  # Disable debug prints

                try:
                    # Get the uproot branch object itself
                    uproot_branch = tree[full_branch]
                    root_typename = uproot_branch.typename
                    # is_likely_vector = "vector" in root_typename.lower()  # Unused variable

                    # if do_debug_print:
                    #     print(f"  ROOT Typename: {root_typename}")
                    #     print(f"  Is Likely Vector (based on typename): {is_likely_vector}")

                    # Read just this branch with a small number of entries
                    arrays = tree.arrays(
                        [full_branch], library="np", entry_stop=max_entries
                    )

                    if do_debug_print:
                        # Limit printing potentially large arrays
                        array_repr = str(
                            arrays[full_branch][:5]
                        )  # Print first 5 entries
                        if len(arrays[full_branch]) > 5:
                            array_repr += "... (truncated)"
                        print(
                            f"  Read Array Sample (first {min(5, max_entries)} entries): {array_repr}"
                        )
                        print(f"  Read Array Length: {len(arrays[full_branch])}")

                    if full_branch in arrays:
                        sample = arrays[full_branch]

                        if len(sample) > 0:
                            first_entry = sample[0]
                            shape = None
                            item_type = "unknown"
                            status = "success"  # Assume success unless changed

                            if do_debug_print:
                                print(
                                    f"  First Entry: {str(first_entry)[:100] + ('... (truncated)' if len(str(first_entry)) > 100 else '')}"
                                )  # Limit print length
                                print(f"  Type of First Entry: {type(first_entry)}")
                                has_shape_attr = hasattr(first_entry, "shape")
                                print(f"  Has 'shape' attribute: {has_shape_attr}")
                                if has_shape_attr:
                                    print(f"    first_entry.shape: {first_entry.shape}")
                                is_list_tuple = isinstance(first_entry, (list, tuple))
                                print(f"  Is list or tuple: {is_list_tuple}")

                            # --- New Shape and Type Inference Logic ---
                            if (
                                hasattr(first_entry, "shape")
                                and first_entry.shape != ()
                            ):
                                # Case 1: First entry is a NumPy array (e.g., from std::vector<float>, but also fixed-size arrays)
                                # Shape per object/track is first_entry.shape
                                # Overall shape is (-1,) + first_entry.shape
                                temp_shape = (-1,) + first_entry.shape
                                item_type = str(first_entry.dtype)

                                # ** Correction for simple vector<T> branches **
                                # If the ROOT type is like vector<T> (not nested) AND the inferred shape is (-1, k != 1),
                                # assume it's scalar-per-track and force shape to (-1, 1).
                                # This corrects for cases where the first event had k > 1 tracks.
                                is_simple_vector = (
                                    "vector<" in root_typename
                                    and "vector<vector" not in root_typename
                                )
                                if (
                                    is_simple_vector
                                    and len(temp_shape) == 2
                                    and temp_shape[1] != 1
                                ):
                                    # if do_debug_print:
                                    #     print(f"    Correcting shape for simple vector: {temp_shape} -> (-1, 1)")
                                    shape = (-1, 1)
                                else:
                                    shape = temp_shape  # Use originally inferred shape

                            elif isinstance(first_entry, (list, tuple)):
                                # Case 2: First entry is a Python sequence (less common with uproot np/stl, but possible)
                                if first_entry:
                                    k = len(first_entry)
                                    shape = (-1, k)
                                    # Infer type from the first element of the sequence
                                    element = first_entry[0]
                                    if hasattr(
                                        element, "dtype"
                                    ):  # Check if elements are numpy types
                                        item_type = str(element.dtype)
                                    else:
                                        item_type = type(element).__name__
                                else:
                                    # Sequence is empty, cannot determine k or element type fully
                                    shape = (-1, -1)  # Indicate unknown inner dimension
                                    item_type = "unknown_empty_sequence"
                                    # Might want to read more entries if this happens often

                            elif isinstance(first_entry, uproot.containers.STLVector):
                                # Case 3: First entry is an uproot STLVector (from nested vectors like vector<vector<float>>)
                                try:
                                    num_tracks = len(
                                        first_entry
                                    )  # Number of tracks in this event
                                    k = (
                                        -1
                                    )  # Features per track, default to -1 (unknown)
                                    shape = (-1, -1)  # Default shape
                                    item_type = "unknown_stl_vector_element"

                                    if num_tracks > 0:
                                        # Get the data for the FIRST track
                                        inner_vector = first_entry[0]
                                        try:
                                            # k is the length of the INNER vector (features per track)
                                            k = len(inner_vector)
                                            shape = (
                                                -1,
                                                k,
                                            )  # Set shape based on inner length

                                            # Try to determine item_type from first element of inner_vector
                                            if k > 0:
                                                inner_element = inner_vector[0]
                                                if isinstance(
                                                    inner_element,
                                                    uproot.containers.STLVector,
                                                ):
                                                    item_type = "deeply_nested_stl_vector"  # vector<vector<vector<T>>>?
                                                elif hasattr(inner_element, "dtype"):
                                                    item_type = str(inner_element.dtype)
                                                else:
                                                    item_type = type(
                                                        inner_element
                                                    ).__name__
                                            else:
                                                item_type = (
                                                    "unknown_empty_inner_stl_vector"
                                                )

                                        except Exception:
                                            k = -1  # Mark inner dimension as unknown on error
                                            shape = (-1, k)
                                            item_type = "unknown_error_inner_stl_vector"
                                            status = "error_inner_stl_processing"
                                    else:
                                        # num_tracks is 0 for this event
                                        shape = (-1, 0)  # Shape is Nx0
                                        item_type = "unknown_empty_outer_stl_vector"

                                except Exception:
                                    # if do_debug_print:
                                    #     print(f"    Error processing outer STLVector: {e_outer}")
                                    shape = (
                                        -1,
                                        -1,
                                    )  # Indicate unknown inner dimension on error
                                    item_type = "unknown_stl_vector_error"
                                    status = "error_outer_stl_processing"

                            else:
                                # Case 4: First entry is a scalar, and not clearly per-object (assume event-level)
                                shape = ()
                                item_type = type(first_entry).__name__
                                if item_type == "float":
                                    item_type = "float64"
                                if item_type == "int":
                                    item_type = "int64"

                            if do_debug_print:
                                print(f"  Determined Shape: {shape}")
                                print(f"  Determined Dtype: {item_type}")
                                print(f"  Final Status: {status}")

                            # Record shape information
                            branch_info[cat][feat]["shape"] = shape
                            branch_info[cat][feat]["dtype"] = item_type
                            branch_info[cat][feat]["status"] = (
                                status  # Use status determined above
                            )

                            # --- Debug Fallback Logic Print ---
                            original_item_type = item_type
                            # --- End Debug Fallback Logic Print ---

                            # Fallback for type using ROOT typename if primary methods fail
                            if item_type.startswith("unknown"):
                                if "<" in root_typename and ">" in root_typename:
                                    parsed_type = root_typename.split("<")[1].split(
                                        ">"
                                    )[0]
                                    # Basic cleanup for common types
                                    if parsed_type.endswith("_t"):
                                        parsed_type = parsed_type[:-2]
                                    if parsed_type == "double":
                                        parsed_type = "float64"
                                    elif parsed_type == "float":
                                        parsed_type = "float32"
                                    elif parsed_type == "unsigned int":
                                        parsed_type = "uint32"  # Example
                                    elif parsed_type == "int":
                                        parsed_type = "int32"  # Example
                                    # Add more mappings as needed
                                    item_type = parsed_type
                                else:
                                    # If it's not a template, it might be a simple type name itself
                                    if root_typename == "double":
                                        item_type = "float64"
                                    elif root_typename == "float":
                                        item_type = "float32"
                                    elif root_typename == "int":
                                        item_type = "int32"
                                    # Add more ROOT primitive types

                            # --- Debug Fallback Logic Print ---
                            if do_debug_print and item_type != original_item_type:
                                print(
                                    f"  Fallback Dtype Applied: {item_type} (was {original_item_type})"
                                )
                            # --- End Debug Fallback Logic Print ---

                            # Update dict if fallback changed dtype
                            branch_info[cat][feat]["dtype"] = item_type

                            # --- End New Logic ---
                        else:
                            branch_info[cat][feat]["status"] = "empty"
                            branch_info[cat][feat]["shape"] = None  # No shape if empty
                            branch_info[cat][feat]["dtype"] = None  # No dtype if empty
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
            from hep_foundation.data.atlas_file_manager import ATLASFileManager
        except ImportError:
            # Simple fallback if running script directly from scripts/
            import sys

            script_dir = Path(__file__).parent.parent
            sys.path.append(str(script_dir))
            from hep_foundation.data.atlas_file_manager import ATLASFileManager

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

        # Get detailed information for ALL branches (or just InDetTrackParticlesAuxDyn in debug mode)
        branch_info = get_branch_info(catalog_path)

        # --- DEBUG: Print InDetTrackParticlesAuxDyn info and exit ---
        # print("\\n\\n--- Collected Branch Info for InDetTrackParticlesAuxDyn ---")
        # if "InDetTrackParticlesAuxDyn" in branch_info:
        #     import pprint
        #     pprint.pprint(branch_info["InDetTrackParticlesAuxDyn"])
        # else:
        #     print("No info collected for InDetTrackParticlesAuxDyn.")
        # print("--- End Debug Output ---")
        # return None # Skip saving file in debug mode
        # --- END DEBUG ---

        # Prepare output file path
        if output_dir is None:
            # Default to src/hep_foundation relative to project root
            # Assumes script is run from project root or hep_foundation/scripts
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "src" / "hep_foundation"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
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
                    "shape": list(info.get("shape"))
                    if info.get("shape") is not None
                    else None,
                    "dtype": info.get("dtype", "unknown"),
                    "status": info.get("status", "unknown"),
                }

        # Save as a JSON file
        print(f"Saving branch information dictionary to {output_path}...")
        metadata = {
            "generation_info": {
                "run_number": run_number,
                "catalog_index": catalog_index,
                "success_rate": f"{successful_branches}/{total_branches} ({success_rate:.1%})",
            },
            "physlite_branches": serializable_branch_info,
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=4)  # Use indent for readability

        print(f"Branch information dictionary saved to {output_path}")
        print(
            f"Success rate: {successful_branches}/{total_branches} branches ({success_rate:.1%})"
        )
        return output_path

    except Exception as e:
        print(f"Error in save_branch_dictionary: {str(e)}")
        import traceback

        traceback.print_exc()  # Print full traceback for debugging
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
