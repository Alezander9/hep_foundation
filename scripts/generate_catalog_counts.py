# This file was used to generate the atlas_index.json file. 
# However it is no longer needed as the data does not change.
# The script is kept here for reference.

# import json
# import time
# from pathlib import Path
# from typing import Any

# import requests

# from hep_foundation.data.atlas_data import get_run_numbers


# def check_rate_limiting(run_number: str) -> bool:
#     """
#     Check if we're being rate limited by testing catalog_0 which should always exist
#     If rate limiting is detected, pause for 10 minutes.

#     Args:
#         run_number: ATLAS run number

#     Returns:
#         True if we're being rate limited, False otherwise
#     """
#     padded_run = run_number.zfill(8)
#     url = f"/record/80001/files/data16_13TeV_Run_{padded_run}_file_index.json_0"
#     response = requests.head(f"https://opendata.cern.ch{url}")

#     if response.status_code == 502:
#         print(f"Detected rate limiting: catalog_0 returned 502 for run {run_number}")
#         print("Pausing for 10 minutes to reset rate limiting...")
#         time.sleep(600)  # 10 minutes in seconds
#         return True
#     return False


# def make_request(
#     url: str, run_number: str, max_retries: int = 3, initial_delay: float = 1.0
# ) -> requests.Response:
#     """
#     Make a request with retries and rate limit checking

#     Args:
#         url: The URL to request
#         run_number: ATLAS run number (for rate limit checking)
#         max_retries: Maximum number of retry attempts
#         initial_delay: Initial delay between retries in seconds

#     Returns:
#         Response object
#     """
#     delay = initial_delay
#     for attempt in range(max_retries):
#         response = requests.head(f"https://opendata.cern.ch{url}")

#         if response.status_code == 502:
#             # Check if we're being rate limited
#             if check_rate_limiting(run_number):
#                 print(f"Rate limited, waiting {delay} seconds...")
#                 time.sleep(delay)
#                 delay *= 2
#                 continue
#             else:
#                 # This is a "real" 502, indicating non-existent file
#                 time.sleep(0.2)  # Small delay between requests
#                 return response

#         time.sleep(0.2)  # Small delay between all requests
#         return response

#     raise Exception(f"Failed after {max_retries} retries")


# def validate_catalog_count(run_number: str, count: int) -> bool:
#     """
#     Validate that a catalog count is correct by checking the boundary condition

#     Args:
#         run_number: ATLAS run number
#         count: Number of catalogs to validate

#     Returns:
#         True if validation passes, False otherwise
#     """
#     # First verify we're not being rate limited
#     if check_rate_limiting(run_number):
#         raise Exception("Rate limited during validation")

#     if count == 0:
#         # This should never happen since catalog_0 should always exist
#         return False

#     # Check that count-1 exists and count doesn't
#     padded_run = run_number.zfill(8)
#     url_exists = (
#         f"/record/80001/files/data16_13TeV_Run_{padded_run}_file_index.json_{count - 1}"
#     )
#     url_not_exists = (
#         f"/record/80001/files/data16_13TeV_Run_{padded_run}_file_index.json_{count}"
#     )

#     response_exists = make_request(url_exists, run_number)
#     response_not_exists = make_request(url_not_exists, run_number)

#     # Verify one last time we're not rate limited
#     if check_rate_limiting(run_number):
#         raise Exception("Rate limited at end of validation")

#     return response_exists.status_code == 200 and response_not_exists.status_code != 200


# def binary_search_catalog_count(run_number: str) -> int:
#     """
#     Binary search to find the last valid catalog index

#     Args:
#         run_number: ATLAS run number

#     Returns:
#         Number of available catalog files
#     """
#     padded_run = run_number.zfill(8)
#     left, right = 0, 1024  # Assuming max 1024 catalogs

#     # First verify we're not being rate limited
#     if check_rate_limiting(run_number):
#         raise Exception("Rate limited at start of binary search")

#     # Binary search for last valid index
#     last_valid = -1  # Start at -1 since we know catalog_0 should exist
#     while left <= right:
#         mid = (left + right) // 2
#         url = f"/record/80001/files/data16_13TeV_Run_{padded_run}_file_index.json_{mid}"
#         response = make_request(url, run_number)
#         print(f"Run {run_number} index {mid}: status {response.status_code}")

#         if response.status_code == 200:
#             last_valid = mid
#             left = mid + 1
#         else:
#             right = mid - 1

#     # Verify the result by checking catalog_0 again
#     if check_rate_limiting(run_number):
#         raise Exception("Rate limited at end of binary search")

#     return last_valid + 1  # Add 1 since we want the count


# def verify_catalog_counts(
#     catalog_data: dict[str, dict[str, Any]],
# ) -> dict[str, dict[str, Any]]:
#     """
#     Verify all catalog counts and update verification statistics

#     Args:
#         catalog_data: Dictionary containing run numbers and their data

#     Returns:
#         Updated catalog data dictionary
#     """
#     print("\nVerifying catalog counts...")

#     for run_number, data in list(catalog_data.items()):
#         count = data["count"]
#         verifications = data.get("verifications", 0)
#         failures = data.get("failures", 0)

#         try:
#             print(f"Verifying run {run_number} (count: {count})...")
#             if validate_catalog_count(run_number, count):
#                 verifications += 1
#                 print(
#                     f"Verification succeeded (total successes: {verifications}, failures: {failures})"
#                 )
#             else:
#                 failures += 1
#                 print(
#                     f"Verification failed (total successes: {verifications}, failures: {failures})"
#                 )

#             # Update the data
#             catalog_data[run_number] = {
#                 "count": count,
#                 "verifications": verifications,
#                 "failures": failures,
#             }

#             # If more failures than successes, delete the entry
#             if failures > verifications:
#                 print(
#                     f"Run {run_number} has more failures than successes, marking for recalculation"
#                 )
#                 del catalog_data[run_number]

#         except Exception as e:
#             print(f"Error verifying run {run_number}: {e}")
#             failures += 1
#             catalog_data[run_number]["failures"] = failures

#     return catalog_data


# def generate_catalog_counts():
#     """Generate a dictionary of run numbers to catalog counts"""
#     catalog_data = {}
#     output_path = Path("atlas_catalog_counts.json")

#     # Load existing progress if any
#     if output_path.exists():
#         with open(output_path) as f:
#             catalog_data = json.load(f)
#         print(f"Loaded {len(catalog_data)} existing entries")

#     print("\nQuerying catalog counts for all run numbers...")

#     # Get run numbers from the new atlas_data module
#     run_numbers = get_run_numbers()

#     for run_number in run_numbers:
#         if run_number in catalog_data:
#             print(f"Skipping run {run_number} (already processed)")
#             continue

#         try:
#             print(f"Checking run {run_number}...")
#             count = binary_search_catalog_count(run_number)
#             print(f"Run {run_number}: Found {count} catalogs")

#             # Initialize entry with verification data
#             catalog_data[run_number] = {
#                 "count": count,
#                 "verifications": 0,
#                 "failures": 0,
#             }

#             # Verify the count immediately
#             if validate_catalog_count(run_number, count):
#                 catalog_data[run_number]["verifications"] = 1
#                 print("Initial verification succeeded")
#             else:
#                 catalog_data[run_number]["failures"] = 1
#                 print("Initial verification failed")

#             # Save progress after each run
#             with open(output_path, "w") as f:
#                 json.dump(catalog_data, f, indent=4)
#             print(f"Saved progress to: {output_path}")

#         except Exception as e:
#             print(f"Error processing run {run_number}: {e}")
#             # Save progress even if there was an error
#             with open(output_path, "w") as f:
#                 json.dump(catalog_data, f, indent=4)

#     # Print final dictionary format
#     print("\nFinal catalog counts:")
#     for run_number, data in sorted(catalog_data.items()):
#         print(f"    '{run_number}': {data},")


# if __name__ == "__main__":
#     generate_catalog_counts()

#     # After generating, verify all counts
#     # print("\nVerifying all catalog counts...")
#     # with open("atlas_catalog_counts.json", 'r') as f:
#     #     catalog_data = json.load(f)

#     # updated_data = verify_catalog_counts(catalog_data)

#     # # Save verified data
#     # with open("atlas_catalog_counts.json", 'w') as f:
#     #     json.dump(updated_data, f, indent=4)

#     # print("\nVerification complete. Results saved.")
