import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.atlas_data import (
    get_catalog_count,
    get_signal_catalog,
    get_signal_catalog_keys,
)


class ATLASFileManager:
    """Manages ATLAS PHYSLITE data access with parallel download capabilities"""

    # Add version as a class attribute
    VERSION = "1.0.1"  # Incremented for parallel download feature

    def __init__(self, base_dir: str = "atlas_data", max_workers: int = 4):
        # Setup logging
        self.logger = get_logger(__name__)

        self.base_dir = Path(base_dir)
        self.max_workers = max_workers
        self._setup_directories()

        # Signal types mapping - get from atlas_data module
        self.signal_types = {
            key: get_signal_catalog(key) for key in get_signal_catalog_keys()
        }

        self.logger.info(
            f"ATLASFileManager initialized with {max_workers} parallel workers"
        )

    def get_version(self) -> str:
        """Return the version of the ATLASFileManager"""
        return self.VERSION

    def get_catalog_count(self, run_number: str) -> int:
        """
        Discover how many catalog files exist for a run by probing the server

        Args:
            run_number: ATLAS run number

        Returns:
            Number of available catalog files
        """
        return get_catalog_count(run_number)

    def download_run_catalog(self, run_number: str, index: int = 0) -> Optional[Path]:
        """
        Download a specific run catalog file.

        Args:
            run_number: ATLAS run number
            index: Catalog index

        Returns:
            Path to the downloaded catalog file or None if file doesn't exist
        """
        padded_run = run_number.zfill(8)
        url = (
            f"/record/80001/files/data16_13TeV_Run_{padded_run}_file_index.json_{index}"
        )
        output_path = (
            self.base_dir / "catalogs" / f"Run_{run_number}_catalog_{index}.root"
        )

        try:
            if self._download_file(
                url, output_path, f"Downloading catalog {index} for Run {run_number}"
            ):
                return output_path
        except Exception as e:
            self.logger.error(
                f"Failed to download catalog {index} for run {run_number}: {str(e)}"
            )
            if output_path.exists():
                output_path.unlink()  # Clean up partial download
            return None

    def download_run_catalogs_parallel(
        self, run_number: str, catalog_limit: Optional[int] = None
    ) -> list[Path]:
        """
        Download multiple run catalogs in parallel.

        Args:
            run_number: ATLAS run number
            catalog_limit: Maximum number of catalogs to download (None for all)

        Returns:
            List of paths to successfully downloaded catalog files
        """
        # Determine how many catalogs to download
        total_catalogs = self.get_catalog_count(run_number)
        num_to_download = min(catalog_limit or total_catalogs, total_catalogs)

        self.logger.info(
            f"Downloading {num_to_download} catalogs for run {run_number} in parallel"
        )

        # Check which catalogs already exist
        download_tasks = []
        existing_catalogs = []

        for catalog_idx in range(num_to_download):
            catalog_path = self.get_run_catalog_path(run_number, catalog_idx)
            if catalog_path.exists():
                existing_catalogs.append(catalog_path)
            else:
                download_tasks.append((run_number, catalog_idx))

        if existing_catalogs:
            self.logger.info(
                f"Found {len(existing_catalogs)} existing catalogs, downloading {len(download_tasks)} new ones"
            )

        # Download missing catalogs in parallel
        successfully_downloaded = []

        if download_tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all download tasks
                future_to_task = {
                    executor.submit(
                        self._download_single_catalog, run_number, catalog_idx
                    ): (run_number, catalog_idx)
                    for run_number, catalog_idx in download_tasks
                }

                # Process completed downloads with progress bar
                with tqdm(
                    total=len(download_tasks),
                    desc=f"Downloading Run {run_number} catalogs",
                ) as pbar:
                    for future in as_completed(future_to_task):
                        run_num, catalog_idx = future_to_task[future]
                        try:
                            catalog_path = future.result()
                            if catalog_path:
                                successfully_downloaded.append(catalog_path)
                                self.logger.debug(
                                    f"Downloaded catalog {catalog_idx} for run {run_num}"
                                )
                            else:
                                self.logger.warning(
                                    f"Failed to download catalog {catalog_idx} for run {run_num}"
                                )
                        except Exception as e:
                            self.logger.error(
                                f"Exception downloading catalog {catalog_idx} for run {run_num}: {str(e)}"
                            )
                        finally:
                            pbar.update(1)

        # Combine existing and newly downloaded catalogs
        all_catalogs = existing_catalogs + successfully_downloaded

        self.logger.info(
            f"Total catalogs available for run {run_number}: {len(all_catalogs)}"
        )
        return sorted(all_catalogs)  # Sort by path for consistent ordering

    def _download_single_catalog(
        self, run_number: str, catalog_idx: int
    ) -> Optional[Path]:
        """
        Download a single catalog file (for use in parallel downloads).

        Args:
            run_number: ATLAS run number
            catalog_idx: Catalog index

        Returns:
            Path to downloaded file or None if failed
        """
        padded_run = run_number.zfill(8)
        url = f"/record/80001/files/data16_13TeV_Run_{padded_run}_file_index.json_{catalog_idx}"
        output_path = (
            self.base_dir / "catalogs" / f"Run_{run_number}_catalog_{catalog_idx}.root"
        )

        try:
            if self._download_file(
                url,
                output_path,
                f"Run {run_number} catalog {catalog_idx}",
                show_progress=False,  # Disable individual progress bars in parallel mode
            ):
                return output_path
            return output_path if output_path.exists() else None
        except Exception as e:
            self.logger.error(
                f"Failed to download catalog {catalog_idx} for run {run_number}: {str(e)}"
            )
            if output_path.exists():
                output_path.unlink()
            return None

    def _setup_directories(self):
        """Create necessary directory structure"""
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / "catalogs").mkdir(exist_ok=True)
        (self.base_dir / "signal_catalogs").mkdir(exist_ok=True)

    def _download_file(
        self, url: str, output_path: Path, desc: str = None, show_progress: bool = True
    ) -> bool:
        """Download a single file if it doesn't exist"""
        if output_path.exists():
            return False

        if show_progress:
            self.logger.info(f"Downloading file: {url}")

        response = requests.get(f"https://opendata.cern.ch{url}", stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))

            # Configure progress bar based on mode
            if show_progress:
                # Check if output is interactive
                is_interactive = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

                if is_interactive:
                    # Use normal progress bar for interactive terminal
                    bar_format = None
                    mininterval = 0.1
                else:
                    # Use simplified progress for log files
                    bar_format = "{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                    mininterval = 30

                with (
                    open(output_path, "wb") as f,
                    tqdm(
                        desc=desc,
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                        mininterval=mininterval,
                        bar_format=bar_format,
                    ) as pbar,
                ):
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)

                self.logger.info(f"Download complete: {desc}")
            else:
                # Silent download for parallel mode
                with open(output_path, "wb") as f:
                    for data in response.iter_content(chunk_size=1024):
                        f.write(data)

            return True
        else:
            raise Exception(f"Download failed with status code: {response.status_code}")

    def get_run_catalog_path(self, run_number: str, index: int = 0) -> Path:
        """Get path to a run catalog file"""
        return self.base_dir / "catalogs" / f"Run_{run_number}_catalog_{index}.root"

    def get_signal_catalog_path(self, signal_key: str, index: int = 0) -> Path:
        """Get path to a signal catalog file"""
        return self.base_dir / "signal_catalogs" / f"{signal_key}_catalog_{index}.root"

    def download_signal_catalog(
        self, signal_key: str, index: int = 0
    ) -> Optional[Path]:
        """Download a specific signal catalog file"""
        if signal_key not in self.signal_types:
            raise ValueError(
                f"Unknown signal key: {signal_key}. Available keys: {list(self.signal_types.keys())}"
            )

        signal_name = self.signal_types[signal_key]
        url = f"/record/80011/files/{signal_name}_file_index.json_{index}"
        output_path = (
            self.base_dir / "signal_catalogs" / f"{signal_key}_catalog_{index}.root"
        )

        try:
            if self._download_file(
                url, output_path, f"Downloading catalog {index} for signal {signal_key}"
            ):
                return output_path
            return output_path if output_path.exists() else None
        except Exception as e:
            self.logger.error(
                f"Failed to download {signal_key} catalog {index}: {str(e)}"
            )
            if output_path.exists():
                output_path.unlink()
            return None

    def download_signal_catalogs_parallel(
        self, signal_keys: list[str]
    ) -> dict[str, list[Path]]:
        """
        Download signal catalogs for multiple signal types in parallel.

        Args:
            signal_keys: List of signal keys to download

        Returns:
            Dictionary mapping signal keys to lists of downloaded catalog paths
        """
        self.logger.info(
            f"Downloading signal catalogs for {len(signal_keys)} signal types in parallel"
        )

        results = {}

        # Process each signal type in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_signal = {
                executor.submit(
                    self._download_signal_catalogs_for_key, signal_key
                ): signal_key
                for signal_key in signal_keys
            }

            for future in as_completed(future_to_signal):
                signal_key = future_to_signal[future]
                try:
                    catalog_paths = future.result()
                    results[signal_key] = catalog_paths
                    self.logger.info(
                        f"Completed downloading {len(catalog_paths)} catalogs for signal {signal_key}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to download catalogs for signal {signal_key}: {str(e)}"
                    )
                    results[signal_key] = []

        total_catalogs = sum(len(paths) for paths in results.values())
        self.logger.info(
            f"Downloaded {total_catalogs} total signal catalogs across {len(signal_keys)} signal types"
        )

        return results

    def _download_signal_catalogs_for_key(self, signal_key: str) -> list[Path]:
        """
        Download all catalogs for a specific signal key.

        Args:
            signal_key: Signal key to download catalogs for

        Returns:
            List of paths to downloaded catalog files
        """
        # For now, most signal types have only one catalog (index 0)
        # But this structure allows for expansion if needed
        catalog_path = self.download_signal_catalog(signal_key, 0)
        return [catalog_path] if catalog_path else []
