#!/usr/bin/env python3
"""
Catalog Download Processor

This script processes YAML configuration files from a _experiment_config_stack directory,
downloading all required ROOT file catalogs for each configuration without processing them into datasets.

This is designed for NERSC SCRATCH systems where:
- Login nodes have unlimited runtime but low resources
- Hundreds of TB of temporary storage available
- Can pre-download all catalogs, then process them into datasets separately

Features:
- Processes all YAML files in _experiment_config_stack/ directory
- Downloads ATLAS run catalogs and signal catalogs as needed
- Avoids duplicate downloads by tracking what's already available
- Saves logs to logs/ directory (one master log)
- Supports dry-run mode for testing
- Handles errors gracefully and continues processing
- Provides detailed progress reporting and download statistics

Usage:
  python scripts/download_catalogs.py                    # Download catalogs for all configs
  python scripts/download_catalogs.py --dry-run          # Preview what would be downloaded
  python scripts/download_catalogs.py --max-configs 3   # Process only first 3 configs
  python scripts/download_catalogs.py --config-dir my_configs  # Use custom config directory
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from hep_foundation.config.config_loader import load_pipeline_config
from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.atlas_file_manager import ATLASFileManager


class CatalogDownloadProcessor:
    """Processes pipeline configuration files to download all required catalogs."""

    def __init__(
        self,
        config_stack_dir: str = "_experiment_config_stack",
        logs_dir: str = "logs",
        max_workers: int = 4,
    ):
        """
        Initialize the catalog download processor.

        Args:
            config_stack_dir: Directory containing YAML configuration files to process
            logs_dir: Directory to save log files
            max_workers: Maximum number of parallel download workers
        """
        self.config_stack_dir = Path(config_stack_dir)
        self.logs_dir = Path(logs_dir)
        self.logger = get_logger(__name__)

        # Create directories if they don't exist
        self.config_stack_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Initialize ATLAS file manager with parallel capabilities
        self.atlas_manager = ATLASFileManager(max_workers=max_workers)

        # Track download requirements and results
        self.required_run_catalogs: set[tuple[str, int]] = (
            set()
        )  # (run_number, catalog_idx)
        self.required_signal_catalogs: set[str] = set()  # signal_key
        self.downloaded_run_catalogs: set[tuple[str, int]] = set()
        self.downloaded_signal_catalogs: set[str] = set()
        self.failed_downloads: list[dict] = []
        self.skipped_downloads: list[dict] = []

        self.logger.info("Catalog download processor initialized")
        self.logger.info(f"  Config stack: {self.config_stack_dir.absolute()}")
        self.logger.info(f"  Logs directory: {self.logs_dir.absolute()}")
        self.logger.info(
            f"  ATLAS data directory: {self.atlas_manager.base_dir.absolute()}"
        )

    def find_config_files(self) -> list[Path]:
        """Find all YAML configuration files in the config stack."""
        yaml_patterns = ["*.yaml", "*.yml"]
        config_files = []

        for pattern in yaml_patterns:
            config_files.extend(self.config_stack_dir.glob(pattern))

        # Sort by modification time (oldest first) for consistent processing order
        config_files.sort(key=lambda x: x.stat().st_mtime)

        self.logger.info(f"Found {len(config_files)} config files to process")
        for i, config_file in enumerate(config_files, 1):
            self.logger.info(f"  {i}. {config_file.name}")

        return config_files

    def load_config_data(self, config_path: Path) -> dict:
        """
        Load pipeline configuration from a config file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing the dataset configuration
        """
        try:
            config = load_pipeline_config(config_path)

            # Extract just the dataset config we need for catalog requirements
            config_data = {
                "dataset_config": config["dataset_config"],
                "source_config_file": config.get("_source_config_file"),
            }

            self.logger.info("Configuration loaded successfully")
            return config_data

        except Exception as e:
            self.logger.error(
                f"Failed to load configuration from {config_path}: {str(e)}"
            )
            raise

    def analyze_catalog_requirements(self, config_path: Path) -> dict:
        """
        Analyze a config file to determine what catalogs are needed.

        Args:
            config_path: Path to the config YAML file

        Returns:
            Dictionary with catalog requirements
        """
        config_name = config_path.stem
        self.logger.info(f"Analyzing catalog requirements for: {config_name}")

        try:
            # Load configuration
            config_data = self.load_config_data(config_path)
            dataset_config = config_data["dataset_config"]

            # Validate dataset config
            dataset_config.validate()

            requirements = {
                "config_name": config_name,
                "run_catalogs": [],
                "signal_catalogs": [],
                "errors": [],
            }

            # Analyze ATLAS run requirements
            if dataset_config.run_numbers:
                self.logger.info(f"  Run numbers: {dataset_config.run_numbers}")

                for run_number in dataset_config.run_numbers:
                    try:
                        # Get catalog count for this run
                        catalog_count = self.atlas_manager.get_catalog_count(run_number)
                        catalog_limit = dataset_config.catalog_limit or catalog_count

                        # Don't exceed available catalogs
                        actual_limit = min(catalog_limit, catalog_count)

                        for catalog_idx in range(actual_limit):
                            catalog_requirement = (run_number, catalog_idx)
                            requirements["run_catalogs"].append(catalog_requirement)
                            self.required_run_catalogs.add(catalog_requirement)

                        self.logger.info(
                            f"    Run {run_number}: {actual_limit} catalogs needed"
                        )

                    except Exception as e:
                        error_msg = f"Error analyzing run {run_number}: {str(e)}"
                        self.logger.error(error_msg)
                        requirements["errors"].append(error_msg)

            # Analyze signal requirements
            if dataset_config.signal_keys:
                self.logger.info(f"  Signal keys: {dataset_config.signal_keys}")

                for signal_key in dataset_config.signal_keys:
                    try:
                        # Add signal catalog requirement
                        requirements["signal_catalogs"].append(signal_key)
                        self.required_signal_catalogs.add(signal_key)

                        self.logger.info(f"    Signal {signal_key}: catalog needed")

                    except Exception as e:
                        error_msg = f"Error analyzing signal {signal_key}: {str(e)}"
                        self.logger.error(error_msg)
                        requirements["errors"].append(error_msg)

            return requirements

        except Exception as e:
            error_msg = f"Error analyzing config {config_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "config_name": config_name,
                "run_catalogs": [],
                "signal_catalogs": [],
                "errors": [error_msg],
            }

    def check_existing_catalogs(self) -> tuple[set[tuple[str, int]], set[str]]:
        """
        Check which catalogs already exist locally.

        Returns:
            Tuple of (existing_run_catalogs, existing_signal_catalogs)
        """
        self.logger.info("Checking for existing catalogs...")

        existing_run_catalogs = set()
        existing_signal_catalogs = set()

        # Check run catalogs
        for run_number, catalog_idx in self.required_run_catalogs:
            catalog_path = self.atlas_manager.get_run_catalog_path(
                run_number, catalog_idx
            )
            if catalog_path.exists():
                existing_run_catalogs.add((run_number, catalog_idx))

        # Check signal catalogs
        for signal_key in self.required_signal_catalogs:
            catalog_path = self.atlas_manager.get_signal_catalog_path(signal_key, 0)
            if catalog_path.exists():
                existing_signal_catalogs.add(signal_key)

        self.logger.info(f"Found {len(existing_run_catalogs)} existing run catalogs")
        self.logger.info(
            f"Found {len(existing_signal_catalogs)} existing signal catalogs"
        )

        return existing_run_catalogs, existing_signal_catalogs

    def download_catalogs(self, dry_run: bool = False) -> bool:
        """
        Download all required catalogs using parallel processing.

        Args:
            dry_run: If True, don't actually download catalogs

        Returns:
            True if all downloads were successful
        """
        self.logger.info("=" * 100)
        self.logger.info("STARTING PARALLEL CATALOG DOWNLOADS")
        if dry_run:
            self.logger.info("DRY RUN MODE - No catalogs will be downloaded")
        self.logger.info("=" * 100)

        # Check what already exists
        existing_run_catalogs, existing_signal_catalogs = self.check_existing_catalogs()

        # Determine what needs to be downloaded
        run_catalogs_to_download = self.required_run_catalogs - existing_run_catalogs
        signal_catalogs_to_download = (
            self.required_signal_catalogs - existing_signal_catalogs
        )

        self.logger.info("Download summary:")
        self.logger.info(
            f"  Total run catalogs required: {len(self.required_run_catalogs)}"
        )
        self.logger.info(f"  Run catalogs already exist: {len(existing_run_catalogs)}")
        self.logger.info(f"  Run catalogs to download: {len(run_catalogs_to_download)}")
        self.logger.info(
            f"  Total signal catalogs required: {len(self.required_signal_catalogs)}"
        )
        self.logger.info(
            f"  Signal catalogs already exist: {len(existing_signal_catalogs)}"
        )
        self.logger.info(
            f"  Signal catalogs to download: {len(signal_catalogs_to_download)}"
        )

        if dry_run:
            self.logger.info("DRY RUN: Would download the following catalogs:")
            for run_number, catalog_idx in sorted(run_catalogs_to_download):
                self.logger.info(f"  Run catalog: {run_number} index {catalog_idx}")
            for signal_key in sorted(signal_catalogs_to_download):
                self.logger.info(f"  Signal catalog: {signal_key}")
            return True

        # Track existing catalogs as "skipped"
        for run_number, catalog_idx in existing_run_catalogs:
            self.skipped_downloads.append(
                {
                    "type": "run",
                    "run_number": run_number,
                    "catalog_idx": catalog_idx,
                    "reason": "already exists",
                }
            )

        for signal_key in existing_signal_catalogs:
            self.skipped_downloads.append(
                {
                    "type": "signal",
                    "signal_key": signal_key,
                    "reason": "already exists",
                }
            )

        success_count = 0
        total_downloads = len(run_catalogs_to_download) + len(
            signal_catalogs_to_download
        )

        # Phase 1: Download run catalogs in parallel (grouped by run)
        if run_catalogs_to_download:
            self.logger.info("=" * 50)
            self.logger.info("PHASE 1: DOWNLOADING RUN CATALOGS")
            self.logger.info("=" * 50)

            # Group catalogs by run number
            run_catalog_groups = {}
            for run_number, catalog_idx in run_catalogs_to_download:
                if run_number not in run_catalog_groups:
                    run_catalog_groups[run_number] = []
                run_catalog_groups[run_number].append(catalog_idx)

            # Download each run's catalogs in parallel
            for run_number, catalog_indices in run_catalog_groups.items():
                self.logger.info(
                    f"Downloading {len(catalog_indices)} catalogs for run {run_number}"
                )

                # Create a temporary catalog limit for this run
                max_catalog_idx = max(catalog_indices)
                catalog_limit = max_catalog_idx + 1

                try:
                    downloaded_paths = (
                        self.atlas_manager.download_run_catalogs_parallel(
                            run_number, catalog_limit
                        )
                    )

                    # Track successful downloads
                    for path in downloaded_paths:
                        # Extract catalog index from path
                        path_parts = path.stem.split("_")
                        if len(path_parts) >= 3:
                            try:
                                catalog_idx = int(path_parts[-1])
                                if (
                                    run_number,
                                    catalog_idx,
                                ) in run_catalogs_to_download:
                                    self.downloaded_run_catalogs.add(
                                        (run_number, catalog_idx)
                                    )
                                    success_count += 1
                            except (ValueError, IndexError):
                                continue

                    self.logger.info(
                        f"Completed downloading catalogs for run {run_number}"
                    )

                except Exception as e:
                    error_msg = (
                        f"Failed to download catalogs for run {run_number}: {str(e)}"
                    )
                    self.logger.error(error_msg)
                    for catalog_idx in catalog_indices:
                        self.failed_downloads.append(
                            {
                                "type": "run",
                                "run_number": run_number,
                                "catalog_idx": catalog_idx,
                                "error": error_msg,
                            }
                        )

        # Phase 2: Download signal catalogs in parallel
        if signal_catalogs_to_download:
            self.logger.info("=" * 50)
            self.logger.info("PHASE 2: DOWNLOADING SIGNAL CATALOGS")
            self.logger.info("=" * 50)

            try:
                signal_results = self.atlas_manager.download_signal_catalogs_parallel(
                    list(signal_catalogs_to_download)
                )

                # Track successful downloads
                for signal_key, paths in signal_results.items():
                    if paths:  # If any paths were downloaded
                        self.downloaded_signal_catalogs.add(signal_key)
                        success_count += 1
                        self.logger.info(f"Downloaded signal catalog for {signal_key}")
                    else:
                        error_msg = f"No catalogs downloaded for signal {signal_key}"
                        self.logger.error(error_msg)
                        self.failed_downloads.append(
                            {
                                "type": "signal",
                                "signal_key": signal_key,
                                "error": error_msg,
                            }
                        )

            except Exception as e:
                error_msg = f"Failed to download signal catalogs: {str(e)}"
                self.logger.error(error_msg)
                for signal_key in signal_catalogs_to_download:
                    self.failed_downloads.append(
                        {
                            "type": "signal",
                            "signal_key": signal_key,
                            "error": error_msg,
                        }
                    )

        # Final summary
        self.logger.info("=" * 100)
        self.logger.info("PARALLEL CATALOG DOWNLOAD SUMMARY")
        self.logger.info("=" * 100)
        self.logger.info(f"Total downloads attempted: {total_downloads}")
        self.logger.info(f"Successful downloads: {success_count}")
        self.logger.info(f"Failed downloads: {len(self.failed_downloads)}")
        self.logger.info(f"Skipped (already exist): {len(self.skipped_downloads)}")

        return len(self.failed_downloads) == 0

    def run(self, dry_run: bool = False, max_configs: int = None) -> bool:
        """
        Run the catalog download processor.

        Args:
            dry_run: If True, don't actually download catalogs
            max_configs: Maximum number of configs to process (None for all)

        Returns:
            True if all catalog requirements were processed successfully
        """
        self.logger.info("=" * 100)
        self.logger.info("STARTING CATALOG DOWNLOAD PROCESSOR")
        if dry_run:
            self.logger.info("DRY RUN MODE - No catalogs will be downloaded")
        self.logger.info("=" * 100)

        # Set up logging for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"catalog_download_{timestamp}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        try:
            # Find all config files
            config_files = self.find_config_files()

            if not config_files:
                self.logger.warning("No config files found in the stack directory")
                return True

            # Limit number of configs if specified
            if max_configs is not None:
                config_files = config_files[:max_configs]
                self.logger.info(f"Processing limited to first {max_configs} configs")

            # Phase 1: Analyze all configs to determine requirements
            self.logger.info("=" * 50)
            self.logger.info("PHASE 1: ANALYZING CATALOG REQUIREMENTS")
            self.logger.info("=" * 50)

            successful_analysis = 0
            failed_analysis = 0

            for i, config_path in enumerate(config_files, 1):
                self.logger.info(
                    f"[{i}/{len(config_files)}] Analyzing: {config_path.name}"
                )

                try:
                    requirements = self.analyze_catalog_requirements(config_path)

                    if requirements["errors"]:
                        self.logger.error(
                            f"  Errors in {config_path.name}: {requirements['errors']}"
                        )
                        failed_analysis += 1
                    else:
                        successful_analysis += 1
                        self.logger.info(
                            f"  ✓ {len(requirements['run_catalogs'])} run catalogs, {len(requirements['signal_catalogs'])} signal catalogs"
                        )

                except Exception as e:
                    self.logger.error(
                        f"  Failed to analyze {config_path.name}: {str(e)}"
                    )
                    failed_analysis += 1

            # Phase 2: Download all required catalogs
            self.logger.info("=" * 50)
            self.logger.info("PHASE 2: DOWNLOADING CATALOGS")
            self.logger.info("=" * 50)

            download_success = self.download_catalogs(dry_run=dry_run)

            # Final summary
            self.logger.info("=" * 100)
            self.logger.info("FINAL SUMMARY")
            self.logger.info("=" * 100)
            self.logger.info(f"Configs analyzed: {len(config_files)}")
            self.logger.info(f"  Successful: {successful_analysis}")
            self.logger.info(f"  Failed: {failed_analysis}")
            self.logger.info(
                f"Unique run catalogs required: {len(self.required_run_catalogs)}"
            )
            self.logger.info(
                f"Unique signal catalogs required: {len(self.required_signal_catalogs)}"
            )

            if not dry_run:
                self.logger.info(
                    f"Run catalogs downloaded: {len(self.downloaded_run_catalogs)}"
                )
                self.logger.info(
                    f"Signal catalogs downloaded: {len(self.downloaded_signal_catalogs)}"
                )
                self.logger.info(
                    f"Downloads skipped (already exist): {len(self.skipped_downloads)}"
                )
                self.logger.info(f"Failed downloads: {len(self.failed_downloads)}")

                if self.failed_downloads:
                    self.logger.info("Failed downloads:")
                    for failure in self.failed_downloads:
                        if failure["type"] == "run":
                            self.logger.info(
                                f"  - Run {failure['run_number']} catalog {failure['catalog_idx']}: {failure['error']}"
                            )
                        else:
                            self.logger.info(
                                f"  - Signal {failure['signal_key']}: {failure['error']}"
                            )

            overall_success = failed_analysis == 0 and download_success

            if overall_success:
                self.logger.info("🎉 All catalog requirements processed successfully!")
                if not dry_run:
                    self.logger.info("Catalogs are ready for dataset creation!")
            else:
                self.logger.warning("⚠️  Some catalog requirements failed processing")

            return overall_success

        except Exception as e:
            self.logger.error(f"Fatal error in catalog download processor: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            return False

        finally:
            # Clean up logging
            root_logger.removeHandler(file_handler)
            file_handler.close()
            self.logger.info(f"Complete log saved to: {log_file}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download catalogs for foundation model pipeline configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_catalogs.py                    # Download catalogs for all configs
  python scripts/download_catalogs.py --dry-run          # Preview what catalogs would be downloaded
  python scripts/download_catalogs.py --max-configs 3   # Process only first 3 configs
  python scripts/download_catalogs.py --max-workers 8   # Use 8 parallel workers
  python scripts/download_catalogs.py --config-dir my_configs  # Use custom config directory

This script is designed for NERSC SCRATCH systems where you can:
1. Run this script on login nodes with unlimited runtime to download all catalogs
2. Run create_datasets.py to process catalogs into datasets (much faster with local catalogs)
3. Transfer the processed datasets to compute nodes for training
        """,
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="_experiment_config_stack",
        help="Directory containing YAML configuration files (default: _experiment_config_stack)",
    )

    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory to save log files (default: logs)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what catalogs would be downloaded without actually downloading them",
    )

    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configs to process (default: all)",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel download workers (default: 4)",
    )

    args = parser.parse_args()

    try:
        # Initialize processor
        processor = CatalogDownloadProcessor(
            config_stack_dir=args.config_dir,
            logs_dir=args.logs_dir,
            max_workers=args.max_workers,
        )

        # Run processor
        success = processor.run(dry_run=args.dry_run, max_configs=args.max_configs)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nCatalog download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
