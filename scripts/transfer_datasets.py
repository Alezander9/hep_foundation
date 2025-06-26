#!/usr/bin/env python3
"""
Dataset Transfer Script

Transfers processed dataset folders to predefined target systems using rsync
with compression for efficient transfer of large HDF5 files.

Usage:
  python scripts/transfer_datasets.py nersc                    # Transfer all datasets to NERSC
  python scripts/transfer_datasets.py nersc --dataset dataset_runs_00298967-2-00311481_769d3f43  # Transfer specific dataset
  python scripts/transfer_datasets.py home --list-datasets     # List available datasets
  python scripts/transfer_datasets.py nersc --dry-run         # Preview transfer

Features:
  - Uses rsync with compression for efficient transfer of large files
  - Supports incremental transfers (only transfers changed files)
  - Validates dataset integrity before transfer
  - Option to delete datasets locally after successful transfer
  - Progress tracking for large transfers
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from hep_foundation.config.logging_config import get_logger
from hep_foundation.utils.remote_transfer import TransferManager


class DatasetTransfer:
    """Manages transfer of processed datasets to remote systems."""

    def __init__(
        self,
        config_file: str = ".env",
        datasets_dir: str = "_processed_datasets",
    ):
        """
        Initialize the dataset transfer manager.

        Args:
            config_file: Path to the system configuration file
            datasets_dir: Directory containing processed datasets
        """
        self.logger = get_logger(__name__)
        self.datasets_dir = Path(datasets_dir)
        self.transfer_manager = TransferManager(config_file)

        # Ensure datasets directory exists
        if not self.datasets_dir.exists():
            self.logger.error(f"Datasets directory not found: {self.datasets_dir}")
            self.logger.error("Please create datasets using the DatasetManager first.")
            sys.exit(1)

        self.logger.info("Dataset Transfer initialized")
        self.logger.info(f"  Datasets directory: {self.datasets_dir.absolute()}")
        self.logger.info(
            f"  Available systems: {list(self.transfer_manager.systems.keys())}"
        )

    def find_datasets(self) -> list[Path]:
        """Find all dataset directories in the datasets folder."""
        dataset_dirs = []

        for item in self.datasets_dir.iterdir():
            if item.is_dir():
                # Check if it looks like a dataset (has required files)
                if self._is_valid_dataset(item):
                    dataset_dirs.append(item)
                else:
                    self.logger.warning(
                        f"Skipping invalid dataset directory: {item.name}"
                    )

        # Sort by modification time for consistent ordering
        dataset_dirs.sort(key=lambda x: x.stat().st_mtime)

        self.logger.info(f"Found {len(dataset_dirs)} valid dataset directories")
        return dataset_dirs

    def _is_valid_dataset(self, dataset_dir: Path) -> bool:
        """Check if a directory contains a valid dataset."""
        required_files = ["_dataset_config.yaml", "_dataset_info.json"]

        for required_file in required_files:
            if not (dataset_dir / required_file).exists():
                return False

        # Check for at least one data file
        data_files = ["dataset.h5", "signal_dataset.h5"]
        has_data = any((dataset_dir / data_file).exists() for data_file in data_files)

        return has_data

    def get_dataset_info(self, dataset_dir: Path) -> dict:
        """Get basic information about a dataset."""
        info = {
            "name": dataset_dir.name,
            "path": dataset_dir,
            "size_mb": 0,
            "files": [],
            "has_atlas_data": False,
            "has_signal_data": False,
        }

        # Calculate total size and count files
        total_size = 0
        file_count = 0

        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                total_size += file_size
                file_count += 1

                # Check for specific data types
                if file_path.name == "dataset.h5":
                    info["has_atlas_data"] = True
                elif file_path.name == "signal_dataset.h5":
                    info["has_signal_data"] = True

        info["size_mb"] = total_size / (1024 * 1024)  # Convert to MB
        info["file_count"] = file_count

        return info

    def list_datasets(self):
        """List all available datasets with their information."""
        datasets = self.find_datasets()

        if not datasets:
            self.logger.info("No datasets found to transfer")
            return

        self.logger.info("Available datasets for transfer:")
        self.logger.info("=" * 80)

        total_size_mb = 0
        for i, dataset_dir in enumerate(datasets, 1):
            info = self.get_dataset_info(dataset_dir)
            total_size_mb += info["size_mb"]

            self.logger.info(f"{i:2d}. {info['name']}")
            self.logger.info(
                f"     Size: {info['size_mb']:.1f} MB ({info['file_count']} files)"
            )

            data_types = []
            if info["has_atlas_data"]:
                data_types.append("ATLAS")
            if info["has_signal_data"]:
                data_types.append("Signal")

            if data_types:
                self.logger.info(f"     Data: {', '.join(data_types)}")

            self.logger.info(f"     Path: {info['path']}")
            self.logger.info("")

        self.logger.info(f"Total: {len(datasets)} datasets, {total_size_mb:.1f} MB")

    def transfer_dataset(
        self,
        dataset_dir: Path,
        system: str,
        dry_run: bool = False,
        compression: bool = True,
        delete_after_transfer: bool = False,
    ) -> bool:
        """
        Transfer a single dataset to the specified system.

        Args:
            dataset_dir: Path to the dataset directory
            system: Target system name
            dry_run: If True, show what would be transferred without doing it
            compression: Whether to use compression during transfer
            delete_after_transfer: Whether to delete local dataset after successful transfer

        Returns:
            True if successful
        """
        # Get transfer client
        transfer_client = self.transfer_manager.create_transfer_client(system)
        if not transfer_client:
            return False

        # Get dataset info
        info = self.get_dataset_info(dataset_dir)

        self.logger.info(f"Preparing to transfer dataset: {info['name']}")
        self.logger.info(
            f"  Size: {info['size_mb']:.1f} MB ({info['file_count']} files)"
        )
        self.logger.info(f"  Target: {system.upper()}")
        self.logger.info(f"  Compression: {'enabled' if compression else 'disabled'}")

        # Create remote datasets directory structure
        system_config = self.transfer_manager.get_system_config(system)
        remote_datasets_path = f"{system_config.path}/_processed_datasets"

        if not dry_run:
            # Create the remote datasets directory
            if not transfer_client.create_remote_directory(remote_datasets_path):
                return False

        # Transfer the dataset directory using rsync
        success = transfer_client.transfer_directory_rsync(
            local_dir=dataset_dir,
            remote_path=remote_datasets_path,
            compression=compression,
            dry_run=dry_run,
            exclude_patterns=[
                ".DS_Store",
                "*.tmp",
                "__pycache__",
                "*.pyc",
            ],
        )

        if success and not dry_run:
            self.logger.info(
                f"Dataset {info['name']} transferred successfully to {system.upper()}"
            )

            # Delete local dataset if requested
            if delete_after_transfer:
                self.logger.info(f"Deleting local dataset: {info['name']}")
                try:
                    shutil.rmtree(dataset_dir)
                    self.logger.info(f"Local dataset deleted: {info['name']}")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to delete local dataset {info['name']}: {e}"
                    )

        return success

    def transfer_datasets(
        self,
        system: str,
        dataset_names: Optional[list[str]] = None,
        dry_run: bool = False,
        compression: bool = True,
        delete_after_transfer: bool = False,
    ) -> bool:
        """
        Transfer datasets to the specified system.

        Args:
            system: Target system name
            dataset_names: List of specific dataset names to transfer (None = all)
            dry_run: If True, show what would be transferred without doing it
            compression: Whether to use compression during transfer
            delete_after_transfer: Whether to delete local datasets after successful transfer

        Returns:
            True if all transfers successful
        """
        # Find available datasets
        all_datasets = self.find_datasets()
        if not all_datasets:
            self.logger.warning("No datasets found to transfer")
            return True

        # Filter datasets if specific names provided
        if dataset_names:
            datasets_to_transfer = []
            for dataset_name in dataset_names:
                dataset_path = self.datasets_dir / dataset_name
                if dataset_path.exists() and dataset_path in all_datasets:
                    datasets_to_transfer.append(dataset_path)
                else:
                    self.logger.error(f"Dataset not found: {dataset_name}")
                    return False
        else:
            datasets_to_transfer = all_datasets

        if not datasets_to_transfer:
            self.logger.warning("No datasets selected for transfer")
            return True

        self.logger.info(
            f"Transferring {len(datasets_to_transfer)} datasets to {system.upper()}"
        )

        # Calculate total transfer size
        total_size_mb = sum(
            self.get_dataset_info(d)["size_mb"] for d in datasets_to_transfer
        )
        self.logger.info(f"Total transfer size: {total_size_mb:.1f} MB")

        if dry_run:
            self.logger.info("DRY RUN: The following datasets would be transferred:")
            for dataset_dir in datasets_to_transfer:
                info = self.get_dataset_info(dataset_dir)
                self.logger.info(f"  - {info['name']} ({info['size_mb']:.1f} MB)")
            return True

        # Transfer each dataset
        successful_transfers = 0
        for i, dataset_dir in enumerate(datasets_to_transfer, 1):
            self.logger.info(
                f"[{i}/{len(datasets_to_transfer)}] Transferring {dataset_dir.name}"
            )

            success = self.transfer_dataset(
                dataset_dir=dataset_dir,
                system=system,
                dry_run=dry_run,
                compression=compression,
                delete_after_transfer=delete_after_transfer,
            )

            if success:
                successful_transfers += 1
            else:
                self.logger.error(f"Failed to transfer dataset: {dataset_dir.name}")

        # Summary
        if successful_transfers == len(datasets_to_transfer):
            self.logger.info(
                f"All {successful_transfers} datasets transferred successfully to {system.upper()}"
            )
            return True
        else:
            self.logger.error(
                f"Only {successful_transfers}/{len(datasets_to_transfer)} datasets transferred successfully"
            )
            return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Transfer processed datasets to predefined target systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/transfer_datasets.py nersc                    # Transfer all datasets to NERSC
  python scripts/transfer_datasets.py nersc --dataset mydata   # Transfer specific dataset
  python scripts/transfer_datasets.py nersc --dry-run         # Preview transfer to NERSC
  python scripts/transfer_datasets.py --list-datasets         # List available datasets
  python scripts/transfer_datasets.py --list-systems          # List available systems

Dataset Transfer:
  Uses rsync with compression for efficient transfer of large HDF5 files.
  Supports incremental transfers - only changed files are transferred.
        """,
    )

    parser.add_argument(
        "system", nargs="?", help="Target system name (nersc, home, etc.)"
    )

    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Specific dataset name to transfer (can be used multiple times)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be transferred without actually doing it",
    )

    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable compression during transfer (faster CPU, more bandwidth)",
    )

    parser.add_argument(
        "--delete-after-transfer",
        action="store_true",
        help="Delete local datasets after successful transfer (DANGEROUS - use with caution)",
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets for transfer",
    )

    parser.add_argument(
        "--list-systems",
        action="store_true",
        help="List available target systems",
    )

    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="_processed_datasets",
        help="Local directory containing datasets (default: _processed_datasets)",
    )

    args = parser.parse_args()

    try:
        # Initialize transfer manager
        manager = DatasetTransfer(datasets_dir=args.datasets_dir)

        if args.list_datasets:
            manager.list_datasets()
            return

        if args.list_systems:
            manager.transfer_manager.list_systems()
            return

        if not args.system:
            parser.print_help()
            print(
                "\nError: Please specify a target system or use --list-datasets/--list-systems"
            )
            sys.exit(1)

        # Confirm dangerous operations
        if args.delete_after_transfer and not args.dry_run:
            response = input(
                "\nWARNING: This will DELETE local datasets after transfer. Continue? (yes/no): "
            )
            if response.lower() != "yes":
                print("Transfer cancelled.")
                sys.exit(0)

        # Transfer datasets
        success = manager.transfer_datasets(
            system=args.system.lower(),
            dataset_names=args.datasets,
            dry_run=args.dry_run,
            compression=not args.no_compression,
            delete_after_transfer=args.delete_after_transfer,
        )
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nTransfer interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
