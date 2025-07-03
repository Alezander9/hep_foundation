#!/usr/bin/env python3
"""
Dataset Creation Processor

This script processes YAML configuration files from a _experiment_config_stack directory,
creating all required datasets for each configuration without running the full pipeline.

Features:
- Processes all YAML files in _experiment_config_stack/ directory
- Creates ATLAS datasets and signal datasets as needed
- Checks for existing datasets to avoid unnecessary re-processing
- Saves logs to logs/ directory (one per config)
- Supports dry-run mode for testing
- Handles errors gracefully and continues processing
- Provides summary of created datasets

This enables efficient two-stage processing:
1. Use CPU machine to create datasets (this script)
2. Transfer datasets to GPU machine for pipeline training
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from hep_foundation.config.config_loader import load_pipeline_config
from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.dataset_manager import DatasetManager


class DatasetCreationProcessor:
    """Processes pipeline configuration files to create datasets from a stack directory."""

    def __init__(
        self, config_stack_dir: str = "_experiment_config_stack", logs_dir: str = "logs"
    ):
        """
        Initialize the dataset creation processor.

        Args:
            config_stack_dir: Directory containing YAML configuration files to process
            logs_dir: Directory to save log files
        """
        self.config_stack_dir = Path(config_stack_dir)
        self.logs_dir = Path(logs_dir)
        self.logger = get_logger(__name__)

        # Create directories if they don't exist
        self.config_stack_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Initialize dataset manager
        self.data_manager = DatasetManager(base_dir="_processed_datasets")

        # Track created datasets
        self.created_datasets = []
        self.skipped_datasets = []
        self.failed_datasets = []

        self.logger.info("Dataset creation processor initialized")
        self.logger.info(f"  Config stack: {self.config_stack_dir.absolute()}")
        self.logger.info(f"  Logs directory: {self.logs_dir.absolute()}")
        self.logger.info(
            f"  Processed datasets: {self.data_manager.base_dir.absolute()}"
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
            Dictionary containing all configuration objects
        """
        self.logger.info(f"Loading configuration from: {config_path}")

        try:
            config = load_pipeline_config(config_path)

            # Extract the config objects we need for dataset creation
            config_data = {
                "dataset_config": config["dataset_config"],
                "task_config": config["task_config"],
                "pipeline_settings": config.get("pipeline_settings", {}),
                "source_config_file": config.get("_source_config_file"),
            }

            self.logger.info("Configuration loaded successfully")
            return config_data

        except Exception as e:
            self.logger.error(
                f"Failed to load configuration from {config_path}: {str(e)}"
            )
            raise

    def setup_config_logging(
        self, config_name: str
    ) -> tuple[Path, logging.FileHandler]:
        """
        Set up logging for a specific config.

        Args:
            config_name: Name of the config (without extension)

        Returns:
            Tuple of (log_file_path, file_handler)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"dataset_creation_{config_name}_{timestamp}.log"

        # Create file handler for this config
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        self.logger.info(f"Logging for config '{config_name}' to: {log_file}")
        return log_file, file_handler

    def cleanup_config_logging(self, file_handler: logging.FileHandler):
        """Clean up logging handler for a config."""
        root_logger = logging.getLogger()
        root_logger.removeHandler(file_handler)
        file_handler.close()

    def check_dataset_exists(
        self, dataset_config, task_config
    ) -> tuple[bool, str, bool, str]:
        """
        Check if datasets already exist for the given configuration.

        Args:
            dataset_config: Dataset configuration
            task_config: Task configuration

        Returns:
            Tuple of (atlas_exists, atlas_dataset_id, signal_exists, signal_dataset_id)
        """
        # Generate dataset IDs
        atlas_dataset_id = self.data_manager.generate_dataset_id(dataset_config)

        # Check if ATLAS dataset exists
        atlas_dataset_path, atlas_config_path = self.data_manager.get_dataset_paths(
            atlas_dataset_id
        )
        atlas_exists = atlas_dataset_path.exists() and atlas_config_path.exists()

        # Check if signal datasets exist (if signal_keys are provided)
        signal_exists = True
        signal_dataset_id = None
        if dataset_config.signal_keys:
            signal_dataset_id = atlas_dataset_id  # Same ID for signal datasets
            signal_dataset_path, signal_config_path = (
                self.data_manager.get_signal_dataset_paths(signal_dataset_id)
            )
            signal_exists = signal_dataset_path.exists() and signal_config_path.exists()
        else:
            signal_exists = True  # No signal datasets needed

        return atlas_exists, atlas_dataset_id, signal_exists, signal_dataset_id

    def create_datasets_for_config(
        self,
        dataset_config,
        task_config,
        pipeline_settings: dict,
        config_name: str,
        dry_run: bool = False,
    ) -> dict:
        """
        Create datasets for a single configuration.

        Args:
            dataset_config: Dataset configuration
            task_config: Task configuration
            pipeline_settings: Pipeline settings from config
            config_name: Name of the config for logging
            dry_run: If True, don't actually create datasets

        Returns:
            Dictionary with creation results
        """
        results = {
            "atlas_created": False,
            "atlas_dataset_id": None,
            "signal_created": False,
            "signal_dataset_id": None,
            "atlas_skipped": False,
            "signal_skipped": False,
            "errors": [],
        }

        try:
            # Check what datasets exist
            atlas_exists, atlas_dataset_id, signal_exists, signal_dataset_id = (
                self.check_dataset_exists(dataset_config, task_config)
            )

            results["atlas_dataset_id"] = atlas_dataset_id
            results["signal_dataset_id"] = signal_dataset_id

            # Get delete_catalogs setting from pipeline config (default to True for backward compatibility)
            delete_catalogs = pipeline_settings.get("delete_catalogs", True)

            self.logger.info(f"Dataset status for {config_name}:")
            self.logger.info(f"  ATLAS dataset ID: {atlas_dataset_id}")
            self.logger.info(f"  ATLAS exists: {atlas_exists}")
            self.logger.info(f"  Delete catalogs: {delete_catalogs}")
            if dataset_config.signal_keys:
                self.logger.info(f"  Signal dataset ID: {signal_dataset_id}")
                self.logger.info(f"  Signal exists: {signal_exists}")
                self.logger.info(f"  Signal keys: {dataset_config.signal_keys}")

            if dry_run:
                self.logger.info("DRY RUN: Would create the following datasets:")
                if not atlas_exists:
                    self.logger.info(f"  - ATLAS dataset: {atlas_dataset_id}")
                else:
                    self.logger.info(
                        f"  - ATLAS dataset: {atlas_dataset_id} (already exists)"
                    )

                if dataset_config.signal_keys and not signal_exists:
                    self.logger.info(f"  - Signal dataset: {signal_dataset_id}")
                elif dataset_config.signal_keys:
                    self.logger.info(
                        f"  - Signal dataset: {signal_dataset_id} (already exists)"
                    )

                return results

            # Create ATLAS dataset if needed
            if not atlas_exists:
                self.logger.info(f"Creating ATLAS dataset: {atlas_dataset_id}")
                try:
                    # Set up plotting if enabled
                    plot_output = None
                    if dataset_config.plot_distributions:
                        plot_output = (
                            self.data_manager.get_dataset_dir(atlas_dataset_id)
                            / "plots"
                            / "atlas_dataset_features.png"
                        )

                    created_atlas_id, created_atlas_path = (
                        self.data_manager._create_atlas_dataset(
                            dataset_config=dataset_config,
                            delete_catalogs=delete_catalogs,  # Use value from config
                            plot_output=plot_output,
                        )
                    )
                    self.logger.info(
                        f"Successfully created ATLAS dataset at: {created_atlas_path}"
                    )
                    results["atlas_created"] = True
                except Exception as e:
                    error_msg = (
                        f"Failed to create ATLAS dataset {atlas_dataset_id}: {str(e)}"
                    )
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            else:
                self.logger.info(f"ATLAS dataset already exists: {atlas_dataset_id}")
                results["atlas_skipped"] = True

            # Create signal datasets if needed
            if dataset_config.signal_keys:
                if not signal_exists:
                    self.logger.info(f"Creating signal dataset: {signal_dataset_id}")
                    try:
                        # Determine background histogram data path for comparison plots
                        background_hist_data_path = None
                        if dataset_config.plot_distributions:
                            background_plot_data_dir = (
                                self.data_manager.get_dataset_dir(atlas_dataset_id)
                                / "plot_data"
                            )
                            potential_bg_hist_path = (
                                background_plot_data_dir
                                / "atlas_dataset_features_hist_data.json"
                            )
                            if potential_bg_hist_path.exists():
                                background_hist_data_path = potential_bg_hist_path

                        created_signal_id, created_signal_path = (
                            self.data_manager._create_signal_dataset(
                                dataset_config=dataset_config,
                                background_hist_data_path=background_hist_data_path,
                            )
                        )
                        self.logger.info(
                            f"Successfully created signal dataset at: {created_signal_path}"
                        )
                        results["signal_created"] = True
                    except Exception as e:
                        error_msg = f"Failed to create signal dataset {signal_dataset_id}: {str(e)}"
                        self.logger.error(error_msg)
                        results["errors"].append(error_msg)
                else:
                    self.logger.info(
                        f"Signal dataset already exists: {signal_dataset_id}"
                    )
                    results["signal_skipped"] = True

        except Exception as e:
            error_msg = f"Error processing datasets for {config_name}: {str(e)}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)

        return results

    def process_config(self, config_path: Path, dry_run: bool = False) -> bool:
        """
        Process a single config file to create its datasets.

        Args:
            config_path: Path to the config YAML file
            dry_run: If True, don't actually create datasets

        Returns:
            True if processing was successful, False otherwise
        """
        config_name = config_path.stem

        self.logger.info("=" * 100)
        self.logger.info(f"PROCESSING CONFIG: {config_name}")
        self.logger.info("=" * 100)

        # Set up logging for this config
        log_file, file_handler = self.setup_config_logging(config_name)

        try:
            # Load configuration
            config_data = self.load_config_data(config_path)
            dataset_config = config_data["dataset_config"]
            task_config = config_data["task_config"]
            pipeline_settings = config_data["pipeline_settings"]

            # Validate configs
            dataset_config.validate()
            self.logger.info("Dataset config validation passed")

            if dry_run:
                self.logger.info(
                    "DRY RUN: Would create datasets for this configuration"
                )
                self.logger.info(f"Run numbers: {dataset_config.run_numbers}")
                self.logger.info(f"Signal keys: {dataset_config.signal_keys}")
                return True

            # Create datasets
            results = self.create_datasets_for_config(
                dataset_config,
                task_config,
                pipeline_settings,
                config_name,
                dry_run=dry_run,
            )

            # Track results
            success = len(results["errors"]) == 0

            if results["atlas_created"]:
                self.created_datasets.append(
                    {
                        "type": "ATLAS",
                        "dataset_id": results["atlas_dataset_id"],
                        "config": config_name,
                    }
                )

            if results["signal_created"]:
                self.created_datasets.append(
                    {
                        "type": "Signal",
                        "dataset_id": results["signal_dataset_id"],
                        "config": config_name,
                        "signal_keys": dataset_config.signal_keys,
                    }
                )

            if results["atlas_skipped"]:
                self.skipped_datasets.append(
                    {
                        "type": "ATLAS",
                        "dataset_id": results["atlas_dataset_id"],
                        "config": config_name,
                        "reason": "already exists",
                    }
                )

            if results["signal_skipped"]:
                self.skipped_datasets.append(
                    {
                        "type": "Signal",
                        "dataset_id": results["signal_dataset_id"],
                        "config": config_name,
                        "reason": "already exists",
                    }
                )

            if results["errors"]:
                for error in results["errors"]:
                    self.failed_datasets.append(
                        {
                            "config": config_name,
                            "error": error,
                        }
                    )

            if success:
                self.logger.info(f"Successfully processed config: {config_name}")
                return True
            else:
                self.logger.error(f"Failed to process config: {config_name}")
                return False

        except Exception as e:
            error_msg = (
                f"Error processing config {config_name}: {type(e).__name__}: {str(e)}"
            )
            self.logger.error(error_msg)
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            self.failed_datasets.append(
                {
                    "config": config_name,
                    "error": error_msg,
                }
            )
            return False

        finally:
            # Clean up logging
            self.cleanup_config_logging(file_handler)
            self.logger.info(f"Log file for config '{config_name}': {log_file}")

    def run(self, dry_run: bool = False, max_configs: int = None) -> bool:
        """
        Run the dataset creation processor.

        Args:
            dry_run: If True, don't actually create datasets
            max_configs: Maximum number of configs to process (None for all)

        Returns:
            True if all configs were processed successfully
        """
        self.logger.info("=" * 100)
        self.logger.info("STARTING DATASET CREATION PROCESSOR")
        if dry_run:
            self.logger.info("DRY RUN MODE - No datasets will be created")
        self.logger.info("=" * 100)

        # Find all config files
        config_files = self.find_config_files()

        if not config_files:
            self.logger.warning("No config files found in the stack directory")
            return True

        # Limit number of configs if specified
        if max_configs is not None:
            config_files = config_files[:max_configs]
            self.logger.info(f"Processing limited to first {max_configs} configs")

        # Process each config
        successful_count = 0
        failed_count = 0

        for i, config_path in enumerate(config_files, 1):
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"CONFIG {i}/{len(config_files)}: {config_path.name}")
            self.logger.info(f"{'=' * 50}")

            success = self.process_config(config_path, dry_run=dry_run)

            if success:
                successful_count += 1
            else:
                failed_count += 1

        # Final summary
        self.print_summary(len(config_files), successful_count, failed_count, dry_run)

        return failed_count == 0

    def print_summary(
        self,
        total_configs: int,
        successful_count: int,
        failed_count: int,
        dry_run: bool,
    ):
        """Print a comprehensive summary of the dataset creation process."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("DATASET CREATION SUMMARY")
        self.logger.info("=" * 100)

        self.logger.info(f"Total configs processed: {total_configs}")
        self.logger.info(f"Successful: {successful_count}")
        self.logger.info(f"Failed: {failed_count}")

        if not dry_run:
            # Group datasets by type for better organization
            atlas_created = [d for d in self.created_datasets if d["type"] == "ATLAS"]
            signal_created = [d for d in self.created_datasets if d["type"] == "Signal"]
            atlas_skipped = [d for d in self.skipped_datasets if d["type"] == "ATLAS"]
            signal_skipped = [d for d in self.skipped_datasets if d["type"] == "Signal"]

            self.logger.info("\nDATASET STATISTICS:")
            self.logger.info(f"  ATLAS datasets created: {len(atlas_created)}")
            self.logger.info(f"  Signal datasets created: {len(signal_created)}")
            self.logger.info(f"  ATLAS datasets skipped: {len(atlas_skipped)}")
            self.logger.info(f"  Signal datasets skipped: {len(signal_skipped)}")
            self.logger.info(f"  Total new datasets: {len(self.created_datasets)}")

            if self.created_datasets:
                self.logger.info(f"\nCREATED DATASETS ({len(self.created_datasets)}):")
                for dataset in self.created_datasets:
                    if dataset["type"] == "ATLAS":
                        self.logger.info(
                            f"  - {dataset['type']}: {dataset['dataset_id']} (from {dataset['config']})"
                        )
                    else:
                        self.logger.info(
                            f"  - {dataset['type']}: {dataset['dataset_id']} (from {dataset['config']}, keys: {dataset['signal_keys']})"
                        )

            if self.skipped_datasets:
                self.logger.info(f"\nSKIPPED DATASETS ({len(self.skipped_datasets)}):")
                for dataset in self.skipped_datasets:
                    self.logger.info(
                        f"  - {dataset['type']}: {dataset['dataset_id']} ({dataset['reason']})"
                    )

            if self.failed_datasets:
                self.logger.info(f"\nFAILED DATASETS ({len(self.failed_datasets)}):")
                for failure in self.failed_datasets:
                    self.logger.info(
                        f"  - Config {failure['config']}: {failure['error']}"
                    )

        if failed_count == 0:
            self.logger.info("\nAll configs processed successfully!")
            if not dry_run and self.created_datasets:
                self.logger.info(
                    f"{len(self.created_datasets)} new datasets are ready for transfer!"
                )
                self.logger.info(
                    f"Datasets location: {self.data_manager.base_dir.absolute()}"
                )
        else:
            self.logger.warning(f"\n{failed_count} config(s) failed processing")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create datasets from foundation model pipeline configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_datasets.py                    # Create datasets for all configs
  python scripts/create_datasets.py --dry-run          # Preview what datasets would be created
  python scripts/create_datasets.py --max-configs 3   # Process only first 3 configs
  python scripts/create_datasets.py --config-dir my_configs  # Use custom config directory

This script enables efficient two-stage processing:
1. Run this script on a CPU machine to create all datasets
2. Transfer datasets to GPU machine for pipeline training
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
        help="Preview what datasets would be created without actually creating them",
    )

    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configs to process (default: all)",
    )

    args = parser.parse_args()

    try:
        # Initialize processor
        processor = DatasetCreationProcessor(
            config_stack_dir=args.config_dir, logs_dir=args.logs_dir
        )

        # Run processor
        success = processor.run(dry_run=args.dry_run, max_configs=args.max_configs)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nDataset creation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
