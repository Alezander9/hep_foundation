#!/usr/bin/env python3
"""
Pipeline Config Processor

This script processes YAML configuration files from a _experiment_config_stack directory,
running the full foundation model pipeline for each configuration.

Features:
- Processes all YAML files in _experiment_config_stack/ directory
- Runs full pipeline (train → regression → anomaly) for each config
- Saves logs to logs/ directory (one per config)
- Deletes processed configs from stack (configs are saved in experiment folders)
- Supports dry-run mode for testing
- Handles errors gracefully and continues processing
"""

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from hep_foundation.config.config_loader import load_pipeline_config
from hep_foundation.config.logging_config import get_logger
from hep_foundation.pipeline.foundation_model_pipeline import FoundationModelPipeline


class PipelineconfigProcessor:
    """Processes pipeline configuration configs from a stack directory."""

    def __init__(
        self, config_stack_dir: str = "_experiment_config_stack", logs_dir: str = "logs"
    ):
        """
        Initialize the config processor.

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

        self.logger.info("config processor initialized")
        self.logger.info(f"  config stack: {self.config_stack_dir.absolute()}")
        self.logger.info(f"  Logs directory: {self.logs_dir.absolute()}")

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

    def load_config_config(self, config_path: Path) -> dict:
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

            self.logger.info("Configuration loaded successfully")
            return config

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
        log_file = self.logs_dir / f"pipeline_{config_name}_{timestamp}.log"

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

    def process_config(self, config_path: Path, dry_run: bool = False) -> bool:
        """
        Process a single config file.

        Args:
            config_path: Path to the config YAML file
            dry_run: If True, don't actually run the pipeline or delete the config

        Returns:
            True if processing was successful, False otherwise
        """
        config_name = config_path.stem

        self.logger.info("=" * 100)
        self.logger.info(f"PROCESSING config: {config_name}")
        self.logger.info("=" * 100)

        # Set up logging for this config
        log_file, file_handler = self.setup_config_logging(config_name)

        try:
            # Load configuration
            config_config = self.load_config_config(config_path)

            if dry_run:
                self.logger.info("DRY RUN: Would run pipeline with this configuration")
                self.logger.info(
                    f"Dataset runs: {config_config['dataset_config'].run_numbers}"
                )
                self.logger.info(
                    f"Signal keys: {config_config['dataset_config'].signal_keys}"
                )
                self.logger.info(
                    f"Foundation model epochs: {config_config['foundation_model_training_config'].epochs}"
                )
                self.logger.info(
                    f"Regression epochs: {config_config['regression_evaluation_config'].epochs}"
                )
                return True

            # Initialize pipeline
            pipeline = FoundationModelPipeline()

            # Set source config file for reproducibility
            if config_config.get("_source_config_file"):
                pipeline.set_source_config_file(config_config["_source_config_file"])

            # Get pipeline settings
            pipeline_settings = config_config.get("pipeline_settings", {})
            delete_catalogs = pipeline_settings.get("delete_catalogs", True)

            self.logger.info(f"Pipeline settings - delete_catalogs: {delete_catalogs}")

            # Run the full pipeline
            self.logger.info("Starting full pipeline execution...")
            success = pipeline.run_full_pipeline(
                dataset_config=config_config["dataset_config"],
                task_config=config_config["task_config"],
                foundation_model_training_config=config_config[
                    "foundation_model_training_config"
                ],
                anomaly_detection_evaluation_config=config_config[
                    "anomaly_detection_evaluation_config"
                ],
                regression_evaluation_config=config_config[
                    "regression_evaluation_config"
                ],
                signal_classification_evaluation_config=config_config[
                    "signal_classification_evaluation_config"
                ],
                delete_catalogs=delete_catalogs,  # Use value from config
            )

            if success:
                self.logger.info(
                    f"Successfully completed pipeline for config: {config_name}"
                )

                # Delete the config file from the stack
                self.logger.info(f"Removing processed config from stack: {config_path}")
                config_path.unlink()

                return True
            else:
                self.logger.error(f"Pipeline failed for config: {config_name}")
                return False

        except Exception as e:
            self.logger.error(
                f"Error processing config {config_name}: {type(e).__name__}: {str(e)}"
            )
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            return False

        finally:
            # Clean up logging
            self.cleanup_config_logging(file_handler)
            self.logger.info(f"Log file for config '{config_name}': {log_file}")

    def run(self, dry_run: bool = False, max_configs: Optional[int] = None) -> bool:
        """
        Run the config processor.

        Args:
            dry_run: If True, don't actually run pipelines or delete configs
            max_configs: Maximum number of configs to process (None for all)

        Returns:
            True if all configs were processed successfully
        """
        self.logger.info("=" * 100)
        self.logger.info("STARTING PIPELINE config PROCESSOR")
        if dry_run:
            self.logger.info("DRY RUN MODE - No pipelines will be executed")
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
            self.logger.info(f"{'=' * 50}")
            self.logger.info(f"config {i}/{len(config_files)}: {config_path.name}")
            self.logger.info(f"{'=' * 50}")

            success = self.process_config(config_path, dry_run=dry_run)

            if success:
                successful_count += 1
                self.logger.info(f"config {i} completed successfully")
            else:
                failed_count += 1
                self.logger.error(f"config {i} failed")

        # Final summary
        self.logger.info("=" * 100)
        self.logger.info("config PROCESSING SUMMARY")
        self.logger.info("=" * 100)
        self.logger.info(f"Total configs processed: {len(config_files)}")
        self.logger.info(f"Successful: {successful_count}")
        self.logger.info(f"Failed: {failed_count}")

        if failed_count == 0:
            self.logger.info("🎉 All configs processed successfully!")
            return True
        else:
            self.logger.warning(f"⚠️  {failed_count} config(s) failed processing")
            return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process foundation model pipeline configs from a stack directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipelines.py                    # Process all configs in  /
  python scripts/run_pipelines.py --dry-run          # Preview what would be processed
  python scripts/run_pipelines.py --max-configs 3   # Process only first 3 configs
  python scripts/run_pipelines.py --config-dir my_configs  # Use custom config directory
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
        help="Preview configs without running pipelines or deleting files",
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
        processor = PipelineconfigProcessor(
            config_stack_dir=args.config_dir, logs_dir=args.logs_dir
        )

        # Run processor
        success = processor.run(dry_run=args.dry_run, max_configs=args.max_configs)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
