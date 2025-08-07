#!/usr/bin/env python3
"""
Standalone Regression Pipeline Execution Script.

This script provides a command-line interface for running standalone DNN regression
experiments using the HEP Foundation Pipeline infrastructure.

Usage:
    python scripts/run_standalone_regression.py [options]
    python scripts/run_standalone_regression.py --config path/to/config.yaml
    python scripts/run_standalone_regression.py --config-stack  # Process all configs in stack
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hep_foundation.config.logging_config import get_logger  # noqa: E402  # noqa: E402
from hep_foundation.pipeline.standalone_regression_pipeline import (  # noqa: E402  # noqa: E402
    StandaloneRegressionPipeline,
)


def setup_logging():
    """Setup logging for the script."""
    logger = get_logger(__name__)
    return logger


def load_yaml_config(config_path: Path) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")


def validate_config(config: dict, logger) -> bool:
    """
    Validate configuration structure for standalone regression.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        True if valid, False otherwise
    """
    required_sections = ["dataset", "task", "models", "training", "evaluation"]
    missing_sections = []

    for section in required_sections:
        if section not in config:
            missing_sections.append(section)

    if missing_sections:
        logger.error(f"Missing required configuration sections: {missing_sections}")
        return False

    # Check for standalone_dnn model configuration
    if "standalone_dnn" not in config.get("models", {}):
        logger.error("Configuration must include 'models.standalone_dnn' section")
        return False

    # Check for standalone_dnn training configuration
    training_config = config.get("training", {})
    if "standalone_dnn" not in training_config:
        logger.error("Configuration must include 'training.standalone_dnn' section")
        return False

    logger.info("Configuration validation passed")
    return True


def run_single_config(
    config_path: Path, pipeline: StandaloneRegressionPipeline, logger
) -> bool:
    """
    Run standalone regression for a single configuration file.

    Args:
        config_path: Path to configuration file
        pipeline: Pipeline instance
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 100)
    logger.info(f"PROCESSING CONFIG: {config_path.name}")
    logger.info("=" * 100)

    try:
        # Load configuration
        config = load_yaml_config(config_path)

        # Validate configuration
        if not validate_config(config, logger):
            return False

        # Extract pipeline settings
        pipeline_settings = config.get("pipeline", {})
        delete_catalogs = pipeline_settings.get("delete_catalogs", False)

        # Log configuration summary
        logger.info("Configuration Summary:")
        logger.info(f"  Name: {config.get('name', 'Unknown')}")
        logger.info(f"  Description: {config.get('description', 'No description')}")
        logger.info(
            f"  Model: {config['models']['standalone_dnn']['architecture']['hidden_layers']}"
        )
        logger.info(f"  Data sizes: {config['evaluation']['regression_data_sizes']}")

        # Check for learning rate scheduler
        lr_scheduler = (
            config.get("training", {}).get("standalone_dnn", {}).get("lr_scheduler")
        )
        if lr_scheduler:
            logger.info(
                f"  LR Scheduler: {lr_scheduler.get('type', 'none')} "
                f"(factor={lr_scheduler.get('factor', 'N/A')}, "
                f"patience={lr_scheduler.get('patience', 'N/A')})"
            )

        # Run pipeline
        start_time = time.time()
        success = pipeline.run_complete_pipeline(
            config, delete_catalogs=delete_catalogs
        )
        elapsed_time = time.time() - start_time

        if success:
            logger.info(
                f"✅ Configuration {config_path.name} completed successfully in {elapsed_time:.1f}s"
            )
        else:
            logger.error(
                f"❌ Configuration {config_path.name} failed after {elapsed_time:.1f}s"
            )

        return success

    except Exception as e:
        logger.error(
            f"❌ Failed to process {config_path.name}: {type(e).__name__}: {str(e)}"
        )
        logger.exception("Detailed traceback:")
        return False


def find_config_files(config_stack_dir: Path) -> list[Path]:
    """
    Find all YAML configuration files in the config stack directory.

    Args:
        config_stack_dir: Directory containing configuration files

    Returns:
        List of configuration file paths, sorted by modification time
    """
    yaml_patterns = ["*.yaml", "*.yml"]
    config_files = []

    for pattern in yaml_patterns:
        config_files.extend(config_stack_dir.glob(pattern))

    # Sort by modification time (oldest first)
    config_files.sort(key=lambda x: x.stat().st_mtime)

    return config_files


def run_config_stack(
    config_stack_dir: Path, pipeline: StandaloneRegressionPipeline, logger
) -> tuple:
    """
    Process all configuration files in the config stack directory.

    Args:
        config_stack_dir: Directory containing configuration files
        pipeline: Pipeline instance
        logger: Logger instance

    Returns:
        Tuple of (successful_count, total_count)
    """
    # Find configuration files
    config_files = find_config_files(config_stack_dir)

    if not config_files:
        logger.warning(f"No configuration files found in {config_stack_dir}")
        return 0, 0

    logger.info(f"Found {len(config_files)} configuration files to process:")
    for i, config_file in enumerate(config_files, 1):
        logger.info(f"  {i}. {config_file.name}")

    # Process each configuration
    successful_count = 0
    total_count = len(config_files)

    for config_file in config_files:
        success = run_single_config(config_file, pipeline, logger)
        if success:
            successful_count += 1
            # Remove processed config file (like the original pipeline)
            try:
                config_file.unlink()
                logger.info(f"Removed processed config file: {config_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove config file {config_file.name}: {e}")

    return successful_count, total_count


def create_example_config(output_path: Path, logger) -> None:
    """
    Create an example configuration file.

    Args:
        output_path: Path where to save the example config
        logger: Logger instance
    """
    example_config = {
        "name": "example_standalone_regression",
        "description": "Example configuration for standalone DNN regression",
        "version": "1.0",
        "created_by": "user",
        "dataset": {
            "run_numbers": ["00298967", "00311481"],
            "signal_keys": ["zprime_tt", "wprime_taunu"],
            "catalog_limit": 5,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "shuffle_buffer": 10000,
            "plot_distributions": True,
            "include_labels": True,
        },
        "task": {
            "event_filters": {},
            "input_features": [],
            "input_array_aggregators": [
                {
                    "input_branches": [
                        "InDetTrackParticlesAuxDyn.d0",
                        "InDetTrackParticlesAuxDyn.z0",
                        "InDetTrackParticlesAuxDyn.phi",
                        "derived.InDetTrackParticlesAuxDyn.eta",
                        "derived.InDetTrackParticlesAuxDyn.pt",
                        "derived.InDetTrackParticlesAuxDyn.reducedChiSquared",
                    ],
                    "filter_branches": [
                        {
                            "branch": "InDetTrackParticlesAuxDyn.d0",
                            "min": -5.0,
                            "max": 5.0,
                        },
                        {
                            "branch": "InDetTrackParticlesAuxDyn.z0",
                            "min": -100.0,
                            "max": 100.0,
                        },
                    ],
                    "sort_by_branch": {"branch": "InDetTrackParticlesAuxDyn.qOverP"},
                    "min_length": 5,
                    "max_length": 20,
                }
            ],
            "label_features": [[]],
            "label_array_aggregators": [
                [
                    {
                        "input_branches": ["AnalysisJetsAuxDyn.pt"],
                        "filter_branches": [
                            {"branch": "AnalysisJetsAuxDyn.pt", "min": 20.0}
                        ],
                        "sort_by_branch": {"branch": "AnalysisJetsAuxDyn.pt"},
                        "min_length": 1,
                        "max_length": 1,
                    }
                ]
            ],
        },
        "models": {
            "standalone_dnn": {
                "model_type": "standalone_dnn_regressor",
                "architecture": {
                    "hidden_layers": [64, 32, 16],
                    "activation": "relu",
                    "output_activation": "linear",
                    "name": "example_regressor",
                },
                "hyperparameters": {
                    "dropout_rate": 0.2,
                    "l2_regularization": 0.001,
                    "batch_normalization": True,
                },
            }
        },
        "training": {
            "standalone_dnn": {
                "batch_size": 512,
                "learning_rate": 0.001,
                "epochs": 50,
                "early_stopping": {"patience": 15, "min_delta": 0.0001},
                "gradient_clip_norm": 1.0,
                "lr_scheduler": {
                    "type": "reduce_on_plateau",
                    "monitor": "val_loss",
                    "factor": 0.5,
                    "patience": 8,
                    "min_lr": 1e-6,
                    "verbose": True,
                },
                "plot_training": True,
            }
        },
        "evaluation": {
            "regression_data_sizes": [1000, 2000, 5000],
            "fixed_epochs": 30,
            "create_detailed_plots": True,
            "include_feature_importance": False,
            "error_analysis_bins": 50,
            "prediction_sample_size": 1000,
        },
        "pipeline": {
            "delete_catalogs": False,
            "experiments_output_dir": "_standalone_experiments",
        },
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(example_config, f, default_flow_style=False, indent=2)
        logger.info(f"Example configuration created at: {output_path}")
        logger.info("You can copy and modify this file for your experiments")
    except Exception as e:
        logger.error(f"Failed to create example configuration: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run standalone DNN regression experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a specific configuration file
  python scripts/run_standalone_regression.py --config my_config.yaml

  # Process all configs in the stack directory
  python scripts/run_standalone_regression.py --config-stack

  # Create an example configuration file
  python scripts/run_standalone_regression.py --create-example example.yaml

  # Use custom directories
  python scripts/run_standalone_regression.py --config my_config.yaml \\
    --experiments-dir custom_experiments --datasets-dir custom_datasets
        """,
    )

    # Configuration options
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config", "-c", type=Path, help="Path to YAML configuration file"
    )
    config_group.add_argument(
        "--config-stack",
        "-s",
        action="store_true",
        help="Process all configuration files in _experiment_config_stack/",
    )
    config_group.add_argument(
        "--create-example",
        type=Path,
        help="Create an example configuration file at the specified path",
    )

    # Directory options
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("_standalone_experiments"),
        help="Directory for experiment outputs (default: _standalone_experiments)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("_processed_datasets"),
        help="Directory for processed datasets (default: _processed_datasets)",
    )
    parser.add_argument(
        "--config-stack-dir",
        type=Path,
        default=Path("_experiment_config_stack"),
        help="Directory containing configuration files (default: _experiment_config_stack)",
    )

    # Logging options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    if args.verbose:
        logger.info("Verbose logging enabled")

    # Handle create-example command
    if args.create_example:
        create_example_config(args.create_example, logger)
        return

    # Create pipeline
    try:
        pipeline = StandaloneRegressionPipeline(
            processed_datasets_dir=args.datasets_dir,
            experiments_output_dir=args.experiments_dir,
        )
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Run based on mode
    start_time = time.time()

    if args.config:
        # Single configuration mode
        success = run_single_config(args.config, pipeline, logger)

        if success:
            logger.info("🎉 Standalone regression completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Standalone regression failed!")
            sys.exit(1)

    elif args.config_stack:
        # Config stack mode
        successful_count, total_count = run_config_stack(
            args.config_stack_dir, pipeline, logger
        )
        elapsed_time = time.time() - start_time

        logger.info("=" * 100)
        logger.info("STANDALONE REGRESSION BATCH PROCESSING SUMMARY")
        logger.info("=" * 100)
        logger.info(f"Total configurations processed: {total_count}")
        logger.info(f"Successful: {successful_count}")
        logger.info(f"Failed: {total_count - successful_count}")
        logger.info(f"Total time: {elapsed_time:.1f} seconds")

        if successful_count == total_count:
            logger.info("🎉 All configurations completed successfully!")
            sys.exit(0)
        elif successful_count > 0:
            logger.warning("⚠️  Some configurations failed, but others succeeded")
            sys.exit(0)
        else:
            logger.error("💥 All configurations failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
