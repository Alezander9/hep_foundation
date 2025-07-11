#!/usr/bin/env python3
"""
Utility script to create training history plots from saved JSON files.

This script demonstrates the new flexible training history plotting system.
Instead of plotting during training, we save training histories to JSON files
and then create plots by combining multiple training runs as needed.

Usage examples:
    # Plot training histories from a directory
    python scripts/plot_training_histories.py --input_dir _foundation_experiments/exp_001/training/

    # Plot specific training history files with custom labels
    python scripts/plot_training_histories.py \
        --input_files path1.json path2.json path3.json \
        --legend_labels "From Scratch" "Fine-Tuned" "Fixed Encoder" \
        --output plots/comparison_plot.png

    # Plot regression evaluation histories
    python scripts/plot_training_histories.py \
        --input_dir _foundation_experiments/exp_001/testing/regression_evaluation/training_histories/ \
        --output regression_training_comparison.png \
        --title "Regression Model Training Comparison"
"""

import argparse
import sys
from pathlib import Path

# Add src to the path to import hep_foundation modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.dataset_visualizer import (
    create_training_history_plot_from_json,
)


def find_training_history_files(directory: Path) -> list[Path]:
    """Find all training history JSON files in a directory."""
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    json_files = list(directory.glob("training_history_*.json"))
    if not json_files:
        raise ValueError(f"No training history JSON files found in: {directory}")

    # Sort by filename for consistent ordering
    return sorted(json_files)


def main():
    parser = argparse.ArgumentParser(
        description="Create training history plots from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing training history JSON files",
    )
    input_group.add_argument(
        "--input_files",
        nargs="+",
        type=Path,
        help="Specific training history JSON files to plot",
    )

    # Output and formatting options
    parser.add_argument(
        "--output",
        type=Path,
        default="training_history_comparison.png",
        help="Output PNG file path (default: training_history_comparison.png)",
    )
    parser.add_argument(
        "--legend_labels",
        nargs="+",
        help="Custom legend labels (must match number of input files)",
    )
    parser.add_argument(
        "--title",
        default="Training History Comparison",
        help="Plot title (default: Training History Comparison)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Specific metrics to plot (default: all loss metrics)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = get_logger(__name__)

    try:
        # Get input files
        if args.input_dir:
            logger.info(f"Finding training history files in: {args.input_dir}")
            input_files = find_training_history_files(args.input_dir)
            logger.info(f"Found {len(input_files)} training history files")
            for file in input_files:
                logger.info(f"  - {file.name}")
        else:
            input_files = args.input_files
            logger.info(f"Using {len(input_files)} specified training history files")

            # Verify files exist
            for file in input_files:
                if not file.exists():
                    raise ValueError(f"Training history file does not exist: {file}")

        # Validate legend labels if provided
        if args.legend_labels and len(args.legend_labels) != len(input_files):
            raise ValueError(
                f"Number of legend labels ({len(args.legend_labels)}) "
                f"must match number of input files ({len(input_files)})"
            )

        # Create the plot
        logger.info(f"Creating training history plot: {args.output}")
        create_training_history_plot_from_json(
            training_history_json_paths=input_files,
            output_plot_path=args.output,
            legend_labels=args.legend_labels,
            title_prefix=args.title,
            metrics_to_plot=args.metrics,
        )

        logger.info(f"Training history plot saved successfully to: {args.output}")

    except Exception as e:
        logger.error(f"Failed to create training history plot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
