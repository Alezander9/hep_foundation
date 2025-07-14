import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from hep_foundation.config.config_loader import load_pipeline_config
from hep_foundation.training.foundation_model_pipeline import FoundationModelPipeline

# Expected file structure after full pipeline completion
# This serves as documentation and validation for pipeline outputs

# Root level structure (_test_results/)
EXPECTED_ROOT_STRUCTURE = {
    "_processed_datasets": {
        "type": "directory",
        "required": True,
        "description": "Processed datasets storage",
    },
    "test_foundation_experiments": {
        "type": "directory",
        "required": True,
        "description": "Foundation model experiments",
    },
}

# Pipeline output root structure (_test_results/test_foundation_experiments/)
EXPECTED_PIPELINE_ROOT_STRUCTURE = {
    "model_index.json": {
        "type": "file",
        "required": True,
        "description": "Model registry index file",
    },
    "test_logs": {
        "type": "directory",
        "required": True,
        "description": "Test execution logs",
    },
    "*_Foundation_VAE_Model": {
        "type": "pattern",
        "required": True,
        "description": "Individual experiment directories",
    },
}

# Individual experiment structure (_test_results/test_foundation_experiments/001_Foundation_VAE_Model/)
EXPECTED_EXPERIMENT_STRUCTURE = {
    # Root experiment files - essential metadata
    "_experiment_config.yaml": {
        "type": "file",
        "required": True,
        "description": "Experiment configuration copy",
    },
    "_experiment_info.json": {
        "type": "file",
        "required": True,
        "description": "Experiment metadata and status",
    },
    # Foundation model training outputs
    "training": {
        "type": "directory",
        "required": True,
        "description": "Foundation model training artifacts",
        "contents": {
            "training_history_*.json": {
                "type": "pattern",
                "required": True,
                "description": "Training history JSON files",
            },
            "training_history.csv": {
                "type": "file",
                "required": True,
                "description": "Training history CSV file",
            },
            "training_history.png": {
                "type": "file",
                "required": True,
                "description": "Training history plot",
            },
            "final_metrics.json": {
                "type": "file",
                "required": True,
                "description": "Final training metrics",
            },
        },
    },
    # Trained model artifacts
    "models": {
        "type": "directory",
        "required": True,
        "description": "Trained model files",
        "contents": {
            "foundation_model": {
                "type": "directory",
                "required": True,
                "description": "Foundation model components",
                "contents": {
                    "model_info.json": {
                        "type": "file",
                        "required": True,
                        "description": "Model metadata",
                    },
                    "encoder": {
                        "type": "directory",
                        "required": True,
                        "description": "Encoder model files",
                    },
                    "deterministic_encoder": {
                        "type": "directory",
                        "required": True,
                        "description": "Deterministic encoder model files",
                    },
                    "decoder": {
                        "type": "directory",
                        "required": True,
                        "description": "Decoder model files",
                    },
                    "full_model": {
                        "type": "directory",
                        "required": True,
                        "description": "Full VAE model files",
                    },
                },
            }
        },
    },
    # Evaluation results
    "testing": {
        "type": "directory",
        "required": True,
        "description": "Evaluation results",
        "contents": {
            "anomaly_detection": {
                "type": "directory",
                "required": True,
                "description": "Anomaly detection evaluation results",
                "contents": {
                    "plot_data": {
                        "type": "directory",
                        "required": True,
                        "description": "Plot data for anomaly detection",
                        "contents": {
                            "loss_bin_edges_metadata.json": {
                                "type": "file",
                                "required": True,
                                "description": "Loss histogram bin edges",
                            },
                            "*.json": {
                                "type": "pattern",
                                "required": False,
                                "description": "Other plot data files",
                            },
                        },
                    },
                    "plots": {
                        "type": "directory",
                        "required": True,
                        "description": "Anomaly detection plots",
                    },
                },
            },
            "regression_evaluation": {
                "type": "directory",
                "required": True,
                "description": "Regression evaluation results",
                "contents": {
                    "regression_data_efficiency_plot.png": {
                        "type": "file",
                        "required": True,
                        "description": "Data efficiency plot",
                    },
                    "regression_data_efficiency_results.json": {
                        "type": "file",
                        "required": True,
                        "description": "Data efficiency results",
                    },
                    "regression_training_comparison.png": {
                        "type": "file",
                        "required": True,
                        "description": "Training comparison plot",
                    },
                    "training_histories": {
                        "type": "directory",
                        "required": True,
                        "description": "Training histories for regression models",
                    },
                },
            },
            "signal_classification": {
                "type": "directory",
                "required": True,
                "description": "Signal classification evaluation results",
                "contents": {
                    "signal_classification_accuracy_plot.png": {
                        "type": "file",
                        "required": True,
                        "description": "Classification accuracy plot",
                    },
                    "signal_classification_data_efficiency_results.json": {
                        "type": "file",
                        "required": True,
                        "description": "Classification data efficiency results",
                    },
                    "signal_classification_loss_plot.png": {
                        "type": "file",
                        "required": True,
                        "description": "Classification loss plot",
                    },
                    "signal_classification_training_comparison.png": {
                        "type": "file",
                        "required": True,
                        "description": "Classification training comparison plot",
                    },
                    "training_histories": {
                        "type": "directory",
                        "required": True,
                        "description": "Training histories for classification models",
                    },
                },
            },
        },
    },
}

# Dataset structure (_test_results/_processed_datasets/dataset_*/)
EXPECTED_DATASET_STRUCTURE = {
    "_dataset_config.yaml": {
        "type": "file",
        "required": True,
        "description": "Dataset configuration",
    },
    "_dataset_info.json": {
        "type": "file",
        "required": True,
        "description": "Dataset metadata",
    },
    "dataset.h5": {
        "type": "file",
        "required": True,
        "description": "Main dataset file",
    },
    "signal_dataset.h5": {
        "type": "file",
        "required": False,
        "description": "Signal dataset file (if signal keys specified)",
    },
    "plot_data": {
        "type": "directory",
        "required": False,
        "description": "Dataset visualization data",
        "contents": {
            "atlas_dataset_features_hist_data.json": {
                "type": "file",
                "required": False,
                "description": "Background histogram data",
            },
            "background_bin_edges_metadata.json": {
                "type": "file",
                "required": False,
                "description": "Background bin edges",
            },
            "*_dataset_features_hist_data.json": {
                "type": "pattern",
                "required": False,
                "description": "Signal histogram data files",
            },
        },
    },
    "plots": {
        "type": "directory",
        "required": False,
        "description": "Dataset visualization plots",
        "contents": {
            "comparison_input_features_background_vs_signals.png": {
                "type": "file",
                "required": False,
                "description": "Feature comparison plot",
            },
        },
    },
}


def validate_experiment_structure(
    experiment_dir: Path, expected_structure: dict[str, Any], path_context: str = ""
) -> tuple[list[str], list[str]]:
    """
    Validate that an experiment directory matches the expected structure.

    Args:
        experiment_dir: Path to the experiment directory
        expected_structure: Expected structure definition
        path_context: Current path context for error messages

    Returns:
        Tuple of (missing_items, unexpected_items)
    """
    missing_items = []
    unexpected_items = []

    if not experiment_dir.exists():
        missing_items.append(f"Directory not found: {path_context or experiment_dir}")
        return missing_items, unexpected_items

    # Track all expected paths for later comparison
    expected_paths = set()

    for name, spec in expected_structure.items():
        current_path = f"{path_context}/{name}" if path_context else name

        if spec["type"] == "file":
            file_path = experiment_dir / name
            expected_paths.add(name)

            if spec["required"] and not file_path.exists():
                missing_items.append(
                    f"Required file missing: {current_path} - {spec['description']}"
                )

        elif spec["type"] == "directory":
            dir_path = experiment_dir / name
            expected_paths.add(name)

            if spec["required"] and not dir_path.exists():
                missing_items.append(
                    f"Required directory missing: {current_path} - {spec['description']}"
                )
            elif dir_path.exists() and "contents" in spec:
                # Recursively validate directory contents
                sub_missing, sub_unexpected = validate_experiment_structure(
                    dir_path, spec["contents"], current_path
                )
                missing_items.extend(sub_missing)
                unexpected_items.extend(sub_unexpected)

        elif spec["type"] == "pattern":
            matching_files = list(experiment_dir.glob(name))

            # For patterns, we expect the parent directory to contain files matching the pattern
            if spec["required"]:
                min_count = spec.get("min_count", 1)
                if len(matching_files) < min_count:
                    missing_items.append(
                        f"Required pattern missing: {current_path} - expected at least {min_count}, found {len(matching_files)} - {spec['description']}"
                    )

            # Add matched files to expected paths (with just the filename)
            for matched_file in matching_files:
                expected_paths.add(matched_file.name)

    # Find unexpected files/directories in this level
    if experiment_dir.exists():
        actual_items = set(item.name for item in experiment_dir.iterdir())
        unexpected_in_this_dir = actual_items - expected_paths

        for unexpected_item in unexpected_in_this_dir:
            item_path = experiment_dir / unexpected_item
            current_path = (
                f"{path_context}/{unexpected_item}" if path_context else unexpected_item
            )

            if item_path.is_file():
                unexpected_items.append(f"Unexpected file: {current_path}")
            elif item_path.is_dir():
                unexpected_items.append(f"Unexpected directory: {current_path}")

    return missing_items, unexpected_items


def assert_complete_pipeline_structure(base_dir: Path, logger: logging.Logger) -> None:
    """
    Assert that the complete pipeline output structure contains all expected files.

    This validates four levels:
    1. Root level (_test_results/)
    2. Pipeline level (_test_results/test_foundation_experiments/)
    3. Individual experiment level (_test_results/test_foundation_experiments/001_Foundation_VAE_Model/)
    4. Dataset level (_test_results/_processed_datasets/dataset_*/)

    Args:
        base_dir: Path to the _test_results directory
        logger: Logger for warnings
    """
    logger.info(f"Validating complete pipeline structure: {base_dir}")

    all_missing = []
    all_unexpected = []

    # 1. Validate root level (_test_results/)
    logger.info(f"Validating root structure: {base_dir}")
    root_missing, root_unexpected = validate_experiment_structure(
        base_dir, EXPECTED_ROOT_STRUCTURE
    )
    all_missing.extend([f"(Root) {item}" for item in root_missing])
    all_unexpected.extend([f"(Root) {item}" for item in root_unexpected])

    # 2. Validate pipeline level (_test_results/test_foundation_experiments/)
    pipeline_root = base_dir / "test_foundation_experiments"
    if pipeline_root.exists():
        logger.info(f"Validating pipeline root structure: {pipeline_root}")
        pipeline_missing, pipeline_unexpected = validate_experiment_structure(
            pipeline_root, EXPECTED_PIPELINE_ROOT_STRUCTURE
        )
        all_missing.extend([f"(Pipeline) {item}" for item in pipeline_missing])
        all_unexpected.extend([f"(Pipeline) {item}" for item in pipeline_unexpected])

        # 3. Validate individual experiment directories
        experiment_dirs = [
            d
            for d in pipeline_root.iterdir()
            if d.is_dir() and d.name.endswith("_Foundation_VAE_Model")
        ]
        if experiment_dirs:
            for exp_dir in experiment_dirs:
                logger.info(f"Validating experiment structure: {exp_dir}")
                exp_missing, exp_unexpected = validate_experiment_structure(
                    exp_dir, EXPECTED_EXPERIMENT_STRUCTURE
                )
                all_missing.extend(
                    [f"(Experiment {exp_dir.name}) {item}" for item in exp_missing]
                )
                all_unexpected.extend(
                    [f"(Experiment {exp_dir.name}) {item}" for item in exp_unexpected]
                )
        else:
            all_missing.append("(Pipeline) No experiment directories found")
    else:
        all_missing.append("(Root) test_foundation_experiments/ directory not found")

    # 4. Validate dataset directories
    datasets_root = base_dir / "_processed_datasets"
    if datasets_root.exists():
        logger.info(f"Validating datasets structure: {datasets_root}")
        dataset_dirs = [
            d
            for d in datasets_root.iterdir()
            if d.is_dir() and d.name.startswith("dataset_")
        ]
        if dataset_dirs:
            for dataset_dir in dataset_dirs:
                logger.info(f"Validating dataset structure: {dataset_dir}")
                dataset_missing, dataset_unexpected = validate_experiment_structure(
                    dataset_dir, EXPECTED_DATASET_STRUCTURE
                )
                all_missing.extend(
                    [f"(Dataset {dataset_dir.name}) {item}" for item in dataset_missing]
                )
                all_unexpected.extend(
                    [
                        f"(Dataset {dataset_dir.name}) {item}"
                        for item in dataset_unexpected
                    ]
                )
        else:
            all_missing.append("(Datasets) No dataset directories found")
    else:
        all_missing.append("(Root) _processed_datasets/ directory not found")

    # Report missing items as test failures
    if all_missing:
        missing_report = "\n".join(f"  - {item}" for item in all_missing)
        raise AssertionError(
            f"Missing required pipeline outputs:\n{missing_report}\n\nThis indicates a regression in the pipeline or missing functionality."
        )

    # Report unexpected items as warnings (not failures)
    if all_unexpected:
        unexpected_report = "\n".join(f"  - {item}" for item in all_unexpected)
        logger.warning("\n" + "=" * 80)
        logger.warning("UNEXPECTED FILES FOUND IN PIPELINE OUTPUT")
        logger.warning("=" * 80)
        logger.warning(f"New files detected:\n{unexpected_report}")
        logger.warning("")
        logger.warning("If these are intentional additions to the pipeline:")
        logger.warning(
            "1. Add them to the appropriate EXPECTED_*_STRUCTURE in tests/test_pipeline.py"
        )
        logger.warning("2. Include appropriate descriptions for documentation")
        logger.warning("3. Mark as required=True if they should always be present")
        logger.warning("")
        logger.warning("This ensures we maintain awareness of all pipeline outputs")
        logger.warning("and can catch regressions when expected files go missing.")
        logger.warning("=" * 80)
    else:
        logger.info("✓ All pipeline outputs match expected structure")


def create_test_configs():
    """Load test configurations from YAML file."""
    # Get the path to the test config file
    test_config_path = Path(__file__).parent / "_test_pipeline_config.yaml"

    # Load configuration using the new config loader
    config = load_pipeline_config(test_config_path)

    # Return the config objects and source file path
    return {
        "dataset_config": config["dataset_config"],
        "vae_model_config": config["vae_model_config"],
        "dnn_model_config": config["dnn_model_config"],
        "vae_training_config": config["vae_training_config"],
        "dnn_training_config": config["dnn_training_config"],
        "task_config": config["task_config"],
        "source_config_file": config.get("_source_config_file"),
    }


@pytest.fixture(scope="session")
def experiment_dir():
    """Create a temporary experiment directory for the test run."""
    # Define a fixed directory for test results
    base_dir = Path.cwd() / "_test_results"
    experiments_dir = base_dir / "test_foundation_experiments"

    # Delete the directory if it exists from a previous run
    if base_dir.exists():
        shutil.rmtree(base_dir)

    # Create the main test directory and the logs subdirectory
    experiments_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiments_dir / "test_logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\nTest results will be stored in: {experiments_dir.absolute()}\n")

    yield str(experiments_dir)  # Convert to string for compatibility

    # No cleanup here, so results persist after the test run.
    # The directory will be cleaned up at the start of the next test session.


@pytest.fixture(scope="module")
def test_configs():
    return create_test_configs()


@pytest.fixture(scope="module")
def pipeline(experiment_dir):
    experiments_output_path = Path(experiment_dir)
    processed_data_parent = experiments_output_path.parent

    return FoundationModelPipeline(
        experiments_output_dir=str(experiments_output_path),
        processed_data_parent_dir=str(processed_data_parent),
    )


@pytest.fixture(scope="session", autouse=True)
def setup_logging(experiment_dir):
    """Configure logging for all tests"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(experiment_dir) / "test_logs"
    log_dir.mkdir(exist_ok=True)

    # Configure root logger to write to both file and console
    log_file = log_dir / f"test_run_{timestamp}.log"

    # Console handler with custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    yield  # Let the tests run

    # Cleanup handlers
    root_logger.removeHandler(console_handler)
    root_logger.removeHandler(file_handler)


def test_run_full_pipeline(pipeline, test_configs, experiment_dir):
    """Test the full pipeline (train → regression → anomaly)"""
    logger = logging.getLogger(__name__)

    logger.info("Starting full pipeline test")
    try:
        # Load test config to get evaluation settings
        test_config_path = Path(__file__).parent / "_test_pipeline_config.yaml"
        full_config = load_pipeline_config(test_config_path)
        evaluation_config = full_config.get("evaluation_config")

        # Set the source config file for reproducibility
        if test_configs.get("source_config_file"):
            pipeline.set_source_config_file(test_configs["source_config_file"])

        result = pipeline.run_full_pipeline(
            dataset_config=test_configs["dataset_config"],
            task_config=test_configs["task_config"],
            vae_model_config=test_configs["vae_model_config"],
            dnn_model_config=test_configs["dnn_model_config"],
            vae_training_config=test_configs["vae_training_config"],
            dnn_training_config=test_configs["dnn_training_config"],
            delete_catalogs=False,
            data_sizes=evaluation_config.regression_data_sizes
            if evaluation_config
            else [1000, 2000],
            fixed_epochs=evaluation_config.fixed_epochs if evaluation_config else 3,
        )

        assert result is True, "Pipeline should return True on success"

        # Validate the complete pipeline structure
        # experiment_dir is the path to the experiments directory (_test_results/test_foundation_experiments)
        # so base_dir is its parent (_test_results)
        base_dir = Path(experiment_dir).parent
        logger.info(f"Validating complete pipeline structure in: {base_dir}")
        assert_complete_pipeline_structure(base_dir, logger)

        logger.info(
            f"Full pipeline test completed successfully. Results in: {experiment_dir}"
        )

    except Exception as e:
        logger.error(f"Full pipeline test failed: {e}")
        raise
