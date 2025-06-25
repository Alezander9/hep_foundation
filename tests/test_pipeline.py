import logging
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from hep_foundation.config.config_loader import load_pipeline_config
from hep_foundation.training.foundation_model_pipeline import FoundationModelPipeline


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
    test_dir = base_dir / "test_foundation_experiments"

    # Delete the directory if it exists from a previous run
    if base_dir.exists():
        shutil.rmtree(base_dir)

    # Create the main test directory and the logs subdirectory
    test_dir.mkdir(parents=True, exist_ok=True)
    log_dir = test_dir / "test_logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\nTest results will be stored in: {test_dir.absolute()}\n")

    yield str(test_dir)  # Convert to string for compatibility

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

        assert result is True, "Full pipeline should return True on success"

        # Find the model that was created by the full pipeline
        experiment_dirs = list(Path(experiment_dir).glob("*"))
        model_dirs = [
            d
            for d in experiment_dirs
            if d.is_dir() and (d / "_experiment_info.json").exists()
        ]

        # Should have at least one model (could be more if previous tests ran)
        assert len(model_dirs) > 0, (
            "No trained foundation model found after full pipeline"
        )

        # Find the most recently created model directory (full pipeline should create a new one)
        latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)

        # Verify all expected outputs exist (no longer checking for experiment_data.json)

        # Check for the new reproducibility files
        experiment_config_path = latest_model_dir / "_experiment_config.yaml"
        assert experiment_config_path.exists(), (
            f"Experiment config file not found at {experiment_config_path}"
        )

        experiment_info_path = latest_model_dir / "_experiment_info.json"
        assert experiment_info_path.exists(), (
            f"Experiment info file not found at {experiment_info_path}"
        )

        # Check anomaly detection outputs
        anomaly_dir = latest_model_dir / "testing" / "anomaly_detection"
        assert anomaly_dir.exists(), f"Anomaly detection dir {anomaly_dir} not found"

        # Check regression evaluation outputs
        regression_dir = latest_model_dir / "testing" / "regression_evaluation"
        assert regression_dir.exists(), (
            f"Regression evaluation dir {regression_dir} not found"
        )

        results_file = regression_dir / "regression_data_efficiency_results.json"
        assert results_file.exists(), (
            f"Regression data efficiency results file {results_file} not found"
        )

        plot_file = regression_dir / "regression_data_efficiency_plot.png"
        assert plot_file.exists(), (
            f"Regression data efficiency plot {plot_file} not found"
        )

        logger.info(
            f"Full pipeline test completed successfully. Model saved at: {latest_model_dir}"
        )

    except Exception as e:
        logger.error(f"Full pipeline test failed: {str(e)}")
        raise
