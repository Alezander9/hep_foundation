import shutil
import tempfile
from pathlib import Path
import logging
import os
from datetime import datetime

import pytest

from hep_foundation.training.foundation_model_pipeline import FoundationModelPipeline
from hep_foundation.config.config_loader import load_pipeline_config


def create_test_configs():
    """Load test configurations from YAML file."""
    # Get the path to the test config file
    test_config_path = Path(__file__).parent / "test_pipeline_config.yaml"
    
    # Load configuration using the new config loader
    config = load_pipeline_config(test_config_path)
    
    # Return the config objects in the same format as before
    return {
        "dataset_config": config["dataset_config"],
        "vae_model_config": config["vae_model_config"],
        "dnn_model_config": config["dnn_model_config"],
        "vae_training_config": config["vae_training_config"],
        "dnn_training_config": config["dnn_training_config"],
        "task_config": config["task_config"],
    }

@pytest.fixture(scope="session")
def experiment_dir():
    """Create a temporary experiment directory for the test run."""
    # Define a fixed directory for test results
    base_dir = Path.cwd() / "test_results"
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
    # The experiment_dir fixture already creates test_results/test_foundation_experiments
    # and yields str(test_results/test_foundation_experiments)
    # We want processed_datasets to be under test_results/, not test_foundation_experiments/
    # So, the parent of experiment_dir is test_results/
    
    experiments_output_path = Path(experiment_dir) # This is test_results/test_foundation_experiments
    processed_data_parent = experiments_output_path.parent # This is test_results/

    return FoundationModelPipeline(
        experiments_output_dir=str(experiments_output_path),
        processed_data_parent_dir=str(processed_data_parent)
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
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
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
        test_config_path = Path(__file__).parent / "test_pipeline_config.yaml"
        full_config = load_pipeline_config(test_config_path)
        evaluation_config = full_config.get('evaluation_config')
        
        result = pipeline.run_full_pipeline(
            dataset_config=test_configs["dataset_config"],
            task_config=test_configs["task_config"],
            vae_model_config=test_configs["vae_model_config"],
            dnn_model_config=test_configs["dnn_model_config"],
            vae_training_config=test_configs["vae_training_config"],
            dnn_training_config=test_configs["dnn_training_config"],
            delete_catalogs=False,
            data_sizes=evaluation_config.regression_data_sizes if evaluation_config else [1000, 2000],
            fixed_epochs=evaluation_config.fixed_epochs if evaluation_config else 3,
        )
        
        assert result is True, "Full pipeline should return True on success"
        
        # Find the model that was created by the full pipeline
        experiment_dirs = list(Path(experiment_dir).glob("*"))
        model_dirs = [d for d in experiment_dirs if d.is_dir() and (d / "experiment_data.json").exists()]
        
        # Should have at least one model (could be more if previous tests ran)
        assert len(model_dirs) > 0, "No trained foundation model found after full pipeline"
        
        # Find the most recently created model directory (full pipeline should create a new one)
        latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
        
        # Verify all expected outputs exist
        experiment_data_path = latest_model_dir / "experiment_data.json"
        assert experiment_data_path.exists(), f"Experiment data file not found at {experiment_data_path}"
        
        # Check anomaly detection outputs
        anomaly_dir = latest_model_dir / "testing" / "anomaly_detection"
        assert anomaly_dir.exists(), f"Anomaly detection dir {anomaly_dir} not found"
        
        # Check regression evaluation outputs
        regression_dir = latest_model_dir / "testing" / "regression_evaluation"
        assert regression_dir.exists(), f"Regression evaluation dir {regression_dir} not found"
        
        results_file = regression_dir / "regression_data_efficiency_results.json"
        assert results_file.exists(), f"Regression data efficiency results file {results_file} not found"
        
        plot_file = regression_dir / "regression_data_efficiency_plot.png"
        assert plot_file.exists(), f"Regression data efficiency plot {plot_file} not found"
        
        logger.info(f"Full pipeline test completed successfully. Model saved at: {latest_model_dir}")
        
    except Exception as e:
        logger.error(f"Full pipeline test failed: {str(e)}")
        raise
