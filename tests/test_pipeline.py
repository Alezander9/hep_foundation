import shutil
import tempfile
from pathlib import Path
import logging
import os
from datetime import datetime

import pytest

from hep_foundation.data.dataset_manager import DatasetConfig
from hep_foundation.data.task_config import TaskConfig
from hep_foundation.training.foundation_model_pipeline import FoundationModelPipeline
from hep_foundation.training.model_trainer import TrainingConfig


def create_test_configs():
    """Create a minimal config for fast testing."""
    # Minimal task config: 2 features, 1 label
    task_config = TaskConfig.create_from_branch_names(
        event_filter_dict={},
        input_features=[],
        input_array_aggregators=[
            {
                "input_branches": [
                    "InDetTrackParticlesAuxDyn.d0",
                    "InDetTrackParticlesAuxDyn.z0",
                    "InDetTrackParticlesAuxDyn.phi",
                    "derived.InDetTrackParticlesAuxDyn.eta",
                    "derived.InDetTrackParticlesAuxDyn.pt",
                    "derived.InDetTrackParticlesAuxDyn.reducedChiSquared",

                    # "InDetTrackParticlesAuxDyn.definingParametersCovMatrixDiag",
                    # "InDetTrackParticlesAuxDyn.definingParametersCovMatrixOffDiag",
                ],
                "filter_branches": [],
                "sort_by_branch": None,
                "min_length": 10,
                "max_length": 30,
            }
        ],
        label_features=[[]],
        label_array_aggregators=[
            [
                {
                    "input_branches": [
                        "MET_Core_AnalysisMETAuxDyn.mpx",
                        "MET_Core_AnalysisMETAuxDyn.mpy",
                        "MET_Core_AnalysisMETAuxDyn.sumet",
                    ],
                    "filter_branches": [],
                    "sort_by_branch": None,
                    "min_length": 1,
                    "max_length": 1,
                }
            ]
        ],
    )

    # Use a very small number of run numbers for speed
    run_numbers = ["00298967", "00311481"] 
    signal_keys = ["zprime_tt", "zprime_bb"]

    dataset_config = DatasetConfig(
        run_numbers=run_numbers,
        signal_keys=signal_keys,
        catalog_limit=2,
        validation_fraction=0.1,
        test_fraction=0.1,
        shuffle_buffer=100,
        plot_distributions=True,
        include_labels=True,
        task_config=task_config,
    )

    vae_model_config = {
        "model_type": "variational_autoencoder",
        "architecture": {
            "latent_dim": 2,
            "encoder_layers": [8],
            "decoder_layers": [8],
            "activation": "relu",
            "name": "test_vae",
        },
        "hyperparameters": {
            "quant_bits": 8,
            "beta_schedule": {
                "start": 0.01,
                "end": 0.1,
                "warmup_epochs": 1,
                "cycle_epochs": 1,
            },
        },
    }

    dnn_model_config = {
        "model_type": "dnn_predictor",
        "architecture": {
            "hidden_layers": [8],
            "label_index": 0,
            "activation": "relu",
            "output_activation": "linear",
            "name": "test_dnn",
        },
        "hyperparameters": {
            "quant_bits": 8,
            "dropout_rate": 0.1,
            "l2_regularization": 1e-5,
        },
    }

    vae_training_config = TrainingConfig(
        batch_size=1024,
        learning_rate=0.001,
        epochs=10,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-3,
        plot_training=False,
    )

    dnn_training_config = TrainingConfig(
        batch_size=1024,
        learning_rate=0.001,
        epochs=10,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-3,
        plot_training=False,
    )

    return {
        "dataset_config": dataset_config,
        "vae_model_config": vae_model_config,
        "dnn_model_config": dnn_model_config,
        "vae_training_config": vae_training_config,
        "dnn_training_config": dnn_training_config,
        "task_config": task_config,
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

# def test_01_train_foundation_model(pipeline, test_configs, experiment_dir):
#     """Test foundation model training"""
#     logger = logging.getLogger(__name__)
    
#     logger.info("Starting foundation model training test")
#     try:
#         result = pipeline.train_foundation_model(
#             dataset_config=test_configs["dataset_config"],
#             model_config=test_configs["vae_model_config"],
#             training_config=test_configs["vae_training_config"],
#             task_config=test_configs["task_config"],
#             delete_catalogs=False,
#         )
        
#         # Now train_foundation_model returns the model path as a string, not True
#         assert isinstance(result, str), f"Pipeline training should return a string path, got {type(result)}"
#         assert result, "Pipeline training returned empty string or None"
        
#         experiment_path = Path(result)
#         assert experiment_path.exists(), f"Experiment dir {experiment_path} not found"
        
#         # Verify that the experiment_data.json file was created
#         experiment_data_path = experiment_path / "experiment_data.json"
#         assert experiment_data_path.exists(), f"Experiment data file not found at {experiment_data_path}"
        
#         logger.info(f"Foundation model training test completed successfully. Model saved at: {result}")
        
#     except Exception as e:
#         logger.error(f"Foundation model training failed: {str(e)}")
#         raise

# def test_02_evaluate_foundation_model_anomaly_detection(pipeline, test_configs, experiment_dir):
#     # We need to find the model that was trained in test_01
#     # Since the experiment name is generated dynamically, we need to find it
#     experiment_dirs = list(Path(experiment_dir).glob("*"))
#     model_dirs = [d for d in experiment_dirs if d.is_dir() and (d / "experiment_data.json").exists()]
    
#     assert len(model_dirs) > 0, "No trained foundation model found. Run test_01_train_foundation_model first."
#     foundation_model_path = str(model_dirs[0])  # Use the first found model
    
#     result = pipeline.evaluate_foundation_model_anomaly_detection(
#         dataset_config=test_configs["dataset_config"],
#         task_config=test_configs["task_config"],
#         delete_catalogs=False,
#         foundation_model_path=foundation_model_path,
#         vae_training_config=test_configs["vae_training_config"],
#     )
#     assert result is True
#     anomaly_dir = Path(foundation_model_path) / "testing" / "anomaly_detection"
#     assert anomaly_dir.exists(), f"Anomaly detection dir {anomaly_dir} not found"

# def test_03_evaluate_foundation_model_regression(pipeline, test_configs, experiment_dir):
#     # We need to find the model that was trained in test_01
#     # Since the experiment name is generated dynamically, we need to find it
#     experiment_dirs = list(Path(experiment_dir).glob("*"))
#     model_dirs = [d for d in experiment_dirs if d.is_dir() and (d / "experiment_data.json").exists()]
    
#     assert len(model_dirs) > 0, "No trained foundation model found. Run test_01_train_foundation_model first."
#     foundation_model_path = str(model_dirs[0])  # Use the first found model
    
#     # Use smaller data sizes for testing to speed up the process
#     test_data_sizes = [1000, 2000, 5000]
    
#     result = pipeline.evaluate_foundation_model_regression(
#         dataset_config=test_configs["dataset_config"],
#         dnn_model_config=test_configs["dnn_model_config"],
#         dnn_training_config=test_configs["dnn_training_config"],
#         task_config=test_configs["task_config"],
#         delete_catalogs=False,
#         foundation_model_path=foundation_model_path,
#         data_sizes=test_data_sizes,
#         fixed_epochs=3,  # Use fewer epochs for testing
#     )
#     assert result is True
    
#     # Check that the regression evaluation results were created
#     regression_dir = Path(foundation_model_path) / "testing" / "regression_evaluation"
#     assert regression_dir.exists(), f"Regression evaluation dir {regression_dir} not found"
    
#     results_file = regression_dir / "regression_data_efficiency_results.json"
#     assert results_file.exists(), f"Regression data efficiency results file {results_file} not found"
    
#     plot_file = regression_dir / "regression_data_efficiency_plot.png"
#     assert plot_file.exists(), f"Regression data efficiency plot {plot_file} not found"

def test_04_run_full_pipeline(pipeline, test_configs, experiment_dir):
    """Test the full pipeline (train → regression → anomaly)"""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting full pipeline test")
    try:
        # Use smaller data sizes for testing to speed up the process
        test_data_sizes = [1000, 2000]
        
        result = pipeline.run_full_pipeline(
            dataset_config=test_configs["dataset_config"],
            task_config=test_configs["task_config"],
            vae_model_config=test_configs["vae_model_config"],
            dnn_model_config=test_configs["dnn_model_config"],
            vae_training_config=test_configs["vae_training_config"],
            dnn_training_config=test_configs["dnn_training_config"],
            delete_catalogs=False,
            data_sizes=test_data_sizes,
            fixed_epochs=3,  # Use fewer epochs for testing
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
