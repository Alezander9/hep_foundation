import shutil
import tempfile
from pathlib import Path

import pytest

from hep_foundation.training.foundation_model_pipeline import FoundationModelPipeline
from hep_foundation.data.task_config import TaskConfig
from hep_foundation.data.dataset_manager import DatasetConfig
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
                ],
                "filter_branches": [],
                "sort_by_branch": None,
                "min_length": 2,
                "max_length": 2,
            }
        ],
        label_features=[[]],
        label_array_aggregators=[
            [
                {
                    "input_branches": [
                        "MET_Core_AnalysisMETAuxDyn.mpx",
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

    dataset_config = DatasetConfig(
        run_numbers=run_numbers,
        signal_keys=["zprime"],  # Use a single signal for speed
        catalog_limit=2,
        validation_fraction=0.1,
        test_fraction=0.1,
        shuffle_buffer=100,
        plot_distributions=False,
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
        epochs=1,
        early_stopping_patience=2,
        early_stopping_min_delta=1e-3,
        plot_training=False,
    )

    dnn_training_config = TrainingConfig(
        batch_size=1024,
        learning_rate=0.001,
        epochs=1,
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

@pytest.fixture(scope="module")
def experiment_dir():
    """Create a temporary experiment directory for the test run."""
    tmpdir = tempfile.mkdtemp(prefix="test_foundation_experiments_")
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture(scope="module")
def test_configs():
    return create_test_configs()

@pytest.fixture(scope="module")
def pipeline(experiment_dir):
    # Use the temp dir as the base_dir for the pipeline
    return FoundationModelPipeline(base_dir=experiment_dir)

def test_train_foundation_model(pipeline, test_configs, experiment_dir):
    experiment_name = "pytest_foundation_test"
    result = pipeline.train_foundation_model(
        dataset_config=test_configs["dataset_config"],
        model_config=test_configs["vae_model_config"],
        training_config=test_configs["vae_training_config"],
        task_config=test_configs["task_config"],
        experiment_name=experiment_name,
        experiment_description="pytest train test",
        delete_catalogs=True,
    )
    assert result is True
    experiment_path = Path(experiment_dir) / experiment_name
    assert experiment_path.exists(), f"Experiment dir {experiment_path} not found"

def test_evaluate_foundation_model_anomaly_detection(pipeline, test_configs, experiment_dir):
    experiment_name = "pytest_foundation_test"
    foundation_model_path = str(Path(experiment_dir) / experiment_name)
    result = pipeline.evaluate_foundation_model_anomaly_detection(
        dataset_config=test_configs["dataset_config"],
        task_config=test_configs["task_config"],
        experiment_name="pytest_anomaly",
        experiment_description="pytest anomaly test",
        delete_catalogs=True,
        foundation_model_path=foundation_model_path,
        vae_training_config=test_configs["vae_training_config"],
    )
    assert result is True
    anomaly_dir = Path(foundation_model_path) / "testing" / "anomaly_detection"
    assert anomaly_dir.exists(), f"Anomaly detection dir {anomaly_dir} not found"

def test_evaluate_foundation_model_regression(pipeline, test_configs, experiment_dir):
    experiment_name = "pytest_foundation_test"
    foundation_model_path = str(Path(experiment_dir) / experiment_name)
    result = pipeline.evaluate_foundation_model_regression(
        dataset_config=test_configs["dataset_config"],
        dnn_model_config=test_configs["dnn_model_config"],
        dnn_training_config=test_configs["dnn_training_config"],
        task_config=test_configs["task_config"],
        experiment_name="pytest_regression",
        experiment_description="pytest regression test",
        delete_catalogs=True,
        foundation_model_path=foundation_model_path,
    )
    assert result is True
    regression_plot = Path(foundation_model_path) / "testing" / "regression_training_comparison.pdf"
    assert regression_plot.exists(), f"Regression plot {regression_plot} not found"
