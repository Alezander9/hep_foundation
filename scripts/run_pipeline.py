import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from hep_foundation.model_pipeline import DatasetConfig, TrainingConfig
from hep_foundation.task_config import TaskConfig
from hep_foundation.utils import ATLAS_RUN_NUMBERS


def create_configs(model_type: str = "vae") -> Dict[str, Any]:
    """Create configuration objects for the model pipeline"""

    # Create TaskConfig with the new structure
    task_config = TaskConfig.create_from_branch_names(
        # No event filters for now
        event_filter_dict={},
        # Input features - empty list since we only have aggregators
        input_features=[],
        # Input array aggregators with separated selectors and filters
        input_array_aggregators=[
            {
                "input_branches": [
                    "InDetTrackParticlesAuxDyn.d0",
                    "InDetTrackParticlesAuxDyn.z0",
                    "InDetTrackParticlesAuxDyn.phi",
                    "InDetTrackParticlesAuxDyn.theta",
                    "InDetTrackParticlesAuxDyn.qOverP",
                    "InDetTrackParticlesAuxDyn.chiSquared",
                    "InDetTrackParticlesAuxDyn.numberDoF",
                ],
                "filter_branches": [
                    {"branch": "InDetTrackParticlesAuxDyn.d0", "min": -5.0, "max": 5.0},
                    {
                        "branch": "InDetTrackParticlesAuxDyn.z0",
                        "min": -100.0,
                        "max": 100.0,
                    },
                    {"branch": "InDetTrackParticlesAuxDyn.chiSquared", "max": 50.0},
                    {"branch": "InDetTrackParticlesAuxDyn.numberDoF", "min": 1.0},
                ],
                "sort_by_branch": {"branch": "InDetTrackParticlesAuxDyn.qOverP"},
                "min_length": 10,
                "max_length": 30,
            }
        ],
        # Empty label features list
        label_features=[[]],
        # Modified label array aggregator
        label_array_aggregators=[
            [
                {
                    "input_branches": [
                        "MET_Core_AnalysisMETAuxDyn.mpx",
                        "MET_Core_AnalysisMETAuxDyn.mpy",
                        "MET_Core_AnalysisMETAuxDyn.sumet",
                    ],
                    "filter_branches": [],  # Add empty filter list
                    "sort_by_branch": None,  # Explicitly set to None
                    "min_length": 1,
                    "max_length": 1,
                }
            ]
        ],
    )

    # Simplified DatasetConfig without track/event selections
    dataset_config = DatasetConfig(
        run_numbers=ATLAS_RUN_NUMBERS[-3:],
        signal_keys=["zprime", "wprime_qq", "zprime_bb"],
        catalog_limit=20,
        validation_fraction=0.15,
        test_fraction=0.15,
        shuffle_buffer=50000,
        plot_distributions=True,
        include_labels=True,
        task_config=task_config,
    )

    # Model configurations

    vae_model_config = {
        "model_type": "variational_autoencoder",
        "architecture": {
            # Input shape is automatically set in the model pipeline
            "latent_dim": 16,
            "encoder_layers": [128, 64, 32],
            "decoder_layers": [32, 64, 128],
            "activation": "relu",
            "name": "variational_autoencoder",
        },
        "hyperparameters": {
            "quant_bits": 8,
            "beta_schedule": {
                "start": 0.01,
                "end": 0.1,
                "warmup_epochs": 5,
                "cycle_epochs": 5,
            },
        },
    }

    dnn_model_config = {
        "model_type": "dnn_predictor",
        "architecture": {
            # Input and output shapes are automatically set in the model pipeline
            "hidden_layers": [128, 64, 32],
            "label_index": 0,  # Use first label set
            "activation": "relu",
            "output_activation": "linear",
            "name": "met_predictor",
        },
        "hyperparameters": {
            "quant_bits": 8,
            "dropout_rate": 0.2,
            "l2_regularization": 1e-4,
        },
    }

    # Select model configuration based on type parameter
    model_configs = {"vae": vae_model_config, "dnn_predictor": dnn_model_config}

    model_config = model_configs.get(model_type)
    if model_config is None:
        raise ValueError(
            f"Unknown model type: {model_type}. Must be one of {list(model_configs.keys())}"
        )

    # Training configuration - common for all models
    training_config = TrainingConfig(
        batch_size=1024,
        learning_rate=0.001,
        epochs=25,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        plot_training=True,
    )

    return {
        "dataset_config": dataset_config,
        "model_config": model_config,
        "training_config": training_config,
        "task_config": task_config,
    }


def run_pipeline(
    model_type: str = "vae",
    experiment_name: str = None,
    experiment_description: str = None,
    custom_configs: Dict[str, Any] = None,
    delete_catalogs: bool = True,
) -> None:
    """
    Run the model pipeline with specified configurations

    Args:
        model_type: Type of model to run ("vae" or "autoencoder")
        experiment_name: Optional name for the experiment
        experiment_description: Optional description for the experiment
        custom_configs: Optional dictionary of custom configurations to override defaults
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_{timestamp}.log"

    # Get default configurations
    configs = create_configs(model_type)

    # Override with custom configurations if provided
    if custom_configs:
        for key, value in custom_configs.items():
            if key in configs:
                configs[key] = value

    # Run the pipeline
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                [
                    "python",
                    "-c",
                    f"""
import sys
from hep_foundation.model_pipeline import model_pipeline
from scripts.run_pipeline import create_configs

configs = create_configs("{model_type}")
success = model_pipeline(
    dataset_config=configs['dataset_config'],
    model_config=configs['model_config'],
    training_config=configs['training_config'],
    task_config=configs['task_config'],
    experiment_name="{experiment_name}",
    experiment_description="{experiment_description}",
    delete_catalogs=True
)
sys.exit(0 if success else 1)
                """,
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,  # This makes it continue running if VSCode is closed
            )

        # Save the process ID to a file for later termination
        # pid_file = logs_dir / f"pipeline_{timestamp}.pid"
        # with open(pid_file, 'w') as pf:
        #     pf.write(str(process.pid))

        print("=" * 80)
        print(f"PIPELINE PROCESS ID: {process.pid}")
        print("=" * 80)
        print(f"Started pipeline process at {timestamp}")
        print(f"Logging output to: {log_file}")
        # print(f"PID saved to: {pid_file}")
        print("Process is running in background. You can close VSCode safely.")
        print(f"To stop the process, run: kill {process.pid}")
        print("=" * 80)

    except Exception as e:
        print(f"Failed to start pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    run_pipeline(model_type="dnn_predictor", delete_catalogs=True)
