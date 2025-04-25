import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from hep_foundation.data.task_config import TaskConfig
from hep_foundation.utils.utils import ATLAS_RUN_NUMBERS
from hep_foundation.data.dataset_manager import DatasetConfig
from hep_foundation.training.model_trainer import TrainingConfig

def create_configs(model_type: str = "vae") -> dict[str, Any]:
    """Create configuration objects for the model pipeline"""

    task_config = TaskConfig.create_from_branch_names(
        event_filter_dict={},
        input_features=[],
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

    dataset_config = DatasetConfig(
        run_numbers=ATLAS_RUN_NUMBERS[-1:],
        # run_numbers=ATLAS_RUN_NUMBERS[-3:],
        signal_keys=["zprime", "wprime_qq", "zprime_bb"],
        catalog_limit=2,
        # catalog_limit=20,
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
            "latent_dim": 32,
            "encoder_layers": [192, 128, 96],
            "decoder_layers": [96, 128, 192],
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

    # Training configurations
    vae_training_config = TrainingConfig(
        batch_size=1024,
        learning_rate=0.001,
        epochs=2,
        early_stopping_patience=10,
        early_stopping_min_delta=1e-4,
        plot_training=True,
    )

    dnn_training_config = TrainingConfig(
        batch_size=1024,
        learning_rate=0.001,
        epochs=2,
        early_stopping_patience=100,
        early_stopping_min_delta=1e-4,
        plot_training=True,
    )

    return {
        "dataset_config": dataset_config,
        "vae_model_config": vae_model_config,
        "dnn_model_config": dnn_model_config,
        "vae_training_config": vae_training_config,
        "dnn_training_config": dnn_training_config,
        "task_config": task_config,
    }


def run_foundation_pipeline(
    process_type: str = "train",
    experiment_name: str = None,
    experiment_description: str = None,
    delete_catalogs: bool = True,
    foundation_model_path: str = None,
) -> None:
    """
    Run the foundation model pipeline with specified process type

    Args:
        process_type: Type of process to run ("train", "anomaly", or "regression")
        experiment_name: Optional name for the experiment
        experiment_description: Optional description for the experiment
        delete_catalogs: Whether to delete catalogs after processing
        foundation_model_path: Path to the foundation model encoder to use for encoding
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"foundation_pipeline_{process_type}_{timestamp}.log"

    # Validate process type
    valid_processes = ["train", "anomaly", "regression"]
    if process_type not in valid_processes:
        raise ValueError(
            f"Invalid process type: {process_type}. Must be one of {valid_processes}"
        )

    # Run the pipeline
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                [
                    "python",
                    "-c",
                    f"""
import sys
from hep_foundation.training.foundation_model_pipeline import FoundationModelPipeline
from scripts.run_foundation_model_pipeline import create_configs

# Get the configs
configs = create_configs()

# Initialize the pipeline
pipeline = FoundationModelPipeline()

# Run the specified process with the configs
success = pipeline.run_process(
    process_name="{process_type}",
    dataset_config=configs['dataset_config'],
    vae_model_config=configs['vae_model_config'],
    dnn_model_config=configs['dnn_model_config'],
    vae_training_config=configs['vae_training_config'],
    dnn_training_config=configs['dnn_training_config'],
    task_config=configs['task_config'],
    experiment_name="{experiment_name}",
    experiment_description="{experiment_description}",
    delete_catalogs={delete_catalogs},
    foundation_model_path="{foundation_model_path}"
)
sys.exit(0 if success else 1)
                """,
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,  # This makes it continue running if VSCode is closed
            )

        print("=" * 80)
        print(f"FOUNDATION PIPELINE PROCESS ID: {process.pid}")
        print("=" * 80)
        print(f"Started foundation pipeline process at {timestamp}")
        print(f"Process type: {process_type}")
        print(f"Logging output to: {log_file}")
        print("Process is running in background. You can close VSCode safely.")
        print(f"To stop the process, run: kill {process.pid}")
        print("=" * 80)

    except Exception as e:
        print(f"Failed to start foundation pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the foundation model pipeline")
    parser.add_argument(
        "--process",
        type=str,
        default="train",
        choices=["train", "anomaly", "regression"],
        help="Type of process to run (train, anomaly, or regression)",
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Name for the experiment"
    )
    parser.add_argument(
        "--description", type=str, default=None, help="Description for the experiment"
    )
    parser.add_argument(
        "--delete-catalogs",
        action="store_true",
        help="Delete catalogs after processing",
    )
    parser.add_argument(
        "--foundation-model",
        type=str,
        default=None,
        help="Path to the foundation model encoder to use for encoding",
    )

    args = parser.parse_args()

    # Run the pipeline with the specified arguments
    run_foundation_pipeline(
        process_type=args.process,
        experiment_name=args.name,
        experiment_description=args.description,
        delete_catalogs=args.delete_catalogs,
        foundation_model_path=args.foundation_model,
    )
