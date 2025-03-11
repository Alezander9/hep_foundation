import subprocess
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Any

from hep_foundation.utils import ATLAS_RUN_NUMBERS
from hep_foundation.model_pipeline import DatasetConfig, ModelConfig, TrainingConfig

def create_configs(model_type: str = "vae") -> Dict[str, Any]:
    """Create configuration objects for the model pipeline"""
    
    # Dataset configuration - common for both models
    dataset_config = DatasetConfig(
        run_numbers=ATLAS_RUN_NUMBERS[-3:],
        signal_keys=["zprime", "wprime_qq", "zprime_bb"],
        catalog_limit=20,
        track_selections={
            'eta': (-2.5, 2.5),
            'chi2_per_ndof': (0.0, 10.0),
        },
        event_selections={},
        max_tracks=30,
        min_tracks=10,
        validation_fraction=0.15,
        test_fraction=0.15,
        shuffle_buffer=50000,
        plot_distributions=True,
        include_labels=False  # Don't include labels for VAE training by default
    )
    
    # Model configurations
    ae_model_config = ModelConfig(
        model_type="autoencoder",
        latent_dim=16,
        encoder_layers=[128, 64, 32],
        decoder_layers=[32, 64, 128],
        quant_bits=8,
        activation='relu',
        beta_schedule=None
    )
    
    vae_model_config = ModelConfig(
        model_type="variational_autoencoder",
        latent_dim=16,
        encoder_layers=[128, 64, 32],
        decoder_layers=[32, 64, 128],
        quant_bits=8,
        activation='relu',
        beta_schedule={
            'start': 0.01,
            'end': 0.1,
            'warmup_epochs': 5,
            'cycle_epochs': 5
        }
    )

    # Training configuration - common for both models
    training_config = TrainingConfig(
        batch_size=1024,
        learning_rate=0.001,
        epochs=50,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        plot_training=True
    )
    
    # Select model configuration based on type
    model_config = vae_model_config if model_type == "vae" else ae_model_config
    
    return {
        'dataset_config': dataset_config,
        'model_config': model_config,
        'training_config': training_config
    }

def run_pipeline(
    model_type: str = "vae",
    experiment_name: str = None,
    experiment_description: str = None,
    custom_configs: Dict[str, Any] = None
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
    
    # Set default experiment name if not provided
    if experiment_name is None:
        experiment_name = f"{model_type}_test_{timestamp}"
    
    if experiment_description is None:
        experiment_description = f"Testing {model_type} model with standard parameters"
    
    # Run the pipeline
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                ["python", "-c", f"""
import sys
from hep_foundation.model_pipeline import test_model_pipeline
from scripts.run_pipeline import create_configs

configs = create_configs("{model_type}")
success = test_model_pipeline(
    dataset_config=configs['dataset_config'],
    model_config=configs['model_config'],
    training_config=configs['training_config'],
    experiment_name="{experiment_name}",
    experiment_description="{experiment_description}"
)
sys.exit(0 if success else 1)
                """],
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp  # This makes it continue running if VSCode is closed
            )
        
        # Save the process ID to a file for later termination
        pid_file = logs_dir / f"pipeline_{timestamp}.pid"
        with open(pid_file, 'w') as pf:
            pf.write(str(process.pid))
        
        print(f"Started pipeline process (PID: {process.pid})")
        print(f"Logging output to: {log_file}")
        print(f"PID saved to: {pid_file}")
        print("Process is running in background. You can close VSCode safely.")
        print(f"To stop the process, run: python -m scripts.stop_pipeline {process.pid}")
        
    except Exception as e:
        print(f"Failed to start pipeline: {str(e)}")
        raise

# Add a new function to stop a running pipeline
def stop_pipeline(pid: int) -> None:
    """
    Stop a running pipeline process by its PID
    
    Args:
        pid: Process ID of the pipeline to stop
    """
    try:
        import signal
        import psutil
        
        # Check if the process exists
        if not psutil.pid_exists(pid):
            print(f"Process with PID {pid} not found.")
            return
            
        # Get the process and all its children
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Send SIGTERM to all child processes first
        for child in children:
            try:
                child.send_signal(signal.SIGTERM)
                print(f"Sent termination signal to child process {child.pid}")
            except psutil.NoSuchProcess:
                pass
                
        # Then terminate the parent
        parent.send_signal(signal.SIGTERM)
        print(f"Sent termination signal to main process {pid}")
        
        # Wait for processes to terminate (with timeout)
        gone, alive = psutil.wait_procs(children + [parent], timeout=5)
        
        # If any processes are still alive, force kill them
        if alive:
            for p in alive:
                print(f"Force killing process {p.pid}")
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
                    
        print(f"Pipeline process {pid} and its children have been terminated.")
        
    except Exception as e:
        print(f"Error stopping pipeline: {str(e)}")

if __name__ == "__main__":
    run_pipeline() 