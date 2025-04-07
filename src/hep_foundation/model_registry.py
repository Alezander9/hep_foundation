import sqlite3
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import uuid
import tensorflow as tf
import platform
import os
import numpy as np
import sys
import psutil
import logging
from hep_foundation.logging_config import setup_logging

class ModelRegistry:
    """
    Enhanced central registry for managing ML experiments, models, and metrics
    Tracks detailed dataset configurations and training metrics
    """
    def __init__(self, base_path: str):
        # Setup logging
        setup_logging()
        
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "registry.db"
        self.model_store = self.base_path / "model_store"
        
        logging.info(f"\nModelRegistry paths:")
        logging.info(f"  Base path: {self.base_path.absolute()}")
        logging.info(f"  DB path: {self.db_path.absolute()}")
        logging.info(f"  Model store: {self.model_store.absolute()}")
        
        # Create directories if they don't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.model_store.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry structure
        self._initialize_model_registry()
        
    def _initialize_model_registry(self):
        """Initialize the model registry folder structure and index tracking"""
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load the model index
        self.index_file = self.base_path / "model_index.json"
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
                self.current_index = index_data.get('current_index', 0)
        else:
            self.current_index = 0
            self._save_index()

    def _save_index(self):
        """Save the current model index"""
        with open(self.index_file, 'w') as f:
            json.dump({'current_index': self.current_index}, f)

    def _create_experiment_folders(self, experiment_name: str) -> Path:
        """
        Create folder structure for a new experiment
        Returns the path to the experiment directory
        """
        # Increment index and format experiment name
        self.current_index += 1
        formatted_name = f"{self.current_index:03d}_{experiment_name}"
        
        # Create experiment directory structure
        exp_dir = self.base_path / formatted_name
        (exp_dir / "models").mkdir(parents=True, exist_ok=True)
        (exp_dir / "training_history").mkdir(parents=True, exist_ok=True)
        (exp_dir / "testing").mkdir(parents=True, exist_ok=True)
        
        # Save updated index
        self._save_index()
        
        return exp_dir

    def _get_environment_info(self) -> Dict:
        """Collect information about the execution environment"""
        # Get memory info
        memory = psutil.virtual_memory()
        
        return {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'python_version': platform.python_version(),
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3)
            },
            'software': {
                'tensorflow': tf.__version__,
                'numpy': np.__version__,
                'cuda_available': tf.test.is_built_with_cuda(),
                'gpu_available': bool(tf.config.list_physical_devices('GPU'))
            },
            'timestamp': str(datetime.now())
        }


    def register_experiment(self,
                          name: str,
                          dataset_id: str,
                          model_config: dict,
                          training_config: dict,
                          description: str = "") -> str:
        """
        Register new experiment using existing dataset
        Returns the experiment directory name (which serves as the experiment ID)
        """
        # Create experiment directory and folder structure
        exp_dir = self._create_experiment_folders(name)
        
        # Prepare experiment data
        experiment_data = {
            'experiment_info': {
                'name': name,
                'description': description,
                'timestamp': str(datetime.now()),
                'status': 'created',
                'environment_info': self._get_environment_info()
            },
            'dataset_config': {
                'dataset_id': dataset_id
            },
            'model_config': {
                'model_type': model_config['model_type'],
                'architecture': model_config['architecture'],
                'hyperparameters': model_config['hyperparameters']
            },
            'training_config': training_config
        }
        
        # Save experiment data
        with open(exp_dir / "experiment_data.json", 'w') as f:
            json.dump(experiment_data, f, indent=2, default=self.ensure_serializable)
        
        # Create a training history file
        with open(exp_dir / "training_history" / "metrics.json", 'w') as f:
            json.dump({
                'epochs_completed': 0,
                'history': {},
                'final_metrics': None
            }, f, indent=2)
        
        return exp_dir.name
        
    def ensure_serializable(self, obj):
        """Recursively convert numpy/tensorflow types to Python native types"""
        if isinstance(obj, dict):
            return {key: self.ensure_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.ensure_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        elif obj is None:
            return "null"  # Convert None to string "null"
        return obj

    def complete_training(self, experiment_id: str, final_metrics: Dict):
        """
        Record final training results and history
        
        Args:
            experiment_id: The experiment directory name
            final_metrics: Dictionary containing final metrics and training history
        """
        exp_dir = self.base_path / experiment_id
        
        # Ensure metrics are serializable
        metrics = self.ensure_serializable(final_metrics)
        
        # Update experiment status in experiment_data.json
        exp_data_path = exp_dir / "experiment_data.json"
        with open(exp_data_path, 'r') as f:
            experiment_data = json.load(f)
        experiment_data['experiment_info']['status'] = 'completed'
        with open(exp_data_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)

        # Save training history to CSV
        history_dir = exp_dir / "training_history"
        
        # Save epoch-wise metrics (including all loss components)
        if 'history' in metrics:
            history_df = []
            for epoch, epoch_metrics in metrics['history'].items():
                epoch_data = {'epoch': int(epoch)}
                epoch_data.update(epoch_metrics)
                history_df.append(epoch_data)
            
            # Save using numpy to handle any remaining numpy types
            import numpy as np
            np.savetxt(
                history_dir / "training_history.csv",
                [list(d.values()) for d in history_df],
                delimiter=',',
                header=','.join(history_df[0].keys()),
                comments=''
            )

        # Save final metrics summary
        final_metrics_path = history_dir / "final_metrics.json"
        final_metrics_data = {
            'training_duration': metrics.get('training_duration', 0.0),
            'epochs_completed': metrics.get('epochs_completed', 0),
            'final_metrics': {
                k: v for k, v in metrics.items() 
                if k not in ['history', 'training_duration', 'epochs_completed']
            }
        }
        
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics_data, f, indent=2)

        logging.info(f"\nTraining results saved to {history_dir}")


    def save_model(self, 
                   experiment_id: str,
                   models: Dict[str, Any],
                   model_name: str = "full_model",
                   metadata: Optional[Dict] = None):
        """
        Save model(s) for an experiment
        
        Args:
            experiment_id: The experiment directory name (e.g. '001_vae_test')
            models: Dictionary of named models to save (e.g. {'encoder': encoder_model, 'decoder': decoder_model})
            model_name: Name for this model version (e.g. 'full_model', 'quantized', 'pruned')
            metadata: Optional metadata about the model (e.g. quantization params, performance metrics)
        """
        exp_dir = self.base_path / experiment_id
        model_dir = exp_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model component
        for component_name, model in models.items():
            model_path = model_dir / component_name
            model.save(model_path)
        
        # Prepare and save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "saved_components": list(models.keys()),
            "save_timestamp": str(datetime.now()),
            "model_path": str(model_dir)
        })
        
        # Save metadata
        with open(model_dir / "model_info.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=self.ensure_serializable)

        logging.info(f"\nModel saved to {model_dir}")

    def load_model(self, 
                   experiment_id: str,
                   model_name: str = "full_model") -> Dict[str, str]:
        """
        Get paths to saved model components
        
        Args:
            experiment_id: The experiment directory name
            model_name: Name of the model version to load
            
        Returns:
            Dictionary of model component names to their saved paths
        """
        exp_dir = self.base_path / experiment_id
        model_dir = exp_dir / "models" / model_name
        
        if not model_dir.exists():
            raise ValueError(f"No model '{model_name}' found for experiment {experiment_id}")
        
        # Load metadata
        try:
            with open(model_dir / "model_info.json", 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Model metadata not found for {model_name}")
        
        # Verify each component exists and return paths
        model_paths = {}
        for component_name in metadata["saved_components"]:
            component_path = model_dir / component_name
            if not component_path.exists():
                logging.warning(f"Warning: Model component '{component_name}' not found at {component_path}")
                continue
            model_paths[component_name] = str(component_path)
        
        if not model_paths:
            raise ValueError(f"No valid model components found in {model_dir}")
        
        return model_paths

    def get_experiment_data(self, experiment_id: str) -> Dict:
        """
        Load experiment data from experiment_data.json
        
        Args:
            experiment_id: The experiment directory name (e.g. '001_vae_test')
            
        Returns:
            Dictionary containing all experiment configuration and metadata
        """
        exp_dir = self.base_path / experiment_id
        exp_data_path = exp_dir / "experiment_data.json"
        
        if not exp_dir.exists():
            raise ValueError(f"No experiment found with ID {experiment_id}")
        
        if not exp_data_path.exists():
            raise ValueError(f"No experiment data file found for {experiment_id}")
        
        try:
            with open(exp_data_path, 'r') as f:
                experiment_data = json.load(f)
                
            # Optionally load final metrics if training is completed
            metrics_path = exp_dir / "training_history" / "final_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    experiment_data['training_results'] = metrics_data
                    
            return experiment_data
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in experiment data file for {experiment_id}")

   