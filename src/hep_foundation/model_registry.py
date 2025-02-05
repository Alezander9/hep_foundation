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

from hep_foundation.processed_dataset_manager import ProcessedDatasetManager

class ModelRegistry:
    """
    Enhanced central registry for managing ML experiments, models, and metrics
    Tracks detailed dataset configurations and training metrics
    """
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "registry.db"
        self.model_store = self.base_path / "model_store"
        
        # Create directories if they don't exist
        self.model_store.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_db()
        
    def _initialize_db(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            # Main experiments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    name TEXT,
                    description TEXT,
                    status TEXT,
                    environment_info JSON
                )
            """)
            
            # Dataset configuration table - include all columns in initial creation
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_configs (
                    experiment_id TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    dataset_path TEXT,
                    creation_date TEXT,
                    atlas_version TEXT,
                    software_versions JSON,
                    run_numbers JSON,
                    track_selections JSON,
                    event_selections JSON,
                    max_tracks_per_event INTEGER,
                    min_tracks_per_event INTEGER,
                    normalization_params JSON,
                    train_fraction FLOAT,
                    validation_fraction FLOAT,
                    test_fraction FLOAT,
                    batch_size INTEGER,
                    shuffle_buffer INTEGER,
                    data_quality_metrics JSON,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            # Model configuration table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_configs (
                    experiment_id TEXT PRIMARY KEY,
                    model_type TEXT,
                    architecture JSON,
                    hyperparameters JSON,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            # Training configuration and results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_info (
                    experiment_id TEXT PRIMARY KEY,
                    config JSON,
                    start_time DATETIME,
                    end_time DATETIME,
                    epochs_completed INTEGER,
                    training_history JSON,
                    final_metrics JSON,
                    hardware_metrics JSON,
                    status TEXT,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            # Checkpoints table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    name TEXT,
                    timestamp DATETIME,
                    metadata JSON,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
    def register_experiment(self,
                          name: str,
                          dataset_id: str,
                          model_config: dict,
                          training_config: dict,
                          description: str = "") -> str:
        """Register new experiment using existing dataset"""
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            # Store experiment info
            conn.execute(
                """
                INSERT INTO experiments 
                (experiment_id, timestamp, name, description, status, environment_info)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    timestamp,
                    name,
                    description,
                    "created",
                    json.dumps(self._get_environment_info())
                )
            )
            
            # Store dataset reference
            conn.execute(
                """
                INSERT INTO dataset_configs 
                (experiment_id, dataset_id) 
                VALUES (?, ?)
                """,
                (experiment_id, dataset_id)
            )
            
            # Store other configs
            conn.execute(
                """
                INSERT INTO model_configs 
                (experiment_id, model_type, architecture, hyperparameters)
                VALUES (?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    model_config["model_type"],
                    json.dumps(model_config["architecture"]),
                    json.dumps(model_config["hyperparameters"])
                )
            )
            
            conn.execute(
                """
                INSERT INTO training_info 
                (experiment_id, config, status)
                VALUES (?, ?, ?)
                """,
                (experiment_id, json.dumps(training_config), "pending")
            )
            
        return experiment_id
        
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

    def update_training_progress(self,
                               experiment_id: str,
                               epoch: int,
                               metrics: Dict[str, float],
                               hardware_metrics: Optional[Dict] = None):
        """Update training progress and metrics"""
        with sqlite3.connect(self.db_path) as conn:
            current = conn.execute(
                "SELECT training_history FROM training_info WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()
            
            if current is None:
                raise ValueError(f"No experiment found with id {experiment_id}")
                
            # Ensure metrics are serializable
            metrics = self.ensure_serializable(metrics)
            if hardware_metrics:
                hardware_metrics = self.ensure_serializable(hardware_metrics)
            
            # Initialize or load history
            history = json.loads(current[0]) if current[0] else {}
            
            # Update history
            if str(epoch) not in history:
                history[str(epoch)] = {}
            history[str(epoch)].update(metrics)
            
            # Update training info
            updates = {
                "epochs_completed": epoch,
                "training_history": json.dumps(history)
            }
            
            if hardware_metrics:
                updates["hardware_metrics"] = json.dumps(hardware_metrics)
                
            update_sql = "UPDATE training_info SET " + \
                        ", ".join(f"{k} = ?" for k in updates.keys()) + \
                        " WHERE experiment_id = ?"
            
            conn.execute(update_sql, list(updates.values()) + [experiment_id])
            
    def complete_training(self, experiment_id: str, final_metrics: Dict):
        """Record final training results"""
        # Start with actual metrics
        metrics = dict(final_metrics)  # Make a copy
        
        print("\nFinal metrics before processing:")  # Debug print
        print(json.dumps(metrics, indent=2))
        
        # Ensure test metrics are properly named
        if 'test_loss' not in metrics and 'test_mse' in metrics:
            metrics['test_loss'] = metrics['test_mse']
        
        # Only fill in missing metrics with defaults
        defaults = {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'test_loss': 0.0,
            'training_duration': 0.0
        }
        
        for key, default_value in defaults.items():
            if key not in metrics:
                metrics[key] = default_value
        
        # Ensure metrics are serializable
        metrics = self.ensure_serializable(metrics)
        
        print("\nFinal metrics after processing:")  # Debug print
        print(json.dumps(metrics, indent=2))
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE training_info 
                SET final_metrics = ?, status = 'completed'
                WHERE experiment_id = ?
                """,
                (json.dumps(metrics), experiment_id)
            )
            
            conn.execute(
                """
                UPDATE experiments 
                SET status = 'completed'
                WHERE experiment_id = ?
                """,
                (experiment_id,)
            )

    def save_checkpoint(self, 
                       experiment_id: str,
                       models: Dict[str, Any],
                       checkpoint_name: str = "latest",
                       metadata: Optional[Dict] = None):
        """
        Save model checkpoints for an experiment
        
        Args:
            experiment_id: Experiment identifier
            models: Dictionary of named models to save
            checkpoint_name: Name for this checkpoint
            metadata: Optional metadata about the checkpoint
        """
        exp_dir = self.model_store / experiment_id
        checkpoint_dir = exp_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_id = str(uuid.uuid4())
        
        # Save each model
        for name, model in models.items():
            model_path = checkpoint_dir / name
            model.save(model_path)
            
        # Save checkpoint metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "saved_models": list(models.keys()),
            "checkpoint_path": str(checkpoint_dir)
        })
            
        # Record checkpoint in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO checkpoints
                (checkpoint_id, experiment_id, name, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    experiment_id,
                    checkpoint_name,
                    datetime.now(),
                    json.dumps(metadata)
                )
            )
            
            # Update experiment status
            conn.execute(
                "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                ("checkpoint_saved", experiment_id)
            )
            
    # def load_checkpoint(self, 
    #                    experiment_id: str,
    #                    checkpoint_name: str = "latest") -> Dict[str, str]:
    #     """
    #     Get paths to saved model checkpoints
        
    #     Returns:
    #         Dictionary of model names to their saved paths
    #     """
    #     with sqlite3.connect(self.db_path) as conn:
    #         result = conn.execute(
    #             """
    #             SELECT metadata FROM checkpoints 
    #             WHERE experiment_id = ? AND name = ?
    #             ORDER BY timestamp DESC LIMIT 1
    #             """,
    #             (experiment_id, checkpoint_name)
    #         ).fetchone()
            
    #     if result is None:
    #         raise ValueError(
    #             f"No checkpoint '{checkpoint_name}' found for experiment {experiment_id}"
    #         )
            
    #     metadata = json.loads(result[0])
    #     checkpoint_path = Path(metadata["checkpoint_path"])
        
    #     if not checkpoint_path.exists():
    #         raise ValueError(f"Checkpoint directory not found: {checkpoint_path}")
            
    #     return {
    #         model_name: str(checkpoint_path / model_name)
    #         for model_name in metadata["saved_models"]
    #         if (checkpoint_path / model_name).exists()
    #     }

    def get_experiment_details(self, experiment_id: str) -> Dict:
        """Get full experiment details including dataset, model, and training configs"""
        with sqlite3.connect(self.db_path) as conn:
            # Get experiment info
            cursor = conn.execute(
                """
                SELECT timestamp, name, description, status, environment_info
                FROM experiments WHERE experiment_id = ?
                """, 
                (experiment_id,)
            )
            exp_row = cursor.fetchone()
            if exp_row is None:
                raise ValueError(f"No experiment found with ID {experiment_id}")
            
            # Get dataset config - use column names explicitly
            cursor = conn.execute(
                """
                SELECT dataset_id, dataset_path, creation_date, atlas_version, 
                       software_versions, run_numbers, track_selections, event_selections
                FROM dataset_configs WHERE experiment_id = ?
                """,
                (experiment_id,)
            )
            dataset_row = cursor.fetchone()
            
            # Get model config - use column names explicitly
            cursor = conn.execute(
                """
                SELECT model_type, architecture, hyperparameters 
                FROM model_configs WHERE experiment_id = ?
                """,
                (experiment_id,)
            )
            model_row = cursor.fetchone()
            
            # Get training info - use column names explicitly
            cursor = conn.execute(
                """
                SELECT config, epochs_completed, training_history, final_metrics 
                FROM training_info WHERE experiment_id = ?
                """,
                (experiment_id,)
            )
            training_row = cursor.fetchone()
            
            # Safely decode JSON with defaults
            def safe_json_decode(json_str, default=None):
                if json_str is None:
                    return default
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return default
            
            return {
                'experiment_info': {
                    'experiment_id': experiment_id,
                    'timestamp': exp_row[0],
                    'name': exp_row[1],
                    'description': exp_row[2],
                    'status': exp_row[3],
                    'environment_info': safe_json_decode(exp_row[4], {})
                },
                'dataset_config': {
                    'dataset_id': dataset_row[0] if dataset_row else None,
                    'dataset_path': dataset_row[1] if dataset_row else None,
                    'creation_date': dataset_row[2] if dataset_row else None,
                    'atlas_version': dataset_row[3] if dataset_row else None,
                    'software_versions': safe_json_decode(dataset_row[4], {}) if dataset_row else {},
                    'run_numbers': safe_json_decode(dataset_row[5], []) if dataset_row else [],
                    'track_selections': safe_json_decode(dataset_row[6], {}) if dataset_row else {},
                    'event_selections': safe_json_decode(dataset_row[7], {}) if dataset_row else {}
                } if dataset_row else None,
                'model_config': {
                    'model_type': model_row[0] if model_row else None,
                    'architecture': safe_json_decode(model_row[1], {}) if model_row else {},
                    'hyperparameters': safe_json_decode(model_row[2], {}) if model_row else {}
                } if model_row else None,
                'training_info': {
                    'config': safe_json_decode(training_row[0], {}) if training_row else {},
                    'epochs_completed': training_row[1] if training_row else 0,
                    'training_history': safe_json_decode(training_row[2], {}) if training_row else {},
                    'final_metrics': safe_json_decode(training_row[3], {}) if training_row else {}
                } if training_row else None
            }

    def get_performance_summary(self, experiment_id: str) -> Dict:
        """Get summary of model performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT epochs_completed, training_history, final_metrics FROM training_info WHERE experiment_id = ?",
                (experiment_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"No training info found for experiment {experiment_id}")
            
            # Use safe_json_decode for metrics
            def safe_json_decode(json_str, default=None):
                if json_str is None:
                    return default
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return default
            
            # Reorganize history by metric name instead of by epoch
            history_by_epoch = safe_json_decode(row[1], {})
            history_by_metric = {}
            
            # Convert epoch-based history to metric-based history
            for epoch, metrics in history_by_epoch.items():
                for metric_name, value in metrics.items():
                    if metric_name not in history_by_metric:
                        history_by_metric[metric_name] = []
                    history_by_metric[metric_name].append(value)
            
            return {
                'epochs_completed': row[0],
                'metric_progression': history_by_metric,
                'final_metrics': safe_json_decode(row[2], {
                    'loss': 0.0,
                    'val_loss': 0.0,
                    'test_loss': 0.0,
                    'training_duration': 0.0
                })
            }

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

   