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
                          dataset_config: dict,
                          model_config: dict,
                          training_config: dict,
                          description: str = "") -> str:
        """
        Register new experiment with enhanced configuration tracking
        
        Args:
            name: Human readable experiment name
            dataset_config: Dataset parameters including:
                - run_numbers: List of ATLAS run numbers
                - track_selections: Dictionary of track-level selection criteria
                - event_selections: Dictionary of event-level selection criteria
                - max_tracks_per_event: Maximum number of tracks to keep per event
                - min_tracks_per_event: Minimum number of tracks required per event
                - normalization_params: Dictionary of normalization parameters
                - train_fraction: Fraction of data for training
                - validation_fraction: Fraction for validation
                - test_fraction: Fraction for testing
                - batch_size: Batch size used
                - shuffle_buffer: Shuffle buffer size
                - data_quality_metrics: Results of data validation
            model_config: Model configuration including:
                - model_type: Type of model (e.g., "autoencoder")
                - architecture: Network architecture details
                - hyperparameters: Model hyperparameters
            training_config: Training parameters
            description: Optional experiment description
        """
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create ProcessedDatasetManager instance
        dataset_manager = ProcessedDatasetManager()
        
        # Create or get dataset
        dataset_id, dataset_path = dataset_manager.create_dataset(dataset_config)
        dataset_info = dataset_manager.get_dataset_info(dataset_id)
        
        # Add to dataset_config - Convert Path to string
        dataset_config.update({
            'dataset_id': dataset_id,
            'dataset_path': str(dataset_path),  # Convert Path to string here
            'creation_date': dataset_info['creation_date'],
            'atlas_version': dataset_info['atlas_version'],
            'software_versions': dataset_info['software_versions']
        })
        
        # Get environment info
        environment_info = {
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count()
        }
        try:
            environment_info["gpu_devices"] = tf.config.list_physical_devices('GPU')
        except:
            environment_info["gpu_devices"] = []
            
        with sqlite3.connect(self.db_path) as conn:
            # Insert main experiment info
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
                    "registered",
                    json.dumps(environment_info)
                )
            )
            
            # Insert dataset configuration
            conn.execute(
                """
                INSERT INTO dataset_configs
                (experiment_id, dataset_id, dataset_path, creation_date, atlas_version, software_versions,
                run_numbers, track_selections, event_selections, max_tracks_per_event, min_tracks_per_event,
                normalization_params, train_fraction, validation_fraction, test_fraction, batch_size, shuffle_buffer,
                data_quality_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    dataset_id,
                    str(dataset_path),  # Make sure it's a string here too
                    dataset_info['creation_date'],
                    dataset_info['atlas_version'],
                    json.dumps(dataset_info['software_versions']),
                    json.dumps(dataset_config['run_numbers']),
                    json.dumps(dataset_config['track_selections']),
                    json.dumps(dataset_config['event_selections']),
                    dataset_config['max_tracks_per_event'],
                    dataset_config['min_tracks_per_event'],
                    json.dumps(dataset_config.get('normalization_params', {})),
                    dataset_config['train_fraction'],
                    dataset_config['validation_fraction'],
                    dataset_config['test_fraction'],
                    dataset_config['batch_size'],
                    dataset_config['shuffle_buffer'],
                    json.dumps(dataset_config.get('data_quality_metrics', {}))
                )
            )
            
            # Insert model configuration
            conn.execute(
                """
                INSERT INTO model_configs
                (experiment_id, model_type, architecture, hyperparameters)
                VALUES (?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    model_config['model_type'],
                    json.dumps(model_config['architecture']),
                    json.dumps(model_config.get('hyperparameters', {}))
                )
            )
            
            # Insert initial training info
            conn.execute(
                """
                INSERT INTO training_info
                (experiment_id, config, start_time, epochs_completed, training_history, final_metrics, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    json.dumps(training_config),
                    None,
                    0,
                    "{}",
                    "{}",
                    "initialized"
                )
            )
        
        # Create experiment directory structure
        exp_dir = self.model_store / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configurations as YAML for easy reading
        configs_dir = exp_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        with open(configs_dir / "dataset_config.yaml", 'w') as f:
            yaml.dump(dataset_config, f)
        with open(configs_dir / "model_config.yaml", 'w') as f:
            yaml.dump(model_config, f)
        with open(configs_dir / "training_config.yaml", 'w') as f:
            yaml.dump(training_config, f)
            
        return experiment_id
        
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
                
            history = json.loads(current[0])
            
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
        # Ensure required metrics exist with defaults
        metrics = {
            'loss': 0.0,
            'val_loss': 0.0,
            'test_loss': 0.0,
            'training_duration': 0.0,
            **final_metrics  # Override defaults with actual values
        }
        
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

   