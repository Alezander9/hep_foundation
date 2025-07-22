import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil
import tensorflow as tf

from hep_foundation.config.logging_config import get_logger


class ModelRegistry:
    """
    Enhanced central registry for managing ML experiments, models, and metrics
    Tracks detailed dataset configurations and training metrics
    """

    def __init__(self, base_path: str):
        # Setup self.logger
        self.logger = get_logger(__name__)

        self.base_path = Path(base_path)
        self.db_path = self.base_path / "registry.db"

        self.logger.info("ModelRegistry paths:")
        self.logger.info(f"Base path: {self.base_path.absolute()}")
        self.logger.info(f"DB path: {self.db_path.absolute()}")

        # Create directories if they don't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize registry structure
        self._initialize_model_registry()

    def _initialize_model_registry(self):
        """Initialize the model registry folder structure and index tracking"""
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize or load the model index
        self.index_file = self.base_path / "model_index.json"
        if self.index_file.exists():
            with open(self.index_file) as f:
                index_data = json.load(f)
                self.current_index = index_data.get("current_index", 0)
        else:
            self.current_index = 0
            self._save_index()

    def _save_index(self):
        """Save the current model index"""
        with open(self.index_file, "w") as f:
            json.dump({"current_index": self.current_index}, f)

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
        (exp_dir / "training").mkdir(parents=True, exist_ok=True)
        (exp_dir / "testing").mkdir(parents=True, exist_ok=True)

        # Save updated index
        self._save_index()

        return exp_dir

    def _get_git_info(self) -> dict:
        """Collect git repository information"""
        git_info = {}

        try:
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                git_info["commit_hash"] = result.stdout.strip()

            # Get current branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()

            # Check if working directory is clean (no uncommitted changes)
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                git_info["is_dirty"] = bool(result.stdout.strip())
                git_info["uncommitted_files"] = (
                    len(result.stdout.strip().split("\n"))
                    if result.stdout.strip()
                    else 0
                )

            # Get remote origin URL (if available)
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                git_info["remote_origin"] = result.stdout.strip()

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            self.logger.warning(f"Could not get git information: {e}")
            git_info["error"] = str(e)

        return git_info

    def _get_environment_info(self) -> dict:
        """Collect information about the execution environment"""
        # Get memory info
        memory = psutil.virtual_memory()

        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
            },
            "software": {
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
                "cuda_available": tf.test.is_built_with_cuda(),
                "gpu_available": bool(tf.config.list_physical_devices("GPU")),
            },
            "git": self._get_git_info(),
            "timestamp": str(datetime.now()),
        }

    def register_experiment(
        self,
        name: str,
        dataset_id: str,
        description: str = "",
        source_config_file: str = None,
    ) -> str:
        """
        Register new experiment
        Returns the experiment directory name (which serves as the experiment ID)

        Args:
            name: Name of the experiment
            dataset_id: ID of the dataset used for this experiment
            description: Optional experiment description
            source_config_file: Path to the source YAML config file containing all configurations
        """
        # Create experiment directory and folder structure
        exp_dir = self._create_experiment_folders(name)

        # Collect experiment info (machine/environment info)
        experiment_info = {
            "name": name,
            "dataset_id": dataset_id,
            "description": description,
            "timestamp": str(datetime.now()),
            "status": "created",
            "environment_info": self._get_environment_info(),
        }

        # Save experiment info as separate JSON file for machine/environment details
        with open(exp_dir / "_experiment_info.json", "w") as f:
            json.dump(experiment_info, f, indent=2, default=self.ensure_serializable)
        self.logger.info(
            f"Experiment info saved to: {exp_dir / '_experiment_info.json'}"
        )

        # Save the source YAML config file for reproducibility
        if source_config_file:
            import shutil
            from pathlib import Path

            source_path = Path(source_config_file)
            if source_path.exists():
                config_dest = exp_dir / "_experiment_config.yaml"
                shutil.copy2(source_path, config_dest)
                self.logger.info(f"Source config file saved as: {config_dest}")
            else:
                self.logger.warning(
                    f"Source config file not found: {source_config_file}"
                )

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

    def complete_training(self, experiment_id: str, final_metrics: dict):
        """
        Record final training results and history

        Args:
            experiment_id: The experiment directory name
            final_metrics: Dictionary containing final metrics and training history
        """
        exp_dir = self.base_path / experiment_id

        # Ensure metrics are serializable
        metrics = self.ensure_serializable(final_metrics)

        # Update experiment status in experiment_info.json
        exp_info_path = exp_dir / "_experiment_info.json"
        if exp_info_path.exists():
            with open(exp_info_path) as f:
                experiment_info = json.load(f)
            experiment_info["status"] = "completed"
            with open(exp_info_path, "w") as f:
                json.dump(
                    experiment_info, f, indent=2, default=self.ensure_serializable
                )

        # Save training history to CSV
        history_dir = exp_dir / "training"

        # Save epoch-wise metrics (including all loss components)
        if "history" in metrics:
            history_df = []
            for epoch, epoch_metrics in metrics["history"].items():
                epoch_data = {"epoch": int(epoch)}
                epoch_data.update(epoch_metrics)
                history_df.append(epoch_data)

            # Save using numpy to handle any remaining numpy types
            import numpy as np

            np.savetxt(
                history_dir / "training_history.csv",
                [list(d.values()) for d in history_df],
                delimiter=",",
                header=",".join(history_df[0].keys()),
                comments="",
            )

        # Save final metrics summary
        final_metrics_path = history_dir / "final_metrics.json"
        final_metrics_data = {
            "training_duration": metrics.get("training_duration", 0.0),
            "epochs_completed": metrics.get("epochs_completed", 0),
            "final_metrics": {
                k: v
                for k, v in metrics.items()
                if k not in ["history", "training_duration", "epochs_completed"]
            },
        }

        with open(final_metrics_path, "w") as f:
            json.dump(final_metrics_data, f, indent=2)

        self.logger.info(f"Training results saved to {history_dir}")

    def save_model(
        self,
        experiment_id: str,
        models: dict[str, Any],
        model_name: str = "full_model",
        metadata: Optional[dict] = None,
    ):
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

        metadata.update(
            {
                "saved_components": list(models.keys()),
                "save_timestamp": str(datetime.now()),
                "model_path": str(model_dir),
            }
        )

        # Save metadata
        with open(model_dir / "model_info.json", "w") as f:
            json.dump(metadata, f, indent=2, default=self.ensure_serializable)

        self.logger.info(f"Model saved to {model_dir}")

    def load_model(
        self, experiment_id: str, model_name: str = "full_model"
    ) -> dict[str, str]:
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
            raise ValueError(
                f"No model '{model_name}' found for experiment {experiment_id}"
            )

        # Load metadata
        try:
            with open(model_dir / "model_info.json") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Model metadata not found for {model_name}")

        # Verify each component exists and return paths
        model_paths = {}
        for component_name in metadata["saved_components"]:
            component_path = model_dir / component_name
            if not component_path.exists():
                self.logger.warning(
                    f"Warning: Model component '{component_name}' not found at {component_path}"
                )
                continue
            model_paths[component_name] = str(component_path)

        if not model_paths:
            raise ValueError(f"No valid model components found in {model_dir}")

        return model_paths

    def get_experiment_data(self, experiment_id: str) -> dict:
        """
        Load experiment data from separate files (_experiment_info.json, _experiment_config.yaml, final_metrics.json)

        Args:
            experiment_id: The experiment directory name (e.g. '001_vae_test')

        Returns:
            Dictionary containing experiment info, training results, and config file path
        """
        exp_dir = self.base_path / experiment_id

        if not exp_dir.exists():
            raise ValueError(f"No experiment found with ID {experiment_id}")

        experiment_data = {}

        # Load experiment info (machine/environment details and status)
        exp_info_path = exp_dir / "_experiment_info.json"
        if exp_info_path.exists():
            try:
                with open(exp_info_path) as f:
                    experiment_data["experiment_info"] = json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(
                    f"Invalid JSON format in experiment info file for {experiment_id}"
                )

        # Load training results if available
        metrics_path = exp_dir / "training" / "final_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    experiment_data["training_results"] = json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(
                    f"Invalid JSON format in training metrics file for {experiment_id}"
                )

        # Note: Configuration is available in _experiment_config.yaml for reproducibility
        config_path = exp_dir / "_experiment_config.yaml"
        if config_path.exists():
            experiment_data["config_file_path"] = str(config_path)

        if not experiment_data:
            raise ValueError(f"No valid experiment files found for {experiment_id}")

        return experiment_data
