import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import h5py

from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.dataset_manager import DatasetConfig, DatasetManager
from hep_foundation.data.task_config import TaskConfig
from hep_foundation.data.dataset_visualizer import create_plot_from_hist_data
from hep_foundation.models.model_factory import ModelFactory
from hep_foundation.models.model_registry import ModelRegistry
from hep_foundation.models.variational_autoencoder import (
    AnomalyDetectionEvaluator,
    VAEConfig,
    VariationalAutoEncoder,
)
from hep_foundation.training.model_trainer import ModelTrainer, TrainingConfig
from hep_foundation.utils.plot_utils import plot_combined_training_histories
from hep_foundation.models.base_model import CustomKerasModelWrapper


class FoundationModelPipeline:
    """
    Pipeline for training and evaluating foundation models.

    This class provides methods for:
    1. Training foundation models
    2. Evaluating foundation models for anomaly detection
    3. Evaluating foundation models for regression tasks
    """

    def __init__(self, experiments_output_dir: str = "foundation_experiments", processed_data_parent_dir: Optional[str] = None):
        """
        Initialize the foundation model pipeline.

        Args:
            experiments_output_dir: Base directory for storing individual experiment results.
            processed_data_parent_dir: Parent directory for 'processed_datasets'. 
                                       If None, 'processed_datasets' is at the workspace root.
        """
        self.logger = get_logger(__name__)

        self.experiments_output_dir = Path(experiments_output_dir)
        self.experiments_output_dir.mkdir(parents=True, exist_ok=True)

        if processed_data_parent_dir is None:
            # Default for script runs: datasets are at the root level in 'processed_datasets'
            self.processed_datasets_dir = Path("processed_datasets")
        else:
            # For tests or if specified: datasets are relative to this given parent
            self.processed_datasets_dir = Path(processed_data_parent_dir) / "processed_datasets"
        
        self.processed_datasets_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Foundation Model Pipeline initialized.")
        self.logger.info(f"  Experiment outputs will be in: {self.experiments_output_dir.absolute()}")
        self.logger.info(f"  Processed datasets will be in: {self.processed_datasets_dir.absolute()}")
        self.logger.info(f"TensorFlow: {tf.__version__} (Eager: {tf.executing_eagerly()})")

    def run_process(
        self,
        process_name: str,
        dataset_config: DatasetConfig,
        vae_model_config: dict,
        dnn_model_config: dict,
        vae_training_config: TrainingConfig,
        dnn_training_config: TrainingConfig,
        task_config: TaskConfig,
        delete_catalogs: bool = True,
        foundation_model_path: str = None,
    ) -> bool:
        """
        Run the specified process with provided configurations.

        Args:
            process_name: Name of the process to run ("train", "anomaly", or "regression")
            dataset_config: Configuration for dataset processing
            vae_model_config: Configuration for VAE model
            dnn_model_config: Configuration for DNN model
            vae_training_config: Configuration for VAE training
            dnn_training_config: Configuration for DNN training
            task_config: Configuration for task processing
            delete_catalogs: Whether to delete catalogs after processing
            foundation_model_path: Path to the foundation model encoder to use for encoding
        """
        valid_processes = ["train", "anomaly", "regression"]
        if process_name not in valid_processes:
            self.logger.error(
                f"Invalid process name: {process_name}. Must be one of {valid_processes}"
            )
            return False

        if process_name == "train":
            return self.train_foundation_model(
                dataset_config=dataset_config,
                model_config=vae_model_config,
                training_config=vae_training_config,
                task_config=task_config,
                delete_catalogs=delete_catalogs,
            )
        elif process_name == "anomaly":
            return self.evaluate_foundation_model_anomaly_detection(
                dataset_config=dataset_config,
                task_config=task_config,
                delete_catalogs=delete_catalogs,
                foundation_model_path=foundation_model_path,
                vae_training_config=vae_training_config,
            )
        elif process_name == "regression":
            return self.evaluate_foundation_model_regression(
                dataset_config=dataset_config,
                dnn_model_config=dnn_model_config,
                dnn_training_config=dnn_training_config,
                task_config=task_config,
                delete_catalogs=delete_catalogs,
                foundation_model_path=foundation_model_path,
            )
        else:
            self.logger.error(f"Unknown process name: {process_name}")
            return False

    def train_foundation_model(
        self,
        dataset_config: DatasetConfig,
        model_config: dict,
        training_config: TrainingConfig,
        task_config: TaskConfig,
        delete_catalogs: bool = True,
    ) -> bool:
        """
        Train a foundation model using provided configurations.
        """
        self.logger.info("=" * 100)
        self.logger.info("Training Foundation Model")
        self.logger.info("=" * 100)

        try:
            # Add logging for signal keys
            if dataset_config.signal_keys:
                self.logger.info(f"Signal keys to process: {dataset_config.signal_keys}")
            else:
                self.logger.warning("No signal keys specified in dataset_config")

            # Helper function for JSON serialization
            def ensure_serializable(obj):
                """Recursively convert numpy types to Python native types"""
                if isinstance(obj, dict):
                    return {
                        key: ensure_serializable(value) for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [ensure_serializable(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            # Initialize registry
            registry = ModelRegistry(str(self.experiments_output_dir))
            self.logger.info(f"Registry initialized at: {registry.db_path}")

            # 1. Initialize managers
            self.logger.info("Initializing managers...")
            data_manager = DatasetManager(base_dir=self.processed_datasets_dir)

            # 2. Validate Configs
            dataset_config.validate()
            self.logger.info("Validated dataset config")

            training_config.validate()
            self.logger.info("Validated training config")

            # 3. Load datasets
            self.logger.info("Loading datasets...")
            
            # Add detailed logging for ATLAS dataset loading
            self.logger.info("Loading ATLAS datasets...")
            train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
                dataset_config=dataset_config,
                validation_fraction=dataset_config.validation_fraction,
                test_fraction=dataset_config.test_fraction,
                batch_size=training_config.batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=dataset_config.include_labels,
                delete_catalogs=delete_catalogs,
            )
            
            # Add logging to inspect dataset structure
            self.logger.info("ATLAS datasets loaded. Inspecting structure...")
            for name, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
                try:
                    for batch in dataset.take(1):
                        if isinstance(batch, tuple):
                            features, labels = batch
                            self.logger.info(f"{name} dataset - Features shape: {features.shape}, Labels shape: {labels.shape}")
                        else:
                            self.logger.info(f"{name} dataset - Batch shape: {batch.shape}")
                except Exception as e:
                    self.logger.error(f"Error inspecting {name} dataset: {str(e)}")

            # Get the dataset ID and verify it exists
            dataset_id = data_manager.get_current_dataset_id()
            self.logger.info(f"Created/loaded dataset with ID: {dataset_id}")
            
            # Verify dataset file exists and log its size
            dataset_path = data_manager.get_current_dataset_path()
            if dataset_path.exists():
                self.logger.info(f"Dataset file exists at: {dataset_path}")
                self.logger.info(f"Dataset file size: {dataset_path.stat().st_size / (1024*1024):.2f} MB")
                
                # Add HDF5 structure inspection
                try:
                    with h5py.File(dataset_path, 'r') as f:
                        self.logger.info("Dataset HDF5 structure:")
                        def print_structure(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                self.logger.info(f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                            elif isinstance(obj, h5py.Group):
                                self.logger.info(f"  Group: {name}")
                        f.visititems(print_structure)
                except Exception as e:
                    self.logger.error(f"Error inspecting HDF5 structure: {str(e)}")
            else:
                self.logger.error(f"Dataset file not found at: {dataset_path}")

            # 4. Register experiment with existing dataset
            self.logger.info("Registering experiment...")
            model_config_dict = {
                "model_type": model_config["model_type"],
                "architecture": {
                    **model_config["architecture"],
                    "input_shape": (
                        task_config.input.get_total_feature_size(),
                    ),  # Must be a tuple
                },
                "hyperparameters": model_config["hyperparameters"],
            }
            training_config_dict = {
                "batch_size": training_config.batch_size,
                "epochs": training_config.epochs,
                "learning_rate": training_config.learning_rate,
                "early_stopping": training_config.early_stopping,
            }
            experiment_id = registry.register_experiment(
                name="Foundation_VAE_Model",
                dataset_id=dataset_id,
                model_config=model_config_dict,
                training_config=training_config_dict,
                description="Training a foundation VAE model for feature encoding",
            )
            self.logger.info(f"Created experiment: {experiment_id}")

            # 5. Create and Build Model
            self.logger.info("Creating model...")
            try:
                model = ModelFactory.create_model(
                    model_type="variational_autoencoder", config=model_config_dict
                )
                model.build()
            except Exception as e:
                self.logger.error(f"Model creation failed: {str(e)}")
                self.logger.error(
                    f"Model config used: {json.dumps(model_config_dict, indent=2)}"
                )
                raise

            self.logger.info("Model created")
            self.logger.info(model.model.summary())

            # 6. Train Model
            self.logger.info("Setting up model and callbacks...")
            trainer = ModelTrainer(model=model, training_config=training_config_dict)

            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=training_config_dict["early_stopping"]["patience"],
                    min_delta=training_config_dict["early_stopping"]["min_delta"],
                    restore_best_weights=True,
                )
            ]

            # Start training
            self.logger.info("Starting training...")
            try:
                training_results = trainer.train(
                    dataset=train_dataset,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    plot_training=True,
                    plots_dir=self.experiments_output_dir / experiment_id / "training",
                )

                # Evaluate Model
                self.logger.info("Evaluating model...")
                test_results = trainer.evaluate(test_dataset)

                # Combine results
                final_metrics = {
                    **training_results["final_metrics"],
                    **test_results,
                    "training_duration": training_results["training_duration"],
                    "epochs_completed": training_results["epochs_completed"],
                    "history": training_results["history"],
                }

                # Update experiment data
                registry.complete_training(
                    experiment_id=experiment_id,
                    final_metrics=ensure_serializable(final_metrics),
                )

                # Save the trained model
                self.logger.info("Saving trained model...")
                model_metadata = {
                    "test_loss": test_results.get("test_loss", 0.0),
                    "test_mse": test_results.get("test_mse", 0.0),
                    "final_train_loss": training_results["final_metrics"].get(
                        "loss", 0.0
                    ),
                    "final_val_loss": training_results["final_metrics"].get(
                        "val_loss", 0.0
                    ),
                    "training_duration": training_results["training_duration"],
                }

                registry.save_model(
                    experiment_id=experiment_id,
                    models={
                        "encoder": model.encoder,
                        "decoder": model.decoder,
                        "full_model": model.model,
                    },
                    model_name="foundation_model",
                    metadata=ensure_serializable(model_metadata),
                )

            except Exception as e:
                self.logger.error(f"\nTraining failed with error: {str(e)}")
                self.logger.info("Dataset inspection:")
                for i, batch in enumerate(train_dataset.take(1)):
                    if isinstance(batch, tuple):
                        features, _ = batch
                        self.logger.info(
                            f"Training batch {i} features shape: {features.shape}"
                        )
                        self.logger.info(
                            f"Sample of features: \n{features[0, :10]}"
                        )  # Show first 10 features of first event
                    else:
                        self.logger.info(f"Training batch {i} shape: {batch.shape}")
                        self.logger.info(
                            f"Sample of data: \n{batch[0, :10]}"
                        )  # Show first 10 features of first event
                raise

            # Display Results
            self.logger.info("=" * 100)
            self.logger.info("Experiment Results")
            self.logger.info("=" * 100)

            experiment_data = registry.get_experiment_data(experiment_id)

            self.logger.info(f"Experiment ID: {experiment_id}")
            self.logger.info(f"Status: {experiment_data['experiment_info']['status']}")

            if "training_results" in experiment_data:
                training_results = experiment_data["training_results"]
                self.logger.info(
                    f"Training Duration: {training_results['training_duration']:.2f}s"
                )
                self.logger.info(
                    f"Epochs Completed: {training_results['epochs_completed']}"
                )

                self.logger.info("Metrics:")

                def print_metrics(metrics, indent=2):
                    """Helper function to print metrics with proper formatting"""
                    for key, value in metrics.items():
                        indent_str = " " * indent
                        if isinstance(value, dict):
                            self.logger.info(f"{indent_str}{key}:")
                            print_metrics(value, indent + 2)
                        elif isinstance(value, (float, int)):
                            self.logger.info(f"{indent_str}{key}: {value:.6f}")
                        else:
                            self.logger.info(f"{indent_str}{key}: {value}")

                print_metrics(training_results["final_metrics"])

            self.logger.info("Foundation model training completed successfully")
            return True

        except Exception as e:
            self.logger.error(
                f"Foundation model training failed: {type(e).__name__}: {str(e)}"
            )
            self.logger.error("Error context:")
            raise


    def evaluate_foundation_model_anomaly_detection(
        self,
        dataset_config: DatasetConfig,
        task_config: TaskConfig,
        delete_catalogs: bool = True,
        foundation_model_path: str = None,
        vae_training_config: TrainingConfig = None,
        eval_batch_size: int = 1024
    ):
        """
        Evaluate a trained foundation model (VAE) for anomaly detection.

        Loads a pre-trained VAE, test/signal datasets, performs evaluation,
        and saves results directly in the foundation model's experiment directory.
        """
        self.logger.info("=" * 100)
        self.logger.info("Evaluating Foundation Model for Anomaly Detection")
        self.logger.info("=" * 100)

        if not foundation_model_path:
            self.logger.error("Foundation model path must be provided for anomaly evaluation.")
            return False

        foundation_model_path = Path(foundation_model_path)
        model_weights_path = foundation_model_path / "models" / "foundation_model" / "full_model"
        config_path = foundation_model_path / "experiment_data.json"

        if not model_weights_path.exists():
            model_weights_path_h5 = model_weights_path.with_suffix(".weights.h5")
            if model_weights_path_h5.exists():
                model_weights_path = model_weights_path_h5
                self.logger.info("Found model weights with .h5 extension.")
            else:
                self.logger.error(f"Foundation model weights not found at: {model_weights_path} or {model_weights_path_h5}")
                return False

        if not config_path.exists():
            self.logger.error(f"Foundation model experiment data not found at: {config_path}")
            return False

        try:
            # Initialize registry
            registry = ModelRegistry(str(self.experiments_output_dir))
            self.logger.info(f"Registry initialized at: {registry.db_path}")

            # 1. Load original model configuration and experiment ID
            self.logger.info(f"Loading original model config from: {config_path}")
            with open(config_path) as f:
                original_experiment_data = json.load(f)

            # Get the experiment ID from the foundation model path
            foundation_experiment_id = foundation_model_path.name

            # Navigate potential nested structure if saved via registry's complete_training
            if "experiment_info" in original_experiment_data and "model_config" in original_experiment_data["experiment_info"]:
                original_model_config = original_experiment_data["experiment_info"]["model_config"]
            elif "model_config" in original_experiment_data:
                original_model_config = original_experiment_data["model_config"]
            else:
                self.logger.error(f"Could not find 'model_config' in experiment data: {config_path}")
                return False

            if not original_model_config or original_model_config.get("model_type") != "variational_autoencoder":
                self.logger.error(f"Loaded config is not for a variational_autoencoder: {config_path}")
                return False

            # 2. Load Datasets
            self.logger.info("Initializing Dataset Manager...")
            data_manager = DatasetManager(base_dir=self.processed_datasets_dir)

            # Use eval_batch_size or derive from vae_training_config if provided
            batch_size = eval_batch_size
            if vae_training_config:
                try:
                    batch_size = vae_training_config.batch_size
                    self.logger.info(f"Using batch size from provided VAE training config: {batch_size}")
                except AttributeError:
                    self.logger.warning("Provided vae_training_config lacks batch_size, using default.")
            else:
                self.logger.info(f"Using default evaluation batch size: {batch_size}")

            # Load test dataset (background) and signal datasets
            self.logger.info("Loading background (test) dataset...")
            _, _, test_dataset = data_manager.load_atlas_datasets(
                dataset_config=dataset_config,
                validation_fraction=0.0,
                test_fraction=1.0,
                batch_size=batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=False,
                delete_catalogs=delete_catalogs,
            )
            background_dataset_id_for_plots = data_manager.get_current_dataset_id()
            self.logger.info("Loaded background (test) dataset.")

            testing_path = foundation_model_path / "testing"
            testing_path.mkdir(parents=True, exist_ok=True)
            anomaly_dir = testing_path / "anomaly_detection"
            anomaly_dir.mkdir(parents=True, exist_ok=True)
            plots_dir = anomaly_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine background histogram data path to pass to signal dataset loading for comparison plot
            background_hist_data_path_for_comparison = None
            if dataset_config.plot_distributions and background_dataset_id_for_plots:
                background_plot_data_dir = data_manager.get_dataset_dir(background_dataset_id_for_plots) / "plots"
                potential_background_hist_path = background_plot_data_dir / "atlas_dataset_features_hist_data.json"
                if potential_background_hist_path.exists():
                    background_hist_data_path_for_comparison = potential_background_hist_path
                else:
                    self.logger.warning(f"Background histogram data for comparison not found at {potential_background_hist_path}. Comparison plot may be skipped by DatasetManager.")

            # Reload signal datasets, now passing the background_hist_data_path for comparison plot generation within DatasetManager
            if dataset_config.signal_keys:
                self.logger.info("Reloading signal datasets for evaluation and comparison plotting...")
                signal_datasets = data_manager.load_signal_datasets(
                    dataset_config=dataset_config,
                    batch_size=batch_size,
                    include_labels=False,
                    background_hist_data_path=background_hist_data_path_for_comparison # Pass the path here
                )
                self.logger.info(f"Reloaded {len(signal_datasets)} signal datasets for evaluation.")
            else:
                self.logger.warning("No signal keys provided. Skipping signal dataset reloading for evaluation.")
                signal_datasets = {} # Ensure signal_datasets is defined
            
            # Create and Load Model
            self.logger.info("Instantiating VAE model from loaded configuration...")

            # Ensure input_shape is present in the loaded config
            if "input_shape" not in original_model_config["architecture"]:
                self.logger.warning("Input shape missing in loaded model config, deriving from task_config.")
                input_shape = (task_config.input.get_total_feature_size(),)
                if input_shape[0] is None or input_shape[0] <= 0:
                    self.logger.error("Could not determine a valid input shape from task_config.")
                    return False
                original_model_config["architecture"]["input_shape"] = list(input_shape)

            # Ensure hyperparameters like beta_schedule are present
            if "hyperparameters" not in original_model_config:
                original_model_config["hyperparameters"] = {}
            if "beta_schedule" not in original_model_config["hyperparameters"]:
                self.logger.warning("beta_schedule missing in loaded config, using default.")
                original_model_config["hyperparameters"]["beta_schedule"] = {
                    "start": 0.0, "end": 1.0, "warmup_epochs": 0, "cycle_epochs": 0
                }

            # Create VAE model
            vae_config = VAEConfig(
                model_type=original_model_config["model_type"],
                architecture=original_model_config["architecture"],
                hyperparameters=original_model_config["hyperparameters"]
            )
            model = VariationalAutoEncoder(config=vae_config)
            model.build(input_shape=tuple(original_model_config["architecture"]["input_shape"]))
            
            # Load weights
            self.logger.info(f"Loading model weights from: {model_weights_path}")
            try:
                model.model.load_weights(str(model_weights_path)).expect_partial()
                self.logger.info("VAE model loaded successfully.")
                self.logger.info(model.model.summary())
            except Exception as e:
                self.logger.error(f"Failed to load model weights: {str(e)}")
                return False

            # 4. Run Anomaly Detection Evaluation directly on the foundation model's directory
            self.logger.info("Running anomaly detection evaluation...")
            
            # Create custom evaluator that will use the foundation model's directory
            evaluator = AnomalyDetectionEvaluator(
                model=model,
                test_dataset=test_dataset,
                signal_datasets=signal_datasets,
                experiment_id=foundation_experiment_id,
                base_path=self.experiments_output_dir
            )
            
            # Run the evaluation
            evaluator.run_anomaly_detection_test()
            
            # 5. Display Results
            self.logger.info("=" * 100)
            self.logger.info("Anomaly Detection Results")
            self.logger.info("=" * 100)
            
            experiment_data = registry.get_experiment_data(foundation_experiment_id)
            self.logger.info(f"Foundation Model ID: {foundation_experiment_id}")

            # Display anomaly detection specific results
            if "test_results" in experiment_data and "anomaly_detection" in experiment_data["test_results"]:
                anomaly_results = experiment_data["test_results"]["anomaly_detection"]
                self.logger.info(f"Timestamp: {anomaly_results.get('timestamp')}")
                self.logger.info(f"Background Events: {anomaly_results.get('background_events')}")
                self.logger.info(f"Plots Directory: {anomaly_results.get('plots_directory')}")
                
                if "signal_results" in anomaly_results:
                    self.logger.info("Signal Results:")
                    for signal_name, results in anomaly_results["signal_results"].items():
                        self.logger.info(f"  Signal: {signal_name} (Events: {results.get('n_events')})")
                        if "reconstruction_metrics" in results:
                            recon_metrics = results["reconstruction_metrics"]
                            self.logger.info("    Reconstruction Metrics:")
                            self.logger.info(f"      AUC: {recon_metrics.get('roc_auc', 'N/A'):.4f}")
                            self.logger.info(f"      Separation: {recon_metrics.get('separation', 'N/A'):.4f}")
                        if "kl_divergence_metrics" in results:
                            kl_metrics = results["kl_divergence_metrics"]
                            self.logger.info("    KL Divergence Metrics:")
                            self.logger.info(f"      AUC: {kl_metrics.get('roc_auc', 'N/A'):.4f}")
                            self.logger.info(f"      Separation: {kl_metrics.get('separation', 'N/A'):.4f}")
            else:
                self.logger.warning("No anomaly detection results found in experiment data.")

            self.logger.info("Anomaly detection evaluation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Anomaly detection evaluation failed: {type(e).__name__}: {str(e)}")
            self.logger.exception("Detailed traceback:")
            return False

    def evaluate_foundation_model_regression(
        self,
        dataset_config: DatasetConfig,
        dnn_model_config: dict,
        dnn_training_config: TrainingConfig,
        task_config: TaskConfig,
        delete_catalogs: bool = True,
        foundation_model_path: str = None,
    ) -> bool:
        """
        Evaluate a foundation model for regression tasks by comparing three approaches:
        1. "From Scratch": An encoder (matching foundation model's encoder architecture) + regressor, trained end-to-end.
        2. "Fine-Tuned": The pre-trained foundation model encoder + regressor, trained end-to-end (encoder is fine-tuned).
        3. "Fixed": The pre-trained foundation model encoder (frozen) + regressor, only regressor is trained.
        """
        self.logger.info("=" * 100)
        self.logger.info("Evaluating Foundation Model for Regression (New Approach)")
        self.logger.info("=" * 100)

        if not foundation_model_path:
            self.logger.error(
                "Foundation model path must be provided for regression evaluation."
            )
            return False
        
        foundation_model_dir = Path(foundation_model_path)
        # New nested path for regression evaluation outputs
        regression_eval_dir = foundation_model_dir / "testing" / "regression_evaluation"
        regression_eval_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize registry (optional, as per original commenting) and data manager
            # registry = ModelRegistry(str(self.experiments_output_dir))
            data_manager = DatasetManager(
                base_dir=self.processed_datasets_dir
            )

            # 1. Load dataset with regression labels
            self.logger.info("Loading datasets with regression labels...")
            train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
                dataset_config=dataset_config,
                validation_fraction=dataset_config.validation_fraction,
                test_fraction=dataset_config.test_fraction,
                batch_size=dnn_training_config.batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=True,  # Ensure labels are included
                delete_catalogs=delete_catalogs,
            )

            dataset_config.validate()
            self.logger.info("Validated dataset config")
            dnn_training_config.validate()
            self.logger.info("Validated training config")

            original_input_shape = (task_config.input.get_total_feature_size(),)
            regression_output_shape = (
                task_config.labels[0].get_total_feature_size(),
            )

            # 2. Load Pre-trained Foundation Encoder & its Config
            self.logger.info(f"Loading foundation model VAE encoder and its configuration from: {foundation_model_dir}")
            
            # Load VAE experiment config to get encoder architecture
            vae_config_path = foundation_model_dir / "experiment_data.json"
            if not vae_config_path.exists():
                self.logger.error(f"Foundation model experiment data not found at: {vae_config_path}")
                return False
            with open(vae_config_path) as f:
                vae_experiment_data = json.load(f)
            
            if "experiment_info" in vae_experiment_data and "model_config" in vae_experiment_data["experiment_info"]:
                original_vae_model_config = vae_experiment_data["experiment_info"]["model_config"]
            elif "model_config" in vae_experiment_data: # Fallback for older structures
                original_vae_model_config = vae_experiment_data["model_config"]
            else:
                self.logger.error(f"Could not find 'model_config' in VAE experiment data: {vae_config_path}")
                return False

            if not original_vae_model_config or original_vae_model_config.get("model_type") != "variational_autoencoder":
                self.logger.error(f"Loaded config from {vae_config_path} is not for a variational_autoencoder.")
                return False

            vae_arch_config = original_vae_model_config["architecture"]
            latent_dim = vae_arch_config["latent_dim"]
            encoder_hidden_layers = vae_arch_config.get("encoder_layers", []) # E.g. [128, 64, 48]
            encoder_activation = vae_arch_config.get("activation", "relu")

            # Load the pre-trained VAE encoder model part
            # This model typically outputs [z_mean, z_log_var, z]. We need z_mean.
            pretrained_vae_full_encoder_path = foundation_model_dir / "models" / "foundation_model" / "encoder"
            if not pretrained_vae_full_encoder_path.exists():
                self.logger.error(f"Pretrained VAE encoder model not found at {pretrained_vae_full_encoder_path}")
                return False
            
            self.logger.info(f"Loading pre-trained VAE full encoder from: {pretrained_vae_full_encoder_path}")
            pretrained_vae_full_encoder = tf.keras.models.load_model(pretrained_vae_full_encoder_path)
            self.logger.info("Pre-trained VAE full encoder loaded.")


            # Helper function to build regressor head
            def build_regressor_head(name_suffix: str) -> tf.keras.Model:
                regressor_config = {
                    "model_type": "dnn_predictor",
                    "architecture": {
                        **dnn_model_config["architecture"],
                        "input_shape": (latent_dim,),
                        "output_shape": regression_output_shape,
                        "name": f"{dnn_model_config['architecture'].get('name', 'regressor')}_{name_suffix}",
                    },
                    "hyperparameters": dnn_model_config["hyperparameters"],
                }
                regressor_model_wrapper = ModelFactory.create_model(
                    model_type="dnn_predictor", config=regressor_config
                )
                if regressor_model_wrapper is None:
                    self.logger.error(f"ModelFactory returned None for dnn_predictor '{name_suffix}'. Cannot build regressor head.")
                    raise ValueError(f"ModelFactory failed for dnn_predictor '{name_suffix}'")
                
                # The DNNEstimator's build method might take input_shape, or derive it from config.
                # The config already has input_shape=(latent_dim,)
                regressor_model_wrapper.build() # This should populate regressor_model_wrapper.model
                
                if regressor_model_wrapper.model is None:
                    self.logger.error(f"Regressor wrapper '{name_suffix}' built, but its .model is still None.")
                    raise ValueError(f"Regressor wrapper '{name_suffix}' .model is None after build.")
                
                return regressor_model_wrapper.model # This is the actual Keras DNN model
                
            # Helper function to train and evaluate a model
            def train_and_evaluate_model(model_name: str, combined_keras_model: tf.keras.Model, plots_subdir: str):
                self.logger.info(f"Training {model_name} model...")
                # combined_keras_model.summary(print_fn=self.logger.info) # Log summary of the Keras model

                # Wrap the Keras model with CustomKerasModelWrapper for ModelTrainer
                wrapped_model_for_trainer = CustomKerasModelWrapper(combined_keras_model, name=model_name)
                wrapped_model_for_trainer.summary() # Log summary via wrapper

                trainer_config_dict = {
                    "batch_size": dnn_training_config.batch_size,
                    "epochs": dnn_training_config.epochs,
                    "learning_rate": dnn_training_config.learning_rate,
                    "early_stopping": dnn_training_config.early_stopping,
                }
                # Note: ModelTrainer will call wrapped_model_for_trainer.compile() and wrapped_model_for_trainer.build()
                trainer = ModelTrainer(model=wrapped_model_for_trainer, training_config=trainer_config_dict)
                
                model_plots_dir = regression_eval_dir / plots_subdir
                model_plots_dir.mkdir(parents=True, exist_ok=True)

                training_results = trainer.train(
                    dataset=train_dataset,
                    validation_data=val_dataset,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            patience=dnn_training_config.early_stopping["patience"],
                            min_delta=dnn_training_config.early_stopping["min_delta"],
                            restore_best_weights=True,
                        )
                    ],
                    plot_training=True,
                    plots_dir=model_plots_dir,
                )
                test_metrics = trainer.evaluate(test_dataset)
                self.logger.info(f"{model_name} - Test Metrics: {test_metrics}")
                return training_results, test_metrics

            all_training_histories = {}
            all_test_metrics = {}

            # --- Model 1: "From Scratch" ---
            self.logger.info("-" * 50)
            self.logger.info("Building Model 1: From Scratch")
            
            scratch_encoder_layers = []
            for units in encoder_hidden_layers:
                scratch_encoder_layers.append(tf.keras.layers.Dense(units, activation=encoder_activation))
            scratch_encoder_layers.append(tf.keras.layers.Dense(latent_dim, name="scratch_latent_space")) # No activation for latent typically

            scratch_encoder_part = tf.keras.Sequential(scratch_encoder_layers, name="scratch_encoder")
            
            scratch_regressor_dnn = build_regressor_head("from_scratch")

            model_inputs = tf.keras.Input(shape=original_input_shape, name="input_features")
            encoded_scratch = scratch_encoder_part(model_inputs)
            predictions_scratch = scratch_regressor_dnn(encoded_scratch)
            model_from_scratch = tf.keras.Model(
                inputs=model_inputs, outputs=predictions_scratch, name="Regressor_From_Scratch"
            )
            
            scratch_results, scratch_test_metrics = train_and_evaluate_model(
                "From_Scratch", model_from_scratch, "from_scratch_plots"
            )
            all_training_histories["From_Scratch"] = scratch_results.get("history", {})
            all_test_metrics["From_Scratch"] = scratch_test_metrics


            # --- Model 2: "Fine-Tuned" ---
            self.logger.info("-" * 50)
            self.logger.info("Building Model 2: Fine-Tuned Pre-trained Encoder")

            # Adapt pre-trained VAE encoder to output only z_mean (or equivalent single tensor)
            vae_encoder_input_layer = tf.keras.Input(shape=original_input_shape, name="vae_encoder_input")
            vae_encoder_all_outputs = pretrained_vae_full_encoder(vae_encoder_input_layer)
            
            # Assuming z_mean is the first output if multiple, or the direct output if single
            if isinstance(vae_encoder_all_outputs, list):
                if not vae_encoder_all_outputs:
                    self.logger.error("Pretrained VAE encoder returned an empty list of outputs.")
                    return False
                latent_representation_output = vae_encoder_all_outputs[0] 
            else:
                latent_representation_output = vae_encoder_all_outputs

            fine_tuned_encoder_part = tf.keras.Model(
                inputs=vae_encoder_input_layer, 
                outputs=latent_representation_output, 
                name="fine_tuned_pretrained_encoder"
            )
            fine_tuned_encoder_part.trainable = True # Ensure it's trainable

            fine_tuned_regressor_dnn = build_regressor_head("fine_tuned")

            model_inputs_ft = tf.keras.Input(shape=original_input_shape, name="input_features_ft")
            encoded_ft = fine_tuned_encoder_part(model_inputs_ft)
            predictions_ft = fine_tuned_regressor_dnn(encoded_ft)
            model_fine_tuned = tf.keras.Model(
                inputs=model_inputs_ft, outputs=predictions_ft, name="Regressor_Fine_Tuned"
            )

            finetuned_results, finetuned_test_metrics = train_and_evaluate_model(
                "Fine_Tuned", model_fine_tuned, "fine_tuned_plots"
            )
            all_training_histories["Fine_Tuned"] = finetuned_results.get("history", {})
            all_test_metrics["Fine_Tuned"] = finetuned_test_metrics

            # --- Model 3: "Fixed" (Feature Extractor) ---
            self.logger.info("-" * 50)
            self.logger.info("Building Model 3: Fixed Pre-trained Encoder")

            # Re-define or clone the encoder part to set trainable to False cleanly
            # We can reuse the same structure as fine_tuned_encoder_part but set trainable status
            fixed_encoder_part = tf.keras.Model(
                inputs=vae_encoder_input_layer, # re-use input definition for clarity
                outputs=latent_representation_output, # re-use output definition
                name="fixed_pretrained_encoder"
            )
            # CRITICAL: Set trainable to False *before* compiling the combined model
            fixed_encoder_part.trainable = False

            fixed_regressor_dnn = build_regressor_head("fixed_encoder")
            
            model_inputs_fx = tf.keras.Input(shape=original_input_shape, name="input_features_fx")
            encoded_fx = fixed_encoder_part(model_inputs_fx) # Encoder is frozen here
            predictions_fx = fixed_regressor_dnn(encoded_fx)
            model_fixed = tf.keras.Model(
                inputs=model_inputs_fx, outputs=predictions_fx, name="Regressor_Fixed_Encoder"
            )

            fixed_results, fixed_test_metrics = train_and_evaluate_model(
                "Fixed_Encoder", model_fixed, "fixed_encoder_plots"
            )
            all_training_histories["Fixed_Encoder"] = fixed_results.get("history", {})
            all_test_metrics["Fixed_Encoder"] = fixed_test_metrics
            
            # 6. Generate combined training plot for all three models
            self.logger.info("-" * 50)
            self.logger.info("Generating combined training plot...")
            try:
                if any(all_training_histories.values()):
                    comparison_plot_path = regression_eval_dir / "regression_training_comparison.png"
                    plot_combined_training_histories(
                        histories=all_training_histories,
                        output_path=comparison_plot_path,
                        title="Regression Training Comparison: Scratch vs Fine-Tuned vs Fixed",
                    )
                    self.logger.info(f"Combined training plot saved to: {comparison_plot_path}")
                else:
                    self.logger.warning("Skipping combined plot: No history data found.")
            except Exception as plot_error:
                self.logger.error(f"Failed to generate combined training plot: {plot_error}")

            # 7. Display and Save Final Evaluation Results
            self.logger.info("=" * 100)
            self.logger.info("Final Regression Evaluation Summary")
            self.logger.info("=" * 100)

            final_summary = {}
            for model_name, test_metrics in all_test_metrics.items():
                self.logger.info(f"--- {model_name} Model ---")
                self.logger.info(f"  Test Loss: {test_metrics.get('test_loss', 'N/A'):.6f}")
                self.logger.info(f"  Test MSE: {test_metrics.get('test_mse', 'N/A'):.6f}")
                # Add other relevant metrics if present
                for metric, value in test_metrics.items():
                    if metric not in ['test_loss', 'test_mse']:
                         self.logger.info(f"  Test {metric}: {value:.6f}")
                final_summary[model_name] = {
                    "training_metrics": all_training_histories[model_name].get("final_metrics", {}), # if available
                    "test_metrics": test_metrics,
                    "epochs_completed": all_training_histories[model_name].get("epochs_completed", "N/A") # if available
                }
            
            # Save summary to a JSON file
            summary_file_path = regression_eval_dir / "regression_evaluation_summary.json"
            try:
                with open(summary_file_path, 'w') as f:
                    # Helper to convert numpy types if any sneak in, though ModelTrainer usually returns native types
                    def ensure_serializable_summary(obj):
                        if isinstance(obj, np.integer): return int(obj)
                        if isinstance(obj, np.floating): return float(obj)
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        if isinstance(obj, dict): return {k: ensure_serializable_summary(v) for k, v in obj.items()}
                        if isinstance(obj, list): return [ensure_serializable_summary(i) for i in obj]
                        return obj
                    json.dump(ensure_serializable_summary(final_summary), f, indent=2)
                self.logger.info(f"Regression evaluation summary saved to: {summary_file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save regression summary: {str(e)}")


            # The registry part was commented out in the original, keeping it that way.
            # If re-enabled, it would need to be adapted for these three models.
            # Example:
            # experiment_id = registry.register_experiment(
            #     name="Foundation Model Regression Evaluation (3-way)",
            #     dataset_id=data_manager.get_current_dataset_id(),
            #     model_config={ 'dnn_regressor_head_config': dnn_model_config, 'vae_encoder_config': original_vae_model_config },
            #     training_config=dnn_training_config, # This might need to be more specific
            #     description="Comparison of regression: From Scratch, Fine-Tuned Pretrained, Fixed Pretrained"
            # )
            # registry.update_experiment_data( # Custom update or store results in a structured way
            #     experiment_id=experiment_id,
            #     data_to_add={'regression_results': final_summary}
            # )


            self.logger.info("Regression evaluation (new approach) completed successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Regression evaluation (new approach) failed: {type(e).__name__}: {str(e)}")
            self.logger.exception("Detailed traceback for regression evaluation failure:") # Logs full traceback
            return False
