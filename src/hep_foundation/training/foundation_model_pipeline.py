import json
from pathlib import Path

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


class FoundationModelPipeline:
    """
    Pipeline for training and evaluating foundation models.

    This class provides methods for:
    1. Training foundation models
    2. Evaluating foundation models for anomaly detection
    3. Evaluating foundation models for regression tasks
    """

    def __init__(self, base_dir: str = "foundation_experiments"):
        """
        Initialize the foundation model pipeline.

        Args:
            base_dir: Base directory for storing experiment results
        """
        # Setup self.logger
        self.logger = get_logger(__name__)

        # Create experiment directory
        self.experiment_dir = Path(base_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Foundation Model Pipeline initialized with base directory: {self.experiment_dir.absolute()}"
        )
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
            registry = ModelRegistry(str(self.experiment_dir))
            self.logger.info(f"Registry initialized at: {registry.db_path}")

            # 1. Initialize managers
            self.logger.info("Initializing managers...")
            data_manager = DatasetManager(base_dir=self.experiment_dir / "processed_datasets")

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
                    plots_dir=Path(f"{self.experiment_dir}/{experiment_id}/training"),
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
            registry = ModelRegistry(str(self.experiment_dir))
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
            data_manager = DatasetManager(base_dir=self.experiment_dir / "processed_datasets")

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
                base_path=self.experiment_dir
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
        Evaluate a foundation model for regression tasks.

        This method will:
        1. Load a dataset with regression labels
        2. Train a regression model to predict the labels
        3. Load a trained foundation model
        4. Encode the dataset with the foundation model
        5. Train another regression model to predict the labels from the encoded dataset
        6. Evaluate the model's regression performance compared to the original regression model
        7. Save the evaluation results
        """
        self.logger.info("=" * 100)
        self.logger.info("Evaluating Foundation Model for Regression")
        self.logger.info("=" * 100)

        try:
            # Initialize registry and data manager
            ModelRegistry(str(self.experiment_dir))
            data_manager = DatasetManager(base_dir=self.experiment_dir / "processed_datasets")

            # 1. Load dataset with regression labels
            self.logger.info("Loading datasets with regression labels...")
            train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
                dataset_config=dataset_config,
                validation_fraction=dataset_config.validation_fraction,
                test_fraction=dataset_config.test_fraction,
                batch_size=dnn_training_config.batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=True,
                delete_catalogs=delete_catalogs,
            )

            dataset_config.validate()
            self.logger.info("Validated dataset config")

            dnn_training_config.validate()
            self.logger.info("Validated training config")

            # 2. Train baseline regression model
            # Add input shape to the model config
            dnn_model_config_dict = {
                "model_type": dnn_model_config["model_type"],
                "architecture": {
                    **dnn_model_config["architecture"],
                    "input_shape": (
                        task_config.input.get_total_feature_size(),
                    ),  # Must be a tuple
                    "output_shape": (
                        task_config.labels[0].get_total_feature_size(),
                    ),  # For MET prediction (mpx, mpy, sumet)
                    # Note using first label set always for now
                },
                "hyperparameters": dnn_model_config["hyperparameters"],
            }

            self.logger.info("Training baseline regression model...")
            baseline_model = ModelFactory.create_model(
                model_type="dnn_predictor", config=dnn_model_config_dict
            )
            baseline_model.build()
            self.logger.info("Baseline model created")
            self.logger.info(baseline_model.model.summary())

            dnn_training_config_dict = {
                "batch_size": dnn_training_config.batch_size,
                "epochs": dnn_training_config.epochs,
                "learning_rate": dnn_training_config.learning_rate,
                "early_stopping": dnn_training_config.early_stopping,
            }

            self.logger.info("Setting up baseline model trainer...")
            baseline_trainer = ModelTrainer(
                model=baseline_model, training_config=dnn_training_config_dict
            )
            # Log sizes and shapes of baseline datasets
            self.logger.info(f"Baseline train dataset size: {train_dataset.cardinality()}")
            for batch in train_dataset.take(1):
                if isinstance(batch, tuple):
                    features, labels = batch
                    self.logger.info("Baseline training dataset shapes:")
                    self.logger.info(f"  Features: {features.shape}")
                    self.logger.info(f"  Labels: {labels.shape}")
                else:
                    self.logger.info(f"Baseline training dataset shape: {batch.shape}")
            self.logger.info("Training baseline regression model...")
            baseline_results = baseline_trainer.train(
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
                plots_dir=Path(f"{foundation_model_path}/baseline_regression/plots"),
            )

            # 3. Load foundation model encoder
            self.logger.info("Loading foundation model...")
            if foundation_model_path:
                # Load from specified path
                foundation_model_path = Path(foundation_model_path)
                encoder_path = foundation_model_path / "models" / "foundation_model" / "encoder"
                self.logger.info(
                    f"Loading foundation model encoder from: {encoder_path}"
                )
                foundation_model = tf.keras.models.load_model(encoder_path)
                # Get the output shape(s) from the loaded encoder
                encoder_output_shape = foundation_model.output_shape
                self.logger.info(f"Encoder raw output shape: {encoder_output_shape}")

                # Check if the output shape is a list (multiple outputs) or a single tuple
                if isinstance(encoder_output_shape, list):
                    # Assuming the first output is the desired latent representation (e.g., z_mean for VAE)
                    target_output_shape = encoder_output_shape[0]
                    self.logger.info(f"Using first output shape from list: {target_output_shape}")
                elif isinstance(encoder_output_shape, tuple):
                    # Single output case
                    target_output_shape = encoder_output_shape
                else:
                    raise TypeError(f"Unexpected encoder output shape type: {type(encoder_output_shape)}")

                # The shape might be (None, latent_dim), so we take the last element
                latent_dim = target_output_shape[-1]
                if not isinstance(latent_dim, int) or latent_dim <= 0:
                    raise ValueError(f"Could not determine a valid latent dimension from target output shape: {target_output_shape}")
                self.logger.info(f"Determined encoder latent dimension: {latent_dim}")
            else:
                raise ValueError("No foundation model path provided")

            # 4. Create encoded datasets
            self.logger.info("Creating encoded datasets...")

            def encode_batch(
                batch, *args
            ):  # *args is for compatibility with the dataset which also comes with an index argument
                # Get features and labels
                features = batch
                labels = args[0] if args else None

                # Encode the features
                encoded_features = foundation_model(features, training=False) # Use training=False for inference

                # Handle case where foundation model returns a list (e.g., VAE returning [z_mean, z_log_var, z])
                # We typically want z_mean or z for downstream tasks. Assuming the first element is the desired latent representation.
                if isinstance(encoded_features, list):
                    encoded_features = encoded_features[0] # Using z_mean (first output)

                # Return encoded features and labels
                if labels is not None:
                    return encoded_features, labels
                return encoded_features

            self.logger.info("Encoded datasets")

            # Create encoded datasets
            self.logger.info("Creating encoded datasets...")
            encoded_train_dataset = train_dataset.map(encode_batch)
            encoded_val_dataset = val_dataset.map(encode_batch)
            encoded_test_dataset = test_dataset.map(encode_batch)
            self.logger.info("Encoded datasets created")
            # Log sizes and shapes of encoded datasets
            self.logger.info(
                f"Encoded train dataset size: {encoded_train_dataset.cardinality()}"
            )
            for batch in encoded_train_dataset.take(1):
                if isinstance(batch, tuple):
                    features, labels = batch
                    self.logger.info("Encoded training dataset shapes:")
                    self.logger.info(f"  Features: {features.shape}")
                    self.logger.info(f"  Labels: {labels.shape}")

            # 5. Train regression model on encoded data
            self.logger.info("Training regression model on encoded data...")
            # Create a new config for the encoded model with adjusted input shape
            encoded_model_config = {
                **dnn_model_config,
                "architecture": {
                    **dnn_model_config["architecture"],
                    "input_shape": (latent_dim,),  # Use determined latent dimension
                    "output_shape": (
                        task_config.labels[0].get_total_feature_size(),
                    ),  # Same as baseline model
                    "name": "encoded_met_predictor",
                },
            }

            self.logger.info("Creating regression model on encoded data...")
            encoded_model = ModelFactory.create_model(
                model_type="dnn_predictor", config=encoded_model_config
            )
            encoded_model.build()
            self.logger.info("Regression model on encoded data created")
            self.logger.info(encoded_model.model.summary())

            self.logger.info("Setting up regression model training on encoded data...")
            encoded_trainer = ModelTrainer(
                model=encoded_model, training_config=dnn_training_config_dict
            )
            self.logger.info("Training regression model on encoded data...")
            encoded_results = encoded_trainer.train(
                dataset=encoded_train_dataset,
                validation_data=encoded_val_dataset,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=dnn_training_config.early_stopping["patience"],
                        min_delta=dnn_training_config.early_stopping["min_delta"],
                        restore_best_weights=True,
                    )
                ],
                plot_training=True,
                plots_dir=Path(f"{foundation_model_path}/encoded_regression/plots"),
            )
            self.logger.info("Regression model on encoded data trained")
            # 6. Evaluate and compare models
            self.logger.info("Evaluating models...")

            # 6a. Generate combined training plot
            try:
                baseline_history = baseline_results.get("history", {})
                encoded_history = encoded_results.get("history", {})

                if (
                    baseline_history or encoded_history
                ):  # Only plot if there is history data
                    comparison_plot_path = (
                        Path(foundation_model_path)
                        / "testing"
                        / "regression_training_comparison.png"
                    )
                    plot_combined_training_histories(
                        histories={
                            "Baseline": baseline_history,
                            "Encoded": encoded_history,
                        },
                        output_path=comparison_plot_path,
                        title="Baseline vs Encoded Regression Training",  # More specific title
                        # metrics_to_plot=['loss', 'val_loss', 'mse', 'val_mse'], # Example: Plot MSE too
                        # metric_labels={'loss': 'Loss', 'val_loss': 'Val Loss', 'mse': 'MSE', 'val_mse': 'Val MSE'}
                    )
                else:
                    self.logger.warning(
                        "Skipping combined plot generation: No history data found in results."
                    )
            except Exception as plot_error:
                self.logger.error(
                    f"Failed to generate combined training plot: {plot_error}"
                )

            baseline_test_results = baseline_trainer.evaluate(test_dataset)
            encoded_test_results = encoded_trainer.evaluate(encoded_test_dataset)

            # 7. Save evaluation results
            self.logger.info("Saving evaluation results...")
            evaluation_results = {
                "baseline_model": {
                    "training_metrics": baseline_results["final_metrics"],
                    "test_metrics": baseline_test_results,
                },
                "encoded_model": {
                    "training_metrics": encoded_results["final_metrics"],
                    "test_metrics": encoded_test_results,
                },
                "comparison": {
                    "test_loss_ratio": encoded_test_results["test_loss"]
                    / baseline_test_results["test_loss"],
                    "test_mse_ratio": encoded_test_results["test_mse"]
                    / baseline_test_results["test_mse"],
                },
            }

            # Save results to registry
            # experiment_id = registry.register_experiment(
            #     name=experiment_name or "Foundation Model Regression Evaluation",
            #     dataset_id=data_manager.get_current_dataset_id(),
            #     model_config={
            #         'baseline_config': dnn_model_config_dict,
            #         'encoded_config': encoded_model_config
            #     },
            #     training_config={
            #         'baseline_config': dnn_training_config_dict,
            #         'encoded_config': dnn_training_config_dict
            #     },
            #     description=experiment_description or "Evaluation of regression performance with and without foundation model encoding"
            # )

            # registry.complete_training(
            #     experiment_id=experiment_id,
            #     final_metrics=evaluation_results
            # )

            # Display results
            self.logger.info("Regression Evaluation Results:")
            self.logger.info("=" * 50)
            self.logger.info("Baseline Model:")
            self.logger.info(f"Test Loss: {baseline_test_results['test_loss']:.6f}")
            self.logger.info(f"Test MSE: {baseline_test_results['test_mse']:.6f}")
            self.logger.info("Encoded Model:")
            self.logger.info(f"Test Loss: {encoded_test_results['test_loss']:.6f}")
            self.logger.info(f"Test MSE: {encoded_test_results['test_mse']:.6f}")
            self.logger.info("Comparison:")
            self.logger.info(
                f"Test Loss Ratio: {evaluation_results['comparison']['test_loss_ratio']:.3f}"
            )
            self.logger.info(
                f"Test MSE Ratio: {evaluation_results['comparison']['test_mse_ratio']:.3f}"
            )

            self.logger.info("Regression evaluation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Regression evaluation failed: {type(e).__name__}: {str(e)}")
            self.logger.error("Error context:")
            raise
