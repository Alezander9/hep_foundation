import json
import shutil
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import tensorflow as tf

from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import TaskConfig
from hep_foundation.config.training_config import TrainingConfig
from hep_foundation.data.dataset_manager import DatasetManager
from hep_foundation.models.base_model import CustomKerasModelWrapper
from hep_foundation.models.dnn_predictor import DNNPredictorConfig
from hep_foundation.models.model_factory import ModelFactory
from hep_foundation.models.model_registry import ModelRegistry
from hep_foundation.models.variational_autoencoder import (
    BetaSchedule,
    VAEConfig,
    VariationalAutoEncoder,
)
from hep_foundation.pipeline.anomaly_detection_evaluator import (
    AnomalyDetectionEvaluator,
)
from hep_foundation.pipeline.foundation_pipeline_utils import log_evaluation_summary
from hep_foundation.plots.foundation_plot_manager import FoundationPlotManager
from hep_foundation.plots.histogram_manager import HistogramManager
from hep_foundation.training.model_trainer import ModelTrainer
from hep_foundation.utils.utils import print_system_usage


class FoundationModelPipeline:
    """
    Pipeline for training and evaluating foundation models.

    This class provides methods for:
    1. Training foundation models
    2. Evaluating foundation models for anomaly detection
    3. Evaluating foundation models for regression tasks
    """

    def __init__(
        self,
        experiments_output_dir: str = "_foundation_experiments",
        processed_data_parent_dir: Optional[str] = None,
    ):
        """
        Initialize the foundation model pipeline.

        Args:
            experiments_output_dir: Base directory for storing individual experiment results.
            processed_data_parent_dir: Parent directory for '_processed_datasets'.
                                       If None, '_processed_datasets' is at the workspace root.
        """
        self.logger = get_logger(__name__)

        self.experiments_output_dir = Path(experiments_output_dir)
        self.experiments_output_dir.mkdir(parents=True, exist_ok=True)

        if processed_data_parent_dir is None:
            # Default for script runs: datasets are at the root level in '_processed_datasets'
            self.processed_datasets_dir = Path("_processed_datasets")
        else:
            # For tests or if specified: datasets are relative to this given parent
            self.processed_datasets_dir = (
                Path(processed_data_parent_dir) / "_processed_datasets"
            )

        self.processed_datasets_dir.mkdir(parents=True, exist_ok=True)

        # Source config file for reproducibility
        self._source_config_file = None

        # Initialize plot managers and histogram manager
        self.foundation_plot_manager = FoundationPlotManager()
        self.histogram_manager = HistogramManager()

        self.logger.info("Foundation Model Pipeline initialized.")
        self.logger.info(
            f"  Experiment outputs will be in: {self.experiments_output_dir.absolute()}"
        )
        self.logger.info(
            f"  Processed datasets will be in: {self.processed_datasets_dir.absolute()}"
        )
        self.logger.info(
            f"TensorFlow: {tf.__version__} (Eager: {tf.executing_eagerly()})"
        )

    def set_source_config_file(self, config_file_path: str):
        """
        Set the source config file path for reproducibility.

        Args:
            config_file_path: Path to the YAML config file used for this pipeline run
        """
        self._source_config_file = config_file_path
        self.logger.info(f"Source config file set to: {config_file_path}")

    def run_process(
        self,
        process_name: str,
        dataset_config: DatasetConfig,
        task_config: TaskConfig,
        vae_model_config: VAEConfig = None,
        dnn_model_config: DNNPredictorConfig = None,
        vae_training_config: TrainingConfig = None,
        dnn_training_config: TrainingConfig = None,
        delete_catalogs: bool = True,
        foundation_model_path: str = None,
        data_sizes: list = None,
        fixed_epochs: int = None,
    ) -> Union[bool, tuple[str, str]]:
        """
        Run the specified process with provided configurations.

        Args:
            process_name: Name of the process to run ("train", "anomaly", or "regression")
            dataset_config: Configuration for dataset processing
            task_config: Configuration for task processing
            vae_model_config: Configuration for VAE model (optional)
            dnn_model_config: Configuration for DNN model (optional)
            vae_training_config: Configuration for VAE training (optional)
            dnn_training_config: Configuration for DNN training (optional)
            delete_catalogs: Whether to delete catalogs after processing
            foundation_model_path: Path to the foundation model encoder to use for encoding (optional)
            data_sizes: List of training data sizes for regression evaluation (optional)
            fixed_epochs: Number of epochs for regression evaluation (optional)

        Returns:
            bool: Success status for anomaly/regression processes
            tuple[str, str]: Tuple containing (foundation_model_path, dataset_path) for train process, or None if training failed
        """
        valid_processes = ["train", "anomaly", "regression", "signal_classification"]
        if process_name not in valid_processes:
            self.logger.error(
                f"Invalid process name: {process_name}. Must be one of {valid_processes}"
            )
            return False

        # Define method mapping with all possible parameters
        method_map = {
            "train": self.train_foundation_model,
            "anomaly": self.evaluate_foundation_model_anomaly_detection,
            "regression": self.evaluate_foundation_model_regression,
            "signal_classification": self.evaluate_foundation_model_signal_classification,
        }

        # Get the method to call
        method = method_map[process_name]

        # Prepare common arguments that all methods accept
        common_args = {
            "dataset_config": dataset_config,
            "task_config": task_config,
            "delete_catalogs": delete_catalogs,
        }

        # Add optional arguments based on what each method accepts
        if process_name == "train":
            common_args.update(
                {
                    "model_config": vae_model_config,
                    "training_config": vae_training_config,
                }
            )
        elif process_name == "anomaly":
            common_args.update(
                {
                    "foundation_model_path": foundation_model_path,
                    "vae_training_config": vae_training_config,
                }
            )
        elif process_name == "regression":
            common_args.update(
                {
                    "dnn_model_config": dnn_model_config,
                    "dnn_training_config": dnn_training_config,
                    "foundation_model_path": foundation_model_path,
                    "data_sizes": data_sizes,
                    "fixed_epochs": fixed_epochs,
                }
            )
        elif process_name == "signal_classification":
            common_args.update(
                {
                    "dnn_model_config": dnn_model_config,
                    "dnn_training_config": dnn_training_config,
                    "foundation_model_path": foundation_model_path,
                    "data_sizes": data_sizes,
                    "fixed_epochs": fixed_epochs,
                }
            )

        return method(**common_args)

    def run_full_pipeline(
        self,
        dataset_config: DatasetConfig,
        task_config: TaskConfig,
        vae_model_config: VAEConfig,
        dnn_model_config: DNNPredictorConfig,
        vae_training_config: TrainingConfig,
        dnn_training_config: TrainingConfig,
        delete_catalogs: bool = True,
        data_sizes: list = None,
        fixed_epochs: int = None,
    ) -> bool:
        """
        Run the complete foundation model pipeline: train → regression → anomaly.

        This method runs all three processes sequentially, using the trained model
        from the training phase for both evaluation tasks.

        Args:
            dataset_config: Configuration for dataset processing
            task_config: Configuration for task processing
            vae_model_config: Configuration for VAE model
            dnn_model_config: Configuration for DNN model
            vae_training_config: Configuration for VAE training
            dnn_training_config: Configuration for DNN training
            delete_catalogs: Whether to delete catalogs after processing
            data_sizes: List of training data sizes for regression evaluation (optional)
            fixed_epochs: Number of epochs for regression evaluation (optional)

        Returns:
            bool: True if all processes completed successfully, False otherwise
        """
        self.logger.info("=" * 100)
        self.logger.info("RUNNING FULL FOUNDATION MODEL PIPELINE")
        self.logger.info(
            "Process: Train → Anomaly Detection → Regression → Signal Classification"
        )
        self.logger.info("=" * 100)
        self.logger.progress("Starting full foundation model pipeline")

        try:
            # Step 1: Train the foundation model
            self.logger.info("=" * 50)
            self.logger.progress("STEP 1/4: TRAINING FOUNDATION MODEL")
            self.logger.info("=" * 50)

            # Print system usage before training
            print_system_usage("Before Training - ")

            training_result = self.run_process(
                process_name="train",
                dataset_config=dataset_config,
                task_config=task_config,
                vae_model_config=vae_model_config,
                vae_training_config=vae_training_config,
                delete_catalogs=delete_catalogs,
            )

            # Print system usage after training
            print_system_usage("After Training - ")

            if (
                not training_result
                or not isinstance(training_result, tuple)
                or len(training_result) != 2
            ):
                self.logger.error(
                    "Training failed or did not return a valid (model_path, dataset_path) tuple"
                )
                return False

            foundation_model_path, dataset_path = training_result
            self.logger.info(
                f"Training completed successfully. Model saved at: {foundation_model_path}"
            )
            self.logger.info(f"Dataset saved at: {dataset_path}")

            # Step 2: Run anomaly detection evaluation
            self.logger.info("=" * 50)
            self.logger.progress("STEP 2/4: ANOMALY DETECTION EVALUATION")
            self.logger.info("=" * 50)

            # Print system usage before anomaly detection
            print_system_usage("Before Anomaly Detection - ")

            anomaly_success = self.run_process(
                process_name="anomaly",
                dataset_config=dataset_config,
                task_config=task_config,
                foundation_model_path=foundation_model_path,
                vae_training_config=vae_training_config,
                delete_catalogs=delete_catalogs,
            )

            # Print system usage after anomaly detection
            print_system_usage("After Anomaly Detection - ")

            if not anomaly_success:
                self.logger.error("Anomaly detection evaluation failed")
                return False

            # Step 3: Run regression evaluation
            self.logger.info("=" * 50)
            self.logger.progress("STEP 3/4: REGRESSION EVALUATION")
            self.logger.info("=" * 50)

            # Print system usage before regression evaluation
            print_system_usage("Before Regression Evaluation - ")

            regression_success = self.run_process(
                process_name="regression",
                dataset_config=dataset_config,
                task_config=task_config,
                dnn_model_config=dnn_model_config,
                dnn_training_config=dnn_training_config,
                foundation_model_path=foundation_model_path,
                data_sizes=data_sizes,
                fixed_epochs=fixed_epochs,
                delete_catalogs=delete_catalogs,
            )

            # Print system usage after regression evaluation
            print_system_usage("After Regression Evaluation - ")

            if not regression_success:
                self.logger.error("Regression evaluation failed")
                return False

            # Step 4: Run signal classification evaluation
            self.logger.info("=" * 50)
            self.logger.progress("STEP 4/4: SIGNAL CLASSIFICATION EVALUATION")
            self.logger.info("=" * 50)

            # Print system usage before signal classification
            print_system_usage("Before Signal Classification - ")

            signal_classification_success = self.run_process(
                process_name="signal_classification",
                dataset_config=dataset_config,
                task_config=task_config,
                dnn_model_config=dnn_model_config,
                dnn_training_config=dnn_training_config,
                foundation_model_path=foundation_model_path,
                data_sizes=data_sizes,
                fixed_epochs=fixed_epochs,
                delete_catalogs=delete_catalogs,
            )

            # Print system usage after signal classification
            print_system_usage("After Signal Classification - ")

            if not signal_classification_success:
                self.logger.error("Signal classification evaluation failed")
                return False
            self.logger.progress("Step 4/4: Signal classification evaluation completed")

            # Copy dataset plots to foundation model directory for easy reference
            self.logger.info("Copying dataset plots to foundation model directory...")
            try:
                # Extract dataset directory from dataset_path
                dataset_dir = Path(dataset_path).parent
                dataset_plots_dir = dataset_dir / "plots"

                if dataset_plots_dir.exists() and dataset_plots_dir.is_dir():
                    # Create destination directory in foundation model folder
                    foundation_dataset_plots_dir = (
                        Path(foundation_model_path) / "dataset_plots"
                    )
                    foundation_dataset_plots_dir.mkdir(parents=True, exist_ok=True)

                    # Copy all files from dataset plots directory
                    for plot_file in dataset_plots_dir.iterdir():
                        if plot_file.is_file():
                            destination = foundation_dataset_plots_dir / plot_file.name
                            shutil.copy2(plot_file, destination)
                            self.logger.info(f"Copied dataset plot: {plot_file.name}")

                    self.logger.info(
                        f"Dataset plots copied to: {foundation_dataset_plots_dir}"
                    )
                else:
                    self.logger.info("No dataset plots directory found to copy")

            except Exception as e:
                self.logger.warning(f"Failed to copy dataset plots: {str(e)}")
                # Don't fail the entire pipeline for plot copying issues

            # Final summary
            self.logger.info("=" * 100)
            self.logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 100)
            self.logger.info(f"Foundation model: {foundation_model_path}")
            self.logger.info(
                f"Dataset plots copied to: {Path(foundation_model_path) / 'dataset_plots'}"
            )
            self.logger.info(
                "All four processes (train → anomaly → regression → signal classification) completed successfully"
            )
            self.logger.info("=" * 100)
            self.logger.progress(
                "Full foundation model pipeline completed successfully!"
            )

            # Print final system usage
            print_system_usage("Full Pipeline Complete - ")

            return True

        except Exception as e:
            self.logger.error(f"Full pipeline failed: {type(e).__name__}: {str(e)}")
            self.logger.exception("Detailed traceback:")

            # Print system usage on failure
            print_system_usage("Full Pipeline Failed - ")
            return False

    def train_foundation_model(
        self,
        dataset_config: DatasetConfig,
        model_config: VAEConfig,
        training_config: TrainingConfig,
        task_config: TaskConfig,
        delete_catalogs: bool = True,
    ) -> Union[tuple[str, str], None]:
        """
        Train a foundation model using provided configurations.

        Returns:
            tuple[str, str]: Tuple containing (foundation_model_path, dataset_path), or None if training failed
        """

        try:
            # Add logging for signal keys
            if dataset_config.signal_keys:
                self.logger.info(
                    f"Signal keys to process: {dataset_config.signal_keys}"
                )
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

            # Get the dataset ID and verify it exists
            dataset_id = data_manager.get_current_dataset_id()
            self.logger.info(f"Created/loaded dataset with ID: {dataset_id}")

            # Verify dataset file exists and log its size
            dataset_path = data_manager.get_current_dataset_path()
            if dataset_path.exists():
                self.logger.info(f"Dataset file exists at: {dataset_path}")
                self.logger.info(
                    f"Dataset file size: {dataset_path.stat().st_size / (1024 * 1024):.2f} MB"
                )

                # Add HDF5 structure inspection
                try:
                    with h5py.File(dataset_path, "r") as f:
                        self.logger.info("Dataset HDF5 structure:")

                        def print_structure(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                self.logger.info(
                                    f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}"
                                )
                            elif isinstance(obj, h5py.Group):
                                self.logger.info(f"  Group: {name}")

                        f.visititems(print_structure)
                except Exception as e:
                    self.logger.error(f"Error inspecting HDF5 structure: {str(e)}")
            else:
                self.logger.error(f"Dataset file not found at: {dataset_path}")

            # 4. Register experiment with complete dataset configuration
            self.logger.info("Registering experiment...")

            # Update the model config with the computed input shape
            model_config.architecture["input_shape"] = (
                task_config.input.get_total_feature_size(),
            )  # Must be a tuple

            # Get source config file path if available
            # Note: This should be passed explicitly when calling train_foundation_model
            source_config_file = getattr(self, "_source_config_file", None)

            experiment_id = registry.register_experiment(
                name="Foundation_VAE_Model",
                dataset_id=dataset_id,
                description="Training a foundation VAE model for feature encoding",
                source_config_file=source_config_file,
            )

            # Convert configs to dictionaries for model trainer
            model_config_dict = model_config.to_dict()
            training_config_dict = training_config.to_dict()
            self.logger.info(f"Created experiment: {experiment_id}")

            # 5. Create Model (building will happen in ModelTrainer within strategy scope)
            self.logger.info("Creating model...")
            try:
                model = ModelFactory.create_model(
                    model_type="variational_autoencoder", config=model_config
                )
                # Note: model.build() will be called by ModelTrainer within strategy scope
            except Exception as e:
                self.logger.error(f"Model creation failed: {str(e)}")
                self.logger.error(
                    f"Model config used: {json.dumps(model_config_dict, indent=2)}"
                )
                raise

            self.logger.info("Model created (will be built during training)")
            # Model summary will be available after building in ModelTrainer

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

            # Add the BetaSchedule callback for VAE training
            # This is essential for proper VAE loss computation (total_loss = reconstruction_loss + beta * kl_loss)
            beta_schedule_config = model_config.hyperparameters.get("beta_schedule", {})
            beta_callback = BetaSchedule(
                start=beta_schedule_config.get("start", 0.0),
                warmup=beta_schedule_config.get("warmup", 50),
                cycle_low=beta_schedule_config.get("cycle_low", 0.0),
                cycle_high=beta_schedule_config.get("cycle_high", 1.0),
                cycle_period=beta_schedule_config.get("cycle_period", 20),
            )
            callbacks.append(beta_callback)

            self.logger.info(
                f"Added BetaSchedule callback: start={beta_schedule_config.get('start', 0.0)}, "
                f"warmup={beta_schedule_config.get('warmup', 50)}, "
                f"cycle_low={beta_schedule_config.get('cycle_low', 0.0)}, "
                f"cycle_high={beta_schedule_config.get('cycle_high', 1.0)}, "
                f"cycle_period={beta_schedule_config.get('cycle_period', 20)}"
            )

            # Start training
            self.logger.info("Starting training...")
            try:
                training_results = trainer.train(
                    dataset=train_dataset,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    training_history_dir=self.experiments_output_dir
                    / experiment_id
                    / "training",
                    model_name="Foundation_VAE_Model",
                    dataset_id=dataset_id,
                    experiment_id=experiment_id,
                    verbose_training="full",  # Full verbosity for main foundation model training
                )

                # Evaluate Model
                self.logger.info("Evaluating model...")
                test_results = trainer.evaluate(
                    test_dataset,
                    save_samples=True,
                    training_history_dir=self.experiments_output_dir
                    / experiment_id
                    / "training",
                    max_samples=5000,
                )

                # Save consolidated training history with all metrics (train/val/test)
                self.logger.info("Saving consolidated training history...")
                trainer.save_consolidated_training_history()

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
                        "deterministic_encoder": model.deterministic_encoder,
                        "decoder": model.decoder,
                        "full_model": model.model,
                    },
                    model_name="foundation_model",
                    metadata=ensure_serializable(model_metadata),
                )

                # Create VAE-specific plots including training history
                self.logger.info("Creating VAE training plots...")
                training_plots_dir = (
                    self.experiments_output_dir / experiment_id / "training"
                )
                training_plots_dir.mkdir(parents=True, exist_ok=True)

                # Use the consolidated training history file
                training_history_json_path = (
                    training_plots_dir / "training_history.json"
                )
                if training_history_json_path.exists():
                    self.logger.info(
                        f"Using consolidated training history from: {training_history_json_path}"
                    )
                else:
                    self.logger.warning(
                        "Consolidated training_history.json file not found for plot generation"
                    )
                    training_history_json_path = None

                try:
                    model.create_plots(training_plots_dir, training_history_json_path)
                    self.logger.info("VAE training plots created successfully")
                except Exception as plot_e:
                    self.logger.error(f"Failed to create VAE plots: {str(plot_e)}")
                    # Don't fail the entire pipeline if plot creation fails
                    self.logger.exception("Failed to create VAE plots")

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

            # Return the path to the trained model and dataset for use in subsequent evaluations
            foundation_model_path = str(self.experiments_output_dir / experiment_id)
            dataset_path = str(data_manager.get_current_dataset_path())
            self.logger.info(f"Foundation model saved at: {foundation_model_path}")
            self.logger.info(f"Dataset saved at: {dataset_path}")
            return foundation_model_path, dataset_path

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
        eval_batch_size: int = 1024,
    ):
        """
        Evaluate a trained foundation model (VAE) for anomaly detection.

        Loads a pre-trained VAE, test/signal datasets, performs evaluation,
        and saves results directly in the foundation model's experiment directory.
        """

        if not foundation_model_path:
            self.logger.error(
                "Foundation model path must be provided for anomaly evaluation."
            )
            return False

        foundation_model_path = Path(foundation_model_path)
        model_weights_path = (
            foundation_model_path / "models" / "foundation_model" / "full_model"
        )
        config_path = foundation_model_path / "_experiment_config.yaml"

        if not model_weights_path.exists():
            model_weights_path_h5 = model_weights_path.with_suffix(".weights.h5")
            if model_weights_path_h5.exists():
                model_weights_path = model_weights_path_h5
                self.logger.info("Found model weights with .h5 extension.")
            else:
                self.logger.error(
                    f"Foundation model weights not found at: {model_weights_path} or {model_weights_path_h5}"
                )
                return False

        if not config_path.exists():
            self.logger.error(f"Foundation model config not found at: {config_path}")
            return False

        try:
            # Initialize registry
            registry = ModelRegistry(str(self.experiments_output_dir))
            self.logger.info(f"Registry initialized at: {registry.db_path}")

            # 1. Load original model configuration and experiment ID
            self.logger.info(f"Loading original model config from: {config_path}")
            from hep_foundation.config.config_loader import PipelineConfigLoader

            config_loader = PipelineConfigLoader()
            original_experiment_data = config_loader.load_config(config_path)

            # Get the experiment ID from the foundation model path
            foundation_experiment_id = foundation_model_path.name

            # Get the VAE model config from the YAML structure
            if (
                "models" in original_experiment_data
                and "vae" in original_experiment_data["models"]
            ):
                original_model_config = original_experiment_data["models"]["vae"]
            else:
                self.logger.error(f"Could not find VAE model config in: {config_path}")
                return False

            if (
                not original_model_config
                or original_model_config.get("model_type") != "variational_autoencoder"
            ):
                self.logger.error(
                    f"Loaded config is not for a variational_autoencoder: {config_path}"
                )
                return False

            # 2. Load Datasets
            self.logger.info("Initializing Dataset Manager...")
            data_manager = DatasetManager(base_dir=self.processed_datasets_dir)

            # Use eval_batch_size or derive from vae_training_config if provided
            batch_size = eval_batch_size
            if vae_training_config:
                try:
                    batch_size = vae_training_config.batch_size
                    self.logger.info(
                        f"Using batch size from provided VAE training config: {batch_size}"
                    )
                except AttributeError:
                    self.logger.warning(
                        "Provided vae_training_config lacks batch_size, using default."
                    )
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
                # Look for background histogram data in the plot_data folder
                background_plot_data_dir = (
                    data_manager.get_dataset_dir(background_dataset_id_for_plots)
                    / "plot_data"
                )
                potential_background_hist_path = (
                    background_plot_data_dir / "atlas_dataset_features_hist_data.json"
                )
                if potential_background_hist_path.exists():
                    background_hist_data_path_for_comparison = (
                        potential_background_hist_path
                    )
                else:
                    self.logger.warning(
                        f"Background histogram data for comparison not found at {potential_background_hist_path}. Comparison plot may be skipped by DatasetManager."
                    )

            # Reload signal datasets, now passing the background_hist_data_path for comparison plot generation within DatasetManager
            if dataset_config.signal_keys:
                self.logger.info(
                    "Reloading signal datasets for evaluation and comparison plotting..."
                )
                signal_datasets = data_manager.load_signal_datasets(
                    dataset_config=dataset_config,
                    batch_size=batch_size,
                    include_labels=False,
                    background_hist_data_path=background_hist_data_path_for_comparison,  # Pass the path here
                    split=False,  # Anomaly detection uses full signal datasets
                )
                self.logger.info(
                    f"Reloaded {len(signal_datasets)} signal datasets for evaluation."
                )
            else:
                self.logger.warning(
                    "No signal keys provided. Skipping signal dataset reloading for evaluation."
                )
                signal_datasets = {}  # Ensure signal_datasets is defined

            # Create and Load Model
            self.logger.info("Instantiating VAE model from loaded configuration...")

            # Ensure input_shape is present in the loaded config
            if "input_shape" not in original_model_config["architecture"]:
                self.logger.warning(
                    "Input shape missing in loaded model config, deriving from task_config."
                )
                input_shape = (task_config.input.get_total_feature_size(),)
                if input_shape[0] is None or input_shape[0] <= 0:
                    self.logger.error(
                        "Could not determine a valid input shape from task_config."
                    )
                    return False
                original_model_config["architecture"]["input_shape"] = list(input_shape)

            # Ensure hyperparameters like beta_schedule are present
            if "hyperparameters" not in original_model_config:
                original_model_config["hyperparameters"] = {}
            if "beta_schedule" not in original_model_config["hyperparameters"]:
                self.logger.warning(
                    "beta_schedule missing in loaded config, using default."
                )
                original_model_config["hyperparameters"]["beta_schedule"] = {
                    "start": 0.0,
                    "end": 1.0,
                    "warmup_epochs": 0,
                    "cycle_epochs": 0,
                }

            # Create VAE model (building will happen when needed)
            vae_config = VAEConfig(
                model_type=original_model_config["model_type"],
                architecture=original_model_config["architecture"],
                hyperparameters=original_model_config["hyperparameters"],
            )
            model = VariationalAutoEncoder(config=vae_config)

            # Build model to load weights (for evaluation, we can build directly)
            input_shape = tuple(original_model_config["architecture"]["input_shape"])
            model.build(input_shape)

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
                base_path=self.experiments_output_dir,
            )

            # Run the evaluation
            evaluator.run_anomaly_detection_test()

            # 5. Display Results
            self.logger.info("=" * 100)
            self.logger.info("Anomaly Detection Results")
            self.logger.info("=" * 100)

            self.logger.info(f"Foundation Model ID: {foundation_experiment_id}")
            self.logger.info(
                "Anomaly detection evaluation completed. Results are available in the testing directory."
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Anomaly detection evaluation failed: {type(e).__name__}: {str(e)}"
            )
            self.logger.exception("Detailed traceback:")
            return False

    def _denormalize_labels(
        self,
        normalized_labels: np.ndarray,
        norm_params: dict,
        label_config_index: int = 0,
    ) -> np.ndarray:
        """
        Denormalize label values using stored normalization parameters.

        Args:
            normalized_labels: Normalized label array of shape (n_samples, n_features)
            norm_params: Normalization parameters dictionary
            label_config_index: Index of the label configuration (default 0)

        Returns:
            Denormalized label array with original scale and units
        """
        if (
            "labels" not in norm_params
            or len(norm_params["labels"]) <= label_config_index
        ):
            self.logger.warning(
                "No label normalization parameters found, returning normalized values"
            )
            return normalized_labels

        label_norm_params = norm_params["labels"][label_config_index]
        denormalized = normalized_labels.copy()

        # For now, assume all labels are aggregated features (this is the common case)
        # If scalar features are present, they would need separate handling
        if "aggregated" in label_norm_params:
            for agg_name, params in label_norm_params["aggregated"].items():
                means = np.array(params["means"])
                stds = np.array(params["stds"])

                # Apply denormalization: denormalized = normalized * std + mean
                denormalized = denormalized * stds + means
                break  # Assuming single aggregator for labels (most common case)
        elif "scalar" in label_norm_params:
            # Handle scalar features if present
            for feature_name, params in label_norm_params["scalar"].items():
                mean = params["mean"]
                std = params["std"]
                # Apply denormalization for scalar features
                denormalized = denormalized * std + mean
                break  # Assuming single scalar feature for labels

        return denormalized

    def evaluate_foundation_model_regression(
        self,
        dataset_config: DatasetConfig,
        dnn_model_config: DNNPredictorConfig,
        dnn_training_config: TrainingConfig,
        task_config: TaskConfig,
        delete_catalogs: bool = True,
        foundation_model_path: str = None,
        data_sizes: list = None,
        fixed_epochs: int = 10,
    ) -> bool:
        """
        Evaluate foundation model for regression tasks using data efficiency study.

        This method trains three models (From Scratch, Fine-Tuned, Fixed) on increasing amounts of training data
        to show how pre-trained weights help with limited data and demonstrate the value of the foundation model.

        Args:
            dataset_config: Configuration for dataset processing
            dnn_model_config: Configuration for DNN model
            dnn_training_config: Configuration for DNN training
            task_config: Configuration for task processing
            delete_catalogs: Whether to delete catalogs after processing
            foundation_model_path: Path to the foundation model encoder
            data_sizes: List of training data sizes to test (e.g., [1000, 2000, 5000, 10000])
            fixed_epochs: Number of epochs to train each model for each data size
        """

        if not foundation_model_path:
            self.logger.error(
                "Foundation model path must be provided for data efficiency evaluation."
            )
            return False

        foundation_model_dir = Path(foundation_model_path)
        regression_dir = foundation_model_dir / "testing" / "regression_evaluation"
        regression_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize data manager
            data_manager = DatasetManager(base_dir=self.processed_datasets_dir)

            # 1. Load full dataset with regression labels
            self.logger.info("Loading full dataset with regression labels...")
            train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
                dataset_config=dataset_config,
                validation_fraction=dataset_config.validation_fraction,
                test_fraction=dataset_config.test_fraction,
                batch_size=dnn_training_config.batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=True,
                delete_catalogs=delete_catalogs,
            )

            # Access normalization parameters for denormalizing labels
            import json

            import h5py

            dataset_path = data_manager.get_current_dataset_path()
            norm_params = None
            try:
                with h5py.File(dataset_path, "r") as f:
                    norm_params = json.loads(f.attrs["normalization_params"])
                self.logger.info(
                    "Successfully loaded normalization parameters for label denormalization"
                )
            except Exception as e:
                self.logger.warning(
                    f"Could not load normalization parameters: {e}. Labels will remain normalized."
                )
                norm_params = None

            # Count total training events
            total_train_events = 0
            for batch in train_dataset:
                batch_size = tf.shape(batch[0])[0]
                total_train_events += batch_size.numpy()

            # Count total test events for plot labeling
            total_test_events = 0
            for batch in test_dataset:
                batch_size = tf.shape(batch[0])[0]
                total_test_events += batch_size.numpy()

            self.logger.info(f"Total training events available: {total_train_events}")
            self.logger.info(f"Total test events: {total_test_events}")

            # Filter data_sizes to only include sizes <= total_train_events
            data_sizes = [
                size
                for size in (data_sizes or [total_train_events])
                if size <= total_train_events
            ]
            self.logger.info(f"Data sizes to test: {data_sizes}")

            # 2. Load Pre-trained Foundation Encoder & its Config
            self.logger.info(
                f"Loading foundation model configuration from: {foundation_model_dir}"
            )

            # Load the VAE model config from the YAML config file
            vae_config_path = foundation_model_dir / "_experiment_config.yaml"
            if not vae_config_path.exists():
                self.logger.error(
                    f"Foundation model config not found at: {vae_config_path}"
                )
                return False

            from hep_foundation.config.config_loader import PipelineConfigLoader

            config_loader = PipelineConfigLoader()
            vae_config_data = config_loader.load_config(vae_config_path)

            if "models" in vae_config_data and "vae" in vae_config_data["models"]:
                original_vae_model_config = vae_config_data["models"]["vae"]
            else:
                self.logger.error(
                    f"Could not find VAE model config in: {vae_config_path}"
                )
                return False

            vae_arch_config = original_vae_model_config["architecture"]
            latent_dim = vae_arch_config["latent_dim"]
            encoder_hidden_layers = vae_arch_config.get("encoder_layers", [])
            encoder_activation = vae_arch_config.get("activation", "relu")

            # Load the pre-trained deterministic encoder directly
            pretrained_deterministic_encoder_path = (
                foundation_model_dir
                / "models"
                / "foundation_model"
                / "deterministic_encoder"
            )
            if not pretrained_deterministic_encoder_path.exists():
                self.logger.error(
                    f"Pretrained deterministic encoder not found at {pretrained_deterministic_encoder_path}"
                )
                return False

            self.logger.info(
                f"Loading pre-trained deterministic encoder from: {pretrained_deterministic_encoder_path}"
            )
            pretrained_deterministic_encoder = tf.keras.models.load_model(
                pretrained_deterministic_encoder_path
            )

            self.logger.info(
                f"Loaded deterministic encoder with output shape: {pretrained_deterministic_encoder.output.shape}"
            )
            self.logger.info("Deterministic encoder layers:")
            for layer in pretrained_deterministic_encoder.layers:
                self.logger.info(
                    f"  {layer.name}: {layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'}"
                )

            original_input_shape = (task_config.input.get_total_feature_size(),)
            regression_output_shape = (task_config.labels[0].get_total_feature_size(),)

            # Helper function to build regressor head
            def build_regressor_head(name_suffix: str) -> tf.keras.Model:
                # Create a copy of the DNN config with modified architecture
                import copy

                regressor_config = copy.deepcopy(dnn_model_config)

                # Update the architecture for the regressor head
                regressor_config.architecture.update(
                    {
                        "input_shape": (latent_dim,),
                        "output_shape": regression_output_shape,
                        "name": f"{dnn_model_config.architecture.get('name', 'regressor')}_{name_suffix}",
                    }
                )

                regressor_model_wrapper = ModelFactory.create_model(
                    model_type="dnn_predictor", config=regressor_config
                )
                regressor_model_wrapper.build()
                return regressor_model_wrapper.model

            # Helper function to create subset dataset
            def create_subset_dataset(dataset, num_events, data_size_index=None):
                """Create a subset of the dataset with exactly num_events events"""
                # Convert to unbatched dataset to count events precisely
                unbatched = dataset.unbatch()
                # Use different seed for each data size to ensure independent sampling
                # This prevents smaller datasets from being strict subsets of larger ones
                seed = 42 + (data_size_index or 0)  # Different seed for each data size
                shuffled = unbatched.shuffle(buffer_size=50000, seed=seed)
                subset = shuffled.take(num_events)
                # Rebatch with original batch size
                return subset.batch(dnn_training_config.batch_size)

            # Helper function to train and evaluate a model for a specific data size
            def train_and_evaluate_for_size(
                model_name: str,
                combined_keras_model: tf.keras.Model,
                train_subset,
                data_size: int,
                save_training_history: bool = False,
                label_distributions_dir_param: Optional[Path] = None,
                label_variable_names_param: Optional[list] = None,
            ):
                self.logger.info(
                    f"Training {model_name} model with {data_size} events..."
                )

                # Wrap the Keras model with CustomKerasModelWrapper for ModelTrainer
                wrapped_model_for_trainer = CustomKerasModelWrapper(
                    combined_keras_model, name=model_name
                )

                trainer_config_dict = {
                    "batch_size": dnn_training_config.batch_size,
                    "epochs": fixed_epochs,  # Use fixed epochs for fair comparison
                    "learning_rate": dnn_training_config.learning_rate,
                    "early_stopping": {
                        "patience": fixed_epochs + 1,
                        "min_delta": 0,
                    },  # Disable early stopping
                }

                trainer = ModelTrainer(
                    model=wrapped_model_for_trainer, training_config=trainer_config_dict
                )

                # Train with reduced verbosity for evaluation
                _ = trainer.train(  # Unused return value
                    dataset=train_subset,
                    validation_data=val_dataset,
                    callbacks=[],  # No callbacks for speed
                    training_history_dir=regression_dir / "training_histories"
                    if save_training_history
                    else None,
                    model_name=model_name,
                    dataset_id=f"regression_eval_{data_size}",
                    experiment_id="regression_evaluation",
                    verbose_training="minimal",  # Reduce verbosity for evaluation models
                    save_individual_history=True,  # Save individual files for comparison plots
                )

                # Evaluate on test set
                test_metrics = trainer.evaluate(test_dataset)
                test_loss = test_metrics.get(
                    "test_loss", test_metrics.get("test_mse", 0.0)
                )

                # Generate predictions and save histogram data (1000 samples)
                try:
                    self.logger.info(
                        f"Generating predictions for {model_name} model..."
                    )
                    predictions_list = []
                    actual_labels_list = []
                    samples_collected = 0
                    max_prediction_samples = 1000

                    for batch in test_dataset:
                        if samples_collected >= max_prediction_samples:
                            break

                        if isinstance(batch, tuple) and len(batch) == 2:
                            features_batch, labels_batch = batch

                            predictions_batch = combined_keras_model.predict(
                                features_batch, verbose=0
                            )

                            # Extract actual labels from batch structure
                            if isinstance(labels_batch, tuple):
                                actual_labels_batch = None
                                for item in labels_batch:
                                    if hasattr(item, "shape") and hasattr(
                                        item, "numpy"
                                    ):
                                        actual_labels_batch = item.numpy()
                                        break
                                if (
                                    actual_labels_batch is None
                                    and len(labels_batch) > 0
                                ):
                                    actual_labels_batch = labels_batch[0]
                            else:
                                actual_labels_batch = (
                                    labels_batch.numpy()
                                    if hasattr(labels_batch, "numpy")
                                    else labels_batch
                                )

                            # Remove extra dimensions if present
                            if (
                                hasattr(actual_labels_batch, "ndim")
                                and actual_labels_batch.ndim == 3
                                and actual_labels_batch.shape[0] == 1
                            ):
                                actual_labels_batch = actual_labels_batch.squeeze(
                                    axis=0
                                )

                            # Convert to numpy and collect samples
                            predictions_np = np.array(predictions_batch)
                            actual_labels_np = np.array(actual_labels_batch)

                            batch_size = predictions_np.shape[0]
                            samples_to_take = min(
                                batch_size, max_prediction_samples - samples_collected
                            )

                            predictions_list.extend(predictions_np[:samples_to_take])
                            actual_labels_list.extend(
                                actual_labels_np[:samples_to_take]
                            )
                            samples_collected += samples_to_take

                    # Convert to numpy arrays
                    predictions_array = np.array(predictions_list)
                    actual_labels_array = np.array(actual_labels_list)

                    if len(predictions_array) > 0 and len(actual_labels_array) > 0:
                        # Denormalize labels and predictions to get original scale/units
                        if norm_params is not None:
                            try:
                                self.logger.info(
                                    f"Denormalizing labels and predictions for {model_name}..."
                                )
                                predictions_array = self._denormalize_labels(
                                    predictions_array, norm_params, label_config_index=0
                                )
                                actual_labels_array = self._denormalize_labels(
                                    actual_labels_array,
                                    norm_params,
                                    label_config_index=0,
                                )
                                self.logger.info(
                                    f"Successfully denormalized {len(predictions_array)} prediction/label pairs"
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to denormalize labels for {model_name}: {e}. Using normalized values."
                                )
                        else:
                            self.logger.info(
                                f"No normalization parameters available for {model_name}, using normalized values"
                            )
                        # Get label variable names (required for histogram keys)
                        if not label_variable_names_param:
                            self.logger.warning(
                                "No label variable names found, using generic names"
                            )
                            num_vars = (
                                predictions_array.shape[1]
                                if predictions_array.ndim > 1
                                else 1
                            )
                            current_label_names = [
                                f"variable_{i}" for i in range(num_vars)
                            ]
                        else:
                            current_label_names = label_variable_names_param

                        # Create data dictionaries for histogram manager (simplified - no special cases)
                        predictions_data = {}
                        differences_data = {}

                        for i, var_name in enumerate(current_label_names):
                            if i < predictions_array.shape[1]:
                                predictions_data[var_name] = predictions_array[
                                    :, i
                                ].tolist()
                                differences_data[var_name] = (
                                    predictions_array[:, i] - actual_labels_array[:, i]
                                ).tolist()

                        # Prepare save directory and file paths
                        hist_save_dir = label_distributions_dir_param or (
                            regression_dir / "label_distributions"
                        )
                        hist_save_dir.mkdir(parents=True, exist_ok=True)

                        data_size_label = (
                            f"{data_size // 1000}k"
                            if data_size >= 1000
                            else str(data_size)
                        )

                        # Extract base model name to avoid duplication (model_name already contains data_size_label)
                        # e.g., "From_Scratch_1k" -> "From_Scratch"
                        base_model_name = (
                            model_name.rsplit("_", 1)[0]
                            if "_" in model_name
                            else model_name
                        )

                        # Save predictions histogram using HistogramManager
                        pred_file_path = (
                            hist_save_dir
                            / f"{base_model_name}_{data_size_label}_predictions_hist.json"
                        )
                        self.histogram_manager.save_to_hist_file(
                            data=predictions_data,
                            file_path=pred_file_path,
                            nbins=50,
                            use_percentile_file=False,
                            update_percentile_file=False,
                            use_percentile_cache=True,  # Use cache for coordinated bins
                        )

                        # Save differences histogram using HistogramManager with separate percentile index
                        diff_file_path = (
                            hist_save_dir
                            / f"{base_model_name}_{data_size_label}_diff_predictions_hist.json"
                        )
                        self.histogram_manager.save_to_hist_file(
                            data=differences_data,
                            file_path=diff_file_path,
                            nbins=50,
                            use_percentile_file=False,
                            update_percentile_file=False,
                            use_percentile_cache=False,  # Don't use coordinated bins for differences
                        )

                        self.logger.info(
                            f"Saved histogram data for {model_name} with {len(predictions_array)} samples"
                        )
                    else:
                        self.logger.warning(
                            f"No predictions or labels generated for {model_name}"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Failed to generate predictions for {model_name}: {str(e)}"
                    )

                self.logger.info(
                    f"{model_name} with {data_size} events - Test Loss: {test_loss:.6f}"
                )
                return test_loss

            # Store results for plotting
            results = {
                "data_sizes": data_sizes,
                "From_Scratch": [],
                "Fine_Tuned": [],
                "Fixed_Encoder": [],
            }

            # Save training histories for all data sizes (not just the largest)
            if not data_sizes:
                self.logger.warning(
                    "No valid data sizes remain after filtering. Using total training events as fallback."
                )
                # Will save histories for all data sizes processed

            # 3. Set up label distributions directory
            label_distributions_dir = regression_dir / "label_distributions"
            label_distributions_dir.mkdir(parents=True, exist_ok=True)

            # Extract label variable names from task config
            label_variable_names = []
            if hasattr(task_config, "labels") and task_config.labels:
                first_label_config = task_config.labels[0]
                if (
                    hasattr(first_label_config, "feature_array_aggregators")
                    and first_label_config.feature_array_aggregators
                ):
                    first_aggregator = first_label_config.feature_array_aggregators[0]
                    if hasattr(first_aggregator, "input_branches"):
                        for branch_selector in first_aggregator.input_branches:
                            if hasattr(branch_selector, "branch") and hasattr(
                                branch_selector.branch, "name"
                            ):
                                label_variable_names.append(branch_selector.branch.name)

            self.logger.info(f"Extracted label variable names: {label_variable_names}")

            # Save actual labels histogram ONCE before model training
            if label_variable_names:
                self.logger.info("Saving actual test labels histogram...")
                actual_labels_list = []
                samples_collected = 0
                max_samples = 1000

                for batch in test_dataset:
                    if samples_collected >= max_samples:
                        break

                    if isinstance(batch, tuple) and len(batch) == 2:
                        features_batch, labels_batch = batch

                        # Extract actual labels from batch structure
                        if isinstance(labels_batch, tuple):
                            actual_labels_batch = None
                            for item in labels_batch:
                                if hasattr(item, "shape") and hasattr(item, "numpy"):
                                    actual_labels_batch = item.numpy()
                                    break
                            if actual_labels_batch is None and len(labels_batch) > 0:
                                actual_labels_batch = labels_batch[0]
                        else:
                            actual_labels_batch = (
                                labels_batch.numpy()
                                if hasattr(labels_batch, "numpy")
                                else labels_batch
                            )

                        # Remove extra dimensions if present
                        if (
                            hasattr(actual_labels_batch, "ndim")
                            and actual_labels_batch.ndim == 3
                            and actual_labels_batch.shape[0] == 1
                        ):
                            actual_labels_batch = actual_labels_batch.squeeze(axis=0)

                        actual_labels_np = np.array(actual_labels_batch)
                        batch_size = actual_labels_np.shape[0]
                        samples_to_take = min(
                            batch_size, max_samples - samples_collected
                        )

                        actual_labels_list.extend(actual_labels_np[:samples_to_take])
                        samples_collected += samples_to_take

                # Convert to numpy and organize by variable names
                actual_labels_array = np.array(actual_labels_list)

                # Denormalize actual test labels to get original scale/units
                if norm_params is not None:
                    try:
                        self.logger.info("Denormalizing actual test labels...")
                        actual_labels_array = self._denormalize_labels(
                            actual_labels_array, norm_params, label_config_index=0
                        )
                        self.logger.info(
                            f"Successfully denormalized {len(actual_labels_array)} actual label samples"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to denormalize actual test labels: {e}. Using normalized values."
                        )
                else:
                    self.logger.info(
                        "No normalization parameters available, using normalized values for actual test labels"
                    )

                actual_data = {}

                for i, var_name in enumerate(label_variable_names):
                    if i < actual_labels_array.shape[1]:
                        actual_data[var_name] = actual_labels_array[:, i].tolist()

                # Save actual labels histogram
                actual_labels_path = (
                    label_distributions_dir / "actual_test_labels_hist.json"
                )
                self.histogram_manager.save_to_hist_file(
                    data=actual_data,
                    file_path=actual_labels_path,
                    nbins=50,
                    use_percentile_file=False,
                    update_percentile_file=False,
                    use_percentile_cache=True,  # Use cache for coordinated bins
                )
                self.logger.info("Saved actual test labels histogram")

            # 4. Run experiments for each data size
            for data_size_index, data_size in enumerate(data_sizes):
                self.logger.info(f"{'=' * 50}")
                self.logger.info(f"Training with {data_size} events")
                self.logger.info(f"{'=' * 50}")

                # Create subset of training data
                train_subset = create_subset_dataset(
                    train_dataset, data_size, data_size_index
                )

                # --- Model 1: From Scratch ---
                self.logger.info("Building From Scratch model...")
                scratch_encoder_layers = []
                for units in encoder_hidden_layers:
                    scratch_encoder_layers.append(
                        tf.keras.layers.Dense(units, activation=encoder_activation)
                    )
                scratch_encoder_layers.append(
                    tf.keras.layers.Dense(latent_dim, name="scratch_latent_space")
                )

                scratch_encoder_part = tf.keras.Sequential(
                    scratch_encoder_layers, name="scratch_encoder"
                )
                scratch_regressor_dnn = build_regressor_head("from_scratch")

                model_inputs = tf.keras.Input(
                    shape=original_input_shape, name="input_features"
                )
                encoded_scratch = scratch_encoder_part(model_inputs)
                predictions_scratch = scratch_regressor_dnn(encoded_scratch)
                model_from_scratch = tf.keras.Model(
                    inputs=model_inputs,
                    outputs=predictions_scratch,
                    name="Regressor_From_Scratch",
                )

                # Save training history for all data sizes
                should_save_history = True

                self.logger.info(
                    f"Enabling training history saving for all models with {data_size} events"
                )

                # Format data size for better labeling
                data_size_label = (
                    f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                )

                scratch_loss = train_and_evaluate_for_size(
                    f"From_Scratch_{data_size_label}",
                    model_from_scratch,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=label_distributions_dir,
                    label_variable_names_param=label_variable_names,
                )
                results["From_Scratch"].append(scratch_loss)

                # --- Model 2: Fine-Tuned ---
                self.logger.info("Building Fine-Tuned model...")
                # Create a functional copy of the deterministic encoder for fine-tuning
                # We can't use clone_model with QKeras layers, so we'll create a new model
                # that uses the same layers but allows training
                fine_tuned_input = tf.keras.Input(
                    shape=original_input_shape, name="fine_tuned_input"
                )
                fine_tuned_encoded = pretrained_deterministic_encoder(fine_tuned_input)

                # Add dtype casting to ensure compatibility with QKeras layers
                # Mixed precision may cause the encoder to output float16, but QKeras expects float32
                fine_tuned_encoded_cast = tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32), name="dtype_cast_fine_tuned"
                )(fine_tuned_encoded)

                fine_tuned_encoder_part = tf.keras.Model(
                    inputs=fine_tuned_input,
                    outputs=fine_tuned_encoded_cast,
                    name="fine_tuned_pretrained_encoder",
                )
                fine_tuned_encoder_part.trainable = True

                fine_tuned_regressor_dnn = build_regressor_head("fine_tuned")

                model_inputs_ft = tf.keras.Input(
                    shape=original_input_shape, name="input_features_ft"
                )
                encoded_ft = fine_tuned_encoder_part(model_inputs_ft)
                predictions_ft = fine_tuned_regressor_dnn(encoded_ft)
                model_fine_tuned = tf.keras.Model(
                    inputs=model_inputs_ft,
                    outputs=predictions_ft,
                    name="Regressor_Fine_Tuned",
                )

                finetuned_loss = train_and_evaluate_for_size(
                    f"Fine_Tuned_{data_size_label}",
                    model_fine_tuned,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=label_distributions_dir,
                    label_variable_names_param=label_variable_names,
                )
                results["Fine_Tuned"].append(finetuned_loss)

                # --- Model 3: Fixed Encoder ---
                self.logger.info("Building Fixed Encoder model...")
                # Create a functional copy of the deterministic encoder for fixed use
                # We can't use clone_model with QKeras layers, so we'll create a new model
                # that uses the same layers but freezes them
                fixed_input = tf.keras.Input(
                    shape=original_input_shape, name="fixed_input"
                )
                fixed_encoded = pretrained_deterministic_encoder(fixed_input)

                # Add dtype casting to ensure compatibility with QKeras layers
                # Mixed precision may cause the encoder to output float16, but QKeras expects float32
                fixed_encoded_cast = tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32), name="dtype_cast_fixed"
                )(fixed_encoded)

                fixed_encoder_part = tf.keras.Model(
                    inputs=fixed_input,
                    outputs=fixed_encoded_cast,
                    name="fixed_pretrained_encoder",
                )
                fixed_encoder_part.trainable = False

                fixed_regressor_dnn = build_regressor_head("fixed_encoder")

                model_inputs_fx = tf.keras.Input(
                    shape=original_input_shape, name="input_features_fx"
                )
                encoded_fx = fixed_encoder_part(model_inputs_fx)
                predictions_fx = fixed_regressor_dnn(encoded_fx)
                model_fixed = tf.keras.Model(
                    inputs=model_inputs_fx,
                    outputs=predictions_fx,
                    name="Regressor_Fixed_Encoder",
                )

                fixed_loss = train_and_evaluate_for_size(
                    f"Fixed_Encoder_{data_size_label}",
                    model_fixed,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=label_distributions_dir,
                    label_variable_names_param=label_variable_names,
                )
                results["Fixed_Encoder"].append(fixed_loss)

            # 5. Save results and create plots
            self.logger.info("Creating data efficiency plot...")

            # Save results to JSON
            results_file = regression_dir / "regression_data_efficiency_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {results_file}")

            # Create combined training history plot if we saved training histories
            training_histories_dir = regression_dir / "training_histories"
            combined_plot_path = regression_dir / "regression_training_comparison.png"
            self.foundation_plot_manager.create_training_history_comparison_plot_from_directory(
                training_histories_dir,
                combined_plot_path,
                title_prefix="Regression Model Training Comparison",
                validation_only=True,
            )

            # Create the data efficiency plot
            plot_file = regression_dir / "regression_data_efficiency_plot.png"
            self.foundation_plot_manager.create_data_efficiency_plot(
                results,
                plot_file,
                plot_type="regression",
                metric_name="Test Loss (MSE)",
                total_test_events=total_test_events,
            )

            # Create label distribution comparison plot
            if label_variable_names:
                # Load physlite plot labels for proper titles
                physlite_plot_labels = None
                try:
                    physlite_labels_path = Path(
                        "src/hep_foundation/data/physlite_plot_labels.json"
                    )
                    if physlite_labels_path.exists():
                        with open(physlite_labels_path) as f:
                            physlite_plot_labels = json.load(f)
                except Exception as e:
                    self.logger.warning(
                        f"Could not load physlite plot labels: {str(e)}"
                    )

                # Use the subplot-based comparison plot
                self.foundation_plot_manager.create_label_distribution_comparison_plot_with_subplots(
                    regression_dir,
                    data_sizes,
                    label_variable_names,
                    physlite_plot_labels,
                )
            else:
                self.logger.warning(
                    "No label variable names found, skipping label distribution comparison plot"
                )

            # 5. Display summary
            log_evaluation_summary(results, evaluation_type="regression")

            return True

        except Exception as e:
            self.logger.error(
                f"Regression evaluation failed: {type(e).__name__}: {str(e)}"
            )
            self.logger.exception("Detailed traceback:")
            return False

    def evaluate_foundation_model_signal_classification(
        self,
        dataset_config: DatasetConfig,
        dnn_model_config: DNNPredictorConfig,
        dnn_training_config: TrainingConfig,
        task_config: TaskConfig,
        delete_catalogs: bool = True,
        foundation_model_path: str = None,
        data_sizes: list = None,
        fixed_epochs: int = 10,
    ) -> bool:
        """
        Evaluate foundation model for signal classification using data efficiency study.

        This method creates balanced datasets mixing background and signal data with binary labels,
        then trains three models (From Scratch, Fine-Tuned, Fixed) on increasing amounts of training data
        to show how pre-trained weights help with limited signal data.

        Args:
            dataset_config: Configuration for dataset processing
            dnn_model_config: Configuration for DNN model
            dnn_training_config: Configuration for DNN training
            task_config: Configuration for task processing
            delete_catalogs: Whether to delete catalogs after processing
            foundation_model_path: Path to the foundation model encoder
            data_sizes: List of training data sizes to test (e.g., [1000, 2000, 5000, 10000])
            fixed_epochs: Number of epochs to train each model for each data size
        """

        if not foundation_model_path:
            self.logger.error(
                "Foundation model path must be provided for signal classification evaluation."
            )
            return False

        if not dataset_config.signal_keys:
            self.logger.error(
                "Signal keys must be configured for signal classification evaluation."
            )
            return False

        foundation_model_dir = Path(foundation_model_path)
        classification_dir = foundation_model_dir / "testing" / "signal_classification"
        classification_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize data manager
            data_manager = DatasetManager(base_dir=self.processed_datasets_dir)

            # 1. Load background dataset (no labels needed)
            self.logger.info("Loading background dataset...")
            background_train, background_val, background_test = (
                data_manager.load_atlas_datasets(
                    dataset_config=dataset_config,
                    validation_fraction=dataset_config.validation_fraction,
                    test_fraction=dataset_config.test_fraction,
                    batch_size=dnn_training_config.batch_size,
                    shuffle_buffer=dataset_config.shuffle_buffer,
                    include_labels=False,  # No labels needed for background
                    delete_catalogs=delete_catalogs,
                )
            )

            # 2. Prepare background histogram path for comparison plot generation
            background_hist_data_path_for_comparison = None
            dataset_id = data_manager.generate_dataset_id(dataset_config)
            potential_background_hist_path = (
                data_manager.get_dataset_dir(dataset_id)
                / "plot_data"
                / "atlas_dataset_features_hist_data.json"
            )
            if potential_background_hist_path.exists():
                background_hist_data_path_for_comparison = (
                    potential_background_hist_path
                )
                self.logger.info(
                    f"Found background histogram data for comparison at {potential_background_hist_path}"
                )
            else:
                self.logger.warning(
                    f"Background histogram data for comparison not found at {potential_background_hist_path}. Comparison plot may be skipped by DatasetManager."
                )

            # 3. Load first signal dataset with proper train/val/test splits
            signal_key = dataset_config.signal_keys[0]
            self.logger.info(f"Loading signal dataset with splits: {signal_key}")

            signal_datasets_splits = data_manager.load_signal_datasets(
                dataset_config=dataset_config,
                validation_fraction=dataset_config.validation_fraction,
                test_fraction=dataset_config.test_fraction,
                batch_size=dnn_training_config.batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=False,  # No labels needed for signals
                background_hist_data_path=background_hist_data_path_for_comparison,  # Pass the path here
                split=True,  # Enable splitting
            )

            if signal_key not in signal_datasets_splits:
                self.logger.error(f"Signal dataset '{signal_key}' not found")
                return False

            signal_train, signal_val, signal_test = signal_datasets_splits[signal_key]

            # 4. Create balanced labeled datasets
            self.logger.info("Creating balanced labeled datasets...")

            def create_balanced_labeled_dataset(bg_dataset, sig_dataset):
                """Create a balanced dataset with binary labels"""
                # Convert to unbatched datasets for easier manipulation
                bg_unbatched = bg_dataset.unbatch()
                sig_unbatched = sig_dataset.unbatch()

                # Add labels: 0 for background, 1 for signal
                bg_labeled = bg_unbatched.map(
                    lambda x: (x, tf.constant(0.0, dtype=tf.float32))
                )
                sig_labeled = sig_unbatched.map(
                    lambda x: (x, tf.constant(1.0, dtype=tf.float32))
                )

                # Combine and shuffle
                combined = bg_labeled.concatenate(sig_labeled)
                combined = combined.shuffle(buffer_size=10000, seed=42)

                # Rebatch
                return combined.batch(dnn_training_config.batch_size)

            # Create balanced train, validation, and test datasets using proper signal splits
            labeled_train_dataset = create_balanced_labeled_dataset(
                background_train, signal_train
            )
            labeled_val_dataset = create_balanced_labeled_dataset(
                background_val, signal_val
            )
            labeled_test_dataset = create_balanced_labeled_dataset(
                background_test, signal_test
            )

            # Count total training events
            total_train_events = 0
            for batch in labeled_train_dataset:
                batch_size = tf.shape(batch[0])[0]
                total_train_events += batch_size.numpy()

            # Count total test events for plot labeling
            total_test_events = 0
            for batch in labeled_test_dataset:
                batch_size = tf.shape(batch[0])[0]
                total_test_events += batch_size.numpy()

            self.logger.info(f"Total training events available: {total_train_events}")
            self.logger.info(f"Total test events: {total_test_events}")

            # Filter data_sizes to only include sizes <= total_train_events
            data_sizes = [
                size
                for size in (data_sizes or [total_train_events])
                if size <= total_train_events
            ]
            self.logger.info(f"Data sizes to test: {data_sizes}")

            # 5. Load Pre-trained Foundation Encoder & its Config
            self.logger.info(
                f"Loading foundation model configuration from: {foundation_model_dir}"
            )

            # Load the VAE model config from the YAML config file
            vae_config_path = foundation_model_dir / "_experiment_config.yaml"
            if not vae_config_path.exists():
                self.logger.error(
                    f"Foundation model config not found at: {vae_config_path}"
                )
                return False

            from hep_foundation.config.config_loader import PipelineConfigLoader

            config_loader = PipelineConfigLoader()
            vae_config_data = config_loader.load_config(vae_config_path)

            if "models" in vae_config_data and "vae" in vae_config_data["models"]:
                original_vae_model_config = vae_config_data["models"]["vae"]
            else:
                self.logger.error(
                    f"Could not find VAE model config in: {vae_config_path}"
                )
                return False

            vae_arch_config = original_vae_model_config["architecture"]
            latent_dim = vae_arch_config["latent_dim"]
            encoder_hidden_layers = vae_arch_config.get("encoder_layers", [])
            encoder_activation = vae_arch_config.get("activation", "relu")

            # Load the pre-trained deterministic encoder directly
            pretrained_deterministic_encoder_path = (
                foundation_model_dir
                / "models"
                / "foundation_model"
                / "deterministic_encoder"
            )
            if not pretrained_deterministic_encoder_path.exists():
                self.logger.error(
                    f"Pretrained deterministic encoder not found at {pretrained_deterministic_encoder_path}"
                )
                return False

            self.logger.info(
                f"Loading pre-trained deterministic encoder from: {pretrained_deterministic_encoder_path}"
            )
            pretrained_deterministic_encoder = tf.keras.models.load_model(
                pretrained_deterministic_encoder_path
            )

            self.logger.info(
                f"Loaded deterministic encoder with output shape: {pretrained_deterministic_encoder.output.shape}"
            )

            original_input_shape = (task_config.input.get_total_feature_size(),)
            classification_output_shape = (1,)  # Binary classification

            # Helper function to build classifier head
            def build_classifier_head(name_suffix: str) -> tf.keras.Model:
                # Create a copy of the DNN config with modified architecture for binary classification
                import copy

                classifier_config = copy.deepcopy(dnn_model_config)

                # Update the architecture for the classifier head
                classifier_config.architecture.update(
                    {
                        "input_shape": (latent_dim,),
                        "output_shape": classification_output_shape,
                        "output_activation": "sigmoid",  # Binary classification
                        "name": f"{dnn_model_config.architecture.get('name', 'classifier')}_{name_suffix}",
                    }
                )

                classifier_model_wrapper = ModelFactory.create_model(
                    model_type="dnn_predictor", config=classifier_config
                )
                classifier_model_wrapper.build()
                return classifier_model_wrapper.model

            # Helper function to create subset dataset
            def create_subset_dataset(dataset, num_events, data_size_index=None):
                """Create a subset of the dataset with exactly num_events events"""
                # Convert to unbatched dataset to count events precisely
                unbatched = dataset.unbatch()
                # Use different seed for each data size to ensure independent sampling
                # This prevents smaller datasets from being strict subsets of larger ones
                seed = 42 + (data_size_index or 0)  # Different seed for each data size
                shuffled = unbatched.shuffle(buffer_size=50000, seed=seed)
                subset = shuffled.take(num_events)
                # Rebatch with original batch size
                return subset.batch(dnn_training_config.batch_size)

            # Helper function to train and evaluate a model for a specific data size
            def train_and_evaluate_for_size(
                model_name: str,
                combined_keras_model: tf.keras.Model,
                train_subset,
                data_size: int,
                save_training_history: bool = False,
                label_distributions_dir_param: Optional[Path] = None,
                label_variable_names_param: Optional[list] = None,
            ):
                self.logger.info(
                    f"Training {model_name} model with {data_size} events..."
                )

                # Wrap the Keras model with CustomKerasModelWrapper for ModelTrainer
                wrapped_model_for_trainer = CustomKerasModelWrapper(
                    combined_keras_model, name=model_name
                )

                trainer_config_dict = {
                    "batch_size": dnn_training_config.batch_size,
                    "epochs": fixed_epochs,  # Use fixed epochs for fair comparison
                    "learning_rate": dnn_training_config.learning_rate,
                    "early_stopping": {
                        "patience": fixed_epochs + 1,
                        "min_delta": 0,
                    },  # Disable early stopping
                }

                trainer = ModelTrainer(
                    model=wrapped_model_for_trainer, training_config=trainer_config_dict
                )

                # Train with reduced verbosity for evaluation
                _ = trainer.train(  # Unused return value
                    dataset=train_subset,
                    validation_data=labeled_val_dataset,
                    callbacks=[],  # No callbacks for speed
                    training_history_dir=classification_dir / "training_histories"
                    if save_training_history
                    else None,
                    model_name=model_name,
                    dataset_id=f"signal_classification_eval_{data_size}",
                    experiment_id="signal_classification_evaluation",
                    verbose_training="minimal",  # Reduce verbosity for evaluation models
                    save_individual_history=True,  # Save individual files for comparison plots
                )

                # Evaluate on test set
                test_metrics = trainer.evaluate(labeled_test_dataset)
                test_loss = test_metrics.get("test_loss", 0.0)
                test_accuracy = test_metrics.get("test_binary_accuracy", 0.0)

                self.logger.info(
                    f"{model_name} with {data_size} events - Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}"
                )
                return test_loss, test_accuracy

            # Store results for plotting
            results = {
                "data_sizes": data_sizes,
                "From_Scratch_loss": [],
                "Fine_Tuned_loss": [],
                "Fixed_Encoder_loss": [],
                "From_Scratch_accuracy": [],
                "Fine_Tuned_accuracy": [],
                "Fixed_Encoder_accuracy": [],
            }

            # 5. Run experiments for each data size
            for data_size_index, data_size in enumerate(data_sizes):
                self.logger.info(f"{'=' * 50}")
                self.logger.info(f"Training with {data_size} events")
                self.logger.info(f"{'=' * 50}")

                # Create subset of training data
                train_subset = create_subset_dataset(
                    labeled_train_dataset, data_size, data_size_index
                )

                # Enable training history saving for all data sizes
                should_save_history = True
                self.logger.info(
                    f"Enabling training history saving for all models with {data_size} events"
                )

                # Format data size for better labeling
                data_size_label = (
                    f"{data_size // 1000}k" if data_size >= 1000 else str(data_size)
                )

                # --- Model 1: From Scratch ---
                self.logger.info("Building From Scratch model...")
                scratch_encoder_layers = []
                for units in encoder_hidden_layers:
                    scratch_encoder_layers.append(
                        tf.keras.layers.Dense(units, activation=encoder_activation)
                    )
                scratch_encoder_layers.append(
                    tf.keras.layers.Dense(latent_dim, name="scratch_latent_space")
                )

                scratch_encoder_part = tf.keras.Sequential(
                    scratch_encoder_layers, name="scratch_encoder"
                )
                scratch_classifier_dnn = build_classifier_head("from_scratch")

                model_inputs = tf.keras.Input(
                    shape=original_input_shape, name="input_features"
                )
                encoded_scratch = scratch_encoder_part(model_inputs)
                predictions_scratch = scratch_classifier_dnn(encoded_scratch)
                model_from_scratch = tf.keras.Model(
                    inputs=model_inputs,
                    outputs=predictions_scratch,
                    name="Classifier_From_Scratch",
                )

                scratch_loss, scratch_accuracy = train_and_evaluate_for_size(
                    f"From_Scratch_{data_size_label}",
                    model_from_scratch,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=None,  # Signal classification doesn't need label distribution analysis
                    label_variable_names_param=None,  # Signal classification uses binary labels
                )
                results["From_Scratch_loss"].append(scratch_loss)
                results["From_Scratch_accuracy"].append(scratch_accuracy)

                # --- Model 2: Fine-Tuned ---
                self.logger.info("Building Fine-Tuned model...")
                fine_tuned_input = tf.keras.Input(
                    shape=original_input_shape, name="fine_tuned_input"
                )
                fine_tuned_encoded = pretrained_deterministic_encoder(fine_tuned_input)

                # Add dtype casting to ensure compatibility with QKeras layers
                # Mixed precision may cause the encoder to output float16, but QKeras expects float32
                fine_tuned_encoded_cast = tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32),
                    name="dtype_cast_fine_tuned_classifier",
                )(fine_tuned_encoded)

                fine_tuned_encoder_part = tf.keras.Model(
                    inputs=fine_tuned_input,
                    outputs=fine_tuned_encoded_cast,
                    name="fine_tuned_pretrained_encoder",
                )
                fine_tuned_encoder_part.trainable = True

                fine_tuned_classifier_dnn = build_classifier_head("fine_tuned")

                model_inputs_ft = tf.keras.Input(
                    shape=original_input_shape, name="input_features_ft"
                )
                encoded_ft = fine_tuned_encoder_part(model_inputs_ft)
                predictions_ft = fine_tuned_classifier_dnn(encoded_ft)
                model_fine_tuned = tf.keras.Model(
                    inputs=model_inputs_ft,
                    outputs=predictions_ft,
                    name="Classifier_Fine_Tuned",
                )

                finetuned_loss, finetuned_accuracy = train_and_evaluate_for_size(
                    f"Fine_Tuned_{data_size_label}",
                    model_fine_tuned,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=None,  # Signal classification doesn't need label distribution analysis
                    label_variable_names_param=None,  # Signal classification uses binary labels
                )
                results["Fine_Tuned_loss"].append(finetuned_loss)
                results["Fine_Tuned_accuracy"].append(finetuned_accuracy)

                # --- Model 3: Fixed Encoder ---
                self.logger.info("Building Fixed Encoder model...")
                fixed_input = tf.keras.Input(
                    shape=original_input_shape, name="fixed_input"
                )
                fixed_encoded = pretrained_deterministic_encoder(fixed_input)

                # Add dtype casting to ensure compatibility with QKeras layers
                # Mixed precision may cause the encoder to output float16, but QKeras expects float32
                fixed_encoded_cast = tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32), name="dtype_cast_fixed_classifier"
                )(fixed_encoded)

                fixed_encoder_part = tf.keras.Model(
                    inputs=fixed_input,
                    outputs=fixed_encoded_cast,
                    name="fixed_pretrained_encoder",
                )
                fixed_encoder_part.trainable = False

                fixed_classifier_dnn = build_classifier_head("fixed_encoder")

                model_inputs_fx = tf.keras.Input(
                    shape=original_input_shape, name="input_features_fx"
                )
                encoded_fx = fixed_encoder_part(model_inputs_fx)
                predictions_fx = fixed_classifier_dnn(encoded_fx)
                model_fixed = tf.keras.Model(
                    inputs=model_inputs_fx,
                    outputs=predictions_fx,
                    name="Classifier_Fixed_Encoder",
                )

                fixed_loss, fixed_accuracy = train_and_evaluate_for_size(
                    f"Fixed_Encoder_{data_size_label}",
                    model_fixed,
                    train_subset,
                    data_size,
                    save_training_history=should_save_history,
                    label_distributions_dir_param=None,  # Signal classification doesn't need label distribution analysis
                    label_variable_names_param=None,  # Signal classification uses binary labels
                )
                results["Fixed_Encoder_loss"].append(fixed_loss)
                results["Fixed_Encoder_accuracy"].append(fixed_accuracy)

            # 6. Save results and create plots
            self.logger.info("Creating data efficiency plots...")

            # Save results to JSON
            results_file = (
                classification_dir
                / "signal_classification_data_efficiency_results.json"
            )
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {results_file}")

            # Create combined training history plot if we saved training histories
            training_histories_dir = classification_dir / "training_histories"
            combined_plot_path = (
                classification_dir / "signal_classification_training_comparison.png"
            )
            self.foundation_plot_manager.create_training_history_comparison_plot_from_directory(
                training_histories_dir,
                combined_plot_path,
                title_prefix="Signal Classification Model Training Comparison",
                validation_only=True,
            )

            # Create the data efficiency plots
            # Plot 1: Loss comparison
            loss_plot_file = classification_dir / "signal_classification_loss_plot.png"
            self.foundation_plot_manager.create_data_efficiency_plot(
                results,
                loss_plot_file,
                plot_type="classification_loss",
                metric_name="Test Loss (Binary Crossentropy)",
                total_test_events=total_test_events,
                signal_key=signal_key,
            )

            # Plot 2: Accuracy comparison
            accuracy_plot_file = (
                classification_dir / "signal_classification_accuracy_plot.png"
            )
            self.foundation_plot_manager.create_data_efficiency_plot(
                results,
                accuracy_plot_file,
                plot_type="classification_accuracy",
                metric_name="Test Accuracy",
                total_test_events=total_test_events,
                signal_key=signal_key,
            )

            # 7. Display summary
            log_evaluation_summary(
                results, evaluation_type="signal_classification", signal_key=signal_key
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Signal classification evaluation failed: {type(e).__name__}: {str(e)}"
            )
            self.logger.exception("Detailed traceback:")
            return False
