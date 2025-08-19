import json
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import tensorflow as tf

from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.logging_config import get_logger
from hep_foundation.config.task_config import TaskConfig
from hep_foundation.config.training_config import TrainingConfig
from hep_foundation.data.dataset_manager import DatasetManager
from hep_foundation.models.model_factory import ModelFactory
from hep_foundation.models.model_registry import ModelRegistry
from hep_foundation.models.variational_autoencoder import (
    BetaSchedule,
    VAEConfig,
)
from hep_foundation.training.model_trainer import ModelTrainer


class FoundationModelTrainer:
    """
    Handles foundation model training functionality.
    """

    def __init__(
        self,
        experiments_output_dir: str,
        processed_datasets_dir: str,
        logger=None,
        source_config_file: str = None,
    ):
        """
        Initialize the foundation model trainer.

        Args:
            experiments_output_dir: Directory for experiment outputs
            processed_datasets_dir: Directory for processed datasets
            logger: Logger instance (optional, will create one if not provided)
            source_config_file: Path to source config file for reproducibility
        """
        self.experiments_output_dir = Path(experiments_output_dir)
        self.processed_datasets_dir = Path(processed_datasets_dir)
        self.logger = logger or get_logger(__name__)
        self._source_config_file = source_config_file

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
                    task_config=task_config,
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
