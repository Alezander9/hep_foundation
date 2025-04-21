import logging
from pathlib import Path
import tensorflow as tf
import json
import numpy as np

from hep_foundation.logging_config import setup_logging
from hep_foundation.model_registry import ModelRegistry
from hep_foundation.model_factory import ModelFactory
from hep_foundation.model_trainer import ModelTrainer, TrainingConfig
from hep_foundation.variational_autoencoder import VariationalAutoEncoder
from hep_foundation.task_config import TaskConfig
from hep_foundation.dataset_manager import DatasetManager, DatasetConfig
from hep_foundation.base_model import ModelConfig
from hep_foundation.utils import ATLAS_RUN_NUMBERS
from hep_foundation.plot_utils import plot_combined_training_histories

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
        # Setup logging
        setup_logging()
        
        # Create experiment directory
        self.experiment_dir = Path(base_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Foundation Model Pipeline initialized with base directory: {self.experiment_dir.absolute()}")
        logging.info(f"TensorFlow: {tf.__version__} (Eager: {tf.executing_eagerly()})")
    
    def run_process(
        self,
        process_name: str,
        dataset_config: DatasetConfig,
        vae_model_config: dict,
        dnn_model_config: dict,
        vae_training_config: TrainingConfig,
        dnn_training_config: TrainingConfig,
        task_config: TaskConfig,
        experiment_name: str = None,
        experiment_description: str = None,
        delete_catalogs: bool = True,
        foundation_model_path: str = None
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
            experiment_name: Optional name for the experiment
            experiment_description: Optional description for the experiment
            delete_catalogs: Whether to delete catalogs after processing
            foundation_model_path: Path to the foundation model encoder to use for encoding
        """
        valid_processes = ["train", "anomaly", "regression"]
        if process_name not in valid_processes:
            logging.error(f"Invalid process name: {process_name}. Must be one of {valid_processes}")
            return False
        
        if process_name == "train":
            return self.train_foundation_model(
                dataset_config=dataset_config,
                model_config=vae_model_config,
                training_config=vae_training_config,
                task_config=task_config,
                experiment_name=experiment_name,
                experiment_description=experiment_description,
                delete_catalogs=delete_catalogs
            )
        elif process_name == "anomaly":
            return self.evaluate_foundation_model_anomaly_detection()
        elif process_name == "regression":
            return self.evaluate_foundation_model_regression(
                dataset_config=dataset_config,
                dnn_model_config=dnn_model_config,
                dnn_training_config=dnn_training_config,
                task_config=task_config,
                experiment_name=experiment_name,
                experiment_description=experiment_description,
                delete_catalogs=delete_catalogs,
                foundation_model_path=foundation_model_path
            )
        else:
            logging.error(f"Unknown process name: {process_name}")
            return False
    
    def train_foundation_model(
        self,
        dataset_config: DatasetConfig,
        model_config: dict,
        training_config: TrainingConfig,
        task_config: TaskConfig,
        experiment_name: str = None,
        experiment_description: str = None,
        delete_catalogs: bool = True
    ) -> bool:
        """
        Train a foundation model using provided configurations.
        """
        logging.info("="*100)
        logging.info("Training Foundation Model")
        logging.info("="*100)
        
        try:
            # Helper function for JSON serialization
            def ensure_serializable(obj):
                """Recursively convert numpy types to Python native types"""
                if isinstance(obj, dict):
                    return {key: ensure_serializable(value) for key, value in obj.items()}
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
            logging.info(f"Registry initialized at: {registry.db_path}")
            
            # 1. Initialize managers
            logging.info("Initializing managers...")
            data_manager = DatasetManager()
            
            # 2. Validate Configs
            dataset_config.validate()
            logging.info("Validated dataset config")

            training_config.validate()
            logging.info("Validated training config")

            # 3. Load datasets

            logging.info("Loading datasets...")
            train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
                dataset_config=dataset_config,
                validation_fraction=dataset_config.validation_fraction,
                test_fraction=dataset_config.test_fraction,
                batch_size=training_config.batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=dataset_config.include_labels,
                delete_catalogs=True
            )
            logging.info("Loaded datasets")
            
            # Get the dataset ID from the data manager
            dataset_id = data_manager.get_current_dataset_id()

            
            # 4. Register experiment with existing dataset
            logging.info("Registering experiment...")
            model_config_dict = {
                'model_type': model_config['model_type'],
                'architecture': {
                    **model_config['architecture'],
                    'input_shape': (task_config.input.get_total_feature_size(),) # Must be a tuple
                },
                'hyperparameters': model_config['hyperparameters']
            }
            training_config_dict = {
                "batch_size": training_config.batch_size,
                "epochs": training_config.epochs,
                "learning_rate": training_config.learning_rate,
                "early_stopping": training_config.early_stopping
            }
            experiment_id = registry.register_experiment(
                name="Foundation VAE Model",
                dataset_id=dataset_id,
                model_config=model_config_dict,
                training_config=training_config_dict,
                description="Training a foundation VAE model for feature encoding"
            )
            logging.info(f"Created experiment: {experiment_id}")
            
            # 5. Create and Build Model
            logging.info("Creating model...")
            try:
                model = ModelFactory.create_model(
                    model_type='variational_autoencoder',
                    config=model_config_dict
                )
                model.build()
            except Exception as e:
                logging.error(f"Model creation failed: {str(e)}")
                logging.error(f"Model config used: {json.dumps(model_config_dict, indent=2)}")
                raise

            logging.info("Model created")
            logging.info(model.model.summary())
            
            # 6. Train Model
            logging.info("Setting up model and callbacks...")
            trainer = ModelTrainer(
                model=model,
                training_config=training_config_dict
            )
            
            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=training_config_dict["early_stopping"]["patience"],
                    min_delta=training_config_dict["early_stopping"]["min_delta"],
                    restore_best_weights=True
                )
            ]
            
            # Start training
            logging.info("Starting training...")
            try:
                training_results = trainer.train(
                    dataset=train_dataset,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    plot_training=True,
                    plots_dir=Path(f"{self.experiment_dir}/{experiment_id}/plots") 
                )
                
                # Evaluate Model
                logging.info("Evaluating model...")
                test_results = trainer.evaluate(test_dataset)
                
                # Combine results
                final_metrics = {
                    **training_results['final_metrics'],
                    **test_results,
                    'training_duration': training_results['training_duration'],
                    'epochs_completed': training_results['epochs_completed'],
                    'history': training_results['history']
                }

                # Update experiment data
                registry.complete_training(
                    experiment_id=experiment_id,
                    final_metrics=ensure_serializable(final_metrics)
                )

                # Save the trained model
                logging.info("Saving trained model...")
                model_metadata = {
                    "test_loss": test_results.get('test_loss', 0.0),
                    "test_mse": test_results.get('test_mse', 0.0),
                    "final_train_loss": training_results['final_metrics'].get('loss', 0.0),
                    "final_val_loss": training_results['final_metrics'].get('val_loss', 0.0),
                    "training_duration": training_results['training_duration']
                }

                registry.save_model(
                    experiment_id=experiment_id,
                    models={
                        "encoder": model.encoder,
                        "decoder": model.decoder,
                        "full_model": model.model
                    },
                    model_name="foundation_model",
                    metadata=ensure_serializable(model_metadata)
                )

            except Exception as e:
                logging.error(f"\nTraining failed with error: {str(e)}")
                logging.info("Dataset inspection:")
                for i, batch in enumerate(train_dataset.take(1)):
                    if isinstance(batch, tuple):
                        features, _ = batch
                        logging.info(f"Training batch {i} features shape: {features.shape}")
                        logging.info(f"Sample of features: \n{features[0, :10]}")  # Show first 10 features of first event
                    else:
                        logging.info(f"Training batch {i} shape: {batch.shape}")
                        logging.info(f"Sample of data: \n{batch[0, :10]}")  # Show first 10 features of first event
                raise
            
            # Display Results
            logging.info("="*100)
            logging.info("Experiment Results")
            logging.info("="*100)

            experiment_data = registry.get_experiment_data(experiment_id)
            
            logging.info(f"Experiment ID: {experiment_id}")
            logging.info(f"Status: {experiment_data['experiment_info']['status']}")

            if 'training_results' in experiment_data:
                training_results = experiment_data['training_results']
                logging.info(f"Training Duration: {training_results['training_duration']:.2f}s")
                logging.info(f"Epochs Completed: {training_results['epochs_completed']}")
                
                logging.info("Metrics:")
                def print_metrics(metrics, indent=2):
                    """Helper function to print metrics with proper formatting"""
                    for key, value in metrics.items():
                        indent_str = " " * indent
                        if isinstance(value, dict):
                            logging.info(f"{indent_str}{key}:")
                            print_metrics(value, indent + 2)
                        elif isinstance(value, (float, int)):
                            logging.info(f"{indent_str}{key}: {value:.6f}")
                        else:
                            logging.info(f"{indent_str}{key}: {value}")

                print_metrics(training_results['final_metrics'])
            
            logging.info("Foundation model training completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Foundation model training failed: {type(e).__name__}: {str(e)}")
            logging.error(f"Error context:")
            raise
    
    def evaluate_foundation_model_anomaly_detection(self):
        """
        Evaluate a foundation model for anomaly detection.
        
        This method will:
        1. Load a trained foundation model
        2. Load test and signal datasets
        3. Evaluate the model's anomaly detection performance
        4. Save the evaluation results
        """
        logging.info("="*100)
        logging.info("Evaluating Foundation Model for Anomaly Detection")
        logging.info("="*100)
        
        # TODO: Implement anomaly detection evaluation
        logging.info("Hello World from evaluate_foundation_model_anomaly_detection!")
        
        return True
    
    def evaluate_foundation_model_regression(
        self,
        dataset_config: DatasetConfig,
        dnn_model_config: dict,
        dnn_training_config: TrainingConfig,
        task_config: TaskConfig,
        experiment_name: str = None,
        experiment_description: str = None,
        delete_catalogs: bool = True,
        foundation_model_path: str = None
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
        logging.info("="*100)
        logging.info("Evaluating Foundation Model for Regression")
        logging.info("="*100)
        
        try:
            # Initialize registry and data manager
            registry = ModelRegistry(str(self.experiment_dir))
            data_manager = DatasetManager()
            
            # 1. Load dataset with regression labels
            logging.info("Loading datasets with regression labels...")
            train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
                dataset_config=dataset_config,
                validation_fraction=dataset_config.validation_fraction,
                test_fraction=dataset_config.test_fraction,
                batch_size=dnn_training_config.batch_size,
                shuffle_buffer=dataset_config.shuffle_buffer,
                include_labels=True,
                delete_catalogs=delete_catalogs
            )

            dataset_config.validate()
            logging.info("Validated dataset config")

            dnn_training_config.validate()
            logging.info("Validated training config")
            
            # 2. Train baseline regression model
            # Add input shape to the model config
            dnn_model_config_dict = {
                'model_type': dnn_model_config['model_type'],
                'architecture': {
                    **dnn_model_config['architecture'],
                    'input_shape': (task_config.input.get_total_feature_size(),), # Must be a tuple
                    'output_shape': (task_config.labels[0].get_total_feature_size(),)# For MET prediction (mpx, mpy, sumet)
                    # Note using first label set always for now
                },
                'hyperparameters': dnn_model_config['hyperparameters']
            }

            logging.info("Training baseline regression model...")
            baseline_model = ModelFactory.create_model(
                model_type='dnn_predictor',
                config=dnn_model_config_dict
            )
            baseline_model.build()
            logging.info("Baseline model created")
            logging.info(baseline_model.model.summary())

            dnn_training_config_dict = {
                "batch_size": dnn_training_config.batch_size,
                "epochs": dnn_training_config.epochs,
                "learning_rate": dnn_training_config.learning_rate,
                "early_stopping": dnn_training_config.early_stopping
            }

            logging.info("Setting up baseline model trainer...")
            baseline_trainer = ModelTrainer(
                model=baseline_model,
                training_config=dnn_training_config_dict
            )
            # Log sizes and shapes of baseline datasets
            logging.info(f"Baseline train dataset size: {train_dataset.cardinality()}")
            for batch in train_dataset.take(1):
                if isinstance(batch, tuple):
                    features, labels = batch
                    logging.info(f"Baseline training dataset shapes:")
                    logging.info(f"  Features: {features.shape}")
                    logging.info(f"  Labels: {labels.shape}")
                else:
                    logging.info(f"Baseline training dataset shape: {batch.shape}")
            logging.info("Training baseline regression model...")
            baseline_results = baseline_trainer.train(
                dataset=train_dataset,
                validation_data=val_dataset,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=dnn_training_config.early_stopping["patience"],
                        min_delta=dnn_training_config.early_stopping["min_delta"],
                        restore_best_weights=True
                    )
                ],
                plot_training=True,
                plots_dir=Path(f"{foundation_model_path}/baseline_regression/plots")
            )
            
            # 3. Load foundation model encoder
            logging.info("Loading foundation model...")
            if foundation_model_path:
                # Load from specified path
                foundation_model_path = Path(foundation_model_path)
                logging.info(f"Loaded foundation model encoder from: {foundation_model_path}/models/foundation_model/encoder")
                foundation_model = tf.keras.models.load_model(f"{foundation_model_path}/models/foundation_model/encoder")
            else:
                raise ValueError("No foundation model path provided")
            
            # 4. Create encoded datasets
            logging.info("Creating encoded datasets...")
            
            def encode_batch(batch, *args): # *args is for compatibility with the dataset which also comes with an index argument
                # Get features and labels
                features = batch
                labels = args[0] if args else None
                
                # Encode the features
                encoded_features = foundation_model(features)
                
                # Handle case where foundation model returns a list
                if isinstance(encoded_features, list):
                    # Take the first element if it's a list (usually the encoded features)
                    encoded_features = encoded_features[0]
                
                # Return encoded features and labels
                if labels is not None:
                    return encoded_features, labels
                return encoded_features

            logging.info("Encoded datasets")
            
            # Create encoded datasets
            logging.info("Creating encoded datasets...")
            encoded_train_dataset = train_dataset.map(encode_batch)
            encoded_val_dataset = val_dataset.map(encode_batch)
            encoded_test_dataset = test_dataset.map(encode_batch)
            logging.info("Encoded datasets created")
            # Log sizes and shapes of encoded datasets
            logging.info(f"Encoded train dataset size: {encoded_train_dataset.cardinality()}")
            for batch in encoded_train_dataset.take(1):
                if isinstance(batch, tuple):
                    features, labels = batch
                    logging.info(f"Encoded training dataset shapes:")
                    logging.info(f"  Features: {features.shape}")
                    logging.info(f"  Labels: {labels.shape}")
            
            # 5. Train regression model on encoded data
            logging.info("Training regression model on encoded data...")
            # Create a new config for the encoded model with adjusted input shape
            encoded_model_config = {
                **dnn_model_config,
                'architecture': {
                    **dnn_model_config['architecture'],
                    'input_shape': (16,),  # Latent dimension from foundation model
                    'output_shape': (task_config.labels[0].get_total_feature_size(),),  # Same as baseline model
                    'name': 'encoded_met_predictor'
                }
            }
            
            logging.info("Creating regression model on encoded data...")
            encoded_model = ModelFactory.create_model(
                model_type='dnn_predictor',
                config=encoded_model_config
            )
            encoded_model.build()
            logging.info("Regression model on encoded data created")
            logging.info(encoded_model.model.summary())

            logging.info("Setting up regression model training on encoded data...")
            encoded_trainer = ModelTrainer(
                model=encoded_model,
                training_config=dnn_training_config_dict
            )
            logging.info("Training regression model on encoded data...")
            encoded_results = encoded_trainer.train(
                dataset=encoded_train_dataset,
                validation_data=encoded_val_dataset,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=dnn_training_config.early_stopping["patience"],
                        min_delta=dnn_training_config.early_stopping["min_delta"],
                        restore_best_weights=True
                    )
                ],
                plot_training=True,
                plots_dir=Path(f"{foundation_model_path}/encoded_regression/plots")
            )
            logging.info("Regression model on encoded data trained")
            # 6. Evaluate and compare models
            logging.info("Evaluating models...")

            # 6a. Generate combined training plot
            try:
                baseline_history = baseline_results.get('history', {})
                encoded_history = encoded_results.get('history', {})
                
                if baseline_history or encoded_history: # Only plot if there is history data
                    comparison_plot_path = Path(foundation_model_path) / "testing" / "regression_training_comparison.pdf"
                    plot_combined_training_histories(
                        histories={
                            "Baseline": baseline_history,
                            "Encoded": encoded_history
                        },
                        output_path=comparison_plot_path,
                        title="Baseline vs Encoded Regression Training", # More specific title
                        # metrics_to_plot=['loss', 'val_loss', 'mse', 'val_mse'], # Example: Plot MSE too
                        # metric_labels={'loss': 'Loss', 'val_loss': 'Val Loss', 'mse': 'MSE', 'val_mse': 'Val MSE'}
                    )
                else:
                    logging.warning("Skipping combined plot generation: No history data found in results.")
            except Exception as plot_error:
                logging.error(f"Failed to generate combined training plot: {plot_error}")

            baseline_test_results = baseline_trainer.evaluate(test_dataset)
            encoded_test_results = encoded_trainer.evaluate(encoded_test_dataset)
            
            # 7. Save evaluation results
            logging.info("Saving evaluation results...")
            evaluation_results = {
                'baseline_model': {
                    'training_metrics': baseline_results['final_metrics'],
                    'test_metrics': baseline_test_results
                },
                'encoded_model': {
                    'training_metrics': encoded_results['final_metrics'],
                    'test_metrics': encoded_test_results
                },
                'comparison': {
                    'test_loss_ratio': encoded_test_results['test_loss'] / baseline_test_results['test_loss'],
                    'test_mse_ratio': encoded_test_results['test_mse'] / baseline_test_results['test_mse']
                }
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
            logging.info("Regression Evaluation Results:")
            logging.info("="*50)
            logging.info("Baseline Model:")
            logging.info(f"Test Loss: {baseline_test_results['test_loss']:.6f}")
            logging.info(f"Test MSE: {baseline_test_results['test_mse']:.6f}")
            logging.info("Encoded Model:")
            logging.info(f"Test Loss: {encoded_test_results['test_loss']:.6f}")
            logging.info(f"Test MSE: {encoded_test_results['test_mse']:.6f}")
            logging.info("Comparison:")
            logging.info(f"Test Loss Ratio: {evaluation_results['comparison']['test_loss_ratio']:.3f}")
            logging.info(f"Test MSE Ratio: {evaluation_results['comparison']['test_mse_ratio']:.3f}")
            
            logging.info("Regression evaluation completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Regression evaluation failed: {type(e).__name__}: {str(e)}")
            logging.error(f"Error context:")
            raise
