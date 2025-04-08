import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

from hep_foundation.model_registry import ModelRegistry
from hep_foundation.model_factory import ModelFactory
from hep_foundation.model_trainer import ModelTrainer, TrainingConfig
from hep_foundation.variational_autoencoder import VariationalAutoEncoder, AnomalyDetectionEvaluator
from hep_foundation.task_config import TaskConfig
from hep_foundation.dataset_manager import DatasetManager, DatasetConfig
from hep_foundation.base_model import ModelConfig
from hep_foundation.logging_config import setup_logging
from hep_foundation.dnn_predictor import DNNPredictor
def model_pipeline(
    dataset_config: DatasetConfig,
    model_config: dict,
    training_config: TrainingConfig,
    task_config: TaskConfig,
    experiment_name: str,
    experiment_description: str,
    delete_catalogs: bool = True
) -> bool:
    """
    Test the complete model pipeline with configuration objects
    
    Args:
        dataset_config: Configuration for dataset processing
        model_config: Dictionary containing model configuration
        training_config: Configuration for model training
        task_config: Configuration for task processing
        experiment_name: Name for the experiment
        experiment_description: Description of the experiment
        delete_catalogs: Whether to delete catalogs after processing
    """

    # Setup logging
    setup_logging()

    # Validate configurations
    dataset_config.validate()
    training_config.validate()

    logging.info("\n" + "="*50)
    logging.info("Starting Model Pipeline Test")
    logging.info("="*50)
    logging.info(f"TensorFlow: {tf.__version__} (Eager: {tf.executing_eagerly()})")
    
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

        # Create experiment directory
        experiment_dir = Path("experiments")
        logging.info(f"\nCreating experiment directory at: {experiment_dir.absolute()}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry
        registry = ModelRegistry(str(experiment_dir))
        logging.info(f"Registry initialized at: {registry.db_path}")
        
        # 1. Initialize managers
        logging.info("Initializing managers...")
        data_manager = DatasetManager()
        
        # 2. Load or create dataset
        logging.info("Loading datasets...")
        train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
            dataset_config=dataset_config,
            validation_fraction=dataset_config.validation_fraction,
            test_fraction=dataset_config.test_fraction,
            batch_size=training_config.batch_size,
            shuffle_buffer=dataset_config.shuffle_buffer,
            include_labels=dataset_config.include_labels,
            delete_catalogs=delete_catalogs
        )
        logging.info("DEBUG: Loaded datasets")
        
        # Get the dataset ID from the data manager
        dataset_id = data_manager.get_current_dataset_id()
        
        # Load signal datasets if specified
        signal_datasets = {}
        if dataset_config.signal_keys:
            logging.info("\nSetting up signal data pipeline...")
            signal_datasets = data_manager.load_signal_datasets(
                dataset_config=dataset_config,
                batch_size=training_config.batch_size,
                include_labels=dataset_config.include_labels,
            )

        if signal_datasets:
            logging.info(f"Loaded {len(signal_datasets)} signal datasets")
        else:
            logging.info("No signal datasets loaded")
        
        # 3. Prepare configs for registry - now just add input shape to architecture
        model_config_dict = {
            'model_type': model_config['model_type'],
            'architecture': {
                **model_config['architecture'],
                'input_shape': (task_config.input.get_total_feature_size(),) # Must be a tuple
            },
            'hyperparameters': model_config['hyperparameters']
        }
        
        # Training config is already in the right format
        training_config_dict = {
            "batch_size": training_config.batch_size,
            "epochs": training_config.epochs,
            "learning_rate": training_config.learning_rate,
            "early_stopping": training_config.early_stopping
        }
        
        # 4. Register experiment with existing dataset
        logging.info("Registering experiment...")
        experiment_id = registry.register_experiment(
            name=experiment_name,
            dataset_id=dataset_id,
            model_config=model_config_dict,
            training_config=training_config_dict,
            description=experiment_description
        )
        logging.info(f"Created experiment: {experiment_id}")
        
        # 5. Create and Build Model
        logging.info("Creating model...")
        try:
            model = ModelFactory.create_model(
                model_type=model_config['model_type'],
                config=model_config_dict
            )
            model.build()
        except Exception as e:
            logging.error(f"Model creation failed: {str(e)}")
            logging.error(f"Model config used: {json.dumps(model_config_dict, indent=2)}")
            raise
        
        # 6. Train Model
        logging.info("Setting up training...")
        trainer = ModelTrainer(
            model=model,
            training_config=training_config_dict
        )
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=training_config.early_stopping["patience"],
                min_delta=training_config.early_stopping["min_delta"],
                restore_best_weights=True
            )
        ]
        
        # Remove direct model compilation as it's now handled in ModelTrainer
        logging.info("\nVerifying dataset shapes:")
        for name, dataset in [
            ('Training', train_dataset),
            ('Validation', val_dataset),
            ('Test', test_dataset)
        ]:
            for batch in dataset.take(1):
                if isinstance(batch, tuple):
                    features, labels = batch
                    logging.info(f"{name} dataset shapes:")
                    logging.info(f"  Features: {features.shape}")
                    if isinstance(labels, (list, tuple)):
                        for i, label_set in enumerate(labels):
                            logging.info(f"  Label set {i}: {label_set.shape}")
                    else:
                        logging.info(f"  Labels: {labels.shape}")
                    
                    # Additional verification for DNN predictor
                    if isinstance(model, DNNPredictor):
                        label_idx = model.label_index
                        if isinstance(labels, (list, tuple)):
                            if label_idx >= len(labels):
                                raise ValueError(f"Label index {label_idx} out of range for {len(labels)} label sets")
                            target_shape = labels[label_idx].shape[1:]  # Remove batch dimension
                            if target_shape != tuple(model.output_shape):
                                raise ValueError(
                                    f"Model output shape {model.output_shape} does not match "
                                    f"label shape {target_shape} at index {label_idx}"
                                )
                else:
                    logging.info(f"{name} dataset shape: {batch.shape}")
                break

        # Start training
        logging.info("\nStarting training...")
        try:
            training_results = trainer.train(
                dataset=train_dataset,
                validation_data=val_dataset,
                callbacks=callbacks,
                plot_training=training_config.plot_training,
                plots_dir=Path(f"experiments/{experiment_id}/plots") 
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

            # Skip anomaly detection test for DNN predictor
            if isinstance(model, VariationalAutoEncoder):
                logging.info("\nRunning anomaly detection tests...")
                tester = AnomalyDetectionEvaluator(
                    model=model,
                    test_dataset=test_dataset,
                    signal_datasets=signal_datasets,
                    experiment_id=experiment_id,
                    base_path=registry.base_path
                )
                additional_test_results = tester.run_anomaly_detection_test()
                logging.info(f"\nTest results: {additional_test_results}")

            # Save the trained model
            logging.info("Saving trained model...")
            model_metadata = {
                "test_loss": test_results.get('test_loss', 0.0),
                "test_mse": test_results.get('test_mse', 0.0),
                "final_train_loss": training_results['final_metrics'].get('loss', 0.0),
                "final_val_loss": training_results['final_metrics'].get('val_loss', 0.0),
                "training_duration": training_results['training_duration']
            }

            # Save model based on type
            if isinstance(model, VariationalAutoEncoder):
                registry.save_model(
                    experiment_id=experiment_id,
                    models={
                        "encoder": model.encoder,
                        "decoder": model.decoder,
                        "full_model": model.model
                    },
                    model_name="full_model",
                    metadata=ensure_serializable(model_metadata)
                )
            else:
                registry.save_model(
                    experiment_id=experiment_id,
                    models={"full_model": model.model},
                    model_name="full_model",
                    metadata=ensure_serializable(model_metadata)
                )

        except Exception as e:
            logging.error(f"\nTraining failed with error: {str(e)}")
            logging.info("\nDataset inspection:")
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
        logging.info("\n" + "="*50)
        logging.info("Experiment Results")
        logging.info("="*50)

        experiment_data = registry.get_experiment_data(experiment_id)
        
        logging.info(f"\nExperiment ID: {experiment_id}")
        logging.info(f"Status: {experiment_data['experiment_info']['status']}")

        if 'training_results' in experiment_data:
            training_results = experiment_data['training_results']
            logging.info(f"Training Duration: {training_results['training_duration']:.2f}s")
            logging.info(f"Epochs Completed: {training_results['epochs_completed']}")
            
            logging.info("\nMetrics:")
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
        
        
        logging.info("Pipeline test completed successfully")
        
        return True
        
    except Exception as e:
        logging.error(f"Pipeline test failed: {type(e).__name__}: {str(e)}")
        logging.error(f"Error context:")
        raise
