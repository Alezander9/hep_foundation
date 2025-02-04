import json
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path

from hep_foundation.model_registry import ModelRegistry
from hep_foundation.model_factory import ModelFactory
from hep_foundation.model_trainer import ModelTrainer
from hep_foundation.processed_dataset_manager import ProcessedDatasetManager 

def test_model_pipeline():
    """Test the complete model pipeline including factory, trainer, and registry"""

    print("\n" + "="*50)
    print("Starting Model Pipeline Test")
    print("="*50)
    print(f"TensorFlow: {tf.__version__} (Eager: {tf.executing_eagerly()})")
    
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

        # 1. Initialize Registry
        print("Initializing model registry...")
        registry = ModelRegistry("experiments")
        
        # 2. Setup Data Pipeline
        print("Setting up data pipeline...")
        data_manager = ProcessedDatasetManager()

        # Define selections
        track_selections = {
            'eta': (-2.5, 2.5),
            'chi2_per_ndof': (0.0, 10.0),  # Increased upper bound
        }

        event_selections = {}  # Remove event selections initially

        # Add debug prints before creating datasets
        print("\nSelection criteria:")
        print(f"Track selections: {track_selections}")
        print(f"Event selections: {event_selections}")

        # Create datasets
        train_dataset, val_dataset, test_dataset = data_manager.load_datasets(
            config={
                'run_numbers': ["00296939", "00296942"],
                'track_selections': track_selections,
                'event_selections': event_selections,
                'max_tracks_per_event': 30,
                'min_tracks_per_event': 3,
                'catalog_limit': 1
            },
            validation_fraction=0.15,
            test_fraction=0.15,
            batch_size=1000,
            shuffle_buffer=10000
        )

        # After creating datasets, update the dataset config
        dataset_config = ensure_serializable({
            # Basic configuration (used to create/identify dataset)
            "run_numbers": ["00296939", "00296942"],
            "track_selections": track_selections,
            "event_selections": event_selections,
            "max_tracks_per_event": 30,
            "min_tracks_per_event": 3,
            "catalog_limit": 1,
            
            # Dataset split configuration
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "batch_size": 1000,
            "shuffle_buffer": 10000,
            
            # These will be filled in by the registry from dataset info
            "dataset_id": None,  # Will be set by registry
            "dataset_path": None,  # Will be set by registry
            "creation_date": None,  # Will be set by registry
            "atlas_version": None,  # Will be set by registry
            "software_versions": None,  # Will be set by registry
            "normalization_params": None,  # Will be set from dataset
        })

        # Add debug prints to verify dataset creation
        print("\nDataset Creation Results:")
        print(f"Dataset ID: {data_manager.generate_dataset_id(dataset_config)}")
        print(f"Dataset Path: {data_manager.datasets_dir / f'{data_manager.generate_dataset_id(dataset_config)}.h5'}")

        # Verify dataset was created and can be loaded
        try:
            dataset_info = data_manager.get_dataset_info(data_manager.generate_dataset_id(dataset_config))
            print("\nDataset Info:")
            print(f"Creation Date: {dataset_info['creation_date']}")
            print(f"Software Versions: {dataset_info['software_versions']}")
        except Exception as e:
            print(f"Error getting dataset info: {e}")

        # When creating model config
        model_config = {
            'model_type': 'autoencoder',
            'input_shape': (30, 6),  # Or get from dataset_config
            'n_features': 6,
            'max_tracks_per_event': 30,
            'latent_dim': 16,
            'encoder_layers': [128, 64],
            'decoder_layers': [64, 128],
            'quant_bits': 8,
            'activation': 'relu'
        }
    
        
        # Model config - nested version for registry
        model_config_registry = {
            "model_type": "autoencoder",
            "architecture": {
                "input_dim": model_config['input_shape'][0],
                "latent_dim": model_config['latent_dim'],
                "encoder_layers": model_config['encoder_layers'],
                "decoder_layers": model_config['decoder_layers']
            },
            "hyperparameters": {
                "activation": model_config['activation'],
                "quant_bits": model_config['quant_bits']
            }
        }
        
        # Training config
        training_config = {
            "batch_size": 1000,
            "epochs": 5,
            "learning_rate": 0.001,
            "early_stopping": {
                "patience": 3,
                "min_delta": 1e-4
            }
        }
        
        # 4. Register Experiment
        print("Registering experiment...")
        experiment_id = registry.register_experiment(
            name="autoencoder_test",
            dataset_config=dataset_config,
            model_config=model_config_registry,
            training_config=training_config,
            description="Testing autoencoder on track data with enhanced monitoring"
        )
        print(f"Created experiment: {experiment_id}")
        
        # 5. Create and Build Model
        print("Creating model...")
        try:
            model = ModelFactory.create_model(
                model_type="autoencoder",
                config=model_config
            )
            model.build()
        except Exception as e:
            print(f"Model creation failed: {str(e)}")
            print(f"Model config used: {json.dumps(model_config, indent=2)}")
            raise
        
        # 6. Setup Training
        print("Setting up training...")
        trainer = ModelTrainer(
            model=model,
            training_config=training_config
        )
        
        # Setup callbacks
        class RegistryCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = ensure_serializable(logs or {})
                registry.update_training_progress(
                    experiment_id=experiment_id,
                    epoch=epoch,
                    metrics=logs,
                )
                
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=training_config["early_stopping"]["patience"],
                min_delta=training_config["early_stopping"]["min_delta"],
                restore_best_weights=True
            ),
            RegistryCallback()
        ]
        
        # Add debug mode for training
        print("\nSetting up model compilation...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_config["learning_rate"]),
            loss='mse',
            run_eagerly=True  # Add this for debugging
        )

        # Verify dataset shapes before training
        print("\nVerifying dataset shapes:")
        for name, dataset in [("Training", train_dataset), 
                             ("Validation", val_dataset), 
                             ("Test", test_dataset)]:
            for batch in dataset.take(1):
                print(f"{name} dataset shape: {batch.shape}")
                if any(dim == 0 for dim in batch.shape):
                    raise ValueError(f"Found zero dimension in {name.lower()} batch: {batch.shape}")

        # Start training with additional debugging
        print("\nStarting training...")
        try:
            training_start_time = datetime.now()
            training_results = trainer.train(
                dataset=train_dataset,
                validation_data=val_dataset,
                callbacks=callbacks
            )
            training_end_time = datetime.now()
        except Exception as e:
            print(f"\nTraining failed with error: {str(e)}")
            print("\nDataset inspection:")
            for i, batch in enumerate(train_dataset.take(1)):
                print(f"Training batch {i} shape: {batch.shape}")
                print(f"Sample of data: \n{batch[0, :5, :]}")  # Show first 5 tracks of first event
            raise
        
        # 8. Evaluate Model
        print("Evaluating model...")
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")  # Add debug print

        # After evaluation
        print("\nCollecting all metrics...")
        all_metrics = {
            **training_results['metrics'],  # Training metrics
            **test_results,                 # Test metrics
            'training_duration': (training_end_time - training_start_time).total_seconds()
        }

        print("\nFinal metrics collected:")
        for metric, value in all_metrics.items():
            print(f"  {metric}: {value:.6f}")

        registry.complete_training(
            experiment_id=experiment_id,
            final_metrics=ensure_serializable(all_metrics)
        )
        
        # 9. Save Model
        print("Saving model checkpoint...")
        checkpoint_metadata = {
            "test_loss": test_results.get('loss', 0.0),
            "test_mse": test_results.get('mse', 0.0),
            "final_train_loss": training_results.get('final_loss', 0.0),
            "final_val_loss": training_results.get('final_val_loss', 0.0)
        }

        registry.save_checkpoint(
            experiment_id=experiment_id,
            models={"autoencoder": model.model},
            checkpoint_name="final",
            metadata=ensure_serializable(checkpoint_metadata)
        )
        
        # 10. Display Results
        print("\n" + "="*50)
        print("Experiment Results")
        print("="*50)

        details = registry.get_experiment_details(experiment_id)
        performance = registry.get_performance_summary(experiment_id)

        print(f"\nExperiment ID: {experiment_id}")
        print(f"Status: {details['experiment_info']['status']}")

        # Handle potential None values for duration
        duration = performance.get('training_duration')
        if duration is not None:
            print(f"Training Duration: {duration:.2f}s")
        else:
            print("Training Duration: Not available")

        print(f"Epochs Completed: {performance['epochs_completed']}")

        print("\nMetrics:")
        def print_metrics(metrics, indent=2):
            """Helper function to print metrics with proper formatting"""
            for key, value in metrics.items():
                indent_str = " " * indent
                if isinstance(value, dict):
                    print(f"{indent_str}{key}:")
                    print_metrics(value, indent + 2)
                elif isinstance(value, (float, int)):
                    print(f"{indent_str}{key}: {value:.6f}")
                else:
                    print(f"{indent_str}{key}: {value}")

        # Print metrics using the helper function
        print_metrics(performance['final_metrics'])
        
        # 11. Visualize Results
        if True:  # Change to control visualization
            plt.figure(figsize=(12, 6))
            history = performance.get('metric_progression', {})
            
            # Add safety checks
            if 'loss' in history and 'val_loss' in history:
                plt.plot(history['loss'], label='Training Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training History')
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                print("Warning: Training history metrics not available for plotting")
        
        print("Pipeline test completed successfully")

        # After training completes successfully, add:
        print("\nVerifying Dataset Reproducibility:")
        try:
            # Get experiment details
            with sqlite3.connect(registry.db_path) as conn:
                cursor = conn.execute(
                    "SELECT dataset_id, dataset_path FROM dataset_configs WHERE experiment_id = ?",
                    (experiment_id,)
                )
                dataset_id, dataset_path = cursor.fetchone()
            
            print(f"Original Dataset ID: {dataset_id}")
            print(f"Original Dataset Path: {dataset_path}")
            
            # Verify dataset can be recreated
            if Path(dataset_path).exists():
                print("Original dataset file exists")
            else:
                print("Original dataset not found - attempting recreation")
                new_path = data_manager.recreate_dataset(dataset_id)
                print(f"Recreated dataset at: {new_path}")
                
                # Verify recreation was successful
                if data_manager.verify_dataset(dataset_id, dataset_config):
                    print("Dataset recreation verified successfully")
                else:
                    print("Warning: Recreated dataset may not match original")
            
        except Exception as e:
            print(f"Dataset verification failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"Pipeline test failed: {type(e).__name__}: {str(e)}")
        print(f"Error context:")
        raise

def main():
    """Main function to run the pipeline test"""
    try:
        success = test_model_pipeline()
        if success:
            print("\nAll tests passed successfully!")
            return 0  # Unix convention for success
        else:
            print("\nTests failed!")
            return 1  # Unix convention for error
    except Exception as e:
        print("\nTest failed with error:")
        print(str(e))
        return 1

if __name__ == "__main__":
    # This ensures the script only runs when executed directly
    exit(main())