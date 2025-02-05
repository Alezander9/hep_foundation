import json
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path
from typing import Dict, Any, List

from hep_foundation.model_registry import ModelRegistry
from hep_foundation.model_factory import ModelFactory
from hep_foundation.model_trainer import ModelTrainer
from hep_foundation.processed_dataset_manager import ProcessedDatasetManager 

def test_model_pipeline(
    # Dataset parameters
    run_numbers: list = ["00296939", "00296942", "00297447"],
    catalog_limit: int = 5,
    
    # Track selection parameters
    track_selections: Dict = {
        'eta': (-2.5, 2.5),
        'chi2_per_ndof': (0.0, 10.0),
    },
    event_selections: Dict = {},
    max_tracks: int = 56,
    min_tracks: int = 3,
    
    # Dataset processing parameters
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    batch_size: int = 128,
    shuffle_buffer: int = 50000,
    plot_distributions: bool = True,
    
    # Model architecture parameters
    latent_dim: int = 32,
    encoder_layers: List[int] = [512, 256, 128],
    decoder_layers: List[int] = [128, 256, 512],
    quant_bits: int = 8,
    activation: str = 'relu',
    
    # Training parameters
    epochs: int = 50,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 1e-4,
    
    # Experiment parameters
    experiment_name: str = "autoencoder_test",
    experiment_description: str = "Testing autoencoder on track data with enhanced monitoring"
) -> bool:
    """
    Test the complete model pipeline with configurable parameters
    
    Args:
        run_numbers: List of ATLAS run numbers to process
        catalog_limit: Maximum number of catalogs to process per run
        track_selections: Selection criteria for individual tracks
        event_selections: Selection criteria for entire events
        max_tracks: Maximum number of tracks per event
        min_tracks: Minimum number of tracks per event
        validation_fraction: Fraction of data for validation
        test_fraction: Fraction of data for testing
        batch_size: Training batch size
        shuffle_buffer: Size of shuffle buffer
        plot_distributions: Whether to create distribution plots
        latent_dim: Dimension of latent space
        encoder_layers: List of layer sizes for encoder
        decoder_layers: List of layer sizes for decoder
        quant_bits: Number of bits for quantization
        activation: Activation function to use
        epochs: Number of training epochs
        learning_rate: Training learning rate
        early_stopping_patience: Patience for early stopping
        early_stopping_min_delta: Minimum delta for early stopping
        experiment_name: Name for the experiment
        experiment_description: Description of the experiment
    """
    
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

        # 1. Initialize managers
        print("Initializing managers...")
        data_manager = ProcessedDatasetManager()
        registry = ModelRegistry("experiments")
        
        # 2. Load or create dataset
        print("Setting up data pipeline...")
        train_dataset, val_dataset, test_dataset = data_manager.load_datasets(
            config={
                'run_numbers': run_numbers,
                'track_selections': track_selections,
                'event_selections': event_selections,
                'max_tracks_per_event': max_tracks,
                'min_tracks_per_event': min_tracks,
                'catalog_limit': catalog_limit
            },
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            batch_size=batch_size,
            shuffle_buffer=shuffle_buffer,
            plot_distributions=plot_distributions
        )
        
        # Prepare configs for registry
        dataset_config = {
            "run_numbers": run_numbers,
            "track_selections": track_selections,
            "event_selections": event_selections,
            "max_tracks_per_event": max_tracks,
            "min_tracks_per_event": min_tracks,
            "catalog_limit": catalog_limit,
            "train_fraction": 1.0 - (validation_fraction + test_fraction),
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "batch_size": batch_size,
            "shuffle_buffer": shuffle_buffer
        }
        
        model_config = {
            'model_type': 'autoencoder',
            'input_shape': (max_tracks, 6),
            'n_features': 6,
            'max_tracks_per_event': max_tracks,
            'latent_dim': latent_dim,
            'encoder_layers': encoder_layers,
            'decoder_layers': decoder_layers,
            'quant_bits': quant_bits,
            'activation': activation
        }
        
        model_config_registry = {
            "model_type": "autoencoder",
            "architecture": {
                "input_dim": max_tracks,
                "latent_dim": latent_dim,
                "encoder_layers": encoder_layers,
                "decoder_layers": decoder_layers
            },
            "hyperparameters": {
                "activation": activation,
                "quant_bits": quant_bits
            }
        }
        
        training_config = {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "early_stopping": {
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta
            }
        }
        
        # 3. Register experiment with existing dataset
        print("Registering experiment...")
        experiment_id = registry.register_experiment(
            name=experiment_name,
            dataset_id=data_manager.get_current_dataset_id(),
            model_config=model_config_registry,
            training_config=training_config,
            description=experiment_description
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
        
        return True
        
    except Exception as e:
        print(f"Pipeline test failed: {type(e).__name__}: {str(e)}")
        print(f"Error context:")
        raise

def main():
    """Main function with parameter configuration"""
    try:
        # Example configuration for medium-scale test
        success = test_model_pipeline(
            # Dataset parameters
            run_numbers=["00296939", "00296942"],
            catalog_limit=1,
            
            # Track selection parameters
            max_tracks=20,
            min_tracks=3,
            
            # Processing parameters
            batch_size=1024,
            shuffle_buffer=1000,
            plot_distributions=True,
            
            # Model architecture
            latent_dim=32,
            encoder_layers=[256, 128, 64],
            decoder_layers=[64, 128, 256],
            
            # Training parameters
            epochs=5,
            learning_rate=0.001,
            
            # Experiment parameters
            experiment_name="small_scale_test",
            experiment_description="Small scale test with 2 runs, 1 catalog each"
        )
        
        if success:
            print("\nAll tests passed successfully!")
            return 0
        else:
            print("\nTests failed!")
            return 1
            
    except Exception as e:
        print("\nTest failed with error:")
        print(str(e))
        return 1

if __name__ == "__main__":
    exit(main())