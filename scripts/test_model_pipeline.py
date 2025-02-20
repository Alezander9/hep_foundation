import json
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from hep_foundation.utils import ATLAS_RUN_NUMBERS
from hep_foundation.model_registry import ModelRegistry
from hep_foundation.model_factory import ModelFactory
from hep_foundation.model_trainer import ModelTrainer
from hep_foundation.processed_dataset_manager import ProcessedDatasetManager 

@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    run_numbers: List[str]
    catalog_limit: int
    track_selections: Dict
    event_selections: Dict
    max_tracks: int
    min_tracks: int
    validation_fraction: float
    test_fraction: float
    batch_size: int
    shuffle_buffer: int
    plot_distributions: bool

    def validate(self) -> None:
        """Validate dataset configuration parameters"""
        if not self.run_numbers:
            raise ValueError("run_numbers cannot be empty")
        if self.catalog_limit < 1:
            raise ValueError("catalog_limit must be positive")
        if self.max_tracks < self.min_tracks:
            raise ValueError("max_tracks must be greater than min_tracks")
        if not 0 <= self.validation_fraction + self.test_fraction < 1:
            raise ValueError("Sum of validation and test fractions must be less than 1")

@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""
    model_type: str
    latent_dim: int
    encoder_layers: List[int]
    decoder_layers: List[int]
    quant_bits: Optional[int]
    activation: str
    learning_rate: float
    epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    plot_training: bool
    # VAE-specific parameters
    beta_schedule: Optional[Dict] = None

    def validate(self) -> None:
        """Validate model configuration parameters"""
        valid_types = ["autoencoder", "variational_autoencoder"]
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")
        
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
            
        if not self.encoder_layers or not self.decoder_layers:
            raise ValueError("encoder_layers and decoder_layers cannot be empty")
            
        if self.model_type == "variational_autoencoder":
            if not self.beta_schedule:
                raise ValueError("beta_schedule required for VAE")
            required_beta_fields = ["start", "end", "warmup_epochs", "cycle_epochs"]
            missing = [f for f in required_beta_fields if f not in self.beta_schedule]
            if missing:
                raise ValueError(f"beta_schedule missing required fields: {missing}")

def test_model_pipeline(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    experiment_name: str,
    experiment_description: str
) -> bool:
    """
    Test the complete model pipeline with configuration objects
    
    Args:
        dataset_config: Configuration for dataset processing
        model_config: Configuration for model architecture and training
        experiment_name: Name for the experiment
        experiment_description: Description of the experiment
    """
    # Validate configurations
    dataset_config.validate()
    model_config.validate()

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

        # Create experiment directory
        experiment_dir = Path("experiments")
        print(f"\nCreating experiment directory at: {experiment_dir.absolute()}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry
        registry = ModelRegistry(str(experiment_dir))
        print(f"Registry initialized at: {registry.db_path}")
        
        # 1. Initialize managers
        print("Initializing managers...")
        data_manager = ProcessedDatasetManager()
        
        # 2. Load or create dataset
        print("Setting up data pipeline...")
        train_dataset, val_dataset, test_dataset = data_manager.load_datasets(
            config={
                'run_numbers': dataset_config.run_numbers,
                'track_selections': dataset_config.track_selections,
                'event_selections': dataset_config.event_selections,
                'max_tracks_per_event': dataset_config.max_tracks,
                'min_tracks_per_event': dataset_config.min_tracks,
                'catalog_limit': dataset_config.catalog_limit
            },
            validation_fraction=dataset_config.validation_fraction,
            test_fraction=dataset_config.test_fraction,
            batch_size=dataset_config.batch_size,
            shuffle_buffer=dataset_config.shuffle_buffer,
            plot_distributions=dataset_config.plot_distributions
        )
        
        # 3. Prepare configs for registry
        model_config_dict = {
            'input_shape': (dataset_config.max_tracks, 6),
            'latent_dim': model_config.latent_dim,
            'encoder_layers': model_config.encoder_layers,
            'decoder_layers': model_config.decoder_layers,
            'quant_bits': model_config.quant_bits,
            'activation': model_config.activation
        }
        
        # Add VAE-specific parameters if needed
        if model_config.model_type == "variational_autoencoder":
            model_config_dict['beta_schedule'] = model_config.beta_schedule
        
        model_config_registry = {
            "model_type": model_config.model_type,
            "architecture": {
                "input_dim": dataset_config.max_tracks,
                "latent_dim": model_config.latent_dim,
                "encoder_layers": model_config.encoder_layers,
                "decoder_layers": model_config.decoder_layers
            },
            "hyperparameters": {
                "activation": model_config.activation,
                "quant_bits": model_config.quant_bits,
                **({"beta_schedule": model_config.beta_schedule} 
                   if model_config.model_type == "variational_autoencoder" else {})
            }
        }
        
        training_config = {
            "batch_size": dataset_config.batch_size,
            "epochs": model_config.epochs,
            "learning_rate": model_config.learning_rate,
            "early_stopping": {
                "patience": model_config.early_stopping_patience,
                "min_delta": model_config.early_stopping_min_delta
            }
        }
        
        # 4. Register experiment with existing dataset
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
                model_type=model_config.model_type,
                config=model_config_dict
            )
            model.build()
        except Exception as e:
            print(f"Model creation failed: {str(e)}")
            print(f"Model config used: {json.dumps(model_config_dict, indent=2)}")
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
                callbacks=callbacks,
                plot_training=model_config.plot_training,
                plots_dir=Path(f"experiments/plots/{experiment_id}")  # Save plots with experiment
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

        # print("\nFinal metrics collected:")
        # for metric, value in all_metrics.items():
        #     print(f"  {metric}: {value:.6f}")

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
    """Main function serving as control panel for experiments"""
    
    # Choose experiment type
    MODEL_TYPE = "vae"  # "vae" or "autoencoder"
    # Dataset configuration - common for both models
    dataset_config = DatasetConfig(
        run_numbers=ATLAS_RUN_NUMBERS[-1:],
        catalog_limit=2,
        track_selections={
            'eta': (-2.5, 2.5),
            'chi2_per_ndof': (0.0, 10.0),
        },
        event_selections={},
        max_tracks=30,
        min_tracks=3,
        validation_fraction=0.15,
        test_fraction=0.15,
        batch_size=1024,
        shuffle_buffer=50000,
        plot_distributions=True
    )
    
    # Model configurations
    ae_config = ModelConfig(
        model_type="autoencoder",
        latent_dim=16,
        encoder_layers=[128, 64, 32],
        decoder_layers=[32, 64, 128],
        quant_bits=8,
        activation='relu',
        learning_rate=0.001,
        epochs=5,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        plot_training=True
    )
    
    vae_config = ModelConfig(
        model_type="variational_autoencoder",
        latent_dim=16,
        encoder_layers=[128, 64, 32],
        decoder_layers=[32, 64, 128],
        quant_bits=8,
        activation='relu',
        learning_rate=0.001,
        epochs=20,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        plot_training=True,
        beta_schedule={
            'start': 0.0,
            'end': 0.01,
            'warmup_epochs': 5,
            'cycle_epochs': 5
        }
    )
    
    # Select model configuration based on type
    model_config = vae_config if MODEL_TYPE == "vae" else ae_config
    
    try:
        success = test_model_pipeline(
            dataset_config=dataset_config,
            model_config=model_config,
            experiment_name=f"{MODEL_TYPE}_test",
            experiment_description=f"Testing {MODEL_TYPE} model with explicit parameters"
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