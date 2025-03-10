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
from hep_foundation.variational_autoencoder import VariationalAutoEncoder
from hep_foundation.dataset_manager import DatasetManager 
from hep_foundation.model_tester import ModelTester

@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    run_numbers: List[str]
    signal_keys: Optional[List[str]]  # New property for signal datasets
    catalog_limit: int
    track_selections: Dict
    event_selections: Dict
    max_tracks: int
    min_tracks: int
    validation_fraction: float
    test_fraction: float
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
    """Configuration for model architecture"""
    model_type: str
    architecture: Dict[str, Any]  # Contains network architecture
    hyperparameters: Dict[str, Any]  # Contains model hyperparameters

    def __init__(
        self,
        model_type: str,
        latent_dim: int,
        encoder_layers: List[int],
        decoder_layers: List[int],
        quant_bits: Optional[int],
        activation: str,
        beta_schedule: Optional[Dict] = None
    ):
        self.model_type = model_type
        # Group architecture-related parameters
        self.architecture = {
            'latent_dim': latent_dim,
            'encoder_layers': encoder_layers,
            'decoder_layers': decoder_layers,
            'activation': activation
        }
        # Group hyperparameters
        self.hyperparameters = {
            'quant_bits': quant_bits
        }
        if beta_schedule:
            self.hyperparameters['beta_schedule'] = beta_schedule

    def validate(self) -> None:
        """Validate model configuration parameters"""
        valid_types = ["autoencoder", "variational_autoencoder"]
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")
        
        # Validate architecture
        if self.architecture['latent_dim'] < 1:
            raise ValueError("latent_dim must be positive")
            
        if not self.architecture['encoder_layers'] or not self.architecture['decoder_layers']:
            raise ValueError("encoder_layers and decoder_layers cannot be empty")
            
        # Validate VAE-specific parameters
        if self.model_type == "variational_autoencoder":
            if 'beta_schedule' not in self.hyperparameters:
                raise ValueError("beta_schedule required for VAE")
            required_beta_fields = ["start", "end", "warmup_epochs", "cycle_epochs"]
            missing = [f for f in required_beta_fields if f not in self.hyperparameters['beta_schedule']]
            if missing:
                raise ValueError(f"beta_schedule missing required fields: {missing}")

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping: Dict[str, Any]
    plot_training: bool

    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
        plot_training: bool
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping = {
            "patience": early_stopping_patience,
            "min_delta": early_stopping_min_delta
        }
        self.plot_training = plot_training

    def validate(self) -> None:
        """Validate training configuration parameters"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.early_stopping["patience"] < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping["min_delta"] < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")

def test_model_pipeline(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    experiment_name: str,
    experiment_description: str
) -> bool:
    """
    Test the complete model pipeline with configuration objects
    
    Args:
        dataset_config: Configuration for dataset processing
        model_config: Configuration for model architecture
        training_config: Configuration for model training
        experiment_name: Name for the experiment
        experiment_description: Description of the experiment
    """
    # Validate configurations
    dataset_config.validate()
    model_config.validate()
    training_config.validate()

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
        data_manager = DatasetManager()
        
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
            batch_size=training_config.batch_size,
            shuffle_buffer=dataset_config.shuffle_buffer,
            plot_distributions=dataset_config.plot_distributions,
            delete_catalogs=True # Delete catalogs after processing
        )
        
        # Get the dataset ID from the data manager
        dataset_id = data_manager.get_current_dataset_id()
        
        # Load signal datasets if specified
        if dataset_config.signal_keys:
            print("\nSetting up signal data pipeline...")
            signal_datasets = data_manager.load_signal_datasets(
                config={
                    'signal_types': dataset_config.signal_keys,
                    'track_selections': dataset_config.track_selections,
                    'event_selections': dataset_config.event_selections,
                    'max_tracks_per_event': dataset_config.max_tracks,
                    'min_tracks_per_event': dataset_config.min_tracks,
                    'catalog_limit': dataset_config.catalog_limit
                },
                batch_size=training_config.batch_size,
                plot_distributions=dataset_config.plot_distributions
            )

        if signal_datasets:
            print(f"Loaded {len(signal_datasets)} signal datasets")
        else:
            print("No signal datasets loaded")
        
        # 3. Prepare configs for registry - now just add input shape to architecture
        model_config_dict = {
            'model_type': model_config.model_type,
            'architecture': {
                **model_config.architecture,
                'input_shape': (dataset_config.max_tracks, 6)
            },
            'hyperparameters': model_config.hyperparameters
        }
        
        # Training config is already in the right format
        training_config_dict = {
            "batch_size": training_config.batch_size,
            "epochs": training_config.epochs,
            "learning_rate": training_config.learning_rate,
            "early_stopping": training_config.early_stopping
        }
        
        # 4. Register experiment with existing dataset
        print("Registering experiment...")
        experiment_id = registry.register_experiment(
            name=experiment_name,
            dataset_id=dataset_id,
            model_config=model_config_dict,
            training_config=training_config_dict,
            description=experiment_description
        )
        print(f"Created experiment: {experiment_id}")
        
        # 5. Create and Build Model
        print("Creating model...")
        try:
            model = ModelFactory.create_model(
                model_type=model_config.model_type,
                config=model_config_dict  # Pass the nested config directly
            )
            model.build()
        except Exception as e:
            print(f"Model creation failed: {str(e)}")
            print(f"Model config used: {json.dumps(model_config_dict, indent=2)}")
            raise
        
        # 6. Train Model
        print("Setting up training...")
        trainer = ModelTrainer(
            model=model,
            training_config=training_config_dict
        )
        
        # Setup callbacks - only essential training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=training_config.early_stopping["patience"],
                min_delta=training_config.early_stopping["min_delta"],
                restore_best_weights=True
            )
        ]
        
        # Add debug mode for training
        print("\nSetting up model compilation...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.learning_rate),
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
            training_results = trainer.train(
                dataset=train_dataset,
                validation_data=val_dataset,
                callbacks=callbacks,
                plot_training=training_config.plot_training,
                plots_dir=Path(f"experiments/{experiment_id}/plots") 
            )
            
            # Evaluate Model
            print("Evaluating model...")
            test_results = trainer.evaluate(test_dataset)
            
            # Combine all results
            final_metrics = {
                **training_results['final_metrics'],  # Final training metrics
                **test_results,                       # Test metrics
                'training_duration': training_results['training_duration'],
                'epochs_completed': training_results['epochs_completed'],
                'history': training_results['history']  # Complete training history
            }

            # Complete training in registry
            registry.complete_training(
                experiment_id=experiment_id,
                final_metrics=ensure_serializable(final_metrics)
            )

            # After training the model, add testing section
            if isinstance(model, VariationalAutoEncoder):
                print("\nRunning model tests...")
                
                # Initialize model tester
                tester = ModelTester(
                    model=model,
                    test_dataset=test_dataset,
                    signal_datasets=signal_datasets,  # From data_manager.load_signal_datasets()
                    experiment_id=experiment_id,
                    base_path=registry.base_path
                )
                
                # Run anomaly detection test
                additional_test_results = tester.run_anomaly_detection_test()
                
                print(f"\nTest results: {additional_test_results}")
            
            # Save the trained model
            print("Saving trained model...")
            model_metadata = {
                "test_loss": test_results.get('test_loss', 0.0),
                "test_mse": test_results.get('test_mse', 0.0),
                "final_train_loss": training_results['final_metrics'].get('loss', 0.0),
                "final_val_loss": training_results['final_metrics'].get('val_loss', 0.0),
                "training_duration": training_results['training_duration']
            }

            if isinstance(model, VariationalAutoEncoder):
                # Save encoder and decoder separately for VAE
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
                # Save single model for standard autoencoder
                registry.save_model(
                    experiment_id=experiment_id,
                    models={"full_model": model.model},
                    model_name="full_model",
                    metadata=ensure_serializable(model_metadata)
                )

        except Exception as e:
            print(f"\nTraining failed with error: {str(e)}")
            print("\nDataset inspection:")
            for i, batch in enumerate(train_dataset.take(1)):
                print(f"Training batch {i} shape: {batch.shape}")
                print(f"Sample of data: \n{batch[0, :5, :]}")  # Show first 5 tracks of first event
            raise
        
        # Display Results
        print("\n" + "="*50)
        print("Experiment Results")
        print("="*50)

        experiment_data = registry.get_experiment_data(experiment_id)
        
        print(f"\nExperiment ID: {experiment_id}")
        print(f"Status: {experiment_data['experiment_info']['status']}")

        if 'training_results' in experiment_data:
            training_results = experiment_data['training_results']
            print(f"Training Duration: {training_results['training_duration']:.2f}s")
            print(f"Epochs Completed: {training_results['epochs_completed']}")
            
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

            print_metrics(training_results['final_metrics'])
        
        
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
        run_numbers=ATLAS_RUN_NUMBERS[-2:],
        signal_keys=["zprime", "wprime_qq", "zprime_bb"],
        catalog_limit=3,
        track_selections={
            'eta': (-2.5, 2.5),
            'chi2_per_ndof': (0.0, 10.0),
        },
        event_selections={},
        max_tracks=30,
        min_tracks=10,
        validation_fraction=0.15,
        test_fraction=0.15,
        shuffle_buffer=50000,
        plot_distributions=True
    )
    
    # Model configurations
    ae_model_config = ModelConfig(
        model_type="autoencoder",
        latent_dim=16,
        encoder_layers=[128, 64, 32],
        decoder_layers=[32, 64, 128],
        quant_bits=8,
        activation='relu',
        beta_schedule=None
    )
    
    vae_model_config = ModelConfig(
        model_type="variational_autoencoder",
        latent_dim=16,
        encoder_layers=[128, 64, 32],
        decoder_layers=[32, 64, 128],
        quant_bits=8,
        activation='relu',
        beta_schedule={
            'start': 0.0,
            'end': 0.01,
            'warmup_epochs': 5,
            'cycle_epochs': 5
        }
    )

    # Training configuration - common for both models
    training_config = TrainingConfig(
        batch_size=1024,  # Move from dataset_config to training_config
        learning_rate=0.001,
        epochs=2,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        plot_training=True
    )
    
    # Select model configuration based on type
    model_config = vae_model_config if MODEL_TYPE == "vae" else ae_model_config
    
    try:
        success = test_model_pipeline(
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
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