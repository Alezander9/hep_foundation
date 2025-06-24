"""
Simplified regression training script for missing energy prediction.

This script trains a DNN model to predict missing energy (MET) from track particles
without any foundation model complexity. It's designed for fast iteration and testing
of different metrics and training approaches.
"""

import json
from pathlib import Path
from typing import Dict, Any

import tensorflow as tf
import numpy as np

from hep_foundation.config.logging_config import get_logger
from hep_foundation.data.atlas_data import get_run_numbers
from hep_foundation.data.dataset_manager import DatasetManager
from hep_foundation.config.dataset_config import DatasetConfig
from hep_foundation.config.task_config import TaskConfig
from hep_foundation.models.model_factory import ModelFactory
from hep_foundation.training.model_trainer import ModelTrainer
from hep_foundation.config.training_config import TrainingConfig
from hep_foundation.models.base_model import CustomKerasModelWrapper


def create_task_config() -> TaskConfig:
    """Create task configuration for missing energy prediction."""
    return TaskConfig.create_from_branch_names(
        event_filter_dict={},
        input_features=[],
        input_array_aggregators=[
            {
                "input_branches": [
                    "InDetTrackParticlesAuxDyn.d0",
                    "InDetTrackParticlesAuxDyn.z0", 
                    "InDetTrackParticlesAuxDyn.phi",
                    "derived.InDetTrackParticlesAuxDyn.eta",
                    "derived.InDetTrackParticlesAuxDyn.pt",
                    "derived.InDetTrackParticlesAuxDyn.reducedChiSquared",
                ],
                "filter_branches": [
                    {"branch": "InDetTrackParticlesAuxDyn.d0", "min": -5.0, "max": 5.0},
                    {"branch": "InDetTrackParticlesAuxDyn.z0", "min": -100.0, "max": 100.0},
                    {"branch": "InDetTrackParticlesAuxDyn.chiSquared", "max": 50.0},
                    {"branch": "InDetTrackParticlesAuxDyn.numberDoF", "min": 1.0},
                ],
                "sort_by_branch": {"branch": "InDetTrackParticlesAuxDyn.qOverP"},
                "min_length": 10,
                "max_length": 30,
            },
        ],
        label_features=[[]],
        label_array_aggregators=[
            [
                {
                    "input_branches": [
                        "MET_Core_AnalysisMETAuxDyn.mpx",
                        "MET_Core_AnalysisMETAuxDyn.mpy", 
                        "MET_Core_AnalysisMETAuxDyn.sumet",
                    ],
                    "filter_branches": [],
                    "sort_by_branch": None,
                    "min_length": 1,
                    "max_length": 1,
                }
            ]
        ],
    )


def create_dataset_config(task_config: TaskConfig) -> DatasetConfig:
    """Create dataset configuration."""
    run_numbers = get_run_numbers()
    
    return DatasetConfig(
        run_numbers=run_numbers[-2:],  # Use fewer runs for faster iteration
        signal_keys=[],  # No signal keys needed for regression
        catalog_limit=10,  # Smaller limit for faster loading
        validation_fraction=0.15,
        test_fraction=0.15,
        shuffle_buffer=50000,
        plot_distributions=True,
        include_labels=True,
        task_config=task_config,
    )


def create_model_config(input_shape: tuple, output_shape: tuple) -> Dict[str, Any]:
    """Create DNN model configuration."""
    return {
        "model_type": "dnn_predictor",
        "architecture": {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "hidden_layers": [64, 32, 16],  # Slightly larger for direct training
            "label_index": 0,
            "activation": "relu",
            "output_activation": "linear",
            "name": "met_predictor",
        },
        "hyperparameters": {
            "quant_bits": 8,
            "dropout_rate": 0.1,
            "l2_regularization": 1e-4,
        },
    }


def create_training_config() -> TrainingConfig:
    """Create training configuration."""
    return TrainingConfig(
        batch_size=1024,
        learning_rate=0.001,
        epochs=20,  # Fewer epochs for fast iteration
        early_stopping_patience=5,
        early_stopping_min_delta=1e-4,
        plot_training=True,
    )


def train_regression_model():
    """Main function to train regression model."""
    logger = get_logger(__name__)
    
    logger.info("=" * 80)
    logger.info("TRAINING REGRESSION MODEL FOR MISSING ENERGY PREDICTION")
    logger.info("=" * 80)
    
    try:
        # 1. Create configurations
        logger.info("Creating configurations...")
        task_config = create_task_config()
        dataset_config = create_dataset_config(task_config)
        training_config = create_training_config()
        
        # Validate configurations
        dataset_config.validate()
        training_config.validate()
        logger.info("Configurations validated successfully")
        
        # 2. Initialize data manager and load datasets
        logger.info("Loading datasets...")
        data_manager = DatasetManager(base_dir="_processed_datasets")
        
        train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
            dataset_config=dataset_config,
            validation_fraction=dataset_config.validation_fraction,
            test_fraction=dataset_config.test_fraction,
            batch_size=training_config.batch_size,
            shuffle_buffer=dataset_config.shuffle_buffer,
            include_labels=True,
            delete_catalogs=True,
        )
        
        logger.info("Datasets loaded successfully")
        
        # 3. Inspect dataset structure and get shapes
        logger.info("Inspecting dataset structure...")
        for batch in train_dataset.take(1):
            features, labels = batch
            input_shape = features.shape[1:]
            output_shape = labels.shape[1:]
            logger.info(f"Input shape: {input_shape}")
            logger.info(f"Output shape: {output_shape}")
            logger.info(f"Batch size: {features.shape[0]}")
            break
        
        # 4. Create model
        logger.info("Creating model...")
        model_config = create_model_config(input_shape, output_shape)
        
        model_wrapper = ModelFactory.create_model(
            model_type="dnn_predictor", 
            config=model_config
        )
        model_wrapper.build()
        
        logger.info("Model created successfully")
        logger.info(model_wrapper.model.summary())
        
        # 5. Train model
        logger.info("Starting training...")
        trainer = ModelTrainer(
            model=model_wrapper, 
            training_config=training_config.__dict__
        )
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=training_config.early_stopping_patience,
                min_delta=training_config.early_stopping_min_delta,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train the model
        training_results = trainer.train(
            dataset=train_dataset,
            validation_data=val_dataset,
            callbacks=callbacks,
            plot_training=True,
            plots_dir="regression_training_plots"
        )
        
        # 6. Evaluate model
        logger.info("Evaluating model on test set...")
        test_metrics = trainer.evaluate(test_dataset)
        
        # 7. Display results
        logger.info("=" * 80)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"Training duration: {training_results['training_duration']:.2f}s")
        logger.info(f"Epochs completed: {training_results['epochs_completed']}")
        
        logger.info("\nFinal Training Metrics:")
        for key, value in training_results["final_metrics"].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("\nTest Metrics:")
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # 8. Save results
        results_dir = Path("regression_results")
        results_dir.mkdir(exist_ok=True)
        
        results = {
            "task_config": task_config.to_dict(),
            "dataset_config": dataset_config.to_dict(),
            "model_config": model_config,
            "training_config": training_config.__dict__,
            "training_results": training_results,
            "test_metrics": test_metrics,
        }
        
        results_file = results_dir / "regression_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        # 9. Save model
        model_dir = results_dir / "trained_model"
        model_dir.mkdir(exist_ok=True)
        model_wrapper.model.save(model_dir / "regression_model")
        
        logger.info(f"Model saved to: {model_dir / 'regression_model'}")
        
        logger.info("=" * 80)
        logger.info("REGRESSION TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Regression training failed: {type(e).__name__}: {str(e)}")
        logger.exception("Detailed traceback:")
        return False


if __name__ == "__main__":
    success = train_regression_model()
    exit(0 if success else 1)
