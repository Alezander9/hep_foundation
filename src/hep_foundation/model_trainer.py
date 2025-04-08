from typing import Dict, List, Any, Optional
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from hep_foundation.base_model import BaseModel
from hep_foundation.plot_utils import (
    set_science_style, get_figure_size, get_color_cycle,
    FONT_SIZES, LINE_WIDTHS
)
from hep_foundation.logging_config import setup_logging
import numpy as np
from hep_foundation.dnn_predictor import DNNPredictor
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


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for clean logging output"""
    def __init__(self, epochs: int):
        super().__init__()
        self.epochs = epochs
        self.epoch_start_time = None
        
        # Setup logging 
        setup_logging()
        
    def on_train_begin(self, logs=None):
        logging.info(f"Starting training for {self.epochs} epochs")
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logging.info(f"Starting epoch {epoch+1}/{self.epochs}")
        
    def on_epoch_end(self, epoch, logs=None):
        time_taken = time.time() - self.epoch_start_time
        metrics = ' - '.join(f'{k}: {v:.6f}' for k, v in logs.items())
        logging.info(f"Epoch {epoch+1}/{self.epochs} completed in {time_taken:.1f}s - {metrics}")

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(
        self,
        model: BaseModel,
        training_config: dict,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: Optional[tf.keras.losses.Loss] = None
    ):
        self.model = model
        self.config = training_config
        
        # Set up training parameters
        self.batch_size = training_config.get("batch_size", 32)
        self.epochs = training_config.get("epochs", 10)
        self.validation_split = training_config.get("validation_split", 0.2)
        
        # Set up optimizer and loss
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            learning_rate=training_config.get("learning_rate", 0.001)
        )
        self.loss = loss or tf.keras.losses.MeanSquaredError()
        
        # Training history
        self.history = None
        self.metrics_history = {}
        
        self.training_start_time = None
        self.training_end_time = None
        
    def compile_model(self):
        """Compile the model with optimizer and loss"""
        if self.model.model is None:
            raise ValueError("Model not built yet")
            
        # Different compilation settings based on model type
        if isinstance(self.model, DNNPredictor):
            self.model.model.compile(
                optimizer=self.optimizer,
                loss='mse',
                metrics=['mse', 'mae'],  # Add mean absolute error for regression
                run_eagerly=True 
            )
        else:
            # Original compilation for autoencoders
            self.model.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=['mse'],
                run_eagerly=True 
            )

    def prepare_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Prepare dataset based on model type
        
        Args:
            dataset: Input dataset that may contain features and labels
            
        Returns:
            Prepared dataset with correct input/target structure
        """
        if isinstance(self.model, DNNPredictor):
            # For predictor models, use features as input and select correct label as target
            def prepare_predictor_data(features, labels):
                # Handle both tuple and non-tuple inputs
                if isinstance(features, (tf.Tensor, np.ndarray)):
                    input_features = features
                else:
                    input_features = features[0]
                
                # Select the correct label set based on label_index
                if isinstance(labels, (list, tuple)):
                    target_labels = labels[self.model.label_index]
                else:
                    target_labels = labels
                    
                return input_features, target_labels
                
            return dataset.map(prepare_predictor_data)
        else:
            # For autoencoders, use input as both input and target
            return dataset.map(lambda x, y: (x, x) if isinstance(x, (tf.Tensor, np.ndarray)) else (x[0], x[0]))

    def _update_metrics_history(self, epoch_metrics: Dict) -> None:
        """Update metrics history with new epoch results"""
        for metric_name, value in epoch_metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(float(value))
            
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary with all metrics and history"""
        if not self.metrics_history:
            return {
                'training_config': self.config,
                'metrics': {},
                'history': {},
                'training_duration': 0.0,
                'epochs_completed': 0
            }
        
        # Calculate training duration
        training_duration = 0.0
        if self.training_start_time and self.training_end_time:
            training_duration = (self.training_end_time - self.training_start_time).total_seconds()
        
        # Get the latest metrics for each metric type
        final_metrics = {
            metric: values[-1] 
            for metric, values in self.metrics_history.items()
        }
        
        # Convert history to epoch-based format
        epoch_history = {}
        for epoch in range(len(next(iter(self.metrics_history.values())))):
            epoch_history[str(epoch)] = {
                metric: values[epoch]
                for metric, values in self.metrics_history.items()
            }
        
        return {
            'training_config': self.config,
            'epochs_completed': len(epoch_history),
            'training_duration': training_duration,
            'final_metrics': final_metrics,
            'history': epoch_history
        }
        
    def train(
        self,
        dataset: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset] = None,
        callbacks: List[tf.keras.callbacks.Callback] = None,
        plot_training: bool = False,
        plots_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Train with enhanced metrics tracking and optional plotting"""
        logging.info("\nStarting training with metrics tracking:")
        
        # Record start time
        self.training_start_time = datetime.now()
        
        if plot_training and plots_dir is None:
            plots_dir = Path("experiments/plots")
        
        if plot_training:
            plots_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"\nWill save training plots to: {plots_dir}")
        
        if self.model.model is None:
            raise ValueError("Model not built yet")
        
        logging.info("\nPreparing datasets for training...")
        
        try:
            # Prepare training and validation datasets
            train_dataset = self.prepare_dataset(dataset)
            
            if validation_data is not None:
                validation_data = self.prepare_dataset(validation_data)
            
            # Log dataset shapes for debugging
            for batch in train_dataset.take(1):
                features, targets = batch
                logging.info(f"\nTraining dataset shapes:")
                logging.info(f"  Features: {features.shape}")
                logging.info(f"  Targets: {targets.shape}")
                break
            
        except Exception as e:
            logging.error(f"Error preparing datasets: {str(e)}")
            raise
        
        # Compile model
        self.compile_model()
        
        # Setup callbacks
        if callbacks is None:
            callbacks = []
        
        # Add our custom progress callback
        callbacks.append(TrainingProgressCallback(epochs=self.epochs))
        
        # Train the model
        logging.info("\nStarting model.fit...")
        history = self.model.model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=True,
            verbose=0  # Turn off default progress bar
        )
        
        # Record end time
        self.training_end_time = datetime.now()
        
        # Update model's history if it's a VAE
        if hasattr(self.model, '_history'):
            self.model._history = history.history
        
        # Update metrics history
        for metric, values in history.history.items():
            self.metrics_history[metric] = [float(v) for v in values]
            
        logging.info("\nTraining completed. Final metrics:")
        for metric, values in self.metrics_history.items():
            logging.info(f"  {metric}: {values[-1]:.6f}")
            
        # After training completes, create plots if requested
        if plot_training:
            logging.info("\nGenerating training plots...")
            self._create_training_plots(plots_dir)
        
        return self.get_training_summary()
        
    def _create_training_plots(self, plots_dir: Path):
        """Create standard training plots with simple formatting"""
        logging.info(f"\nCreating training plots in: {plots_dir.absolute()}")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from hep_foundation.plot_utils import (
                set_science_style, get_figure_size, get_color_cycle,
                FONT_SIZES, LINE_WIDTHS
            )
            
            set_science_style(use_tex=False)
            
            plt.figure(figsize=get_figure_size('single', ratio=1.2))
            history = self.metrics_history
            colors = get_color_cycle('high_contrast')
            color_idx = 0
            
            # Plot metrics with simple labels
            for metric in history.keys():
                if 'loss' in metric.lower() and not metric.lower().startswith(('val_', 'test_')):
                    label = metric.replace('_', ' ').title()
                    plt.plot(
                        history[metric], 
                        label=label,
                        color=colors[color_idx % len(colors)],
                        linewidth=LINE_WIDTHS['thick']
                    )
                    color_idx += 1
            
            plt.yscale('log')  # Set y-axis to logarithmic scale
            plt.xlabel('Epoch', fontsize=FONT_SIZES['large'])
            plt.ylabel('Loss (log)', fontsize=FONT_SIZES['large'])
            plt.title('Training History', fontsize=FONT_SIZES['xlarge'])
            plt.legend(fontsize=FONT_SIZES['normal'], loc='upper right')
            plt.grid(True, alpha=0.3, which='both')  # Grid lines for both major and minor ticks
            
            plt.savefig(plots_dir / 'training_history.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Let the model create any model-specific plots
            if hasattr(self.model, 'create_plots'):
                self.model.create_plots(plots_dir)

            logging.info(f"Plots saved to: {plots_dir}")
            
        except Exception as e:
            logging.error(f"Error creating plots: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def evaluate(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Evaluate with enhanced metrics tracking"""
        logging.info("\nEvaluating model...")
        if self.model.model is None:
            raise ValueError("Model not built yet")
        
        try:
            # Prepare test dataset
            test_dataset = self.prepare_dataset(dataset)
            
            # Evaluate and get results
            results = self.model.model.evaluate(
                test_dataset,
                return_dict=True,
                verbose=0
            )
            
            # Add test_ prefix to metrics
            test_metrics = {
                'test_' + k: float(v) for k, v in results.items()
            }
            
            logging.info("\nEvaluation metrics:")
            for metric, value in test_metrics.items():
                logging.info(f"  {metric}: {value:.6f}")
            
            # Store test metrics in history
            self.metrics_history.update(test_metrics)
            
            return test_metrics
            
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise