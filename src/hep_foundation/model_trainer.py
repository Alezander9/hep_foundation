from typing import Dict, List, Any, Optional
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from hep_foundation.base_model import BaseModel
from hep_foundation.plot_utils import (
    set_science_style, get_figure_size, get_color_cycle,
    FONT_SIZES, LINE_WIDTHS
)

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
        
    def compile_model(self):
        """Compile the model with optimizer and loss"""
        if self.model.model is None:
            raise ValueError("Model not built yet")
            
        self.model.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['mse'],
            run_eagerly=True 
        )
        
    def _update_metrics_history(self, epoch_metrics: Dict) -> None:
        """Update metrics history with new epoch results"""
        for metric_name, value in epoch_metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(float(value))
            
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary with all metrics"""
        if not self.metrics_history:
            return {
                'training_config': self.config,
                'metrics': {},
                'history': {}
            }
            
        # Get the latest metrics
        final_metrics = {
            metric: values[-1] 
            for metric, values in self.metrics_history.items()
        }
        
        # Add prefixes to distinguish metric types
        formatted_metrics = {
            'train_' + k: v for k, v in final_metrics.items() 
            if not k.startswith(('val_', 'test_'))
        }
        formatted_metrics.update({
            k: v for k, v in final_metrics.items() 
            if k.startswith(('val_', 'test_'))
        })
        
        return {
            'training_config': self.config,
            'metrics': formatted_metrics,
            'history': self.metrics_history
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
        print("\nStarting training with metrics tracking:")
        
        if plot_training and plots_dir is None:
            plots_dir = Path("experiments/plots")
        
        if plot_training:
            plots_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nWill save training plots to: {plots_dir}")
        
        if self.model.model is None:
            raise ValueError("Model not built yet")
        
        print("\nChecking datasets before training...")
        # Check if datasets have any data
        try:
            print("Checking training dataset...")
            for i, batch in enumerate(dataset):
                print(f"Training batch {i} shape: {batch.shape}")
                if i == 0:  # Just check first batch
                    break
        except Exception as e:
            print(f"Error checking training dataset: {e}")
        
        if validation_data is not None:
            try:
                print("\nChecking validation dataset...")
                for i, batch in enumerate(validation_data):
                    print(f"Validation batch {i} shape: {batch.shape}")
                    if i == 0:  # Just check first batch
                        break
            except Exception as e:
                print(f"Error checking validation dataset: {e}")
        
        # For autoencoder, use input data as both input and target
        train_dataset = dataset.map(lambda x: (x, x))
        if validation_data is not None:
            validation_data = validation_data.map(lambda x: (x, x))
        
        # Compile model
        self.compile_model()
        
        # Setup callbacks
        if callbacks is None:
            callbacks = []
        
        # Train the model
        print("\nStarting model.fit...")
        history = self.model.model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=True
        )
        
        # Update model's history if it's a VAE
        if hasattr(self.model, '_history'):
            self.model._history = history.history
        
        # Update metrics history
        for metric, values in history.history.items():
            self.metrics_history[metric] = [float(v) for v in values]
            
        print("\nTraining completed. Final metrics:")
        for metric, values in self.metrics_history.items():
            print(f"  {metric}: {values[-1]:.6f}")
            
        # After training completes, create plots if requested
        if plot_training:
            print("\nGenerating training plots...")
            self._create_training_plots(plots_dir)
        
        return self.get_training_summary()
        
    def _create_training_plots(self, plots_dir: Path):
        """Create standard training plots with LaTeX formatting"""
        print(f"\nCreating training plots in: {plots_dir.absolute()}")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from hep_foundation.plot_utils import (
                set_science_style, get_figure_size, get_color_cycle,
                FONT_SIZES, LINE_WIDTHS
            )
            
            set_science_style(use_tex=True)
            
            plt.figure(figsize=get_figure_size('single', ratio=1.2))
            history = self.metrics_history
            colors = get_color_cycle('high_contrast')
            color_idx = 0
            
            # Plot metrics with LaTeX labels
            for metric in history.keys():
                if 'loss' in metric.lower() and not metric.lower().startswith(('val_', 'test_')):
                    label = metric.replace('_', ' ').title()
                    label = rf'$\mathcal{{L}}_\mathrm{{{label}}}$'
                    plt.plot(
                        history[metric], 
                        label=label,
                        color=colors[color_idx % len(colors)],
                        linewidth=LINE_WIDTHS['thick']
                    )
                    color_idx += 1
            
            plt.xlabel(r'\textbf{Epoch}', fontsize=FONT_SIZES['large'])
            plt.ylabel(r'\textbf{Loss}', fontsize=FONT_SIZES['large'])
            plt.title(r'\textbf{Training History}', fontsize=FONT_SIZES['xlarge'])
            plt.legend(fontsize=FONT_SIZES['normal'], loc='upper right')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(plots_dir / 'training_history.pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Let the model create any model-specific plots
            if hasattr(self.model, 'create_plots'):
                self.model.create_plots(plots_dir)

            print(f"Plots saved to: {plots_dir}")
            
        except Exception as e:
            print(f"Error creating plots: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def evaluate(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Evaluate with enhanced metrics tracking"""
        print("\nEvaluating model...")
        if self.model.model is None:
            raise ValueError("Model not built yet")
        
        # For autoencoder, use input data as both input and target
        test_dataset = dataset.map(lambda x: (x, x))
        
        # Evaluate and get results
        results = self.model.model.evaluate(test_dataset, return_dict=True, verbose=1)
        
        # Add test_ prefix to metrics
        test_metrics = {
            'test_' + k: float(v) for k, v in results.items()
        }
        
        print("\nEvaluation metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        # Store test metrics in history
        self.metrics_history.update(test_metrics)
        
        return test_metrics