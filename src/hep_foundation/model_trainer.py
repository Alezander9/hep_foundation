from typing import Dict, List, Any, Optional
import tensorflow as tf

from .base_model import BaseModel

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
        callbacks: List[tf.keras.callbacks.Callback] = None
    ) -> Dict[str, Any]:
        """Train with enhanced metrics tracking"""
        print("\nStarting training with metrics tracking:")
        
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
        
        # Update metrics history
        for metric, values in history.history.items():
            self.metrics_history[metric] = [float(v) for v in values]
            
        print("\nTraining completed. Final metrics:")
        for metric, values in self.metrics_history.items():
            print(f"  {metric}: {values[-1]:.6f}")
            
        return self.get_training_summary()
        
    def evaluate(
        self,
        dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        """Evaluate with enhanced metrics tracking"""
        print("\nEvaluating model...")
        if self.model.model is None:
            raise ValueError("Model not built yet")
            
        results = self.model.model.evaluate(dataset, return_dict=True, verbose=1)
        
        # Add test_ prefix to metrics
        test_metrics = {
            f'test_{k}': float(v) for k, v in results.items()
        }
        
        print("\nEvaluation metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.6f}")
            
        return test_metrics