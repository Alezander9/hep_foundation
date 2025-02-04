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
        
    def train(
        self,
        dataset: tf.data.Dataset,
        validation_data: Optional[tf.data.Dataset] = None,
        callbacks: List[tf.keras.callbacks.Callback] = None
    ) -> Dict[str, Any]:
        """Train the model"""
        if self.model.model is None:
            raise ValueError("Model not built yet")
        
        print("\nChecking datasets before training:")
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
        
        # Compile model
        self.compile_model()
        
        # Setup callbacks
        if callbacks is None:
            callbacks = []
        
        # Train the model
        print("\nStarting model.fit...")
        self.history = self.model.model.fit(
            dataset,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=True
        )
        
        return self.get_training_summary()
        
    def evaluate(
        self,
        dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        """
        Evaluate the model
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model.model is None:
            raise ValueError("Model not built yet")
            
        results = self.model.model.evaluate(dataset, return_dict=True)
        return results
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results"""
        if self.history is None:
            raise ValueError("Model not trained yet")
            
        return {
            "training_config": self.config,
            "final_loss": float(self.history.history['loss'][-1]),
            "final_val_loss": float(self.history.history['val_loss'][-1]),
            "history": {
                metric: [float(val) for val in values]
                for metric, values in self.history.history.items()
            }
        }