from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all models"""
    def __init__(self):
        self.model = None
        
    @abstractmethod
    def build(self, input_shape: tuple) -> None:
        """Build the model architecture"""
        pass
        
    @abstractmethod
    def get_config(self) -> dict:
        """Get model configuration"""
        pass
        
    def compile(self, *args, **kwargs):
        """Compile the underlying Keras model"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.compile(*args, **kwargs)
        
    def fit(self, *args, **kwargs):
        """Train the underlying Keras model"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.fit(*args, **kwargs)
        
    def evaluate(self, *args, **kwargs):
        """Evaluate the underlying Keras model"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.evaluate(*args, **kwargs)
        
    def save(self, path: str) -> None:
        """Save model weights and config"""
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.save(path)
        
    def load(self, path: str) -> None:
        """Load model weights"""
        if self.model is None:
            raise ValueError("Model not built yet")
        self.model.load_weights(path)