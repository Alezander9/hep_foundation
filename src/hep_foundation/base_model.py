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