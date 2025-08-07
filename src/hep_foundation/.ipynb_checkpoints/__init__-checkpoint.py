# Set TensorFlow environment variables BEFORE any imports that might import TensorFlow
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "2"  # Suppress tensorflow INFO and WARNING messages
)

from hep_foundation.config.logging_config import setup_logging

# Initialize logging once for the entire package
setup_logging()
