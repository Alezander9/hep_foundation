import logging
import os


def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration for the entire package"""
    
    # Suppress TensorFlow warnings to reduce log clutter
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
    
    # Suppress specific TensorFlow Python warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    
    # Set TensorFlow logging level
    try:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)  # Only show ERROR messages
    except ImportError:
        # TensorFlow not available, skip TF logging configuration
        pass
    
    # Create formatter
    if level == logging.DEBUG:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    return root_logger


def get_logger(name):
    """Get a logger for a module"""
    return logging.getLogger(name)


def enable_tensorflow_debug_logging():
    """
    Temporarily enable TensorFlow verbose logging for debugging.
    Call this function if you need to see TensorFlow's internal warnings and info messages.
    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all messages
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.INFO)
        print("TensorFlow debug logging enabled. TensorFlow INFO and WARNING messages will now be displayed.")
    except ImportError:
        print("TensorFlow not available, cannot enable TF debug logging.")
