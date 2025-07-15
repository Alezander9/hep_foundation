import logging

# Add custom PROGRESS log level (between INFO=20 and WARNING=30)
PROGRESS_LEVEL = 25
logging.addLevelName(PROGRESS_LEVEL, "PROGRESS")


def progress(self, message, *args, **kwargs):
    """Log a progress message at PROGRESS level."""
    if self.isEnabledFor(PROGRESS_LEVEL):
        self._log(PROGRESS_LEVEL, message, args, **kwargs)


# Add the progress method to Logger class
logging.Logger.progress = progress


def log_progress(message):
    """Convenience function to log a progress message."""
    logging.getLogger().progress(message)


def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration for the entire package

    Note: TensorFlow C++ logging level is controlled by TF_CPP_MIN_LOG_LEVEL
    environment variable which is set in hep_foundation.__init__.py before
    any TensorFlow imports can occur.
    """

    # Suppress specific TensorFlow Python warnings
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

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

    Note: This will only affect the Python logger level. The C++ logging level
    (TF_CPP_MIN_LOG_LEVEL) is set at import time and cannot be changed after
    TensorFlow has been imported.
    """
    try:
        import tensorflow as tf

        tf.get_logger().setLevel(logging.INFO)
        print(
            "TensorFlow Python debug logging enabled. TensorFlow Python INFO and WARNING messages will now be displayed."
        )
        print(
            "Note: C++ logging level is controlled by TF_CPP_MIN_LOG_LEVEL environment variable set at import time."
        )
    except ImportError:
        print("TensorFlow not available, cannot enable TF debug logging.")
