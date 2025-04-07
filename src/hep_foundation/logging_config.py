import logging

def setup_logging(level=logging.INFO):
    """Setup logging configuration for the entire package"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # This ensures the configuration is applied even if logging was already configured
    )
