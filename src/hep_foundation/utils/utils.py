import gc
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import psutil
import tensorflow as tf
import yaml


# Memory usage
def get_system_usage():
    """Get current system memory and CPU usage with proper cleanup"""
    # Force garbage collection before checking memory
    gc.collect()

    # Get memory info
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)  # Shorter interval

    # Calculate memory in GB
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    used_gb = memory.used / (1024**3)

    return {
        "memory": {
            "total_gb": total_gb,
            "available_gb": available_gb,
            "used_gb": used_gb,
            "percent": memory.percent,
        },
        "cpu": {"percent": cpu_percent},
    }


def print_system_usage(prefix=""):
    """Print current system usage with optional prefix"""
    usage = get_system_usage()
    logging.info(f"{prefix}System Usage:")
    logging.info(
        f"Memory: {usage['memory']['used_gb']:.1f}GB / {usage['memory']['total_gb']:.1f}GB ({usage['memory']['percent']}%)"
    )
    logging.info(f"Available Memory: {usage['memory']['available_gb']:.1f}GB")
    logging.info(f"CPU Usage: {usage['cpu']['percent']}%")


# Type conversion


class TypeConverter:
    """Utility class for handling type conversions in the ML pipeline"""

    @staticmethod
    def to_numpy(data: Any, dtype=np.float32) -> np.ndarray:
        """Convert any numeric data to numpy array with specified dtype"""
        return np.array(data, dtype=dtype)

    @staticmethod
    def to_tensorflow(data: Any, dtype=tf.float32) -> tf.Tensor:
        """Convert any numeric data to tensorflow tensor with specified dtype"""
        return tf.convert_to_tensor(data, dtype=dtype)

    @staticmethod
    def to_python(data: Any) -> Union[int, float, list, dict]:
        """Convert numpy/tensorflow types to Python native types"""
        if isinstance(data, (np.integer, np.floating)):
            return data.item()
        elif isinstance(data, (np.ndarray, tf.Tensor)):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: TypeConverter.to_python(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [TypeConverter.to_python(x) for x in data]
        return data


class ConfigSerializer:
    """Handles serialization of configuration data"""

    @staticmethod
    def setup_yaml():
        """Configure YAML serializer for our needs"""

        def represent_none(dumper, _):
            return dumper.represent_scalar("tag:yaml.org,2002:null", "")

        # Add custom representers
        yaml.add_representer(type(None), represent_none)

        # Use safe dumper as base
        class SafeConfigDumper(yaml.SafeDumper):
            pass

        # Disable Python-specific tags
        SafeConfigDumper.ignore_aliases = lambda *args: True

        return SafeConfigDumper

    @staticmethod
    def to_yaml(data: dict, file_path: Path):
        """Save configuration to YAML file"""

        # Convert Python types to simple types
        def simplify(obj):
            if isinstance(obj, (tuple, list)):
                return [simplify(x) for x in obj]
            elif isinstance(obj, dict):
                return {str(k): simplify(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        # Prepare data
        simplified_data = simplify(data)

        # Save with custom dumper
        with open(file_path, "w") as f:
            yaml.dump(
                simplified_data,
                f,
                Dumper=ConfigSerializer.setup_yaml(),
                default_flow_style=False,
                sort_keys=False,
            )

    @staticmethod
    def from_yaml(file_path: Path) -> dict:
        """Load configuration from YAML file"""
        with open(file_path) as f:
            return yaml.safe_load(f)
