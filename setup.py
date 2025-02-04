from setuptools import setup, find_packages

setup(
    name="hep_foundation",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Core scientific and ML packages
        "numpy",
        "tensorflow",
        "qkeras",  # For quantized neural networks
        
        # Data handling and processing
        "uproot",  # For ROOT file handling
        
        # Visualization
        "matplotlib",
        
        # Progress bars and utilities
        "tqdm",
        
        # System utilities
        "psutil",  # For system resource monitoring
        
        # Data formats and storage
        "pyyaml",  # For YAML file handling
        
        # HTTP requests
        "requests"
    ],
    python_requires=">=3.7",  # Specify minimum Python version
    
    # Optional: Add more metadata
    author="Your Name",
    description="HEP Foundation Package for ML in High Energy Physics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
    ],
)