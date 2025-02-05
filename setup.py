from setuptools import setup, find_packages

setup(
    name="hep_foundation",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Core scientific and ML packages
        "numpy>=1.24.3",
        "tensorflow>=2.13.1",
        "qkeras>=0.9.0",  # For quantized neural networks
        "pandas>=2.2.3",  # Used in registry inspector and data management
        
        # Data handling and processing
        "uproot>=5.5.1",  # For ROOT file handling
        "h5py>=3.12.1",   # For HDF5 dataset storage
        "awkward>=2.7.2", # Required for some uproot operations
        
        # Visualization
        "matplotlib>=3.9.4",
        "seaborn>=0.13.0",  # Added seaborn
        
        # Progress bars and utilities
        "tqdm>=4.67.1",
        
        # System utilities
        "psutil>=6.1.0",  # For system resource monitoring
        
        # Data formats and storage
        "pyyaml>=6.0.2",  # For YAML file handling
        
        # HTTP requests
        "requests>=2.32.3",  # For downloading ATLAS data
    ],
    extras_require={
        'dev': [
            'pytest',          # For testing
            'ipykernel',      # For notebook development
            'jupyter',        # For notebook development
        ],
        'docs': [
            'sphinx',         # For documentation
            'sphinx-rtd-theme'
        ]
    },
    python_requires=">=3.7",
    
    # Metadata
    author="Your Name",
    author_email="your.email@example.com",
    description="HEP Foundation Package for ML in High Energy Physics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hep_foundation",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
)