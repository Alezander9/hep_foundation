# HEP Foundation

ML tools for High Energy Physics analysis, focusing on ATLAS PHYSLITE data processing and autoencoder training.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Alezander9/hep_foundation
cd hep_foundation

# Create and activate virtual environment using uv
# Requires uv to be installed (e.g., pip install uv or via package manager)
uv venv --python 3.9  # Specify your desired Python version
source .venv/bin/activate # On Windows: .venv\\Scripts\\activate

# Install package in editable mode
uv pip install -e .
```

## Usage

### Basic Pipeline
```python
from hep_foundation.model_factory import ModelFactory
from hep_foundation.dataset_manager import DatasetManager

# Setup data pipeline
data_manager = DatasetManager()
train_dataset, val_dataset, test_dataset = data_manager.load_atlas_datasets(
config={
'run_numbers': ["00296939", "00296942", "00297447"],
'track_selections': {
'eta': (-2.5, 2.5),
'chi2_per_ndof': (0.0, 10.0),
},
'max_tracks_per_event': 20,
'min_tracks_per_event': 3,
'catalog_limit': 5
}
)
# Create and train model
model = ModelFactory.create_model(
model_type="autoencoder",
config={
'input_shape': (20, 6),
'latent_dim': 32,
'encoder_layers': [128, 64, 32],
'decoder_layers': [32, 64, 128],
'activation': 'relu'
}
)
```
For full training pipeline example, see scripts/model_pipeline.py
```bash
Run full pipeline test
python scripts/model_pipeline.py
```

## Project Structure
- `src/hep_foundation/`: Core package code
- `scripts/`: Example scripts and tests
- `experiments/`: Output directory for model registry and results
- `processed_datasets/`: Storage for processed datasets

## Dependencies
Core requirements are automatically handled by `uv pip install -e .` using `pyproject.toml`.

For development dependencies (like linters, testing tools), ensure they are listed under `[project.optional-dependencies]` in `pyproject.toml` (e.g., in a group named `dev`) and install them using:

```bash
uv pip install -e ".[dev]"
```

To add a new runtime dependency:
```bash
uv add <package_name>  # e.g., uv add numpy==1.24.3
```

To add a new development dependency:
```bash
uv add --dev <package_name> # e.g., uv add --dev pytest
```

## Code Quality

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

- **Check Code**: To check for linting errors and style issues, run:
  ```bash
  uv run ruff check .
  ```
- **Format Code**: To automatically format the code according to the project's style guide, run:
  ```bash
  uv run ruff format .
  ```

## Testing

Run the foundation model pipeline tests with:
```bash
pytest tests/test_pipeline.py -v
```

This will execute the test suite with verbose output, showing real-time logging of the training and evaluation processes. Test logs are stored in `./test_results/test_foundation_experiments_<timestamp>/test_logs/`.

It's recommended to run both code quality checks and tests after making significant changes and before committing code.

## Data Access
The package automatically handles ATLAS PHYSLITE data download from CERN OpenData.

## Questions?
Contact: alexyue@stanford.edu