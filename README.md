# HEP Foundation

ML tools for High Energy Physics analysis, focusing on ATLAS PHYSLITE data processing and autoencoder training.

## Quick Start
```bash
Clone repository
git clone https://github.com/yourusername/hep_foundation.git
cd hep_foundation
Create and activate virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
Install package
pip install -e .
```

## Usage

### Basic Pipeline
```python
from hep_foundation.model_factory import ModelFactory
from hep_foundation.processed_dataset_manager import ProcessedDatasetManager

# Setup data pipeline
data_manager = ProcessedDatasetManager()
train_dataset, val_dataset, test_dataset = data_manager.load_datasets(
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
Create and train model
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
For full training pipeline example, see scripts/test_model_pipeline.py
```bash
Run full pipeline test
python scripts/test_model_pipeline.py
```

## Project Structure
- `src/hep_foundation/`: Core package code
- `scripts/`: Example scripts and tests
- `experiments/`: Output directory for model registry and results

## Dependencies
Core requirements are handled by setup.py. For development:

```bash
pip install -e ".[dev]"
```

## Data Access
The package automatically handles ATLAS PHYSLITE data download from CERN OpenData.

## Questions?
Contact: alexyue@stanford.edu