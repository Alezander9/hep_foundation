# Config Stack System - Quick Start Guide

## Overview
Automated system to run multiple foundation model pipeline experiments sequentially using YAML configuration files.

## Setup

1. **Activate environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Create experiment configurations:**
   ```bash
   # Copy template and customize
   cp _experiment_config_stack/example_config.yaml _experiment_config_stack/my_experiment.yaml
   ```

3. **Edit YAML files** to specify:
   - ATLAS run numbers (`dataset.run_numbers`)
   - Signal types (`dataset.signal_keys`)
   - Training epochs (`training.vae.epochs`, `training.dnn.epochs`)
   - Model architectures (`models.vae.architecture`)

## Running Pipelines

### Test First (Recommended)
```bash
python scripts/run_pipelines.py --dry-run
```

### Run All Configs
```bash
python scripts/run_pipelines.py
```

### Run Limited Number
```bash
python scripts/run_pipelines.py --max-configs 3
```

## Monitoring

### Check Progress
```bash
# View latest log
tail -f logs/pipeline_*_$(date +%Y%m%d)*.log

# List all running processes
ps aux | grep run_pipelines
```

### Results Location
- **Experiments**: `_foundation_experiments/{experiment_id}/`
- **Logs**: `logs/pipeline_{config_name}_{timestamp}.log`

## Key Features

- **Sequential Processing**: Runs one experiment at a time
- **Auto-Cleanup**: Deletes processed configs from stack (saved in experiment folders)
- **Error Handling**: Continues processing remaining configs if one fails
- **Reproducibility**: Original configs saved with results

## NERSC Considerations

### Resource Management
```bash
# For long runs, use compute nodes
salloc -N 1 -t 8:00:00 -q interactive
# or submit as batch job

# Monitor disk usage
df -h $SCRATCH
```

### Large Scale Runs
- Each experiment can take 2-8 hours depending on data size
- Monitor disk space (datasets + model files can be large)
- Consider using `--max-configs` for testing resource limits

### Troubleshooting
- **Failed configs**: Check specific log file for errors
- **Stuck processing**: Failed configs remain in stack for retry
- **Out of memory**: Reduce batch sizes in YAML configs
- **Disk full**: Clean up old experiments or use `$SCRATCH`

## File Structure
```
_experiment_config_stack/           # Your experiment configs (auto-created, gitignored)
├── experiment1.yaml
├── experiment2.yaml
└── README.md

logs/                    # Individual experiment logs
├── pipeline_experiment1_20250623_160011.log
└── pipeline_experiment2_20250623_170045.log

_foundation_experiments/  # Results (auto-created)
├── {experiment_id}/
│   ├── _experiment_config.yaml    # Original config
│   ├── models/                    # Trained models
│   └── testing/                   # Evaluation results
```

**Note**: Each config is automatically deleted after successful processing. Original configs are preserved in experiment folders for reproducibility.
