# .gitignore Guide

## Overview

The `.gitignore` file is configured to exclude datasets, internal files, and development artifacts while keeping all enterprise-required code, configuration, and documentation.

## What's Excluded

### Data and Datasets
- `data/` - All data directories
- `results/` - Training and analysis results
- `models/` - Trained model files
- `logs/` - Log files
- All data file formats: `*.parquet`, `*.pkl`, `*.h5`, `*.hdf5`, `*.csv` (in data directories)

### Internal/Development Directories
- `dep/` - Deprecated/development code
- `docs/internal/` - Internal documentation
- `docs/audit/` - Audit trails
- `NOTES/` - Personal notes
- `UPDATE/` - Update logs

### Training Artifacts
- `TRAINING/modular_output/` - Training outputs
- `TRAINING/EXPERIMENTS/output/` - Experiment outputs
- `TRAINING/EXPERIMENTS/metadata/` - Experiment metadata
- `TRAINING/EXPERIMENTS/logs/` - Experiment logs
- `catboost_info/` - CatBoost training artifacts
- Model checkpoints: `*.ckpt`, `*.pt`, `*.pth`, `*.h5`, `*.pb`, `*.onnx`

### Build Artifacts
- `__pycache__/` - Python cache
- `build/`, `dist/` - Build directories
- `*.pyc`, `*.pyo` - Compiled Python files
- C++ build artifacts: `*.o`, `*.a`, `*.so`, `*.dll`

### IDE and OS Files
- `.vscode/`, `.idea/` - IDE configuration
- `.DS_Store`, `Thumbs.db` - OS files
- `*.swp`, `*.swo` - Editor temporary files

### Environment and Secrets
- `.env` files (except examples)
- `secrets/` directory
- `*.key`, `*.pem` - Key files
- `credentials.json` - Credential files

## What's Included (Enterprise Required)

### Code
- All Python files: `**/*.py`
- All shell scripts: `**/*.sh`
- All C++ files: `**/*.cpp`, `**/*.h`

### Configuration
- `CONFIG/` - All configuration files
- `*.yaml`, `*.yml`, `*.json` - Configuration files
- Small example CSVs in `CONFIG/`

### Documentation
- `docs/` - All documentation (except `docs/internal/`)
- `INFORMATION/` - Information documentation
- `legal/` - Legal documentation

### Trading Systems
- `ALPACA_trading/` - Alpaca trading code
- `IBKR_trading/` - IBKR trading code (code files)
- `DATA_PROCESSING/` - Data processing code
- `TRAINING/` - Training code (code files, not outputs)

## Usage

### Adding New Exclusions

If you need to exclude additional files or directories, add them to `.gitignore`:

```bash
# Example: Exclude a new data directory
new_data_directory/
```

### Checking What's Ignored

```bash
# Check if a file/directory is ignored
git check-ignore -v path/to/file

# List all ignored files
git ls-files --others --ignored --exclude-standard
```

### Force Adding Ignored Files

If you need to add a file that's normally ignored (e.g., a small example dataset):

```bash
git add -f path/to/file
```

## Best Practices

1. **Never commit datasets**: All data files should be excluded
2. **Never commit models**: Trained models are large and user-specific
3. **Never commit secrets**: All credential and key files are excluded
4. **Keep configs small**: Only commit small example/config files, not large datasets
5. **Document exceptions**: If you need to commit something normally ignored, document why

## Common Scenarios

### Scenario 1: Adding a New Dataset Directory

The dataset will be automatically excluded. No action needed.

### Scenario 2: Adding a New Configuration File

Configuration files in `CONFIG/` are included. Ensure it's a small file (not a dataset).

### Scenario 3: Adding Training Outputs

Training outputs are automatically excluded. No action needed.

### Scenario 4: Adding Internal Documentation

Move to `docs/internal/` - it's automatically excluded.

## Verification

To verify your `.gitignore` is working correctly:

```bash
# Check what would be committed
git status

# Check what's ignored
git status --ignored

# Test specific paths
git check-ignore -v data/
git check-ignore -v CONFIG/
```

