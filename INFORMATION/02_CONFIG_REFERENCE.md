# Configuration Reference

Configuration files and parameters.

## Directory Structure

```
CONFIG/
├── config_loader.py              # Python utility for loading configs
├── model_config/                 # Model-specific configs (17 files)
│   ├── lightgbm.yaml
│   ├── xgboost.yaml
│   ├── ensemble.yaml
│   ├── multi_task.yaml
│   ├── mlp.yaml
│   ├── transformer.yaml
│   ├── lstm.yaml
│   ├── cnn1d.yaml
│   ├── vae.yaml
│   ├── gan.yaml
│   ├── gmm_regime.yaml
│   ├── ngboost.yaml
│   ├── quantile_lightgbm.yaml
│   ├── change_point.yaml
│   ├── ftrl_proximal.yaml
│   ├── reward_based.yaml
│   └── meta_learning.yaml
└── training_config/              # Training workflow configs
    ├── first_batch_specs.yaml
    ├── sequential_config.yaml
    └── family_config.yaml
```

## Config Loader API

### Loading Configs
```python
from CONFIG.config_loader import load_model_config, load_training_config

# Load model config (default variant)
config = load_model_config("lightgbm")

# Load specific variant
config = load_model_config("xgboost", variant="conservative")

# Load with overrides
config = load_model_config("mlp", overrides={"epochs": 100})

# Load training config
training_cfg = load_training_config("first_batch_specs")
```

### Available Functions
- `load_model_config(model_name, variant=None, overrides=None)` - Load model config
- `load_training_config(config_name, overrides=None)` - Load training workflow config
- `get_available_model_configs()` - List all available model configs
- `get_available_training_configs()` - List all available training configs

## Environment Variables

```bash
# Set default variant for all models
export MODEL_VARIANT=conservative

# Override specific config paths
export MODEL_CONFIG_DIR=/custom/path/to/configs
```

## Configuration File Format

Each YAML file has 3 sections:

```yaml
# Default configuration (balanced)
default:
  param1: value1
  param2: value2

# Variants
variants:
  conservative:
    param1: conservative_value

  balanced:
    param1: balanced_value

  aggressive:
    param1: aggressive_value
```

## Key Parameters by Model Family

### Tree Models (LightGBM, XGBoost)
- `n_estimators` - Number of trees (default: 1000-1500)
- `learning_rate` - Step size (default: 0.03-0.05)
- `max_depth` - Tree depth (default: 5-8)
- `min_child_weight` / `min_data_in_leaf` - Min samples per leaf (default: 100-400)
- `subsample` / `bagging_fraction` - Row sampling (default: 0.7-0.8)
- `colsample_bytree` / `feature_fraction` - Column sampling (default: 0.7-0.8)
- `reg_alpha` / `lambda_l1` - L1 regularization (default: 0.0-0.1)
- `reg_lambda` / `lambda_l2` - L2 regularization (default: 1.0-5.0)
- `early_stopping_rounds` - Patience (default: 50-100)

### Neural Networks (MLP, Transformer, LSTM, CNN1D)
- `epochs` - Training epochs (default: 50-100)
- `batch_size` - Batch size (default: 256-512)
- `learning_rate` - Learning rate (default: 0.001)
- `dropout` - Dropout rate (default: 0.1-0.3)
- `patience` - Early stopping patience (default: 10-20)

### Ensemble
- `hgb_max_iter` - HistGradientBoosting iterations (default: 300)
- `hgb_max_depth` - HGB tree depth (default: 8)
- `rf_n_estimators` - RandomForest trees (default: 150)
- `rf_max_depth` - RF tree depth (default: 12)
- `ridge_alpha` - Ridge regularization (default: 1.0)
- `use_stacking` - Enable stacking (default: true)

### Multi-Task
- `epochs`, `batch_size`, `learning_rate`, `dropout` - Same as neural networks
- `use_multi_head` - Use separate heads per task (default: true)
- `loss_weights` - Weights for each task loss (default: [1.0, 0.5, 0.5])
- `target_names` - Task names (default: ["TTH", "MDD", "MFE"])

## Backward Compatibility

All trainers maintain backward compatibility:
- If config file is missing, falls back to hardcoded defaults
- If `config_loader` import fails, uses hardcoded defaults
- Old parameter names are supported (e.g., `hidden` → `hidden_layers` for MLP)

## Editing Configs

Edit YAML files in `CONFIG/model_config/`. Save and re-run training. No code changes needed.

Example (`CONFIG/model_config/lightgbm.yaml`):
```yaml
default:
  n_estimators: 2000        # Changed from 1500
  learning_rate: 0.02       # Changed from 0.03
  max_depth: 6              # Changed from 5
```

## Creating Custom Variants

Add new variants to any config file:

```yaml
variants:
  conservative: {...}
  balanced: {...}
  aggressive: {...}

  # Add custom variant
  my_experiment:
    n_estimators: 3000
    learning_rate: 0.01
    max_depth: 4
```

Load it:
```python
config = load_model_config("lightgbm", variant="my_experiment")
```
