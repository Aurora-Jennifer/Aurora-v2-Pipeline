# Configuration Overlays

Use configuration overlays to combine multiple config files.

## Overview

Overlays allow you to:
- Start with a base configuration
- Apply environment-specific overrides
- Merge multiple config sources
- Maintain separate configs for different use cases

## Basic Overlay

```python
import yaml

# Load base config
with open("CONFIG/base.yaml") as f:
    base = yaml.safe_load(f)

# Load overlay
with open("CONFIG/overlays/production.yaml") as f:
    overlay = yaml.safe_load(f)

# Merge (overlay takes precedence)
config = {**base, **overlay}
```

> **Note**: `load_config` function does not exist in `CONFIG.config_loader`. Use `yaml.safe_load()` directly.

## Environment-Based Overlays

```python
import os
import yaml

env = os.getenv("ENVIRONMENT", "development")

# Load base
with open("CONFIG/base.yaml") as f:
    base = yaml.safe_load(f)

# Load environment overlay
if env == "production":
    with open("CONFIG/overlays/production.yaml") as f:
        overlay = yaml.safe_load(f)
elif env == "development":
    with open("CONFIG/overlays/development.yaml") as f:
        overlay = yaml.safe_load(f)

config = {**base, **overlay}
```

## Deep Merging

For nested dictionaries, use deep merge:

```python
def deep_merge(base, overlay):
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

config = deep_merge(base, overlay)
```

## Overlay Structure

Create overlay files in `CONFIG/overlays/`:

```yaml
# CONFIG/overlays/production.yaml
models:
  lightgbm:
    variant: "conservative"
  xgboost:
    variant: "conservative"

training:
  early_stopping_rounds: 100
```

## Use Cases

### Production vs Development

```yaml
# development.yaml
training:
  early_stopping_rounds: 20
  n_estimators: 100

# production.yaml
training:
  early_stopping_rounds: 100
  n_estimators: 1000
```

### Model-Specific Overlays

```yaml
# lightgbm_overlay.yaml
lightgbm:
  max_depth: 10
  learning_rate: 0.05
```

## See Also

- [Config Loader API](CONFIG_LOADER_API.md) - Loader API
- [Advanced Config](../../01_tutorials/configuration/ADVANCED_CONFIG.md) - Advanced techniques

