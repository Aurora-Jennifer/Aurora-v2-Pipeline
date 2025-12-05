# Project Overview

High-level overview of the trading ML pipeline.

## Purpose

Build machine learning models to predict short-term market movements for systematic trading strategies.

Focus: Intraday prediction (5m-2h horizons) using barrier labels and excess returns.

## Architecture

```
Raw Market Data
    ↓
DATA_PROCESSING → Features + Targets
    ↓
TRAINING → Trained Models
    ↓
Strategies (Future) → Trading Signals
    ↓
Execution (Future) → Orders
```

## Directory Structure

```
trader/
│
├── CONFIG/                        # Centralized configurations
│   ├── model_config/             # Model hyperparameters (17 models)
│   ├── training_config/          # Training workflows
│   └── config_loader.py          # Config loading utility
│
├── DATA_PROCESSING/              # Data pipeline
│   ├── features/                 # Feature engineering
│   ├── targets/                  # Target generation
│   ├── pipeline/                 # Processing workflows
│   ├── utils/                    # Utilities
│   └── data/                     # Data storage
│
├── TRAINING/                     # Model training
│   ├── model_fun/                # 17+ model trainers
│   ├── strategies/               # Training strategies
│   ├── walkforward/              # Walk-forward validation
│   ├── memory/                   # Memory management
│   └── EXPERIMENTS/              # Experimental workflows
│
├── INFORMATION/                  # Documentation
│   ├── 01_QUICK_START.md        # Config quick start
│   ├── 02_CONFIG_REFERENCE.md   # Config reference
│   ├── 03_MIGRATION_NOTES.md    # Migration details
│   ├── 04_DATA_PIPELINE.md      # Data workflow
│   ├── 05_MODEL_TRAINING.md     # Training guide
│   ├── 06_COLUMN_REFERENCE.md   # Column docs
│   └── 07_PROJECT_OVERVIEW.md   # This file
│
└── data/                         # Primary data directory
    └── data_labeled/             # Labeled datasets
        └── interval=5m/          # 5-minute bars
            ├── A.parquet
            ├── AAPL.parquet
            └── ...
```

## Workflow

### 1. Data Acquisition
Input: Raw OHLCV data (5-minute bars, NYSE RTH)  
Output: Cleaned, normalized session data

```
data/data_labeled/interval=5m/*.parquet
    ↓
[Normalize session]
    ↓
DATA_PROCESSING/data/raw/
```

### 2. Feature Engineering
Input: Normalized OHLCV  
Output: 200+ engineered features

```
[Feature builders]
    ↓
Returns, volatility, momentum, technical indicators
    ↓
DATA_PROCESSING/data/processed/
```

### 3. Target Generation
Input: Processed features  
Output: Prediction labels

```
[Target generators]
    ↓
Barrier labels, excess returns, HFT forward returns
    ↓
DATA_PROCESSING/data/labeled/
```

### 4. Model Training
Input: Labeled datasets  
Output: Trained models

```
[Training strategies]
    ↓
LightGBM, XGBoost, Ensemble, MultiTask, Deep Learning
    ↓
models/*.pkl + configs
```

### 5. Evaluation
Input: Trained models  
Output: Performance metrics

```
[Walk-forward validation]
    ↓
Sharpe, drawdown, hit rate, profit factor
    ↓
results/
```

## Key Components

### CONFIG (Centralized Configuration)
Centralized configuration system.

Features:
- 17 model configs with 3 variants each (conservative/balanced/aggressive)
- Runtime overrides
- Environment variable support
- Version control friendly

Usage:
```python
from CONFIG.config_loader import load_model_config

config = load_model_config("lightgbm", variant="conservative")
trainer = LightGBMTrainer(config)
```

### DATA_PROCESSING (ETL Pipeline)
Transforms raw data into ML-ready features.

Modules:
- `features/`: 3 feature builders (simple, comprehensive, streaming)
- `targets/`: 3 target types (barrier, excess returns, HFT)
- `pipeline/`: End-to-end workflows (normalization, batch processing)
- `utils/`: Memory management, logging, validation

Usage:
```python
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder
from DATA_PROCESSING.targets import compute_barrier_targets

builder = ComprehensiveFeatureBuilder(config_path="config/features.yaml")
features = builder.build_features(input_paths, output_dir, universe_config)

targets = compute_barrier_targets(prices=df['close'], horizon_minutes=15)
```

### TRAINING (Model Training)
Trains 17+ models on labeled data.

Available Models:
- Core: LightGBM, XGBoost, Ensemble, MultiTask
- Deep Learning: MLP, Transformer, LSTM, CNN1D
- Feature Engineering: VAE, GAN, GMMRegime
- Probabilistic: NGBoost, QuantileLightGBM
- Advanced: ChangePoint, FTRL, RewardBased, MetaLearning

Training Strategies:
- SingleTask: One model per target
- MultiTask: Shared model for correlated targets
- Cascade: Sequential dependencies

Usage:
```python
from TRAINING.model_fun import LightGBMTrainer
from CONFIG.config_loader import load_model_config

config = load_model_config("lightgbm", variant="conservative")
trainer = LightGBMTrainer(config)
trainer.train(X_train, y_train, X_val, y_val)
```

## Configuration Philosophy

Centralized configuration replaces hardcoded parameters.

Example:
```python
class LightGBMTrainer:
    def __init__(self, config=None):
        if config is None:
            config = load_model_config("lightgbm")  # Loads from CONFIG/
        self.config = config
```

Benefits:
- Change configs without editing code
- Version control configs separately
- Easy experimentation
- Reproducibility

## Data Schema

### Labeled Dataset Structure

**OHLCV (6 columns):**
- `ts`, `open`, `high`, `low`, `close`, `volume`

**Features (~200 columns):**
- Returns: `ret_1m`, `ret_5m`, ...
- Volatility: `vol_5m`, `vol_15m`, ...
- Technical: `sma_*`, `ema_*`, `rsi_*`, ...
- Custom: Cross-sectional, interactions

**Targets (~20+ columns):**
- Barrier: `y_will_peak`, `y_will_valley`, `y_first_touch`
- Probabilities: `p_up`, `p_down`
- Excess Returns: `excess_ret_5d`, `y_class`
- HFT: `fwd_ret_15m`, `fwd_ret_30m`, ...

See `06_COLUMN_REFERENCE.md` for complete documentation.

## Training Approach

### Prevent Overfitting
1. High regularization (conservative variant)
2. Early stopping (monitor validation)
3. Feature selection (top 50-60 features)
4. Walk-forward validation (simulate real trading)

### Model Selection
1. Start simple - LightGBM/XGBoost first
2. Ensemble - Combine diverse models
3. Deep learning - For sequential patterns
4. Specialized - NGBoost for uncertainty, VAE for features

### Hyperparameter Tuning
1. Use variants - Conservative/Balanced/Aggressive presets
2. Manual tuning - Edit YAML configs
3. Grid search - If needed (expensive)

## Common Tasks

### Add New Feature
1. Edit `DATA_PROCESSING/features/simple_features.py`
2. Add to feature definitions dict
3. Rerun feature builder
4. Validate output

### Add New Target
1. Edit `DATA_PROCESSING/targets/barrier.py` (or create new)
2. Define target function
3. Update `targets/__init__.py`
4. Rerun target generation

### Add New Model
1. Create trainer in `TRAINING/model_fun/`
2. Create config in `CONFIG/model_config/`
3. Update `TRAINING/model_fun/__init__.py`
4. Test with small dataset

### Run Experiment
1. Create variant in config YAML
2. Load config with variant name
3. Train model
4. Compare metrics to baseline

## Performance Metrics

### Model Evaluation
- MSE/RMSE - Regression error
- R² - Variance explained
- Classification metrics - Accuracy, precision, recall, F1

### Trading Simulation
- Sharpe Ratio - Risk-adjusted returns
- Max Drawdown - Worst decline
- Hit Rate - % correct predictions
- Profit Factor - Gross profit / loss
- Win/Loss Ratio - Avg win / avg loss

## Documentation Navigation

- [01_QUICK_START.md](01_QUICK_START.md) - Config quick reference
- [02_CONFIG_REFERENCE.md](02_CONFIG_REFERENCE.md) - Config API
- [03_MIGRATION_NOTES.md](03_MIGRATION_NOTES.md) - Migration details
- [04_DATA_PIPELINE.md](04_DATA_PIPELINE.md) - Data processing workflow
- [05_MODEL_TRAINING.md](05_MODEL_TRAINING.md) - Training guide
- [06_COLUMN_REFERENCE.md](06_COLUMN_REFERENCE.md) - Column documentation
- [07_PROJECT_OVERVIEW.md](07_PROJECT_OVERVIEW.md) - This file

## Related Documentation

- `CONFIG/` - Configuration files
- `DATA_PROCESSING/` - Data processing code
- `TRAINING/` - Training code
- `TRAINING/EXPERIMENTS/` - Experimental workflows
