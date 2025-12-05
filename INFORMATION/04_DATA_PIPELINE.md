# Data Pipeline

Workflow from raw market data to ML-ready labeled datasets.

## Pipeline Overview

```
Raw Market Data
    ↓
[1] Normalization → Session-aligned, grid-corrected data
    ↓
[2] Feature Engineering → 200+ technical features
    ↓
[3] Target Generation → Labels (barrier, excess returns, HFT)
    ↓
Labeled Dataset → Ready for model training
```

## Stage 1: Raw Data Acquisition

### Data Location
```
data/
└── data_labeled/
    └── interval=5m/
        ├── A.parquet
        ├── AAPL.parquet
        └── ...
```

### Expected Format
- Timeframe: 5-minute bars
- Columns: `ts`, `open`, `high`, `low`, `close`, `volume`
- Timezone: UTC timestamps
- Coverage: NYSE Regular Trading Hours (RTH) only

### Quality Checks
```python
from DATA_PROCESSING.pipeline import normalize_interval, assert_bars_per_day

# Normalize to RTH and grid
df_clean = normalize_interval(df, interval="5m")

# Verify bar count per day
assert_bars_per_day(df_clean, interval="5m", min_full_day_frac=0.90)
```

## Stage 2: Feature Engineering

### Available Feature Builders

#### Simple Features (~50 features)
```python
from DATA_PROCESSING.features import SimpleFeatureComputer

computer = SimpleFeatureComputer()
feature_names = computer.get_all_features()
# Returns, momentum, volatility, basic technical indicators
```

#### Comprehensive Features (200+ features)
```python
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder

builder = ComprehensiveFeatureBuilder(config_path="config/features.yaml")
features = builder.build_features(
    input_paths=["data/data_labeled/interval=5m/*.parquet"],
    output_dir="DATA_PROCESSING/data/processed/",
    universe_config="config/universe.yaml"
)
```

#### Streaming Builder (Memory-Efficient)
```python
# For large datasets that don't fit in memory
# Uses Polars lazy API for efficient streaming
# See DATA_PROCESSING/features/streaming_builder.py
```

### Feature Categories

**Core Features:**
- Returns: `ret_1m`, `ret_5m`, `ret_15m`, `ret_30m`, `ret_60m`
- Volatility: `vol_5m`, `vol_15m`, `vol_30m`, `vol_60m`
- Momentum: `mom_1d`, `mom_3d`, `mom_5d`, `mom_10d`
- Volume: `dollar_volume`, `turnover_20`, `turnover_5`

**Technical Indicators:**
- Moving Averages: `sma_5`, `sma_10`, `sma_20`, `ema_5`, `ema_10`, ...
- Oscillators: `rsi_14`, `macd`, `stoch_k`, `stoch_d`, ...
- Momentum: `roc`, `williams_r`, `cci`, ...
- Volatility: `atr_14`, `bollinger_width`, ...

**Advanced Features:**
- Cross-sectional rankings
- Regime indicators
- Interaction features
- Custom composite indicators

## Stage 3: Target Generation

### Target Types

#### Barrier Targets (First-Passage Labels)
Predict which barrier (up/down) will be hit first.

```python
from DATA_PROCESSING.targets import compute_barrier_targets

targets = compute_barrier_targets(
    prices=df['close'],
    horizon_minutes=15,      # Lookahead window
    barrier_size=0.5,        # Size in volatility units
    vol_window=20,           # Volatility estimation window
    min_touch_prob=0.15      # Minimum probability threshold
)

# Returns:
# - y_first_touch: {-1, 0, +1} (down-first, none, up-first)
# - y_will_peak: {0, 1} (will hit up barrier)
# - y_will_valley: {0, 1} (will hit down barrier)
# - p_up: Probability of hitting up barrier first
# - p_down: Probability of hitting down barrier first
```

#### Excess Return Targets (Market-Adjusted)
Predict future returns adjusted for market exposure.

```python
from DATA_PROCESSING.targets import future_excess_return, classify_excess_return

# Calculate beta-adjusted excess returns
excess_ret = future_excess_return(
    asset_ret=df['returns'],
    mkt_ret=spy['returns'],
    H=5  # 5-day horizon
)

# Classify into 3 classes using neutral band
labels = classify_excess_return(
    excess_ret=excess_ret,
    epsilon=0.01  # Neutral band size
)
# Returns: {-1: SELL, 0: HOLD, +1: BUY}
```

#### HFT Forward Returns (Short-Horizon)
Generate targets for high-frequency trading.

```python
from DATA_PROCESSING.targets import hft_forward

# Generates forward returns for 15m, 30m, 60m, 120m
df_with_targets = hft_forward.add_hft_targets(
    data_dir="data/data_labeled/interval=5m/",
    output_dir="DATA_PROCESSING/data/labeled/"
)

# Creates columns:
# - fwd_ret_15m
# - fwd_ret_30m
# - fwd_ret_60m
# - fwd_ret_120m
```

## Stage 4: Data Quality & Validation

### Schema Validation
```python
from DATA_PROCESSING.utils import SchemaExpectations, validate_schema

# Define expected schema
schema = SchemaExpectations(
    required={'ts', 'open', 'high', 'low', 'close', 'volume'},
    optional_interactions={'ret_1m_x_vol_5m', 'rsi_x_mom'}
)

# Validate
warnings = validate_schema(
    cols=df.columns,
    spec=schema,
    strict_interactions=False
)
```

### Memory Management
```python
from DATA_PROCESSING.utils import MemoryManager

mem = MemoryManager()
mem.check_memory("Before processing")

# Process data...

mem.check_memory("After processing")
mem.force_cleanup()  # If needed
```

## Complete Pipeline Example

```python
import polars as pl
from DATA_PROCESSING.pipeline import normalize_interval
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder
from DATA_PROCESSING.targets import compute_barrier_targets
from DATA_PROCESSING.utils import MemoryManager

# Initialize
mem = MemoryManager()
mem.check_memory("Start")

# 1. Load raw data
df = pl.read_parquet("data/data_labeled/interval=5m/AAPL.parquet")

# 2. Normalize session
df_clean = normalize_interval(df, interval="5m")

# 3. Build features
builder = ComprehensiveFeatureBuilder(config_path="config/features.yaml")
df_features = builder.build_features(
    input_paths=["AAPL.parquet"],
    output_dir="DATA_PROCESSING/data/processed/",
    universe_config="config/universe.yaml"
)

# 4. Generate targets
targets = compute_barrier_targets(
    prices=df_features['close'],
    horizon_minutes=15,
    barrier_size=0.5
)

# 5. Combine and save
df_labeled = df_features.join(targets, on='ts')
df_labeled.write_parquet("DATA_PROCESSING/data/labeled/AAPL_labeled.parquet")

mem.check_memory("Complete")
```

## Batch Processing

### Smart Barrier Pipeline (Resumable)
```bash
# Process all symbols with barrier targets
python DATA_PROCESSING/pipeline/barrier_pipeline.py \
    --input-dir data/data_labeled/interval=5m/ \
    --output-dir DATA_PROCESSING/data/labeled/ \
    --horizon 15 \
    --barrier-size 0.5 \
    --parallel-jobs 4
```

Features:
- Automatic resumption (checks what's already processed)
- Progress tracking
- Parallel processing
- Error recovery

## Output Schema

### Final Labeled Dataset Columns

**OHLCV (Original):**
- `ts`, `open`, `high`, `low`, `close`, `volume`

**Features (~200+):**
- Returns: `ret_*`
- Volatility: `vol_*`
- Technical: `sma_*`, `ema_*`, `rsi_*`, `macd_*`, ...
- Custom: See `06_COLUMN_REFERENCE.md`

**Targets:**
- Barrier: `y_will_peak`, `y_will_valley`, `y_first_touch`
- Probabilities: `p_up`, `p_down`
- Excess Returns: `excess_ret_5d`, `y_class`
- HFT: `fwd_ret_15m`, `fwd_ret_30m`, `fwd_ret_60m`

## Storage Organization

```
DATA_PROCESSING/data/
├── raw/              # Raw unprocessed data
├── processed/        # Features added, no targets
└── labeled/          # Complete dataset with targets
```

Keep intermediate stages to allow reprocessing with different target definitions.

## Troubleshooting

**Memory errors during feature engineering**: Use `streaming_builder.py` with Polars lazy API

**Missing bars in session data**: Use `normalize_interval()` to enforce grid and remove partial days

**Duplicate timestamps**: Check for timezone issues, use `unique()` after normalization

**Target labels imbalanced**: Adjust `barrier_size` or `min_touch_prob` parameters

## Related Documentation

- [05_MODEL_TRAINING.md](05_MODEL_TRAINING.md) - Training models on labeled data
- [06_COLUMN_REFERENCE.md](06_COLUMN_REFERENCE.md) - Column documentation
- `DATA_PROCESSING/README.md` - Module documentation
