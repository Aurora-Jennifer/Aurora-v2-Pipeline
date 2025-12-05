# Live Trading System - IBKR Integration

Live trading system integrating all trained models (tabular + sequential + multi-task) across all horizons and strategies for IBKR trading.

## Architecture

### Core Components

1. **ModelPredictor** - Unified model prediction engine
   - Handles prediction from all trained models across horizons and strategies
   - Supports tabular, sequential, and multi-task models
   - Implements caching and TTL for performance

2. **HorizonBlender** - Per-horizon model blending
   - Uses OOF-trained ridge regression + simplex projection
   - Blends all regression models per horizon into single alpha
   - Supports adaptive weighting based on performance

3. **BarrierGate** - Timing and risk attenuator
   - Uses barrier probabilities to scale final alpha multiplicatively
   - Implements calibrated barrier probability gating
   - Supports horizon-specific gates

4. **CostArbitrator** - Cost model and horizon arbitration
   - Estimates trading costs per symbol/horizon
   - Implements winner-takes-most and softmax arbitration
   - Supports session-aware and correlation-aware adjustments

5. **PositionSizer** - Alpha to weights conversion
   - Vol scaling with cross-sectional standardization
   - Risk parity optimization with ridge regression
   - No-trade band for turnover control

6. **LiveTradingSystem** - Main integration loop
   - Orchestrates all components in live trading loop
   - Handles market data, features, predictions, and execution
   - Implements error handling and state management

## Usage

### Basic Setup

```python
from live_trading import LiveTradingSystem

# Configuration
config = {
    'symbols': ['AAPL', 'MSFT', 'GOOGL'],
    'horizons': ['5m', '10m', '15m', '30m', '60m', '120m', '1d', '5d', '20d'],
    'model_dir': 'TRAINING/models',
    'blender_dir': 'TRAINING/blenders',
    'device': 'cpu',
    'update_interval': 60,
    'g_min': 0.2,
    'gamma': 1.0,
    'delta': 0.5,
    'z_max': 3.0,
    'max_weight': 0.05,
    'target_gross': 0.5,
    'no_trade_band': 0.008
}

# Create and start system
system = LiveTradingSystem(config)
system.start()

# Get portfolio status
status = system.get_portfolio_status()
print(status)

# Stop system
system.stop()
```

### Advanced Configuration

```python
from live_trading import LiveTradingSystem, LiveTradingManager

# Multiple systems
configs = [
    {
        'symbols': ['AAPL', 'MSFT'],
        'horizons': ['5m', '15m', '60m'],
        'target_gross': 0.3
    },
    {
        'symbols': ['GOOGL', 'AMZN'],
        'horizons': ['1d', '5d'],
        'target_gross': 0.7
    }
]

# Manager for multiple systems
manager = LiveTradingManager(configs)
manager.start_all()

# Get all status
all_status = manager.get_all_status()
```

## Configuration

All configuration in `config/ibkr_enhanced.yaml`. Key sections:

- Models: Model families, horizons, inference settings
- Execution: Order types, TIF, bracket settings
- Risk: Position limits, kill switches
- Safety: Market integrity, time windows, risk limits

## Performance

- Decision latency: < 350ms (p99)
- Order routing: < 2s (p99)
- Model inference: < 100ms (p95)

## Related Documentation

- [IBKR Trading System](../README.md)
- [Architecture Documentation](../../docs/02_reference/systems/IBKR_SYSTEM_REFERENCE.md)
