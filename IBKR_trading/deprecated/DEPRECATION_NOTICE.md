# Deprecation Notice - IBKR Trading System

Files moved to deprecated folder.

## Deprecation Date

2024-01-XX

## Replacement System

New Live Trading Integration

The old trading system components have been replaced by a comprehensive new system that integrates all trained models (tabular + sequential + multi-task) across all horizons and strategies.

## Deprecated Files

### Live Trading Components (Old)
Moved to: `deprecated/live_trading_old/`

| Old File | New File | Reason for Deprecation |
|----------|----------|------------------------|
| `barrier_gate.py` | `live_trading/barrier_gate.py` | Replaced with enhanced barrier gating system |
| `cost_arbitrator.py` | `live_trading/cost_arbitrator.py` | Replaced with advanced cost model & arbitration |
| `horizon_blender.py` | `live_trading/horizon_blender.py` | Replaced with per-horizon model blending |
| `model_predictor.py` | `live_trading/model_predictor.py` | Replaced with unified prediction engine |
| `position_sizer.py` | `live_trading/position_sizer.py` | Replaced with advanced position sizing |
| `main_loop.py` | `live_trading/main_loop.py` | Replaced with comprehensive trading system |

### Old System Components
Moved to: `deprecated/old_components/`

| Old File | New File | Reason for Deprecation |
|----------|----------|------------------------|
| `ibkr_trading_system.py` | `live_trading/main_loop.py` | Replaced with new LiveTradingSystem |
| `enhanced_decision_pipeline.py` | `live_trading/main_loop.py` | Integrated into new system |
| `ibkr_live_exec.py` | `live_trading/main_loop.py` | Replaced with LiveTradingManager |

## New System Features

### What's New
- Complete Model Zoo Integration: All 20+ models across all strategies
- Sequential Mode Support: Proper (N, T, F) data handling for CNN1D, LSTM, Transformer
- Per-Horizon Blending: OOF-trained ridge → simplex blending
- Advanced Barrier Gating: Calibrated peak/valley probability gates
- Cost-Aware Arbitration: Realistic trading costs with horizon selection
- Risk Management: Position caps, no-trade bands, turnover control
- Live Buffers: Rolling sequences for sequential models
- Error Handling: Graceful degradation on failures
- Comprehensive Testing: Full test suite for all components

### Key Improvements
1. Model Coverage: Now supports all trained models (tabular + sequential + multi-task)
2. Data Handling: Proper sequential data for CNN1D, LSTM, Transformer models
3. Blending: Sophisticated per-horizon model blending with missing model handling
4. Gating: Advanced barrier probability gating for timing risk attenuation
5. Arbitration: Cost-aware horizon arbitration with winner-takes-most or softmax
6. Sizing: Advanced position sizing with vol scaling, risk parity, and caps
7. Integration: Complete system integration with comprehensive error handling

## Migration Guide

### For Developers
If you were using the old components, here's how to migrate:

#### Old Code
```python
from live_trading.barrier_gate import BarrierGate
from live_trading.cost_arbitrator import CostArbitrator
from live_trading.horizon_blender import HorizonBlender
```

#### New Code
```python
from live_trading.barrier_gate import BarrierGate
from live_trading.cost_arbitrator import CostArbitrator
from live_trading.horizon_blender import HorizonBlender
from live_trading.main_loop import LiveTradingSystem, LiveTradingManager
```

### Configuration Changes
- Old config files remain compatible
- New config file: `config/live_trading_config.yaml`
- Enhanced configuration options for all components

### API Changes
- All old APIs remain functional for backward compatibility
- New APIs provide enhanced functionality
- See `LIVE_TRADING_INTEGRATION.md` for complete documentation

## Performance Improvements

### Model Integration
- Before: Limited model support, basic blending
- After: All 20+ models, sophisticated per-horizon blending

### Data Handling
- Before: Tabular models only
- After: Tabular + sequential models with proper buffer management

### Decision Quality
- Before: Simple weighted averaging
- After: Cost-aware arbitration with barrier gating

## Support

For questions about the new system, see:
- `IBKR_trading/live_trading/README.md` - Component documentation
- `IBKR_trading/LIVE_TRADING_INTEGRATION.md` - Complete integration guide
- `IBKR_trading/README.md` - System overview
