# ALPACA Paper Trading Service

Alpaca paper trading service package.

## Structure

```
ALPACA_trading/
├── scripts/
│   ├── paper_runner.py              # Main paper trading runner
│   └── data/
│       ├── alpaca_batch_optimized.py
│       └── alpaca_batch.py
├── core/
│   ├── engine/
│   │   └── paper.py                 # Core paper trading engine
│   ├── enhanced_logging.py
│   ├── feature_reweighter.py
│   ├── notifications.py
│   ├── performance.py
│   ├── regime_detector.py
│   ├── strategy_selector.py
│   ├── utils.py
│   ├── data_sanity.py
│   ├── risk/
│   │   └── guardrails.py            # Optional
│   └── telemetry/
│       └── snapshot.py              # Optional
├── brokers/
│   ├── paper.py                     # Paper broker implementation
│   ├── interface.py                 # Broker interface
│   ├── data_provider.py
│   └── ibkr_broker.py
├── strategies/
│   ├── factory.py
│   └── regime_aware_ensemble.py
├── ml/
│   ├── model_interface.py
│   ├── registry.py
│   └── runtime.py
├── utils/
│   └── ops_runtime.py
├── tools/
│   └── provenance.py
├── cli/
│   └── paper.py                     # CLI interface
└── config/
    ├── base.yaml
    ├── models.yaml
    ├── paper_trading_config.json
    ├── paper_config.json
    ├── paper-trading.env
    └── [other paper configs]
```

## Entry Points

- `scripts/paper_runner.py` - Main paper trading runner
- `core/engine/paper.py` - Core trading engine
- `cli/paper.py` - CLI interface

## Configuration

Configuration files in `config/`:
- `base.yaml` - Base configuration
- `models.yaml` - Model registry
- `paper_trading_config.json` - Paper trading settings
- `paper-trading.env` - Environment variables

## Dependencies

### Python Packages
- `alpaca-trade-api` or `alpaca-py`
- `pandas`
- `numpy`
- `yfinance`
- `yaml`
- `requests`

### Environment Variables
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL` (optional)

## Usage

From repo root:
```bash
python ALPACA_trading/scripts/paper_runner.py --symbols SPY,TSLA --profile risk_balanced
```

Or using CLI:
```bash
python ALPACA_trading/cli/paper.py [options]
```

## Notes

- Some files are optional (marked in structure above)
- Import paths may need adjustment if running from this folder directly
