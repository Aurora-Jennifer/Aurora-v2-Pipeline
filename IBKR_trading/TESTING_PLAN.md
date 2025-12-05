# IBKR Testing Plan

## Testing Strategy

### Phase 1: C++ Component Validation
- Test C++ components in isolation
- Validate Python-C++ bindings
- Performance benchmarking
- Memory leak testing

### Phase 2: IBKR Integration Testing
- Run Alpaca in parallel for comparison
- Copy models to IBKR test environment
- Side-by-side performance validation
- Risk management validation

## Phase 1: C++ Component Testing

### 1. Build and Test C++ Components

```bash
# Build C++ components
cd IBKR_trading/cpp_engine
./build.sh

# Run C++ tests
./test_cpp_components.sh

# Run performance benchmarks
./benchmark_cpp_components.sh
```

### 2. Python-C++ Integration Testing

```bash
# Test Python bindings
python IBKR_trading/test_cpp_integration.py

# Test with sample data
python IBKR_trading/test_cpp_with_data.py
```

### 3. Memory and Performance Validation

```bash
# Memory leak testing
python IBKR_trading/test_memory_leaks.py

# Performance comparison
python IBKR_trading/benchmark_python_vs_cpp.py
```

## Phase 2: IBKR Integration Testing

### 1. Parallel Alpaca + IBKR Testing

```bash
# Start Alpaca (keep running for comparison)
python run_alpaca_trading.py --config config/alpaca_config.yaml

# Start IBKR testing (parallel)
python IBKR_trading/test_ibkr_integration.py --config config/ibkr_daily_test.yaml
```

### 2. Model Copying and Validation

```bash
# Copy models from Alpaca to IBKR
python IBKR_trading/copy_models_from_alpaca.py

# Validate model compatibility
python IBKR_trading/validate_model_compatibility.py
```

### 3. Side-by-Side Performance Comparison

```bash
# Run comparison test
python IBKR_trading/compare_alpaca_ibkr.py --duration 1h
```

## Testing Checklist

### C++ Components
- [ ] C++ builds successfully
- [ ] Python bindings work
- [ ] Performance is 2-5x faster than Python
- [ ] No memory leaks
- [ ] Handles edge cases (empty data, NaN, etc.)

### IBKR Integration
- [ ] Connects to IBKR TWS/Gateway
- [ ] Market data streaming works
- [ ] Order placement works
- [ ] Position tracking works
- [ ] Risk management works

### Model Compatibility
- [ ] Models load correctly
- [ ] Features are compatible
- [ ] Predictions are identical
- [ ] Performance is maintained

### Risk Management
- [ ] Position limits enforced
- [ ] Drawdown controls work
- [ ] Kill switches work
- [ ] Emergency flatten works

## Expected Results

### C++ Components
- Performance: 2-5x faster than Python equivalent
- Memory: No memory leaks after 24h continuous running
- Accuracy: Identical results to Python version
- Stability: No crashes with edge cases

### IBKR Integration
- Connection: Stable connection to IBKR
- Data: Real-time market data streaming
- Orders: Orders execute correctly
- Risk: All risk controls work
- Performance: Matches or exceeds Alpaca performance

### Model Compatibility
- Accuracy: Identical predictions to Alpaca
- Speed: Faster inference than Python
- Memory: Lower memory usage
- Stability: No crashes during inference

## Troubleshooting

### C++ Build Issues
```bash
# Check if build directory exists
ls -la IBKR_trading/cpp_engine/build/

# Rebuild if needed
cd IBKR_trading/cpp_engine
rm -rf build/
./build.sh
```

### IBKR Connection Issues
```bash
# Check if IBKR TWS/Gateway is running
netstat -an | grep 7497

# Check IBKR logs
tail -f ~/IBKR/TWS/logs/tws.log
```

### Model Loading Issues
```bash
# Check model directory
ls -la IBKR_trading/models/daily_models/

# Validate model files
python IBKR_trading/validate_models.py
```

## Success Criteria

### C++ Components
- C++ builds successfully
- Python bindings work
- Performance is 2-5x faster than Python
- No memory leaks
- Handles edge cases (empty data, NaN, etc.)

### IBKR Integration
- Connects to IBKR TWS/Gateway
- Market data streaming works
- Order placement works
- Position tracking works
- Risk management works

### Model Compatibility
- Models load correctly
- Features are compatible
- Predictions are identical
- Performance is maintained

### Risk Management
- Position limits enforced
- Drawdown controls work
- Kill switches work
- Emergency flatten works

## Next Steps

1. Run C++ tests - Validate all C++ components
2. Start Alpaca - Keep running for comparison
3. Copy models - Transfer your existing models
4. Test IBKR connection - Ensure stable connection
5. Run parallel testing - Compare Alpaca vs IBKR
6. Validate results - Ensure performance matches
7. Go live - Switch to IBKR when ready
