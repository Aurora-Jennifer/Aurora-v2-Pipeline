# Model Training

Training models on labeled data.

## Training Overview

```
Labeled Data
    ↓
[1] Data Preparation → Feature selection, train/test split
    ↓
[2] Model Selection → Choose model family
    ↓
[3] Configuration → Load/override hyperparameters
    ↓
[4] Training → Fit model with early stopping
    ↓
[5] Evaluation → Metrics, feature importance, diagnostics
    ↓
Trained Model
```

## Available Models

### Core Models

**LightGBM** - Gradient boosting (highly regularized)
```python
from TRAINING.model_fun import LightGBMTrainer
from CONFIG.config_loader import load_model_config

config = load_model_config("lightgbm", variant="conservative")
trainer = LightGBMTrainer(config)
trainer.train(X_train, y_train)
```

**XGBoost** - Gradient boosting (highly regularized)
```python
config = load_model_config("xgboost", variant="balanced")
trainer = XGBoostTrainer(config)
```

**Ensemble** - Stacking (HGB + RF + Ridge)
```python
config = load_model_config("ensemble")
trainer = EnsembleTrainer(config)
```

**MultiTask** - Multi-task neural network (TTH, MDD, MFE)
```python
config = load_model_config("multi_task")
trainer = MultiTaskTrainer(config)
```

### Deep Learning

**MLP** - Multi-layer perceptron
```python
config = load_model_config("mlp", overrides={"epochs": 100})
trainer = MLPTrainer(config)
```

**Transformer** - Attention-based architecture
```python
config = load_model_config("transformer")
trainer = TransformerTrainer(config)
```

**LSTM** - Recurrent network
```python
config = load_model_config("lstm")
trainer = LSTMTrainer(config)
```

**CNN1D** - 1D convolutional network
```python
config = load_model_config("cnn1d")
trainer = CNN1DTrainer(config)
```

### Feature Engineering Models

**VAE** - Variational autoencoder (latent features)
```python
config = load_model_config("vae")
trainer = VAETrainer(config)
latent = trainer.train(X_train, y_train)  # Returns latent features
```

**GAN** - Generative adversarial network
```python
config = load_model_config("gan")
trainer = GANTrainer(config)
```

**GMMRegime** - Gaussian mixture for regime detection
```python
config = load_model_config("gmm_regime")
trainer = GMMRegimeTrainer(config)
```

### Probabilistic Models

**NGBoost** - Natural gradient boosting (distributional)
```python
config = load_model_config("ngboost")
trainer = NGBoostTrainer(config)
```

**QuantileLightGBM** - Quantile regression
```python
config = load_model_config("quantile_lightgbm", overrides={"alpha": 0.5})
trainer = QuantileLightGBMTrainer(config)
```

### Advanced Models

**ChangePoint** - Change point detection  
**FTRLProximal** - Follow-the-regularized-leader  
**RewardBased** - Reward-weighted regression  
**MetaLearning** - Meta-learning ensemble

## Training Strategies

### Single Target Training

One model per target variable.

```python
from TRAINING.strategies import SingleTaskStrategy

strategy = SingleTaskStrategy(
    model_name="lightgbm",
    model_config=load_model_config("lightgbm", variant="conservative")
)

model = strategy.train(
    X_train=X_train,
    y_train=y_train['y_will_peak'],
    X_val=X_val,
    y_val=y_val['y_will_peak']
)
```

### Multi-Task Learning

Correlated targets (TTH, MDD, MFE).

```python
from TRAINING.strategies import MultiTaskStrategy

strategy = MultiTaskStrategy(
    model_name="multi_task",
    target_names=["TTH", "MDD", "MFE"],
    loss_weights=[1.0, 0.5, 0.5]
)

model = strategy.train(
    X_train=X_train,
    y_train=y_train[["TTH", "MDD", "MFE"]],
    X_val=X_val,
    y_val=y_val[["TTH", "MDD", "MFE"]]
)
```

### Cascade Training

Sequential dependencies between targets.

```python
from TRAINING.strategies import CascadeStrategy

strategy = CascadeStrategy(
    models=["lightgbm", "xgboost", "ensemble"]
)

final_model = strategy.train(X_train, y_train, X_val, y_val)
```

## Configuration Management

### Using Variants

```python
# Conservative (least overfitting)
config = load_model_config("lightgbm", variant="conservative")

# Balanced (default)
config = load_model_config("xgboost", variant="balanced")

# Aggressive (faster training)
config = load_model_config("mlp", variant="aggressive")
```

### Runtime Overrides

```python
config = load_model_config("lightgbm", overrides={
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "early_stopping_rounds": 50
})
```

### Environment Variables

```bash
export MODEL_VARIANT=conservative

# Then in Python
trainer = LightGBMTrainer()  # Automatically uses conservative
```

## Training Pipeline

### Phase 1: Feature Engineering

```bash
python TRAINING/EXPERIMENTS/phase1_features/build_vae_features.py \
    --input-dir DATA_PROCESSING/data/labeled/ \
    --output-dir TRAINING/EXPERIMENTS/phase1_features/output/ \
    --latent-dim 32
```

### Phase 2: Core Model Training

```bash
bash TRAINING/train_all_symbols.sh \
    --phase core \
    --config CONFIG/model_config/lightgbm.yaml \
    --variant conservative
```

### Phase 3: Sequential/Advanced Models

```bash
bash TRAINING/train_all_symbols.sh \
    --phase sequential \
    --models lstm,transformer,ngboost
```

## Walk-Forward Validation

Simulates real trading conditions by training on past data and testing on future data.

```python
from TRAINING.walkforward import WalkForwardEngine

engine = WalkForwardEngine(
    data=df_labeled,
    train_days=252,      # 1 year training
    test_days=63,        # 1 quarter testing
    step_days=21,        # Roll forward monthly
    min_train_samples=1000
)

results = engine.run(
    model_name="lightgbm",
    config=load_model_config("lightgbm", variant="conservative"),
    metrics=["sharpe", "max_drawdown", "hit_rate"]
)
```

### Metrics Tracked

- Sharpe Ratio: Risk-adjusted returns
- Max Drawdown: Worst peak-to-trough decline
- Hit Rate: Percentage of correct predictions
- Profit Factor: Gross profit / Gross loss
- Win/Loss Ratio: Average win / Average loss

## Feature Selection

### Importance-Based Selection

```python
trainer = LightGBMTrainer()
trainer.train(X_train, y_train)

importance = trainer.get_feature_importance()
top_features = importance.head(60).index.tolist()
X_train_selected = X_train[top_features]
```

### Recursive Feature Elimination

```python
from sklearn.feature_selection import RFECV

selector = RFECV(
    estimator=trainer.model,
    step=1,
    cv=5,
    scoring='neg_mean_squared_error'
)
X_train_selected = selector.fit_transform(X_train, y_train)
```

## Hyperparameter Tuning

### Manual Tuning

Edit config files in `CONFIG/model_config/`:

```yaml
# CONFIG/model_config/lightgbm.yaml
variants:
  my_experiment:
    n_estimators: 2000
    learning_rate: 0.02
    max_depth: 6
    min_data_in_leaf: 200
```

Load and use:
```python
config = load_model_config("lightgbm", variant="my_experiment")
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [1000, 1500, 2000],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [4, 5, 6]
}

grid = GridSearchCV(
    estimator=trainer.model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid.fit(X_train, y_train)
```

## Training Checklist

### Before Training
- [ ] Data loaded and validated
- [ ] Features scaled/normalized (if needed)
- [ ] Train/validation split created
- [ ] Config loaded or overridden
- [ ] Memory usage checked

### During Training
- [ ] Early stopping enabled
- [ ] Progress logged
- [ ] Validation metrics monitored
- [ ] Feature importance tracked

### After Training
- [ ] Model saved
- [ ] Metrics logged
- [ ] Feature importance exported
- [ ] Predictions validated
- [ ] Config saved with results

## Common Issues

### Overfitting

Symptoms: High train accuracy, low validation accuracy

Solutions:
1. Use `conservative` variant
2. Increase regularization (`reg_lambda`, `reg_alpha`)
3. Reduce model complexity (`max_depth`, `num_leaves`)
4. Increase `early_stopping_rounds`
5. Add more dropout (neural networks)

### Underfitting

Symptoms: Low train and validation accuracy

Solutions:
1. Use `aggressive` variant
2. Increase model complexity
3. Add more features
4. Reduce regularization
5. Train longer

### Memory Errors

Solutions:
1. Reduce `batch_size`
2. Use sequential training
3. Enable gradient checkpointing (neural networks)
4. Use streaming feature builder

### Slow Training

Solutions:
1. Reduce `n_estimators`
2. Increase `learning_rate`
3. Use `aggressive` variant
4. Enable GPU (neural networks)
5. Parallelize across symbols

## Evaluation & Diagnostics

### Model Evaluation

```python
y_pred = trainer.predict(X_val)

from sklearn.metrics import mean_squared_error, r2_score, classification_report

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# For classification
report = classification_report(y_val, y_pred)
```

### Feature Importance

```python
importance = trainer.get_feature_importance(target="y_will_peak")
print(importance.head(20))
```

### Learning Curves

```python
import matplotlib.pyplot as plt

history = trainer.model.evals_result()
plt.plot(history['validation_0']['rmse'], label='Train')
plt.plot(history['validation_1']['rmse'], label='Val')
plt.legend()
plt.show()
```

## Saving & Loading Models

### Save Model

```python
import joblib

joblib.dump(trainer.model, "models/lightgbm_AAPL_2025-11-13.pkl")

import json
with open("models/lightgbm_AAPL_2025-11-13_config.json", "w") as f:
    json.dump(trainer.config, f, indent=2)
```

### Load Model

```python
model = joblib.load("models/lightgbm_AAPL_2025-11-13.pkl")
y_pred = model.predict(X_new)
```

## Related Documentation

- [01_QUICK_START.md](01_QUICK_START.md) - Config reference
- [04_DATA_PIPELINE.md](04_DATA_PIPELINE.md) - Data preparation
- [06_COLUMN_REFERENCE.md](06_COLUMN_REFERENCE.md) - Feature descriptions
- `CONFIG/` - Configuration files
