# Combined WR/TE Model - Quick Start Guide

## Results Summary

**Competitive with Benchmark**: R² 0.2733 vs 0.2754 (0.7% behind)
**MAE**: 4.81 (vs benchmark 4.73)
**Key Feature**: `is_te` position indicator
**Training Data**: 18,559 combined WR/TE games

## Production Model

**File**: `wr_te_combined_ensemble.pkl` (1.1MB)

```python
import pickle

# Load model
with open('wr_te_combined_ensemble.pkl', 'rb') as f:
    model = pickle.load(f)

# IMPORTANT: Must include is_te feature
X['is_te'] = (X['INL%'] > 15).astype(int)

# Make predictions
predictions = model.predict(X)
```

## Why Combined?

1. **More data**: 18,559 games vs 12,836 (WR) or 5,723 (TE) separately
2. **Shared patterns**: Both positions share receiving metrics
3. **Position indicator**: `is_te` allows model to differentiate
4. **Mirrors benchmark**: Same approach as comparison model

## File Structure

```
wr_te_combined_model/
├── README.md                        # Full documentation
├── QUICKSTART.md                    # This file
├── wr_te_combined_ensemble.pkl      # Main production model
├── wr_te_combined_predictions.csv   # Validation predictions
└── wr_te_combined_pipeline.py       # Training pipeline
```

## Retrain Model

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna tqdm

# Prerequisites: Must have WR and TE engineered data
# Run these first if needed:
python wr_feature_engineering.py
python te_feature_engineering.py

# Run combined training pipeline (~35 minutes)
python wr_te_combined_pipeline.py
```

## Key Metrics

| Split | R² | MAE | RMSE |
|-------|-----|-----|------|
| Validation (2022) | 0.2733 | 4.81 | 6.77 |
| Test (2023-2024) | 0.2772 | 4.79 | 6.64 |
| Benchmark | 0.2754 | 4.73 | - |

## Model Architecture

**Stacking Ensemble**:
1. XGBoost (tuned with 150 Optuna trials)
2. LightGBM (tuned with 150 Optuna trials)
3. CatBoost (tuned with 150 Optuna trials)
4. Ridge meta-learner (5-fold CV)

## Position Indicator

**Critical Feature**: `is_te`

```python
# How it's created
df['is_te'] = (df['INL%'] > 15).astype(int)

# TEs line up inline (high INL%), WRs don't
# Zach Ertz: INL% = 62.8% → is_te = 1
# Julio Jones: INL% = 0.3% → is_te = 0
```

## Comparison Table

| Model | R² | MAE | Status |
|-------|-----|-----|--------|
| Combined (Ours) | 0.2733 | 4.81 | Competitive |
| Benchmark | 0.2754 | 4.73 | Target |
| WR-only | 0.2494 | 5.24 | +52.7% vs baseline |
| TE-only | 0.2525 | 3.95 | +13.2% vs baseline |

## When to Use This Model

Use **combined model** when:
- Predicting both WR and TE in same pipeline
- Want single model for all receiving positions
- Limited training data for TEs

Use **separate models** when:
- Want position-specific feature engineering
- Need specialized TE features (blocking, inline%)
- Want to optimize each position independently

---

**Author**: Minjun
**Date**: November 2025
**Status**: Production Ready
