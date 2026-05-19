# WR Model - Quick Start Guide

## Results Summary

**Beat Baseline Model**: R² 0.2494 vs 0.1633 (+52.7%)
**MAE**: 5.24 (vs baseline 5.64)
**Top Features**: Rolling averages, career stats, target metrics
**New Features**: 52 engineered features

## Production Model

**File**: `wr_ensemble_model.pkl` (666KB)

```python
import pickle

# Load model
with open('wr_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X)
```

## Training Data

- **12,836 WR games** from 383 unique WRs
- Years: 2012-2024
- Train: Pre-2022 | Val: 2022 | Test: 2023-2024

## File Structure

```
wr_model/
├── README.md                      # Full documentation
├── QUICKSTART.md                  # This file
├── wr_ensemble_model.pkl          # Main production model
├── wr_ensemble_predictions.csv    # Validation predictions
├── wr_full_pipeline.py            # Training pipeline
└── wr_feature_engineering.py      # Feature engineering
```

## Retrain Model

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna tqdm

# Run feature engineering (creates wr_data_engineered.csv)
python wr_feature_engineering.py

# Run full training pipeline (~30 minutes)
python wr_full_pipeline.py
```

## Key Metrics

| Split | R² | MAE | RMSE |
|-------|-----|-----|------|
| Validation (2022) | 0.2494 | 5.24 | 6.89 |
| Test (2023-2024) | 0.2772 | 5.18 | 6.64 |
| Baseline | 0.1633 | 5.64 | - |

## Model Architecture

**Stacking Ensemble**:
1. XGBoost (tuned with 150 Optuna trials)
2. LightGBM (tuned with 150 Optuna trials)
3. CatBoost (tuned with 150 Optuna trials)
4. Ridge meta-learner (5-fold CV)

## Top Feature Categories

1. Rolling PPR averages (last_game_ppr, last_3_games_ppr, season_avg_ppr)
2. Career statistics (career_avg_ppr, career_games)
3. Target metrics (last_3_games_tgt, career_tgt_per_game)
4. WR-specific features (is_alpha_wr, is_deep_threat)

---

**Author**: Minjun
**Date**: November 2025
**Status**: Production Ready
