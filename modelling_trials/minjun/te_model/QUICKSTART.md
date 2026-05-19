# TE Model - Quick Start Guide

## Results Summary

**Beat Baseline Model**: R² 0.2525 vs 0.2230 (+13.2%)
**MAE**: 3.95 (vs baseline 4.27)
**Top Features**: Rolling averages, career stats, blocking metrics
**New Features**: 53 engineered features

## Production Model

**File**: `te_ensemble_model.pkl` (713KB)

```python
import pickle

# Load model
with open('te_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X)
```

## Training Data

- **5,723 TE games** from 196 unique TEs
- Years: 2012-2024
- Train: Pre-2022 | Val: 2022 | Test: 2023-2024

## File Structure

```
te_model/
├── README.md                      # Full documentation
├── QUICKSTART.md                  # This file
├── te_ensemble_model.pkl          # Main production model
├── te_ensemble_predictions.csv    # Validation predictions
├── te_full_pipeline.py            # Training pipeline
└── te_feature_engineering.py      # Feature engineering
```

## Retrain Model

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna tqdm

# Run feature engineering (creates te_data_engineered.csv)
python te_feature_engineering.py

# Run full training pipeline (~25 minutes)
python te_full_pipeline.py
```

## Key Metrics

| Split | R² | MAE | RMSE |
|-------|-----|-----|------|
| Validation (2022) | 0.2525 | 3.95 | 5.18 |
| Test (2023-2024) | 0.2599 | 3.91 | 5.03 |
| Baseline | 0.2230 | 4.27 | - |

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
4. TE-specific features (is_elite_te, is_dual_threat, inline_pct, blocking metrics)

## TE-Specific Features

- **is_dual_threat**: TEs who both block and receive effectively
- **is_elite_te**: TE1s with high target share
- **inline_pct**: Inline usage (key separator from WRs)
- **pblk_grade_x_snaps**: Pass blocking effectiveness
- Lower boom threshold: 15 PPR (vs 20 for WRs)

---

**Author**: Minjun
**Date**: November 2025
**Status**: Production Ready
