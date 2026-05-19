# Combined WR/TE Fantasy Football Prediction Model - Minjun

**Final Results: Competitive with Benchmark Model**

## Model Performance

### Validation Results (2022 Season)
- **R²: 0.2733** (Benchmark: 0.2754) - **Only 0.7% behind benchmark**
- **MAE: 4.81** (Benchmark: 4.73)
- **RMSE: 6.77**

### Test Results (2023-2024)
- **R²: 0.2772**
- **MAE: 4.79**
- **RMSE: 6.64**

## Why Combined Model?

This model combines WR and TE data into a single training set, similar to the benchmark approach. Key advantages:

1. **More training data**: 18,559 games vs 12,836 (WR) or 5,723 (TE) separately
2. **Shared receiver patterns**: Both positions share core receiving metrics (targets, receptions, yards)
3. **Position indicator**: `is_te` feature allows model to learn universal patterns, then apply position-specific adjustments
4. **Benefits TEs**: Limited TE data (5,723 games) benefits from learning alongside 12,836 WR games

## Model Architecture

**Stacking Ensemble** combining:
1. XGBoost
2. LightGBM
3. CatBoost

**Meta-learner**: Ridge Regression (alpha=1.0)

## Key Features

**Top Features** include:
- Rolling PPR averages (last_game_ppr, last_3_games_ppr, last_5_games_ppr)
- Career statistics (career_avg_ppr, season_avg_ppr)
- Target metrics (last_3_games_tgt, career_tgt_per_game)
- Team performance (team_win_rate, team_point_diff)
- **Position indicator**: `is_te` (1 if TE, 0 if WR)
- Advanced metrics (value_over_replacement, recent_form)

**Uses common features** between WR and TE datasets + position indicator

## Files

### Models
- `wr_te_combined_ensemble.pkl` - Main production model (1.1MB)

### Results
- `wr_te_combined_predictions.csv` - Validation predictions (with position labels)

### Code
- `wr_te_combined_pipeline.py` - 7-step training pipeline

## Pipeline Steps

1. **Load Data** - Combine WR and TE engineered datasets
2. **Noise Removal** - K-NN outlier detection (remove top 15%)
3. **Feature Selection** - Top 35 common features + is_te indicator
4. **Tune XGBoost** - 150 trials with Optuna
5. **Tune LightGBM** - 150 trials with Optuna
6. **Tune CatBoost** - 150 trials with Optuna
7. **Stacking Ensemble** - Ridge meta-learner with 5-fold CV

## Usage

```python
import pickle
import pandas as pd
import numpy as np

# Load model
with open('wr_te_combined_ensemble.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare data (must include is_te feature)
X['is_te'] = (X['INL%'] > 15).astype(int)  # 1 for TE, 0 for WR

# Make predictions
predictions = model.predict(X)
```

## Training Data

- **18,559 total games** (2012-2024)
  - 12,836 WR games (383 unique WRs)
  - 5,723 TE games (196 unique TEs)
- Train: Pre-2022, Val: 2022, Test: 2023-2024
- Position separation: INL% > 15 for TE

## Comparison: Combined vs Separate

| Approach | R² (Val) | MAE (Val) | Data Size |
|----------|----------|-----------|-----------|
| **Combined** | 0.2733 | 4.81 | 18,559 games |
| **Benchmark** | 0.2754 | 4.73 | 18,559 games |
| WR-only | 0.2494 | 5.24 | 12,836 games |
| TE-only | 0.2525 | 3.95 | 5,723 games |

**Trade-offs**:
- Combined model achieves near-benchmark performance (0.7% gap)
- Separate models allow position-specific feature engineering
- Combined approach has more training data but averages across positions

## Position Indicator Feature

The `is_te` feature is critical for the combined model:

```python
# In wr_te_combined_pipeline.py
df['is_te'] = (df['INL%'] > 15).astype(int)
top_features.append('is_te')  # Added to feature set
```

This allows the model to:
1. Learn shared receiver patterns (targets, catches, yards)
2. Apply position-specific adjustments via the `is_te` feature
3. Handle different scoring ranges (WRs have higher variance, TEs more consistent)

## Requirements

- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- optuna (for hyperparameter tuning)
- tqdm (progress bars)

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna tqdm
```

## Notes

- This approach mirrors the benchmark methodology (combined training)
- Achieves competitive performance with more sophisticated ensemble stacking
- For position-specific optimization, see separate `wr_model/` and `te_model/` folders

**Author**: Minjun | **Date**: November 2025
