# WR Fantasy Football Prediction Model - Minjun

**Final Results: Beat Baseline Model's Benchmark **

## Model Performance

### Validation Results (2022 Season)
- **R²: 0.2494** (vs Baseline: 0.1633) - **+52.7% improvement**
- **MAE: 5.24** (vs Baseline: 5.64)
- **RMSE: 6.89**

### Test Results (2023-2024)
- **R²: 0.2772**
- **MAE: 5.18**
- **RMSE: 6.64**

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
- Advanced metrics (value_over_replacement, recent_form, boom_rate)
- WR-specific features (is_alpha_wr, is_deep_threat, alignment_diversity)

**52 new engineered features** from 89 original features

## Files

### Models
- `wr_ensemble_model.pkl` - Main production model (666KB)

### Results
- `wr_ensemble_predictions.csv` - Validation predictions

### Code
- `wr_full_pipeline.py` - 7-step training pipeline
- `wr_feature_engineering.py` - Feature engineering script

## Pipeline Steps

1. **Load Data** - WR games filtered from combined WR/TE dataset (INL% ≤ 15)
2. **Noise Removal** - K-NN outlier detection (remove top 15%)
3. **Feature Selection** - Top 35 features from importance ranking
4. **Tune XGBoost** - 150 trials with Optuna
5. **Tune LightGBM** - 150 trials with Optuna
6. **Tune CatBoost** - 150 trials with Optuna
7. **Stacking Ensemble** - Ridge meta-learner with 5-fold CV

## Usage

```python
import pickle
import pandas as pd

# Load model
with open('wr_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X)
```

## Training Data

- **12,836 WR games** (2012-2024)
- **383 unique WRs**
- Train: Pre-2022, Val: 2022, Test: 2023-2024
- Position separation: INL% ≤ 15 (WRs line up wide/slot, not inline)

## Feature Engineering Categories

1. **Rolling Averages** (10 features) - last_game_ppr, last_3_games_ppr, last_5_games_ppr, season_avg_ppr, etc.
2. **Career History** (7 features) - career_avg_ppr, career_games, career_ppr_std, is_wr1, is_wr2, etc.
3. **Team Performance** (5 features) - team_win_rate, team_point_diff, opp_win_rate, etc.
4. **Situational** (10 features) - is_home, is_short_week, season_phase, age_group, etc.
5. **Interactions** (10 features) - tgt_x_rec_rate, recv_grade_x_tgt, alignment_diversity, etc.
6. **Advanced** (8 features) - boom_rate, bust_rate, value_over_replacement, is_alpha_wr, is_deep_threat, etc.

## Requirements

- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- optuna (for hyperparameter tuning)
- tqdm (progress bars)

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna tqdm
```

## Notes

- WR position separated using INL% ≤ 15 threshold (TEs have higher inline percentage)
- Model trained separately from TE model for position-specific optimization
- See `wr_te_combined_model/` for combined WR/TE approach

**Author**: Minjun | **Date**: November 2025
