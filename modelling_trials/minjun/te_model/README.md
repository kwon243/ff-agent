# TE Fantasy Football Prediction Model - Minjun

**Final Results: Beat Baseline Model's Benchmark **

## Model Performance

### Validation Results (2022 Season)
- **R²: 0.2525** (vs Baseline: 0.2230) - **+13.2% improvement**
- **MAE: 3.95** (vs Baseline: 4.27)
- **RMSE: 5.18**

### Test Results (2023-2024)
- **R²: 0.2599**
- **MAE: 3.91**
- **RMSE: 5.03**

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
- TE-specific features (is_elite_te, is_dual_threat, inline_pct, pblk_grade_x_snaps)

**53 new engineered features** from 89 original features

## Files

### Models
- `te_ensemble_model.pkl` - Main production model (713KB)

### Results
- `te_ensemble_predictions.csv` - Validation predictions

### Code
- `te_full_pipeline.py` - 7-step training pipeline
- `te_feature_engineering.py` - Feature engineering script

## Pipeline Steps

1. **Load Data** - TE games filtered from combined WR/TE dataset (INL% > 15)
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
with open('te_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X)
```

## Training Data

- **5,723 TE games** (2012-2024)
- **196 unique TEs**
- Train: Pre-2022, Val: 2022, Test: 2023-2024
- Position separation: INL% > 15 (TEs line up inline, WRs don't)

## Feature Engineering Categories

1. **Rolling Averages** (10 features) - last_game_ppr, last_3_games_ppr, last_5_games_ppr, season_avg_ppr, etc.
2. **Career History** (7 features) - career_avg_ppr, career_games, career_ppr_std, is_te1, is_te2, etc.
3. **Team Performance** (5 features) - team_win_rate, team_point_diff, opp_win_rate, etc.
4. **Situational** (10 features) - is_home, is_short_week, season_phase, age_group, etc.
5. **Interactions** (11 features) - tgt_x_rec_rate, recv_grade_x_tgt, pblk_grade_x_snaps, inline_pct, etc.
6. **Advanced** (8 features) - boom_rate, bust_rate, value_over_replacement, is_elite_te, is_dual_threat, etc.

## TE-Specific Features

- **is_dual_threat**: TEs who both block (PBLK_SNAPS > 10) and receive (TGT > 3)
- **is_elite_te**: TE1s with high target share (top 25% targets)
- **inline_pct**: Inline usage percentage (key differentiator from WRs)
- **pblk_grade_x_snaps**: Pass blocking effectiveness metric
- **Age groups**: TEs peak later (24-29) compared to WRs

## Requirements

- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- optuna (for hyperparameter tuning)
- tqdm (progress bars)

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna tqdm
```

## Notes

- TE position separated using INL% > 15 threshold (TEs line up inline, WRs don't)
- Model trained separately from WR model for position-specific optimization
- TEs have unique features: blocking metrics, different age curves, lower boom threshold (15 vs 20 PPR)
- See `wr_te_combined_model/` for combined WR/TE approach

**Author**: Minjun | **Date**: November 2025
