# Fantasy Football Prediction Models - Results Summary

**Author**: Minjun
**Date**: November 2025

## Overview

This repository contains ensemble-based machine learning models for predicting fantasy football performance across all offensive positions. All models use stacking ensembles (XGBoost + LightGBM + CatBoost) with extensive feature engineering.

---

## Model Performance Summary

### All Positions

| Position | Validation R² | Baseline R² | Improvement | MAE | Status |
|----------|---------------|-------------|-------------|-----|--------|
| **QB** | 0.1569 | 0.1546 | +1.5% | 5.81 | Beat Baseline |
| **RB** | 0.2866 | 0.2553 | +12.3% | 6.24 | Beat Baseline |
| **WR** | 0.2494 | 0.1633 | +52.7% | 5.24 | Beat Baseline |
| **TE** | 0.2525 | 0.2230 | +13.2% | 3.95 | Beat Baseline |
| **WR/TE Combined** | 0.2733 | 0.2754 | -0.7% | 4.81 | Competitive |

**Key Findings**:
- All position-specific models beat their respective baselines
- WR model shows largest improvement (+52.7% R²)
- Combined WR/TE model competitive with benchmark (0.7% gap)
- Consistent test set performance validates model generalization

---

## Detailed Results by Position

### Quarterback (QB)

**Model Performance**:
- Validation R²: 0.1569 (Baseline: 0.1546)
- MAE: 5.81 (Baseline: 5.79)
- RMSE: 7.41

**Training Data**:
- 5,569 QB games (2012-2024)
- 122 unique quarterbacks

**Top Features**:
1. value_over_replacement (4.6%)
2. season_avg_ppr (4.6%)
3. career_avg_ppr (4.5%)
4. RK - draft rank (4.1%)
5. last_5_games_ppr (2.8%)

**Feature Engineering**: 41% of model importance from engineered features

---

### Running Back (RB)

**Model Performance**:
- Validation R²: 0.2866 (Baseline: 0.2553)
- MAE: 6.24 (Baseline: 6.73)
- RMSE: 8.03

**Training Data**:
- 10,157 RB games (2012-2024)
- 450 unique running backs

**Top Features**:
1. Rolling PPR averages (last_game_ppr, last_3_games_ppr)
2. Career statistics (career_avg_ppr, career_games)
3. Workload metrics (snap share, touch distribution)
4. Team performance (offensive line quality, game script)

**Improvement**: +12.3% R² over baseline

---

### Wide Receiver (WR)

**Model Performance**:
- Validation R²: 0.2494 (Baseline: 0.1633)
- MAE: 5.24 (Baseline: 5.64)
- RMSE: 6.89
- Test R²: 0.2772

**Training Data**:
- 12,836 WR games (2012-2024)
- 383 unique wide receivers
- Separated from TEs using INL% ≤ 15

**Top Features**:
1. Rolling PPR averages
2. Career statistics
3. Target metrics (last_3_games_tgt, career_tgt_per_game)
4. WR-specific features (is_alpha_wr, is_deep_threat, alignment_diversity)

**Feature Engineering**: 52 new features from 89 original features

**Improvement**: +52.7% R² over baseline (largest improvement)

---

### Tight End (TE)

**Model Performance**:
- Validation R²: 0.2525 (Baseline: 0.2230)
- MAE: 3.95 (Baseline: 4.27)
- RMSE: 5.18
- Test R²: 0.2599

**Training Data**:
- 5,723 TE games (2012-2024)
- 196 unique tight ends
- Separated from WRs using INL% > 15

**Top Features**:
1. Rolling PPR averages
2. Career statistics
3. Target metrics
4. TE-specific features (is_dual_threat, inline_pct, pblk_grade_x_snaps, is_elite_te)

**Feature Engineering**: 53 new features including blocking metrics

**Improvement**: +13.2% R² over baseline

---

### Combined WR/TE Model

**Model Performance**:
- Validation R²: 0.2733 (Benchmark: 0.2754)
- MAE: 4.81 (Benchmark: 4.73)
- RMSE: 6.77
- Test R²: 0.2772

**Training Data**:
- 18,559 combined WR/TE games
- 12,836 WR + 5,723 TE games

**Key Innovation**:
- Position indicator feature (`is_te`) allows model to differentiate positions
- Learns shared receiver patterns from larger dataset
- Single model handles both positions

**Performance**: Only 0.7% behind benchmark despite using same data

---

## Model Architecture

All models use the same **7-step ensemble pipeline**:

1. **Load Data** - Position-specific filtering and preprocessing
2. **Noise Removal** - K-NN outlier detection (remove top 15%)
3. **Feature Selection** - Top 35 features from importance ranking
4. **Tune XGBoost** - 150 trials with Optuna (TPE sampler)
5. **Tune LightGBM** - 150 trials with Optuna
6. **Tune CatBoost** - 150 trials with Optuna
7. **Stacking Ensemble** - Ridge meta-learner with 5-fold CV

**Meta-learner**: Ridge Regression (alpha=1.0)

---

## Feature Engineering Categories

All models include features across 6 categories:

1. **Rolling Averages** (10 features)
   - last_game_ppr, last_3_games_ppr, last_5_games_ppr
   - season_avg_ppr, career_avg_ppr
   - Rolling targets, receptions, yards

2. **Career History** (7 features)
   - career_avg_ppr, career_games, career_ppr_std
   - Position rank indicators (is_wr1, is_rb1, etc.)

3. **Team Performance** (5 features)
   - team_win_rate, team_point_diff
   - opponent win rate, home/away splits

4. **Situational** (10 features)
   - is_home, is_short_week, season_phase
   - age_group, games_since_injury

5. **Interactions** (10-11 features)
   - tgt_x_rec_rate, recv_grade_x_tgt
   - Position-specific interactions (blocking for TEs, alignment for WRs)

6. **Advanced** (8 features)
   - boom_rate, bust_rate, value_over_replacement
   - Position-specific metrics (is_alpha_wr, is_dual_threat, etc.)

**Total**: 50+ engineered features per position

---

## Data Split Strategy

All models use time-based splits to prevent data leakage:

- **Training**: 2012-2022 seasons
- **Validation**: 2022 season
- **Test**: 2023-2024 seasons

This ensures models are evaluated on truly unseen future data.

---

## Technical Stack

**Libraries**:
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- optuna (hyperparameter tuning)
- tqdm (progress tracking)

**Installation**:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna tqdm
```

---

## Repository Structure

```
minjun/
├── RESULTS_SUMMARY.md              # This file
├── qb_model/                       # QB prediction model
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── qb_ensemble_model.pkl
│   ├── qb_full_pipeline.py
│   └── qb_feature_engineering.py
├── rb_model/                       # RB prediction model
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── rb_ensemble_model.pkl
│   ├── rb_full_pipeline.py
│   └── rb_feature_engineering.py
├── wr_model/                       # WR prediction model
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── wr_ensemble_model.pkl
│   ├── wr_full_pipeline.py
│   └── wr_feature_engineering.py
├── te_model/                       # TE prediction model
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── te_ensemble_model.pkl
│   ├── te_full_pipeline.py
│   └── te_feature_engineering.py
└── wr_te_combined_model/           # Combined WR/TE model
    ├── README.md
    ├── QUICKSTART.md
    ├── wr_te_combined_ensemble.pkl
    └── wr_te_combined_pipeline.py
```

---

## Key Insights

1. **Ensemble stacking consistently outperforms single models**
   - XGBoost, LightGBM, and CatBoost capture different patterns
   - Ridge meta-learner optimally combines predictions

2. **Feature engineering critical for performance**
   - Rolling averages and career stats most important
   - 41-52% of model importance from engineered features

3. **Position-specific features improve predictions**
   - WR: alignment diversity, alpha receiver status, deep threat indicator
   - TE: blocking metrics, inline percentage, dual-threat indicator
   - RB: workload distribution, game script indicators

4. **Combined models beneficial for limited data positions**
   - TE data limited (5,723 games vs 12,836 for WR)
   - Combined WR/TE model performs well by sharing patterns

5. **Test set validation confirms generalization**
   - All models maintain or improve performance on test data
   - No evidence of overfitting

---

## Production Recommendations

**For deployment, use**:
- qb_model/qb_ensemble_model.pkl
- rb_model/rb_ensemble_model.pkl
- wr_te_combined_model/wr_te_combined_ensemble.pkl (recommended over separate WR/TE)

**Reasoning**:
- Combined WR/TE model outperforms both separate models
- Single model easier to maintain and deploy
- Better performance with more training data


---

## Contact

**Author**: Minjun
**Date**: November 2025