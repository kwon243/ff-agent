# RB Fantasy Football Prediction Model

Ensemble machine learning model for predicting weekly RB PPR fantasy points.

## Model Performance

**Validation Set (2022 Season):**
- **R² Score: 0.2866** (vs Baseline: 0.2553)
- **MAE: 5.00** (vs Baseline: 5.23)
- **RMSE: 6.88** (vs Baseline: 7.18)
- **Improvement: +12.3%** over baseline

**Test Set (2023-2024):**
- **R² Score: 0.2618**
- **MAE: 5.33**
- **RMSE: 7.39**

RBs are more predictable than QBs due to more consistent role and usage patterns.

## Model Architecture

**Stacking Ensemble** combining:
1. XGBoost Regressor
2. LightGBM Regressor
3. CatBoost Regressor
4. Ridge meta-learner (5-fold CV)

Each base model was tuned with Optuna (150 trials, 450 total).

## Top Features

The 10 most important features for RB prediction:

| Feature | Importance | Type |
|---------|-----------|------|
| last_5_games_ppr | 14.6% | Rolling Average |
| SNP | 11.2% | Snap Count |
| last_3_games_ppr | 10.8% | Rolling Average |
| ATT | 8.4% | Rushing Attempts |
| season_avg_ppr | 6.9% | Season Stats |
| career_avg_ppr | 5.7% | Career Stats |
| YDS | 4.3% | Rushing Yards |
| last_game_ppr | 3.8% | Recent Form |
| TGT | 3.2% | Receiving Targets |
| REC | 2.9% | Receptions |

**Key Insights:**
- Recent performance (last 3-5 games) accounts for 25%+ of prediction power
- Volume metrics (SNP, ATT, TGT) are critical (23%+ combined)
- Career history provides stability to predictions
- Receiving work (TGT, REC) important for PPR scoring

## Training Data

- **12,420 RB games** from 2018-2024
- **1,789 unique RBs**
- **52 engineered features** from 81 original features
- **Noise removal**: 14.2% outliers removed with K-NN

## Feature Engineering

Created 52 new features across 6 categories:

1. **Rolling Averages** (10 features): last_3/5_games_ppr, last_3_games_yds/td/rec
2. **Career History** (6 features): career_avg_ppr, career_games, career_td_rate
3. **Team Performance** (5 features): team_win_rate, team_ppg, opp_win_rate
4. **Situational** (10 features): is_home, is_short_week, season_phase, age_group
5. **Interactions** (9 features): td_x_yds, att_x_ypa, catch_rate
6. **Advanced** (8 features): boom_rate, bust_rate, value_over_replacement, is_workhorse

New features account for 62% of total model importance.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run feature engineering
python rb_feature_engineering.py

# Run full pipeline (training + evaluation)
python rb_full_pipeline.py
```

## Pipeline Steps

1. **Load Data**: Read RB dataset (12,420 games)
2. **Noise Removal**: K-NN outlier detection (removes 14.2%)
3. **Feature Selection**: Top 35 features via Random Forest
4. **XGBoost Tuning**: Optuna optimization (150 trials)
5. **LightGBM Tuning**: Optuna optimization (150 trials)
6. **CatBoost Tuning**: Optuna optimization (150 trials)
7. **Stacking Ensemble**: Ridge meta-learner with 5-fold CV

Total training time: ~45 minutes on modern CPU.

## Files

- `rb_ensemble_model.pkl` - Production ensemble model
- `rb_full_pipeline.py` - Complete training pipeline
- `rb_feature_engineering.py` - Feature creation script
- `rb_ensemble_predictions.csv` - Validation set predictions
- `requirements.txt` - Python dependencies

## Model Insights

**Why RBs are More Predictable:**
- Volume-based scoring (carries, targets)
- More consistent weekly usage
- Fewer game-script variations
- Clearer role definitions (workhorse vs committee)

**Key Predictors:**
- Recent PPR performance (last 3-5 games)
- Snap count and touch volume
- Receiving involvement (crucial for PPR)
- Team offensive strength

**Limitations:**
- Injuries not captured in historical data
- Game script changes unpredictable
- Coaching changes affect usage
- Committee backfields harder to predict
