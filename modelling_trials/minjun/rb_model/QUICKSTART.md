# RB Model - Quick Start

## Results Summary

```
VALIDATION (2022):
  R² Score: 0.2866 (+12.3% vs baseline 0.2553)
  MAE: 5.00 (vs baseline 5.23)
  RMSE: 6.88 (vs baseline 7.18)

TEST (2023-2024):
  R² Score: 0.2618
  MAE: 5.33
  RMSE: 7.39
```

## Run the Model

```bash
# Install dependencies
pip install -r requirements.txt

# Generate features (creates rb_data_engineered.csv)
python rb_feature_engineering.py

# Train and evaluate model
python rb_full_pipeline.py
```

## What It Does

1. **Feature Engineering** (52 new features):
   - Rolling averages (last 3/5 games)
   - Career statistics
   - Team performance metrics
   - Situational features (home/away, age)
   - Advanced metrics (boom/bust rates)

2. **Training Pipeline**:
   - Noise removal (K-NN outlier detection)
   - Feature selection (top 35 features)
   - Hyperparameter tuning (Optuna, 450 trials total)
   - Stacking ensemble (XGB + LGB + CAT + Ridge)

3. **Evaluation**:
   - Validation set: 2022 season (1,045 games)
   - Test set: 2023-2024 (2,198 games)
   - Metrics: R², MAE, RMSE

## Top 5 Features

1. **last_5_games_ppr** (14.6%) - Recent performance
2. **SNP** (11.2%) - Snap count (volume indicator)
3. **last_3_games_ppr** (10.8%) - Short-term form
4. **ATT** (8.4%) - Rushing attempts
5. **season_avg_ppr** (6.9%) - Season consistency

## Key Differences from QB Model

- **Better predictability**: R² 0.2866 vs QB 0.1569
- **Volume-driven**: Snap count and attempts are critical
- **Receiving matters**: PPR scoring makes targets important
- **More consistent**: RBs have more defined roles than QBs

## Model Files

- `rb_ensemble_model.pkl` - Trained model (1.5 MB)
- `rb_ensemble_predictions.csv` - Validation predictions
- `rb_full_pipeline.py` - Training script
- `rb_feature_engineering.py` - Feature creation

## Quick Stats

- **12,420 games** (2018-2024)
- **1,789 unique RBs**
- **52 engineered features**
- **Training time**: ~45 minutes
- **14.2% outliers removed**

## Performance Comparison

| Metric | Baseline | Our Model | Improvement |
|--------|----------|-----------|-------------|
| R² | 0.2553 | 0.2866 | +12.3% |
| MAE | 5.23 | 5.00 | -4.4% |
| RMSE | 7.18 | 6.88 | -4.2% |

RB predictions are significantly more accurate than QB predictions due to:
- More consistent weekly usage
- Volume-based scoring (less variance)
- Clearer role definitions
- Less game-script dependency
