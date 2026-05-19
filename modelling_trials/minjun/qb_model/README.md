# QB Fantasy Football Prediction Model - Minjun

**Final Results: Beat Baseline Model's Benchmark **

## Model Performance

### Validation Results (2022 Season)
- **R²: 0.1569** (vs Baseline: 0.1546) - **+1.5% improvement**
- **MAE: 5.81** (vs Baseline: 5.79) - Essentially tied
- **RMSE: 7.41**

## Model Architecture

**Stacking Ensemble** combining:
1. XGBoost (R² 0.1565)
2. LightGBM (R² 0.1520)  
3. CatBoost (R² 0.1548)

## Top 10 Features (41% from new engineered features)

1. value_over_replacement (4.6%)
2. season_avg_ppr (4.6%)
3. career_avg_ppr (4.5%)
4. RK - draft rank (4.1%)
5. last_5_games_ppr (2.8%)
6. last_game_ppr (2.7%)
7. average_rank (2.6%)
8. last_3_games_ppr (2.4%)
9. best_rank (2.4%)
10. ppr_trend (2.2%)

## Files

### Models
- `qb_ensemble_model.pkl` - Main production model
- `qb_xgb_final.pkl` - XGBoost model
- `qb_lgb_final.pkl` - LightGBM model
- `qb_cat_final.pkl` - CatBoost model

### Results
- `qb_ensemble_predictions.csv` - Validation predictions
- `qb_final_features.txt` - Feature list
- `model_performance_comparison.png` - Visualizations

### Code
- `qb_full_pipeline.py` - Training pipeline
- `qb_feature_engineering.py` - Feature engineering

## Usage

```python
import pickle

# Load model
with open('qb_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X)
```

## Training Data
- 5,569 QB games (2012-2024)
- 122 unique QBs
- Train: Pre-2022, Val: 2022, Test: Post-2022

## Requirements
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- optuna (for retraining)

**Author**: Minjun | **Date**: November 2025
