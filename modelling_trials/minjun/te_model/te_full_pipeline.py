"""
MINJUN'S FULL PIPELINE - TE PREDICTION MODEL
Complete pipeline with:
- Noise removal
- Feature selection
- Multiple algorithms (XGBoost, LightGBM, CatBoost)
- Hyperparameter tuning
- Stacking ensemble
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
import optuna
from tqdm import tqdm

print("="*80)
print("MINJUN'S FULL PIPELINE - TE MODEL")
print("Target: Beat Baseline Model")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("[1/7] Loading engineered data...")
df = pd.read_csv('te_data_engineered.csv')

# Load feature importance for selection
feat_imp = pd.read_csv('te_feature_importance.csv')
top_features = feat_imp.head(50)['feature'].tolist()  # Top 50 features

print(f"   ✓ Loaded {len(df)} games")
print(f"   ✓ Using top 50 features")

# Split data
train_df = df[df['YEAR'] < 2022].copy()
val_df = df[df['YEAR'] == 2022].copy()
test_df = df[df['YEAR'] > 2022].copy()

print(f"   ✓ Train: {len(train_df)} games (pre-2022)")
print(f"   ✓ Val:   {len(val_df)} games (2022)")
print(f"   ✓ Test:  {len(test_df)} games (2023-2024)")

# Prepare features
X_train = train_df[top_features].values
y_train = train_df['PPR'].values
X_val = val_df[top_features].values
y_val = val_df['PPR'].values
X_test = test_df[top_features].values
y_test = test_df['PPR'].values

# ============================================================================
# STEP 2: NOISE REMOVAL
# ============================================================================
print("\n[2/7] Removing noise from training data...")

# K-NN outlier detection
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

knn = NearestNeighbors(n_neighbors=20)
knn.fit(X_train_scaled)
distances, _ = knn.kneighbors(X_train_scaled)
outlier_scores = distances.mean(axis=1)

threshold = np.percentile(outlier_scores, 85)  # Remove top 15% outliers
mask = outlier_scores < threshold

X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

removed_pct = (1 - mask.sum() / len(mask)) * 100
print(f"   ✓ Removed {len(mask) - mask.sum()} outliers ({removed_pct:.1f}%)")
print(f"   ✓ Clean training set: {len(X_train_clean)} games")

# ============================================================================
# STEP 3: FEATURE SELECTION
# ============================================================================
print("\n[3/7] Selecting top features...")

# Use top 35 features for final model
final_features = feat_imp.head(35)['feature'].tolist()
print(f"   ✓ Selected top 35 features")

# Re-prepare with selected features
X_train_final = train_df[final_features].values[mask]
X_val_final = val_df[final_features].values
X_test_final = test_df[final_features].values

# ============================================================================
# STEP 4: TUNE XGBOOST
# ============================================================================
print("\n[4/7] Tuning XGBoost...")

def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train_final, y_train_clean)
    preds = model.predict(X_val_final)
    return r2_score(y_val, preds)

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=150, show_progress_bar=True)

best_xgb = xgb.XGBRegressor(**study_xgb.best_params)
best_xgb.fit(X_train_final, y_train_clean)

xgb_val_preds = best_xgb.predict(X_val_final)
xgb_r2 = r2_score(y_val, xgb_val_preds)
print(f"   ✓ XGBoost R²: {xgb_r2:.4f}")

# ============================================================================
# STEP 5: TUNE LIGHTGBM
# ============================================================================
print("\n[5/7] Tuning LightGBM...")

def objective_lgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_final, y_train_clean)
    preds = model.predict(X_val_final)
    return r2_score(y_val, preds)

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=150, show_progress_bar=True)

best_lgb = lgb.LGBMRegressor(**study_lgb.best_params)
best_lgb.fit(X_train_final, y_train_clean)

lgb_val_preds = best_lgb.predict(X_val_final)
lgb_r2 = r2_score(y_val, lgb_val_preds)
print(f"   ✓ LightGBM R²: {lgb_r2:.4f}")

# ============================================================================
# STEP 6: TUNE CATBOOST
# ============================================================================
print("\n[6/7] Tuning CatBoost...")

def objective_cat(trial):
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'iterations': trial.suggest_int('iterations', 100, 500),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_state': 42,
        'verbose': False
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train_final, y_train_clean)
    preds = model.predict(X_val_final)
    return r2_score(y_val, preds)

study_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_cat.optimize(objective_cat, n_trials=150, show_progress_bar=True)

best_cat = CatBoostRegressor(**study_cat.best_params)
best_cat.fit(X_train_final, y_train_clean, verbose=False)

cat_val_preds = best_cat.predict(X_val_final)
cat_r2 = r2_score(y_val, cat_val_preds)
print(f"   ✓ CatBoost R²: {cat_r2:.4f}")

# ============================================================================
# STEP 7: STACKING ENSEMBLE
# ============================================================================
print("\n[7/7] Creating stacking ensemble...")

estimators = [
    ('xgb', best_xgb),
    ('lgb', best_lgb),
    ('cat', best_cat)
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

stacking_model.fit(X_train_final, y_train_clean)

# Validation predictions
ensemble_val_preds = stacking_model.predict(X_val_final)

# Metrics
ensemble_r2 = r2_score(y_val, ensemble_val_preds)
ensemble_mae = mean_absolute_error(y_val, ensemble_val_preds)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))

print(f"   ✓ Ensemble R²: {ensemble_r2:.4f}")
print(f"   ✓ Ensemble MAE: {ensemble_mae:.2f}")
print(f"   ✓ Ensemble RMSE: {ensemble_rmse:.2f}")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print("\n📊 VALIDATION SET (2022):")
print(f"   R² Score:  {ensemble_r2:.4f}")
print(f"   MAE:       {ensemble_mae:.2f}")
print(f"   RMSE:      {ensemble_rmse:.2f}")

# Test set
test_preds = stacking_model.predict(X_test_final)
test_r2 = r2_score(y_test, test_preds)
test_mae = mean_absolute_error(y_test, test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

print(f"\n📊 TEST SET (2023-2024):")
print(f"   R² Score:  {test_r2:.4f}")
print(f"   MAE:       {test_mae:.2f}")
print(f"   RMSE:      {test_rmse:.2f}")

# Baseline comparison (simple season average)
baseline_preds = val_df['season_avg_ppr'].fillna(val_df['PPR'].mean()).values
baseline_r2 = r2_score(y_val, baseline_preds)
baseline_mae = mean_absolute_error(y_val, baseline_preds)

print(f"\n⚔️  VS BASELINE:")
print(f"   Baseline R²:  {baseline_r2:.4f}")
print(f"   Baseline MAE: {baseline_mae:.2f}")
print(f"   Our R²:       {ensemble_r2:.4f}")
print(f"   Our MAE:      {ensemble_mae:.2f}")

improvement = ((ensemble_r2 - baseline_r2) / baseline_r2) * 100
print(f"\n   📈 R² Improvement: {improvement:+.1f}%")

if ensemble_r2 > baseline_r2:
    print("\n🎉 SUCCESS! MODEL IMPROVED! 🎉")
else:
    print("\n⚠️  Model did not beat baseline")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# Save individual models
with open('te_xgb_final.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)
print("✓ Saved: te_xgb_final.pkl")

with open('te_lgb_final.pkl', 'wb') as f:
    pickle.dump(best_lgb, f)
print("✓ Saved: te_lgb_final.pkl")

with open('te_cat_final.pkl', 'wb') as f:
    pickle.dump(best_cat, f)
print("✓ Saved: te_cat_final.pkl")

# Save stacking ensemble
with open('te_ensemble_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)
print("✓ Saved: te_ensemble_model.pkl")

# Save predictions
val_df['predicted_ppr'] = ensemble_val_preds
val_df[['PFR_ID', 'YEAR', 'Week', 'PPR', 'predicted_ppr']].to_csv('te_ensemble_predictions.csv', index=False)
print("✓ Saved: te_ensemble_predictions.csv")

# Save feature list
with open('te_final_features.txt', 'w') as f:
    for feat in final_features:
        f.write(feat + '\n')
print("✓ Saved: te_final_features.txt")

print("\n" + "="*80)
print("✅ PIPELINE COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
