"""
MINJUN'S FULL PIPELINE - QB PREDICTION MODEL
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
print("MINJUN'S FULL PIPELINE - QB MODEL")
print("Target: Beat Baseline by 40%+")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("[1/7] Loading engineered data...")
df = pd.read_csv('qb_data_engineered.csv')

# Load feature importance for selection
feat_imp = pd.read_csv('feature_importance.csv')
top_features = feat_imp.head(50)['feature'].tolist()  # Top 50 features

print(f"   ✓ Loaded {len(df)} games")
print(f"   ✓ Using top 50 features")

# Split data
train_df = df[df['YEAR'] < 2022].copy()
val_df = df[df['YEAR'] == 2022].copy()
test_df = df[df['YEAR'] > 2022].copy()

print(f"   ✓ Train: {len(train_df)} games")
print(f"   ✓ Val:   {len(val_df)} games")
print(f"   ✓ Test:  {len(test_df)} games")

# ============================================================================
# STEP 2: NOISE REMOVAL (Like Baseline does!)
# ============================================================================
print("\n[2/7] Removing noisy samples (K-NN outlier detection)...")

def find_noisy_samples(X, y, n_neighbors=10, threshold=1.5):
    """
    Find noisy samples using K-NN
    Similar to what Baseline does in his notebook
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    noise_scores = []
    for i in range(len(X)):
        neighbor_indices = indices[i][1:]  # Exclude self
        neighbor_targets = y[neighbor_indices]

        target_std = np.std(np.append(neighbor_targets, y[i]))
        target_diff = np.abs(y[i] - np.mean(neighbor_targets))

        noise_scores.append(target_diff / (target_std + 1e-10))

    noise_scores = np.array(noise_scores)
    noisy_mask = noise_scores > threshold

    return ~noisy_mask  # Return clean mask

# Prepare features
y_train = train_df['PPR'].values
X_train = train_df[top_features].values

# Find clean samples
clean_mask = find_noisy_samples(X_train, y_train, n_neighbors=10, threshold=1.5)

# Filter training data
X_train_clean = X_train[clean_mask]
y_train_clean = y_train[clean_mask]
train_df_clean = train_df[clean_mask]

removed = len(X_train) - len(X_train_clean)
print(f"   ✓ Removed {removed} noisy samples ({removed/len(X_train)*100:.1f}%)")
print(f"   ✓ Clean training set: {len(X_train_clean)} samples")

# Validation and test (no noise removal)
X_val = val_df[top_features].values
y_val = val_df['PPR'].values
X_test = test_df[top_features].values
y_test = test_df['PPR'].values

# ============================================================================
# STEP 3: FEATURE SELECTION (Keep best 35 features)
# ============================================================================
print("\n[3/7] Feature selection (keeping top 35 features)...")

# Use top 35 most important features
final_features = top_features[:35]

X_train_final = X_train_clean[:, :35]
X_val_final = X_val[:, :35]
X_test_final = X_test[:, :35]

print(f"   ✓ Selected {len(final_features)} features")
print(f"   📊 Top 10 features:")
for i, feat in enumerate(final_features[:10], 1):
    imp = feat_imp[feat_imp['feature'] == feat]['importance'].values[0]
    is_new = feat_imp[feat_imp['feature'] == feat]['is_new'].values[0]
    marker = "🆕" if is_new == "NEW" else "📌"
    print(f"      {i:2d}. {marker} {feat:30s} ({imp:.4f})")

# ============================================================================
# STEP 4: BUILD & TUNE XGBOOST
# ============================================================================
print("\n[4/7] Building and tuning XGBoost...")

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_xgboost(X_tr, y_tr, X_va, y_va, n_trials=150):
    """Optimize XGBoost hyperparameters"""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        y_pred = model.predict(X_va)
        mae = mean_absolute_error(y_va, y_pred)
        return mae

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

    with tqdm(total=n_trials, desc="   XGBoost", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best MAE': f'{study.best_value:.2f}'})

        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    return study.best_params

best_xgb_params = optimize_xgboost(X_train_final, y_train_clean, X_val_final, y_val, n_trials=150)
xgb_model = xgb.XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_final, y_train_clean, eval_set=[(X_val_final, y_val)], verbose=False)

xgb_val_pred = xgb_model.predict(X_val_final)
xgb_val_r2 = r2_score(y_val, xgb_val_pred)
xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)

print(f"   ✅ XGBoost: Val R² = {xgb_val_r2:.4f}, MAE = {xgb_val_mae:.2f}")

# ============================================================================
# STEP 5: BUILD & TUNE LIGHTGBM
# ============================================================================
print("\n[5/7] Building and tuning LightGBM...")

def optimize_lightgbm(X_tr, y_tr, X_va, y_va, n_trials=150):
    """Optimize LightGBM hyperparameters"""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        y_pred = model.predict(X_va)
        mae = mean_absolute_error(y_va, y_pred)
        return mae

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=43))

    with tqdm(total=n_trials, desc="   LightGBM", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best MAE': f'{study.best_value:.2f}'})

        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    return study.best_params

best_lgb_params = optimize_lightgbm(X_train_final, y_train_clean, X_val_final, y_val, n_trials=150)
lgb_model = lgb.LGBMRegressor(**best_lgb_params, random_state=42, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train_final, y_train_clean, eval_set=[(X_val_final, y_val)])

lgb_val_pred = lgb_model.predict(X_val_final)
lgb_val_r2 = r2_score(y_val, lgb_val_pred)
lgb_val_mae = mean_absolute_error(y_val, lgb_val_pred)

print(f"   ✅ LightGBM: Val R² = {lgb_val_r2:.4f}, MAE = {lgb_val_mae:.2f}")

# ============================================================================
# STEP 6: BUILD & TUNE CATBOOST
# ============================================================================
print("\n[6/7] Building and tuning CatBoost...")

def optimize_catboost(X_tr, y_tr, X_va, y_va, n_trials=150):
    """Optimize CatBoost hyperparameters"""

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 800),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'verbose': False
        }

        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

        y_pred = model.predict(X_va)
        mae = mean_absolute_error(y_va, y_pred)
        return mae

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=44))

    with tqdm(total=n_trials, desc="   CatBoost", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'Best MAE': f'{study.best_value:.2f}'})

        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)

    return study.best_params

best_cat_params = optimize_catboost(X_train_final, y_train_clean, X_val_final, y_val, n_trials=150)
cat_model = CatBoostRegressor(**best_cat_params, random_seed=42, verbose=False)
cat_model.fit(X_train_final, y_train_clean, eval_set=(X_val_final, y_val), verbose=False)

cat_val_pred = cat_model.predict(X_val_final)
cat_val_r2 = r2_score(y_val, cat_val_pred)
cat_val_mae = mean_absolute_error(y_val, cat_val_pred)

print(f"   ✅ CatBoost: Val R² = {cat_val_r2:.4f}, MAE = {cat_val_mae:.2f}")

# ============================================================================
# STEP 7: STACKING ENSEMBLE
# ============================================================================
print("\n[7/7] Building stacking ensemble...")

# Create base models
base_models = [
    ('xgb', xgb.XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)),
    ('lgb', lgb.LGBMRegressor(**best_lgb_params, random_state=42, n_jobs=-1, verbose=-1)),
    ('cat', CatBoostRegressor(**best_cat_params, random_seed=42, verbose=False))
]

# Meta-learner (Ridge regression)
meta_model = Ridge(alpha=1.0)

# Create stacking regressor
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

print("   Training stacking ensemble...")
stacking_model.fit(X_train_final, y_train_clean)

# Predictions
stack_train_pred = stacking_model.predict(X_train_final)
stack_val_pred = stacking_model.predict(X_val_final)
stack_test_pred = stacking_model.predict(X_test_final)

# Metrics
stack_train_r2 = r2_score(y_train_clean, stack_train_pred)
stack_train_mae = mean_absolute_error(y_train_clean, stack_train_pred)

stack_val_r2 = r2_score(y_val, stack_val_pred)
stack_val_mae = mean_absolute_error(y_val, stack_val_pred)

stack_test_r2 = r2_score(y_test, stack_test_pred)
stack_test_mae = mean_absolute_error(y_test, stack_test_pred)

print(f"   ✅ Ensemble: Val R² = {stack_val_r2:.4f}, MAE = {stack_val_mae:.2f}")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print("\n📊 INDIVIDUAL MODELS (Validation):")
print(f"   XGBoost:  R² = {xgb_val_r2:.4f}, MAE = {xgb_val_mae:.2f}")
print(f"   LightGBM: R² = {lgb_val_r2:.4f}, MAE = {lgb_val_mae:.2f}")
print(f"   CatBoost: R² = {cat_val_r2:.4f}, MAE = {cat_val_mae:.2f}")

print("\n🏆 STACKING ENSEMBLE:")
print(f"\n   Training:")
print(f"      R²:   {stack_train_r2:.4f}")
print(f"      MAE:  {stack_train_mae:.2f}")

print(f"\n   Validation:")
print(f"      R²:   {stack_val_r2:.4f}")
print(f"      MAE:  {stack_val_mae:.2f}")

print(f"\n   Test:")
print(f"      R²:   {stack_test_r2:.4f}")
print(f"      MAE:  {stack_test_mae:.2f}")

# Compare to Baseline
baseline_r2 = 0.1546
baseline_mae = 5.79

print("\n" + "="*80)
print("⚔️  VS BASELINE")
print("="*80)

print(f"\n   Baseline (Kernel Ridge):")
print(f"      Val R²:  {baseline_r2:.4f}")
print(f"      Val MAE: {baseline_mae:.2f}")

print(f"\n   Minjun (Stacking Ensemble):")
print(f"      Val R²:  {stack_val_r2:.4f}")
print(f"      Val MAE: {stack_val_mae:.2f}")

r2_improvement = stack_val_r2 - baseline_r2
r2_pct = (stack_val_r2 / baseline_r2 - 1) * 100
mae_improvement = stack_val_mae - baseline_mae
mae_pct = (stack_val_mae / baseline_mae - 1) * 100

print(f"\n   📈 Improvement:")
print(f"      R² delta:  {r2_improvement:+.4f} ({r2_pct:+.1f}%)")
print(f"      MAE delta: {mae_improvement:+.2f} ({mae_pct:+.1f}%)")

if stack_val_r2 > baseline_r2 and stack_val_mae < baseline_mae:
    print("\n" + "🎉"*20)
    print("🎉 SUCCESS! MODEL IMPROVED! 🎉")
    print("🎉"*20)
elif stack_val_r2 > baseline_r2:
    print("\n✅ WE BEAT BASELINE ON R²!")
elif stack_val_mae < baseline_mae:
    print("\n✅ WE BEAT BASELINE ON MAE!")
else:
    print("\n💪 Close fight! Ensemble is competitive!")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# Save ensemble
with open('qb_ensemble_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)
print("\n✅ Saved: qb_ensemble_model.pkl")

# Save individual models
with open('qb_xgb_final.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
with open('qb_lgb_final.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)
with open('qb_cat_final.pkl', 'wb') as f:
    pickle.dump(cat_model, f)
print("✅ Saved: Individual models")

# Save predictions
results_df = pd.DataFrame({
    'YEAR': val_df['YEAR'].values,
    'Week': val_df['Week'].values,
    'PFR_ID': val_df['PFR_ID'].values,
    'Actual_PPR': y_val,
    'Ensemble_Pred': stack_val_pred,
    'XGB_Pred': xgb_val_pred,
    'LGB_Pred': lgb_val_pred,
    'CAT_Pred': cat_val_pred,
    'Error': y_val - stack_val_pred,
    'Abs_Error': np.abs(y_val - stack_val_pred)
})
results_df.to_csv('qb_ensemble_predictions.csv', index=False)
print("✅ Saved: qb_ensemble_predictions.csv")

# Save final features
with open('qb_final_features.txt', 'w') as f:
    f.write("Final 35 Features Used:\n")
    f.write("="*50 + "\n")
    for i, feat in enumerate(final_features, 1):
        f.write(f"{i:2d}. {feat}\n")
print("✅ Saved: qb_final_features.txt")

print("\n" + "="*80)
print("🏁 PIPELINE COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
