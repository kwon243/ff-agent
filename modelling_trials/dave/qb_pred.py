"""
qb_model_v2.py
Predicts weekly QB fantasy performance (PPR) using stacked models with neural and Bayesian regression.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

TARGET = "PPR"
TARGET_YEAR = 2024
NOISE_Z = 3
state = 42

def rolling_feats(df, player_col, year_col, week_col, target_col, windows=(3,5)):
    df = df.sort_values([player_col, year_col, week_col]).copy()
    d = df.groupby(player_col, group_keys=False)
    for w in windows:
        df[f"{target_col}_roll{w}_mean"] = d[target_col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"{target_col}_roll{w}_std"] = d[target_col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).std())
    df["games_played_to_date"] = d.cumcount()
    return df

def make_year_weights(year_series, target_year, floor=0.2):
    gap = (target_year - year_series).clip(lower=1)
    weights = 1.0 / gap.astype(float)
    return (weights * (1 - floor)) + floor

def rem_noise(df, target_col, year_col, z=3.0):
    def robust_z(s):
        med = np.median(s)
        mad = np.median(np.abs(s - med)) + 1e-6
        return 0.6745 * (s - med) / mad
    z_score = df.groupby(year_col)[target_col].transform(robust_z).abs()
    return df[z_score <= z].copy(), int((z_score > z).sum())

def evaluate_results(y_true, y_pred):
    # Metrics that work across sklearn versions
    mae = mean_absolute_error(y_true, y_pred)
    
    # RMSE without using the `squared` kwarg
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    
    # Safe MAPE via sklearn (ok if some y_true are zero; may yield inf which is informative)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    r2 = r2_score(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)

    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}, R²: {r2:.3f}, EVS: {ev:.3f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "EVS": ev}


df = pd.read_csv(r"C:\Users\dpatel\Downloads\final_qb_data.csv")
df.drop(columns=[c for c in ["Unnamed: 0", "PFFPlayerKey", "FantasyProsPlayerKey"] if c in df.columns], inplace=True, errors="ignore")
df = df.dropna(subset=[TARGET])

year_col, player_col, week_col = "YEAR", "PFR_ID", "Week"
df = rolling_feats(df, player_col, year_col, week_col, TARGET, windows=(3,5))
df, dropped = rem_noise(df, TARGET, year_col, NOISE_Z)
if dropped: print(f"Removed {dropped} noisy rows.")

train_df = df[df[year_col] < TARGET_YEAR].copy()
pred_df = df[df[year_col] == TARGET_YEAR].copy()

num_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c]) and c not in {player_col, week_col, year_col, TARGET}]
cat_cols = [c for c in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[c]) and c not in {player_col, week_col, year_col}]

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocess = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

base_learners = [
    ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.08, max_depth=5, random_state=state)),
    ("rf", RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=3, random_state=state, n_jobs=-1)),
    ("cat", CatBoostRegressor(verbose=0, iterations=400, depth=8, learning_rate=0.06, random_state=state))
]

meta_model = BayesianRidge()
stack_model = StackingRegressor(estimators=base_learners, final_estimator=meta_model, n_jobs=-1)

neural_net = MLPRegressor(hidden_layer_sizes=(128,64,32), activation="relu", solver="adam",
                          alpha=0.0005, max_iter=400, random_state=state, early_stopping=True)

models = {
    "StackedModel": Pipeline([("prep", preprocess), ("model", stack_model)]),
    "NeuralNet": Pipeline([("prep", preprocess), ("model", neural_net)]),
    "GradientBoosting": Pipeline([("prep", preprocess), ("model", GradientBoostingRegressor(random_state=state))])
}

weeks = sorted(pred_df[week_col].unique())
all_preds = []
weights = make_year_weights(train_df[year_col], TARGET_YEAR)

for week in weeks:
    hist = pred_df[pred_df[week_col] < week]
    combined_train = pd.concat([train_df, hist], ignore_index=True)
    X_train = combined_train.drop(columns=[TARGET])
    y_train = combined_train[TARGET].values
    target_rows = pred_df[pred_df[week_col] == week].copy()
    X_test = target_rows.drop(columns=[TARGET])

    print(f"\n===== Week {week} Training ({len(X_train)} rows) =====")
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        target_rows[f"{name}_pred"] = y_pred
        print(f"{name:15}: ", end="")
        evaluate_results(target_rows[TARGET], y_pred)
    all_preds.append(target_rows)

final_df = pd.concat(all_preds, ignore_index=True)
final_df["final_pred"] = final_df[[f"{m}_pred" for m in models]].mean(axis=1)


print("\n===== AGGREGATED 2024 PERFORMANCE =====")
evaluate_results(final_df[TARGET], final_df["final_pred"])

print("\nSample predictions:")
print(final_df[[player_col, week_col, TARGET, "final_pred"]].head(10))

final_df.to_csv(f"qb_predictions_{TARGET_YEAR}_weekly_v2.csv", index=False)
print(f"\nSaved qb_predictions_{TARGET_YEAR}_weekly_v2.csv")
