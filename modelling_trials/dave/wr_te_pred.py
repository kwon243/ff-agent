"""
wrte_model_v2.py
Predicts weekly WR/TE fantasy PPR using multiple nonlinear and linear regressors,
with an ensemble blending layer.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, explained_variance_score
from sklearn.linear_model import ElasticNetCV, BayesianRidge
from sklearn.svm import SVR

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ModuleNotFoundError:
    HAS_LGBM = False

CSV = "final_wr_and_te_data.csv"
TARGET = "PPR"
TARGET_YEAR = 2024
NOISE_Z = 3.0
STATE = 42
use_features = True

top20_features = {
    "RK","POSITION_RANK","average_rank","best_rank","worst_rank","rank_variance","TIERS",
    "game_num","Age","ADOT","DRP","DRP%","DROP","YAC/REC","MTF","RTG","INL%",
    "PRSH_opponent","PBLK_opponent","RBLK_opponent","RUN_opponent",
    "COV_opponent","PA_opponent","PASS_opponent","RDEF_opponent","TACK_opponent",
    "DEF_opponent","PF_opponent","OFF_opponent",
    "PPR_roll3_mean","PPR_roll5_mean","games_played_to_date"
}

PAIR_PREFS = [("COV_opponent","DEF_opponent")]

def rolling_feats(df, player_col, year_col, week_col, target_col, windows=(3,5)):
    df = df.sort_values([player_col, year_col, week_col]).copy()
    g = df.groupby(player_col, group_keys=False)
    for w in windows:
        df[f"{target_col}_roll{w}_mean"] = g[target_col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"{target_col}_roll{w}_std"]  = g[target_col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).std())
    df["games_played_to_date"] = g.cumcount()
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
    keep = z_score <= z
    return df[keep].copy(), int((~keep).sum())

def apply_features(X, idlike, use_features=True):
    if not use_features:
        return X
    keep = set(c for c in X.columns if (c in top20_features) or (c in idlike))
    for a, b in PAIR_PREFS:
        if a in X.columns:
            keep.add(a); keep.discard(b)
        elif b in X.columns:
            keep.add(b)
    cols = [c for c in X.columns if c in keep]
    return X if not cols else X[cols]

def build_preprocess(X, idlike, use_features=True):
    X = apply_features(X, idlike, use_features)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in idlike]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c not in idlike]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    if len(cat_cols):
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    else:
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])
    prep = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
    return prep, num_cols, cat_cols

def rmse_safe(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate_results(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = rmse_safe(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    evs  = explained_variance_score(y_true, y_pred)
    prefix = f"{label}: " if label else ""
    print(f"{prefix}MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}, R²: {r2:.3f}, EVS: {evs:.3f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "EVS": evs}


df = pd.read_csv(r"C:\Users\dpatel\Downloads\final_wr_and_te_data.csv" )
year_col, player_col, week_col = "YEAR", "PFR_ID", "Week"

drop_cols = ["Unnamed: 0","PFFPlayerKey","FantasyProsPlayerKey","PFFTeamKey"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df = df.dropna(subset=[TARGET]).copy()

df = rolling_feats(df, player_col, year_col, week_col, TARGET, windows=(3,5))
train_df = df[df[year_col] < TARGET_YEAR].copy()
pred_df  = df[df[year_col] == TARGET_YEAR].copy()

train_df, dropped = rem_noise(train_df, TARGET, year_col, z=NOISE_Z)
if dropped:
    print(f"Removed {dropped} noisy rows from training (z > {NOISE_Z}).")

idlike = {player_col, week_col, year_col}
weeks = sorted(pred_df[week_col].unique())
all_rows = []

for wk in weeks:
    hist = pred_df[pred_df[week_col] < wk]
    comb = pd.concat([train_df, hist], ignore_index=True)
    X_train = apply_features(comb.drop(columns=[TARGET]), idlike, use_features)
    y_train = comb[TARGET].values
    X_test  = apply_features(pred_df[pred_df[week_col] == wk].drop(columns=[TARGET]), idlike, use_features)
    target_rows = pred_df[pred_df[week_col] == wk].copy()

    prep, _, _ = build_preprocess(X_train, idlike, use_features)
    w = make_year_weights(comb[year_col], TARGET_YEAR)

    # Define diverse models
    models = {
        "ElasticNetCV": Pipeline([("prep", prep),
            ("model", ElasticNetCV(l1_ratio=[.1,.5,.9], alphas=np.logspace(-3,1,20), max_iter=5000, cv=5, random_state=STATE))]),
        "SVR_RBF": Pipeline([("prep", prep),
            ("model", SVR(kernel="rbf", C=1.5, epsilon=0.1))]),
        "CatBoost": Pipeline([("prep", prep),
            ("model", CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8, loss_function="RMSE", verbose=False, random_state=STATE))])
    }

    if HAS_LGBM:
        models["LightGBM"] = Pipeline([("prep", prep),
            ("model", LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                    colsample_bytree=0.8, random_state=STATE, n_jobs=-1))])

    # Train + predict
    for mname, pipe in models.items():
        try:
            pipe.fit(X_train, y_train, model__sample_weight=w)
        except Exception:
            pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        target_rows[f"{mname}_pred"] = preds
        evaluate_results(target_rows[TARGET], preds, label=f"Week {wk:>2} {mname}")

    # --- Ensemble blending ---
    blend_cols = [c for c in target_rows.columns if c.endswith("_pred")]
    blended = target_rows[blend_cols].mean(axis=1)
    # Meta smoothing with Bayesian Ridge
    meta = BayesianRidge()
    meta.fit(target_rows[blend_cols], target_rows[TARGET])
    target_rows["final_pred"] = meta.predict(target_rows[blend_cols])

    all_rows.append(target_rows)


out = pd.concat(all_rows, ignore_index=True).sort_values([player_col, week_col])
print("\n===== 2024 Blended Final Evaluation =====")
evaluate_results(out[TARGET], out["final_pred"], label="BlendedFinal")

# Per-model summary
season_rows = []
for c in out.columns:
    if c.endswith("_pred"):
        season_rows.append({
            "model": c.replace("_pred",""),
            "MAE": mean_absolute_error(out[TARGET], out[c]),
            "RMSE": rmse_safe(out[TARGET], out[c]),
            "R2": r2_score(out[TARGET], out[c]),
            "EVS": explained_variance_score(out[TARGET], out[c])
        })
print("\n===== FULL-SEASON MODEL PERFORMANCE =====")
print(pd.DataFrame(season_rows).sort_values("R2", ascending=False).to_string(index=False))

# Baseline
y_all = out[TARGET].to_numpy()
y_mean = np.full_like(y_all, fill_value=y_all.mean(), dtype=float)
baseline_mae = mean_absolute_error(y_all, y_mean)
baseline_rmse = rmse_safe(y_all, y_mean)
print(f"\nBaseline (mean predictor): MAE={baseline_mae:.3f}, RMSE={baseline_rmse:.3f}")

# Save results
save_cols = [year_col, player_col, week_col, TARGET, "final_pred"] + [c for c in out.columns if c.endswith("_pred")]
print("\nSample WR/TE weekly predictions:")
print(out[save_cols].head(10).round(3).to_string(index=False))
out[save_cols].to_csv(f"wrte_predictions_{TARGET_YEAR}_weekly_v2.csv", index=False)
print(f"\nSaved wrte_predictions_{TARGET_YEAR}_weekly_v2.csv")
