"""
rb_model_v2.py
RB weekly fantasy (PPR) prediction with:
  - ElasticNet (linear w/ L1+L2)
  - TweedieRegressor (compound Poisson-ish; handles zero-inflation-ish targets)
  - CatBoostRegressor (handles mixed features well)
  - LightGBM Quantile models for floor/median/ceiling bands (p20/p50/p80)

Outputs:
  - Week-by-week predictions per player (final_pred = blend of ElasticNet + CatBoost + Tweedie)
  - Quantile bands: p20, p50, p80 for risk-aware decisions
  - Season metrics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, explained_variance_score
from sklearn.linear_model import ElasticNet, TweedieRegressor

# --- Optional CatBoost with auto-install fallback (handy across machines) ---
try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    import sys, subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
        from catboost import CatBoostRegressor
    except Exception:
        CatBoostRegressor = None

# --- Optional LightGBM for quantile regression (skip gracefully if missing) ---
try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ModuleNotFoundError:
    _HAS_LGBM = False


CSV = "final_rb_data.csv"
TARGET = "PPR"
TARGET_YEAR = 2024
NOISE_Z = 3.0
STATE = 42

use_features = True

top20_features = {
    "RK","POSITION_RANK","average_rank","best_rank","worst_rank","rank_variance","TIERS","ecr_adp_gap",
    "game_num","Age","ATT","TD","1ST","DYDS","YDS","YCO","YCO/A","RSNP","SNP","BAY","D15+","10+","TGT",
    "PRSH_opponent","RUN_opponent","COV_opponent","RDEF_opponent","PBLK_opponent","RBLK_opponent",
    "PA_opponent","PASS_opponent","TACK_opponent","DEF_opponent","OVER_opponent","PF_opponent","OFF_opponent",
    "PPR_roll3_mean","PPR_roll5_mean","games_played_to_date"
}

PAIR_PREFS = [("RUN_opponent","OFF_opponent")]  # prefer RUN_opponent over OFF_opponent


def rolling_feats(df, player_col, year_col, week_col, target_col, windows=(3,5)):
    df = df.sort_values([player_col, year_col, week_col]).copy()
    g = df.groupby(player_col, group_keys=False)
    for w in windows:
        df[f"{target_col}_roll{w}_mean"] = g[target_col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"{target_col}_roll{w}_std"]  = g[target_col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).std())
    # (Keep your generic rolling opportunities if those cols exist)
    for c in ["PA", "YPA", "ADOT", "PRSH", "NFL", "SCR", "TTT"]:
        if c in df.columns:
            for w in windows:
                df[f"{c}_roll{w}_mean"] = g[c].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
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


df = pd.read_csv(r"C:\Users\dpatel\Downloads\final_rb_data.csv")

year_col = "YEAR"
player_col = "PFR_ID"
week_col = "Week"

if year_col is None: raise ValueError("Need YEAR column")
if player_col is None:
    player_col = "_anon_player"; df[player_col] = "player_" + df.index.astype(str)
if week_col is None:
    week_col = "_anon_week"; df[week_col] = df.groupby([player_col, year_col]).cumcount() + 1

drop_cols = ["Unnamed: 0","PFFPlayerKey","FantasyProsPlayerKey","PFFTeamKey"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df = df.dropna(subset=[TARGET]).copy()

df = rolling_feats(df, player_col, year_col, week_col, TARGET, windows=(3,5))

if TARGET_YEAR not in df[year_col].unique():
    raise ValueError(f"{TARGET_YEAR} not in {year_col}. Available: {sorted(df[year_col].unique())}")

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

    X_train_full = apply_features(comb.drop(columns=[TARGET]), idlike, use_features)
    y_train_full = comb[TARGET].values
    X_test_full  = apply_features(pred_df[pred_df[week_col] == wk].drop(columns=[TARGET]), idlike, use_features)
    target_rows  = pred_df[pred_df[week_col] == wk].copy()

    prep, _, _ = build_preprocess(X_train_full, idlike, use_features)
    w = make_year_weights(comb[year_col], TARGET_YEAR)

    # --- Define models ---
    models = {}

    # ElasticNet (robust linear baseline; handles collinearity and trims noise)
    models["ElasticNet"] = Pipeline([
        ("prep", prep),
        ("model", ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=2000, random_state=STATE))
    ])

    # Tweedie (Poisson-like; robust for skewed target with many low outcomes)
    models["Tweedie"] = Pipeline([
        ("prep", prep),
        ("model", TweedieRegressor(power=1.3, alpha=0.001, max_iter=1000))
    ])

    # CatBoost (if available)
    if CatBoostRegressor is not None:
        models["CatBoost"] = Pipeline([
            ("prep", prep),
            ("model", CatBoostRegressor(
                iterations=500, depth=8, learning_rate=0.06, loss_function="RMSE",
                random_state=STATE, verbose=False
            ))
        ])

    # Fit + predict each model; log quick metrics on this week's holdout
    for mname, pipe in models.items():
        # sample_weight only for estimators that accept it; try/except keeps it simple
        try:
            pipe.fit(X_train_full, y_train_full, model__sample_weight=w)
        except Exception:
            pipe.fit(X_train_full, y_train_full)

        preds = pipe.predict(X_test_full)
        target_rows[f"{mname}_pred"] = preds

        # Print week-local metrics (informational)
        evaluate_results(target_rows[TARGET], preds, label=f"Week {wk:>2} {mname}")

    # Simple blend for point prediction (robust across weeks): mean of available strong models
    blend_cols = [c for c in [f"ElasticNet_pred", f"Tweedie_pred", f"CatBoost_pred"] if c in target_rows.columns]
    target_rows["final_pred"] = target_rows[blend_cols].mean(axis=1)

    # --- Quantile LightGBM range (if LightGBM is available) ---
    # We fit 3 separate quantile models on the same preprocessed data (p20, p50, p80)
    if _HAS_LGBM:
        q_preds = {}
        for q in [0.2, 0.5, 0.8]:
            q_model = Pipeline([
                ("prep", prep),
                ("model", LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=63,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=STATE,
                    objective="quantile",
                    alpha=q
                ))
            ])
            try:
                q_model.fit(X_train_full, y_train_full, model__sample_weight=w)
            except Exception:
                q_model.fit(X_train_full, y_train_full)
            q_preds[q] = q_model.predict(X_test_full)

        target_rows["pred_p20"] = q_preds.get(0.2, np.nan)
        target_rows["pred_p50"] = q_preds.get(0.5, np.nan)  # median
        target_rows["pred_p80"] = q_preds.get(0.8, np.nan)
    else:
        target_rows["pred_p20"] = np.nan
        target_rows["pred_p50"] = np.nan
        target_rows["pred_p80"] = np.nan

    all_rows.append(target_rows)

out = pd.concat(all_rows, ignore_index=True).sort_values([player_col, week_col])

# Week-by-week evaluation of our blended "final_pred"
print("\n===== Strict-weekly evaluation on TARGET_YEAR (Blended Final) =====")
evaluate_results(out[TARGET], out["final_pred"], label="BlendedFinal")

# Per-model season metrics
print("\n===== FULL-SEASON MODEL PERFORMANCE (2024) =====")
season_rows = []
for col in ["ElasticNet_pred", "Tweedie_pred", "CatBoost_pred"]:
    if col in out.columns:
        season_rows.append({
            "model": col.replace("_pred",""),
            "MAE": mean_absolute_error(out[TARGET], out[col]),
            "RMSE": rmse_safe(out[TARGET], out[col]),
            "R2": r2_score(out[TARGET], out[col]),
            "EVS": explained_variance_score(out[TARGET], out[col])
        })
if season_rows:
    sd = pd.DataFrame(season_rows).sort_values("R2", ascending=False)
    print(sd.to_string(index=False))

# Baseline (mean) context
y_all = out[TARGET].to_numpy()
y_mean = np.full_like(y_all, fill_value=y_all.mean(), dtype=float)
baseline_mae = mean_absolute_error(y_all, y_mean)
baseline_rmse = rmse_safe(y_all, y_mean)
print(f"\nBaseline (mean predictor): MAE={baseline_mae:.3f}, RMSE={baseline_rmse:.3f}")

# Save
cols_to_save = [year_col, player_col, week_col, TARGET, "final_pred", "pred_p20", "pred_p50", "pred_p80"]
for c in ["ElasticNet_pred","Tweedie_pred","CatBoost_pred"]:
    if c in out.columns: cols_to_save.append(c)

print("\nSample RB weekly predictions:")
print(out[cols_to_save].head(12).round(3).to_string(index=False))

out[cols_to_save].to_csv(f"rb_predictions_{TARGET_YEAR}_weekly_v2.csv", index=False)
print(f"\nSaved rb_predictions_{TARGET_YEAR}_weekly_v2.csv")
