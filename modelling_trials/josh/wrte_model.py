#Currently creates blended model taking best of ridge, xgboost, lgbm, random forest each week.
#still takes 
#models were based from old project code + GPT and finetuned from there (with added models now)
#metric diagnostic text was also GPT code

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
import joblib

#metric imports
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

#remove useless lightgbm warning for receiving ndarray
import warnings
warnings.filterwarnings("ignore",message= ".*does not have valid feature names.*")

#setup basic model config
CSV = "final_wr_and_te_data.csv"
TARGET = "PPR"
TARGET_YEAR = 2024
#z threathold for noise, adjusted to 3 from 3.5
NOISE_Z = 3 
#boolean to use preselected 20 vars from feature importance testing
use_features = True
#make sure model is repeatable to compare after fine tuning
state = 42
#booleans for using ensembl models
lgbm=True
xgboost =True

#selected features
#note: removed rolling features: "PPR_roll3_mean","PPR_roll5_mean","games_played_to_date"
top20_features = {
    "RK","POSITION_RANK","average_rank","best_rank","worst_rank","rank_variance","TIERS",
    "game_num","Age","ADOT","DRP","DRP%","DROP","YAC/REC","MTF","RTG","INL%",
    "PRSH_opponent","PBLK_opponent","RBLK_opponent","RUN_opponent",
    "COV_opponent","PA_opponent","PASS_opponent","RDEF_opponent","TACK_opponent",
    "DEF_opponent","PF_opponent","OFF_opponent",
    
}
#what to replace in either/or (1st over 2nd)
PAIR_PREFS = PAIR_PREFS = [("COV_opponent","DEF_opponent")]                    # keep YPA over PA



#leak-proof rolling stats using a 1 year shift
def rolling_feats(df, player_col, year_col, week_col, target_col, windows=(3,5)):
    df=df.sort_values([player_col, year_col, week_col]).copy()
    d =df.groupby(player_col,group_keys=False)
    for w in windows:
        #perform shift to prevent data leak
        df[f"{target_col}_roll{w}_mean"] = d[target_col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"{target_col}_roll{w}_std"]  = d[target_col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).std())
    for c in ["PA","YPA", "ADOT", "PRSH","NFL","SCR","TTT"]:
        if c in df.columns:
            for w in windows:
                df[f"{c}_roll{w}_mean"] = d[c].apply(lambda s:s.shift(1).rolling(w, min_periods=1).mean())
    #add gamecount
    df["games_played_to_date"] = d.cumcount()
    return df
#weigh recent seasons higher in 5 year backtrain
def make_year_weights(year_series, target_year, floor=0.2):
    gap = (target_year - year_series).clip(lower=1)   # 2023->1, 2022->2, ...
    weights = 1.0 / gap.astype(float)
    return (weights * (1 - floor)) + floor

#remove outliers per year sing median/mad
def rem_noise(df, target_col, year_col, z=3.0):
    def robust_z(s):
        med = np.median(s)
        mad = np.median(np.abs(s - med)) + 1e-6
        return 0.6745 * (s - med) / mad
    z_score = df.groupby(year_col)[target_col].transform(robust_z).abs()
    keep = z_score <= z
    return df[keep].copy(), int((~keep).sum())
#print diagnostics
def evaluate_block(df_preds, y_col: str = "PPR",pred_col: str = "predicted_PPR",idcols: list | None = None,bins_actual = [(0,10),(10,15),(15,20),(20,25),(25,200)],deciles: int = 10) -> None:
    if idcols is None: 
        idcols=[]
    df = df_preds.copy()
    y_true = df[y_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    err = y_true - y_pred
    abs_err = np.abs(err)

    #print errors per week for each model
    mae= mean_absolute_error(y_true, y_pred)
    rmse= root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("\n=== OVERALL ===")
    print(f"MAE: {mae:0.3f}, RMSE: {rmse:0.3f}, R²: {r2:0.3f}")
    print(f"Mean errpr: {err.mean():0.3f}, Median errpr: {np.median(err):0.3f}, Stdev: {err.std(ddof=1):0.3f}")

    #calibrate by prediction decile w safe qcut
    try:
        q =min(deciles, max(2,len(df) // 50))
        bins = pd.qcut(y_pred, q=q,duplicates="drop")
    except Exception:
        edges = np.linspace(y_pred.min(), y_pred.max(), num=deciles+1)
        bins = pd.cut(y_pred, bins=edges, include_lowest=True, duplicates="drop")
    df["calib_bin"] = bins
    df["abs_err"] = abs_err

    cal = (df.groupby("calib_bin").agg(n=(pred_col, "size"),pred_med=(pred_col, "median"),y_med=(y_col, "median"),mae=("abs_err", "median"),))
    print("\n- Calibration by pred decile -")
    print(cal.to_string())

    #Error by band (median abs err)
    rows = []
    for lo, hi in bins_actual:
        m = (df[y_col] >= lo) & (df[y_col] < hi)
        rows.append({"band": f"{lo}-{hi if hi<200 else '25+'}","n": int(m.sum()),"mae": float(np.median(np.abs(df.loc[m, y_col] - df.loc[m, pred_col]))) if m.any() else np.nan})
    print("\n- Error by actual PPR band (median MAE) -")
    print(pd.DataFrame(rows).to_string(index=False))

    #Extremes
    if idcols:
        label = df[idcols].astype(str).agg(" ".join, axis=1)
    else:
        label = df.index.astype(str)
    tops = pd.DataFrame({"label": label, "abs_err": abs_err}).sort_values("abs_err")
    print("\n=== Top 10 Best Predictions ===")
    print(tops.head(10).to_string(index=False))
    print("\n=== Top 10 Worst Predictions ===")
    print(tops.tail(10).sort_values("abs_err", ascending=False).to_string(index=False))

#use list of features (keeps ID-like columns; handles choose-one pairs)
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

#preprocess and convert cols to encoders/imputers for better model handeling
def preprocess(X, idlike, use_features=True):
    X = apply_features(X, idlike, use_features)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in idlike]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c not in idlike]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]) if len(cat_cols) else Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])
    preprocess = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
    return preprocess, num_cols, cat_cols

#following models were build using primarily GPT to get base for each models pipeline, finetuning from there
def build_models(preprocess):
    models = {
        "Ridge": Pipeline([("prep", preprocess), ("model", Ridge(alpha=1.0, random_state=state))]),
        "RandomForest": Pipeline([("prep", preprocess),("model", RandomForestRegressor(n_estimators=800, max_depth=8, min_samples_leaf=3,max_features=.6, random_state=state, n_jobs=-1))]),
        "HistGBR": Pipeline([("prep", preprocess), ("model", HistGradientBoostingRegressor(learning_rate=0.06, max_iter=600, max_depth=6,l2_regularization=.2, early_stopping=True, random_state=state))]),}
    if xgboost:
        models["XGB"] = Pipeline([("prep", preprocess),
                                  ("model", XGBRegressor(
                                      n_estimators=300, max_depth=5, learning_rate=0.08,
                                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                      random_state=state, tree_method="hist", n_jobs=-1))])
    if lgbm:
        models["LGBM"] = Pipeline([("prep", preprocess),
                                   ("model", LGBMRegressor(
                                       n_estimators=400, num_leaves=63, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                       random_state=state))])
    return models


#feature engineering
df = pd.read_csv(CSV)
#target col check
year_col = "YEAR"
player_col = 'PFR_ID'
week_col="Week"

if year_col is None: raise ValueError("Need YEAR/year column")
if player_col is None:
    player_col = "_anon_player"; df[player_col] = "player_" + df.index.astype(str)
if week_col is None:
    week_col = "_anon_week"; df[week_col] = df.groupby([player_col, year_col]).cumcount() + 1

drop_cols = ["Unnamed: 0","PFFPlayerKey","FantasyProsPlayerKey","PFFTeamKey"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df = df.dropna(subset=[TARGET]).copy()
df = rolling_feats(df, player_col, year_col, week_col, TARGET, windows=(3,5))

if TARGET_YEAR not in df[year_col].unique():
    raise ValueError(f"{TARGET_YEAR} not in {year_col}. Available years: {sorted(df[year_col].unique())}")

train_df = df[df[year_col] < TARGET_YEAR].copy()
pred_df = df[df[year_col] == TARGET_YEAR].copy()

#noise prune training only
train_df, dropped = rem_noise(train_df, TARGET, year_col, z=NOISE_Z)
if dropped:
    print(f"Removed {dropped} noisy rows from training (z > {NOISE_Z}).")

idlike = {player_col, week_col, year_col}


#perform rolling split
weeks = sorted(pred_df[week_col].unique())
preds_list = []

#to compute full-season metrics per model, store every model's predictions for each week on the true target rows
all_model_preds = []   # rows: [model, YEAR, PLAYER, WEEK, y_true, y_pred]
all_val_rows = []      # optional: rolling internal validation snapshots (for printing like before)

base_pool = train_df.copy()

for week in weeks:
    hist_this_year = pred_df[pred_df[week_col] < week]
    train = pd.concat([base_pool, hist_this_year], ignore_index=True)

    X_train = apply_features(train.drop(columns=[TARGET]), idlike, use_features)
    y_train = train[TARGET].values

    target_rows = pred_df[pred_df[week_col] == week].copy()
    X_pred = apply_features(target_rows.drop(columns=[TARGET]), idlike, use_features)

    preproc, num_cols, cat_cols = preprocess(X_train, idlike, False)  # already featuresed
    models = build_models(preproc)

    # time-valid selection on last train year (for logging only; we’ll still record per-model preds)
    sub_years = np.sort(train[year_col].unique())
    val_year = sub_years[-1]
    core  = train[train[year_col] < val_year] if len(sub_years) > 1 else train
    valid = train[train[year_col] == val_year] if len(sub_years) > 1 else train.iloc[:0]
    X_core = apply_features(core.drop(columns=[TARGET]), idlike, use_features)
    y_core = core[TARGET].values
    X_valid =apply_features(valid.drop(columns=[TARGET]),idlike, use_features)
    y_valid = valid[TARGET].values

    # Fit each model on core to log validation, then refit on X_train for prediction
    val_mae = {}
    for name, pipe in models.items():
        # validation fit
        pipe.fit(X_core, y_core)
        if len(valid):
            yv = pipe.predict(X_valid)
            mae = mean_absolute_error(y_valid, yv)
            rmse = root_mean_squared_error(y_valid, yv)
            r2 = r2_score(y_valid, yv)
            print(f"[VALID {val_year}] {name:12}  MAE {mae:6.3f}  RMSE {rmse:6.3f}  R² {r2:6.3f}")
            val_mae[name] = mae
        else:
            val_mae[name] = np.inf

    best = min(val_mae, key=val_mae.get) if val_mae else "Ridge"

    #apply recency weights on full train
    w = make_year_weights(train[year_col], TARGET_YEAR)

    #generate and store predictions for ALL models (season-wide comparison)
    for name, pipe in models.items():
        # refit on full train with weights
        fitted = pipe.fit(X_train, y_train, model__sample_weight=w)
        preds = fitted.predict(X_pred)
        # store per-model predictions
        block = target_rows[[year_col, player_col, week_col, TARGET]].copy()
        block["model"] = name
        block["y_pred"] = preds
        all_model_preds.append(block)

    #use the week's best model for the blended/operational forecast
    best_model = models[best].fit(X_train, y_train, model__sample_weight=w)
    target_rows = target_rows.copy()
    target_rows["predicted_PPR"] = best_model.predict(X_pred)
    preds_list.append(target_rows[[year_col, player_col, week_col, TARGET, "predicted_PPR"]]
                      if TARGET in target_rows.columns else
                      target_rows[[year_col, player_col, week_col, "predicted_PPR"]])

    if len(valid):
        tmp = valid[[year_col, player_col, week_col, TARGET]].copy()
        tmp["predicted_PPR"] = models[best].predict(X_valid)
        all_val_rows.append(tmp)

    print(f"Week {week:>2} | trained on <{TARGET_YEAR} + weeks < {week} | model={best}")

#outputs
out = pd.concat(preds_list, ignore_index=True).sort_values([player_col, week_col])

#1) Blended/operational result (week-by-week best model)
if TARGET in out.columns:
    print("\nStrict-weekly evaluation on TARGET_YEAR (BLENDED MODEL):")
    evaluate_block(out, y_col=TARGET, pred_col="predicted_PPR",
                   idcols=[player_col, year_col, week_col])

#2) full szn metrics per model (aggregate across all weeks)
all_preds_df = pd.concat(all_model_preds, ignore_index=True)
season_rows = []
for model_name, dfm in all_preds_df.groupby("model"):
    y_true = dfm[TARGET].to_numpy()
    y_pred = dfm["y_pred"].to_numpy()
    season_rows.append({"model": model_name,"MAE": mean_absolute_error(y_true, y_pred), "RMSE": root_mean_squared_error(y_true, y_pred),"R2": r2_score(y_true, y_pred)})
season_df = pd.DataFrame(season_rows).sort_values("R2", ascending=False)
print("\n- FULL-SEASON MODEL PERFORMANCE (2024) -")
print(season_df.to_string(index=False))
# 2b) save all per-model predictions (LONG format)
all_preds_df.to_csv(f"wrte_predictions_{TARGET_YEAR}_all_models_long.csv", index=False)
print(f"\nSaved wrte_predictions_{TARGET_YEAR}_all_models_long.csv")

# 2c) WIDE format with one column per model
wrte_wide = (
    all_preds_df
      .pivot_table(
          index=[year_col, player_col, week_col, TARGET],
          columns="model",
          values="y_pred"
      )
      .reset_index()
)

wrte_wide.columns.name = None

wrte_wide.to_csv(f"wrte_predictions_{TARGET_YEAR}_all_models_wide.csv", index=False)
print(f"Saved wrte_predictions_{TARGET_YEAR}_all_models_wide.csv")

#3) Baseline (mean predictor) for context
y_all = all_preds_df[TARGET].to_numpy()
y_mean = np.full_like(y_all, fill_value=y_all.mean(), dtype=float)
baseline_mae = mean_absolute_error(y_all, y_mean)
baseline_rmse = root_mean_squared_error(y_all, y_mean)
print(f"\nBaseline (mean predictor): MAE={baseline_mae:.3f}, RMSE={baseline_rmse:.3f}")

#4)Aggregated internal validation snapshots (optional summary like before)
if all_val_rows:
    val_all = pd.concat(all_val_rows, ignore_index=True)
    print("\nAggregated internal validation (rolling):")
    evaluate_block(val_all, y_col=TARGET, pred_col="predicted_PPR",
                   idcols=[player_col, year_col, week_col])

#from cnhatgpt: print and save predictions
# ============================================================
# STATIC SEASON-LONG WR/TE MODELS FOR GYM (5-year weighted)
# ============================================================

print("\n=== Training static (season-long) WR/TE models for gym use ===")

# 1) Full training data: all seasons before TARGET_YEAR, including rolling features and denoising
X_train_full = apply_features(
    train_df.drop(columns=[TARGET]),
    idlike,
    use_features
)
y_train_full = train_df[TARGET].values

# 2) Build a fresh preprocessor + model dict for these columns
#    (we set use_features=False because we already applied the whitelist above)
preproc_full, _, _ = preprocess(X_train_full, idlike, use_features=False)
models_full = build_models(preproc_full)

# 3) Recency weights: newer seasons get higher weight
w_full = make_year_weights(train_df[year_col], TARGET_YEAR)

# 4) Fit and save each model as its own pickle
prefix = "wrte"  # filenames: wrte_ridge.pkl, wrte_randomforest.pkl, etc.

for name, pipe in models_full.items():
    print(f"  Fitting {name} on full pre-{TARGET_YEAR} WR/TE data...")
    pipe.fit(X_train_full, y_train_full, model__sample_weight=w_full)

    pkl_name = f"{prefix}_{name.lower()}.pkl"
    joblib.dump(pipe, pkl_name)
    print(f"    Saved {pkl_name}")

print("\nFinished training + saving static WR/TE models.")
