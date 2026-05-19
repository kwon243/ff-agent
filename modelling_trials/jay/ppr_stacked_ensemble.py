# ppr_stacked_ensemble.py
# Requires: pandas, numpy, scikit-learn >= 1.1, scipy
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform, loguniform

from sklearn import set_config
set_config(enable_metadata_routing=True)


# =========================
# Config
# =========================
TARGET = "PPR"
ID_COLS = ["PFR_ID"]  # keep IDs out of features
CAT_COLS_DEFAULT = ["Team", "Opp", "Home_Away", "Day"]  # extend as needed
NUMERIC_EXCLUDE = set([TARGET] + ID_COLS)

SEED = 42
N_JOBS = 6
CV_SPLITS = 5  # GroupKFold by YEAR within training years
USE_FEATURE_GATING = True  # toggle feature gating here


# =========================
# Utility
# =========================
def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def report_block(name: str, y_true, y_pred) -> Dict[str, float]:
    return {
        "name": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

def print_report_table(rows: List[Dict[str, float]]):
    df = pd.DataFrame(rows).set_index("name")
    print("\n=== Metrics ===")
    print(df.round(4).to_string())
    return df

def _make_ohe():
    """Return an OneHotEncoder that works across sklearn versions."""
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# =========================
# Preprocessing builders
# =========================
def build_preprocessors(df: pd.DataFrame, cat_cols):
    # Infer numerics by dtype minus target/ids/categoricals
    NUMERIC_EXCLUDE = set([TARGET] + ID_COLS)
    candidate_numeric = [c for c in df.columns if c not in NUMERIC_EXCLUDE and c not in cat_cols]
    numeric_cols = [c for c in candidate_numeric if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols_kept = [c for c in cat_cols if c in df.columns]

    ohe = _make_ohe()

    # Trees: impute numerics; one-hot cats; no scaling
    preproc_trees = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", ohe, cat_cols_kept),
        ],
        remainder="drop"
    )

    # SVR/MLP: impute + scale numerics; one-hot cats
    preproc_scaled = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ]), numeric_cols),
            ("cat", ohe, cat_cols_kept),
        ],
        remainder="drop"
    )

    return preproc_trees, preproc_scaled, numeric_cols, cat_cols_kept

def build_feature_selector(preproc_for_selector: ColumnTransformer) -> SelectFromModel:
    """
    Feature gating via model-based selection (ExtraTrees) on preprocessed space.
    We put this in each pipeline so it fits only on train folds (no leakage).
    """
    et_selector = SelectFromModel(
        estimator=ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            random_state=SEED,
            n_jobs=N_JOBS
        ),
        threshold="median",  # keep features with >= median importance (adjustable)
        max_features=None
    )
    # We'll wrap as Pipeline steps (preproc -> selector -> estimator)
    return et_selector


# =========================
# Models & search spaces
# =========================
def base_estimators(preproc_trees, preproc_scaled, use_feature_gating: bool) -> Dict[str, Pipeline]:
    selector_trees = build_feature_selector(preproc_trees) if use_feature_gating else "passthrough"
    selector_scaled = build_feature_selector(preproc_scaled) if use_feature_gating else "passthrough"

    models = {
        "rf": Pipeline([
            ("pre", preproc_trees),
            ("gate", selector_trees),
            ("est", RandomForestRegressor(random_state=SEED, n_jobs=N_JOBS)),
        ]),
        "et": Pipeline([
            ("pre", preproc_trees),
            ("gate", selector_trees),
            ("est", ExtraTreesRegressor(random_state=SEED, n_jobs=N_JOBS)),
        ]),
        "hgb": Pipeline([
            ("pre", preproc_trees),
            ("gate", selector_trees),
            ("est", HistGradientBoostingRegressor(random_state=SEED)),
        ]),
        "svr": Pipeline([
            ("pre", preproc_scaled),
            ("gate", selector_scaled),
            ("est", SVR()),
        ]),
        "mlp": Pipeline([
            ("pre", preproc_scaled),
            ("gate", selector_scaled),
            ("est", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                learning_rate="adaptive",
                learning_rate_init=1e-3,
                alpha=1e-4,              # ↑ L2 to reduce overfit
                early_stopping=True,     # ← key
                validation_fraction=0.1,
                n_iter_no_change=20,
                max_iter=2000,           # allow convergence with early stopping
                random_state=SEED
            )),
        ]),
    }
    return models


def search_spaces() -> Dict[str, Dict[str, object]]:
    # Parameter grids reference pipeline step names: "est__param"
    return {
        "rf": {
            "est__n_estimators": randint(400, 1200),
            "est__max_depth": randint(6, 24),
            "est__min_samples_split": randint(2, 12),
            "est__min_samples_leaf": randint(1, 8),
            "est__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        },
        "et": {
            "est__n_estimators": randint(500, 1400),
            "est__max_depth": randint(6, 24),
            "est__min_samples_split": randint(2, 12),
            "est__min_samples_leaf": randint(1, 8),
            "est__max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        },
        "hgb": {
            "est__learning_rate": loguniform(1e-2, 3e-1),
            "est__max_depth": randint(3, 12),
            "est__max_iter": randint(200, 1200),
            "est__l2_regularization": loguniform(1e-8, 1e-1),
        },
        "svr": {
            "est__C": loguniform(1e-1, 1e2),
            "est__epsilon": loguniform(1e-3, 1.0),
            "est__gamma": ["scale", "auto"],
            "est__kernel": ["rbf"],  # keep to RBF for robustness
        },
        "mlp": {
            "est__hidden_layer_sizes": [(128, 64), (64, 32), (128,), (64,)],
            "est__alpha": loguniform(1e-5, 1e-3),
            "est__learning_rate_init": loguniform(5e-5, 2e-3),
            "est__activation": ["relu"],
        },
    }


# =========================
# Tuning
# =========================
def tune_model(
    name: str,
    pipe: Pipeline,
    param_dist: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_iter: int = 25,
) -> RandomizedSearchCV:
    cv = GroupKFold(n_splits=CV_SPLITS)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        n_jobs=N_JOBS,
        cv=cv,
        random_state=SEED,
        verbose=1
    )
    search.fit(X, y, groups=groups)
    print(f"\n>>> {name} best MAE (cv): {-search.best_score_:.4f}")
    print(f">>> {name} best params: {search.best_params_}")
    return search

# =========================
# Extract features
# =========================
def extract_selected_features_from_pipeline(pipe: Pipeline) -> pd.DataFrame:
    """
    Given a fitted pipeline with steps ["pre", "gate", "est"],
    return a DataFrame with columns: feature_name, selected (bool), importance (if available).
    """
    pre = pipe.named_steps["pre"]
    gate = pipe.named_steps.get("gate", None)

    if gate is None or gate == "passthrough":
        raise ValueError("Pipeline has no feature gating step or gating is 'passthrough'.")

    # 1) Feature names after preprocessing
    # Works in sklearn >= 1.0
    try:
        feature_names = pre.get_feature_names_out()
    except AttributeError:
        # Fallback: build them manually if needed
        # (Simple: just create generic names)
        n_features = gate.estimator_.n_features_in_
        feature_names = np.array([f"feat_{i}" for i in range(n_features)])

    feature_names = np.array(feature_names)

    # 2) Which features are kept
    support_mask = gate.get_support()
    selected = feature_names[support_mask]

    # 3) (Optional) importance from the underlying ExtraTrees
    et = gate.estimator_
    importances = getattr(et, "feature_importances_", None)
    if importances is not None:
        imp_selected = importances[support_mask]
    else:
        imp_selected = np.full(len(selected), np.nan)

    df_fs = pd.DataFrame({
        "feature_name": selected,
        "importance": imp_selected
    }).sort_values("importance", ascending=False)

    return df_fs


# =========================
# Main training function
# =========================
def train_and_evaluate(df: pd.DataFrame):
    # ---- splits (as requested)
    train_df = df[df["YEAR"] < 2022].copy()
    val_df   = df[df["YEAR"] == 2022].copy()
    test_df  = df[df["YEAR"] > 2022].copy()

    # Identify columns
    cat_cols = [c for c in CAT_COLS_DEFAULT if c in df.columns]
    pre_trees, pre_scaled, num_cols, cat_cols = build_preprocessors(df, cat_cols)

    # Features/targets
    def XY(d):
        X = d.drop(columns=[TARGET] + ID_COLS, errors="ignore")
        y = d[TARGET].astype(float)
        return X, y

    X_train, y_train = XY(train_df)
    X_val,   y_val   = XY(val_df)
    X_test,  y_test  = XY(test_df)

    # Groups for GroupKFold (year)
    groups_train = train_df["YEAR"]

    # Build & tune base models
    pipes = base_estimators(pre_trees, pre_scaled, USE_FEATURE_GATING)
    spaces = search_spaces()

    tuned_models = {}
    for key in pipes:
        tuned = tune_model(
            key, pipes[key], spaces[key],
            X=X_train, y=y_train, groups=groups_train, n_iter=25
        )
        tuned_models[key] = tuned.best_estimator_

    if USE_FEATURE_GATING:
        print("\n=== Feature Selection (gating) per base model ===")
        feature_selection_results = {}
        for name, model in tuned_models.items():
            try:
                fs_df = extract_selected_features_from_pipeline(model)
                feature_selection_results[name] = fs_df
                print(f"\n>>> {name}: {len(fs_df)} features selected")
                print(fs_df.head(20).to_string(index=False))  # top 20 by importance
            except ValueError as e:
                print(f"\n>>> {name}: {e}")

        # you can return this dict if you want to use it outside
    else:
        feature_selection_results = {}

    # Evaluate base models on val
    base_rows = []
    for name, est in tuned_models.items():
        pred_tr = est.predict(X_train)
        pred_va = est.predict(X_val)
        base_rows += [report_block(f"{name}-train", y_train, pred_tr)]
        base_rows += [report_block(f"{name}-val", y_val, pred_va)]
    print_report_table(base_rows)

    # Build stacker (use tuned base models)
    estimators = [(k, tuned_models[k]) for k in tuned_models.keys()]
    gkf = GroupKFold(n_splits=CV_SPLITS)
    cv_splits = list(gkf.split(X_train, y_train, groups=groups_train))

    stacker = StackingRegressor(
        estimators=estimators,
        final_estimator=ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=SEED),
        passthrough=False,
        n_jobs=N_JOBS,
        cv=cv_splits,
    )

    stacker.fit(X_train, y_train)

    # Predictions
    yhat_train = stacker.predict(X_train)
    yhat_val   = stacker.predict(X_val)
    yhat_test  = stacker.predict(X_test)

    # Reports
    rows = [
        report_block("Stacked-TRAIN", y_train, yhat_train),
        report_block("Stacked-VAL",   y_val,   yhat_val),
        report_block("Stacked-TEST",  y_test,  yhat_test),
    ]
    print_report_table(rows)

    # Meta-learner weights (if linear)
    if isinstance(stacker.final_estimator_, ElasticNet):
        print("\n=== Meta-learner (ElasticNet) coefficients ===")
        fe = stacker.final_estimator_
        # Coeffs correspond to [base1_pred, base2_pred, ..., passthrough original features]
        # We’ll just show weights on base learner predictions:
        base_names = list(tuned_models.keys())
        base_w = fe.coef_[:len(base_names)]
        for n, w in zip(base_names, base_w):
            print(f"{n:>4s}: {w:+.4f}")

    # Permutation importance of the final stacker on VAL
    # (This is expensive; sample rows if needed.)
    print("\n=== Permutation Importance on VAL (stacker) ===")
    SAMPLE_SIZE = 2000  # or None for full val
    if SAMPLE_SIZE and len(X_val) > SAMPLE_SIZE:
        val_sample = X_val.sample(SAMPLE_SIZE, random_state=SEED)
        y_sample = y_val.loc[val_sample.index]
    else:
        val_sample, y_sample = X_val, y_val


    perm = permutation_importance(
        estimator=stacker,
        X=val_sample,
        y=y_sample,
        n_repeats=5,
        random_state=SEED,
        n_jobs=N_JOBS,
        scoring="neg_mean_absolute_error"
    )
    # Feature names are not straightforward because each base pipeline transforms features internally.
    # We therefore report importances of the stacker input columns positionally:
    importances = pd.DataFrame({
        "feature_idx": np.arange(len(perm.importances_mean)),
        "imp_mean": perm.importances_mean,
        "imp_std": perm.importances_std
    }).sort_values("imp_mean", ascending=False)
    print(importances.head(20).round(5).to_string(index=False))

    return {
        "stacker": stacker,
        "tuned_models": tuned_models,
        "val_metrics": rows[1],
        "test_metrics": rows[2],
        "feature_selection": feature_selection_results,  # <-- added
    }


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Example: df must already be loaded with your columns.
    # df = pd.read_csv("qb_games.csv")
    # Ensure expected columns exist, e.g., YEAR, PPR, Team, Opp, Home_Away, Day, etc.
    # Here we just raise if not present:
    try:
        df = pd.read_csv("../../final_qb_data.csv", index_col=0)  # noqa: F821
    except NameError:
        raise SystemExit("Please define/load a DataFrame named `df` before running this script.")

    result = train_and_evaluate(df)
    print("\nDone.")
