# ================================================
# feature_selection.py
# Fast feature-importance extraction for QB/RB/WR
# ================================================

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor

# =========================
# Config
# =========================
TARGET = "PPR"
ID_COLS = ["PFR_ID"]
CAT_COLS_DEFAULT = ["Team", "Opp", "Home_Away", "Day"]

SEED = 42
N_JOBS = 6


# =========================
# Utility
# =========================
def _make_ohe():
    """Return an OneHotEncoder across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# =========================
# Preprocessing
# =========================
def build_preprocessors(df: pd.DataFrame):
    """Build the same preprocessing as the main PPR script (but without feature gating)."""

    cat_cols = [c for c in CAT_COLS_DEFAULT if c in df.columns]

    numeric_exclude = set([TARGET] + ID_COLS)
    candidate_numeric = [c for c in df.columns if c not in numeric_exclude and c not in cat_cols]
    numeric_cols = [c for c in candidate_numeric if pd.api.types.is_numeric_dtype(df[c])]

    ohe = _make_ohe()

    # Trees: Impute numerics, one-hot cats
    preproc_trees = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop"
    )

    return preproc_trees, numeric_cols, cat_cols


# =========================
# Feature Importance
# =========================
def compute_feature_importances_only(df: pd.DataFrame, n_estimators: int = 300):
    """Compute fast feature importances using ExtraTrees (no tuning, no stacking)."""

    # Use only training years for consistency
    train_df = df[df["YEAR"] < 2022].copy()

    # Build preprocessing
    preproc_trees, num_cols, cat_cols = build_preprocessors(df)

    # Split into X/y
    X_train = train_df.drop(columns=[TARGET] + ID_COLS, errors="ignore")
    y_train = train_df[TARGET].astype(float)

    # Build model
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=SEED,
        n_jobs=N_JOBS
    )

    pipe = Pipeline([
        ("pre", preproc_trees),
        ("est", model)
    ])

    print("\nFitting ExtraTrees model for feature selection...")
    pipe.fit(X_train, y_train)

    # Extract processed feature names
    pre = pipe.named_steps["pre"]
    est = pipe.named_steps["est"]

    try:
        feature_names = pre.get_feature_names_out()
    except AttributeError:
        feature_names = np.array([f"feat_{i}" for i in range(est.n_features_in_)])

    # Build DataFrame
    importances = est.feature_importances_
    imp_df = pd.DataFrame({
        "feature_name": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n=== Top 30 Features ===")
    print(imp_df.head(30).to_string(index=False))

    # Select features above median importance (similar to gating)
    median_imp = imp_df["importance"].median()
    selected = imp_df[imp_df["importance"] >= median_imp]

    print(f"\n=== Features Selected (≥ median importance threshold) ===")
    print(f"Total selected: {len(selected)}")
    print(selected.head(30).to_string(index=False))

    return imp_df, selected


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Load your dataset here
    # Example: final_qb_data.csv
    path = "../../final_qb_data.csv"

    print(f"\nLoading dataset: {path}")
    df = pd.read_csv(path, index_col=0)

    full_importances, selected_features = compute_feature_importances_only(df)

    # Save results
    full_importances.to_csv("feature_importances_full.csv", index=False)
    selected_features.to_csv("feature_importances_selected.csv", index=False)

    print("\nSaved: feature_importances_full.csv")
    print("Saved: feature_importances_selected.csv")
    print("\nDone.")
