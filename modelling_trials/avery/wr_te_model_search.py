# Standard library
import pickle
from tqdm import tqdm
import warnings

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from joblib import parallel_backend

# Scikit-learn preprocessing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Scikit-learn models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neighbors import NearestNeighbors

# Scikit-learn model selection and evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE

# Scikit-learn utilities
from sklearn.pipeline import Pipeline

def find_noisy_samples(X, y, n_neighbors=5, threshold=2.0):
    """
    Find samples where nearby points have very different target values
    """
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # For each point, check if its target differs from neighbors
    noise_scores = []
    for i in range(len(X)):
        neighbor_indices = indices[i][1:]  # Exclude self
        neighbor_targets = y[neighbor_indices]

        # Calculate how different this point's target is from neighbors
        target_std = np.std(np.append(neighbor_targets, y[i]))
        target_diff = np.abs(y[i] - np.mean(neighbor_targets))

        noise_scores.append(target_diff / (target_std + 1e-10))

    noise_scores = np.array(noise_scores)

    # Flag samples with high noise scores
    noisy_mask = noise_scores > threshold

    return noisy_mask, noise_scores

wr_and_te_df = pd.read_csv("../../final_wr_and_te_data.csv", index_col=0)
wr_and_te_df = wr_and_te_df.drop(['PFR_ID', 'Team', 'Opp'], axis=1)

wr_and_te_df = pd.get_dummies(wr_and_te_df)

train_df = wr_and_te_df[wr_and_te_df['YEAR'] < 2022]
validation_df = wr_and_te_df[wr_and_te_df['YEAR'] == 2022]
test_df = wr_and_te_df[wr_and_te_df['YEAR'] > 2022]

y_train = train_df['PPR'].values
X_train = train_df.drop(['PPR', 'YEAR'], axis=1).values
train_year = train_df['YEAR'].values
X_train_df = train_df.drop(['PPR', 'YEAR'], axis=1)

y_valid = validation_df['PPR'].values
X_valid = validation_df.drop(['PPR', 'YEAR'], axis=1).values
X_valid_df = validation_df.drop(['PPR', 'YEAR'], axis=1)

y_test = test_df['PPR'].values
X_test = test_df.drop(['PPR', 'YEAR'], axis=1).values
X_test_df = test_df.drop(['PPR', 'YEAR'], axis=1)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)

# --- Prepare data ---
final_features = [
    'average_rank', 'YAC', 'RECV_PLAYER', 'OFF_PLAYER', 'REC', 'MTF', 'YDS',
    'INL%', 'RECV_TEAM', 'TGT', '1ST', 'Y/RR', 'INL', 'PBLK_PLAYER',
    'PRSH_opponent', 'COV_opponent'
]

X_train_new = X_train_df[final_features].values
X_valid_new = X_valid_df[final_features].values
X_test_new = X_test_df[final_features].values

# --- Remove noisy samples ---
noisy_mask, noise_scores = find_noisy_samples(X_train_new, y_train, n_neighbors=9, threshold=0.8)
clean_mask = ~noisy_mask
X_train_new = X_train_new[clean_mask]
y_train_new = y_train[clean_mask]
train_year_new = train_year[clean_mask]

# --- Define objective function for Optuna ---
def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-3, 1e2, log=True)
    gamma = trial.suggest_float('gamma', 1e-4, 1e1, log=True)
    coef0 = trial.suggest_float('coef0', 0, 2)
    degree = trial.suggest_int('degree', 2, 5)
    kernel = trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid', 'linear'])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KernelRidge(
            alpha=alpha, kernel=kernel, degree=degree,
            gamma=gamma, coef0=coef0
        ))
    ])

    weights = 1 / (2022 - train_year_new + 1)
    pipeline.fit(X_train_new, y_train_new, model__sample_weight=weights)

    y_pred_valid = pipeline.predict(X_valid_new)
    mse = mean_squared_error(y_valid, y_pred_valid)
    return mse

# --- Create and optimize the study ---
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

# Create a progress bar
pbar = tqdm(total=1000, desc="Optimization")

# Callback to update progress bar with best R²
def update_progress(study, trial):
    # Calculate R² from best MSE
    best_mse = study.best_value
    best_r2 = 1 - (best_mse / np.var(y_valid))

    pbar.set_postfix({'Best Val R²': f'{best_r2:.4f}'})
    pbar.update(1)

# Let Optuna handle parallel threads
study.optimize(objective, n_trials=1000, n_jobs=-1, catch=(Exception,), callbacks=[update_progress])

# Close the progress bar when done
pbar.close()

# --- Best trial ---
print("\nBest Trial:")
print(study.best_trial.params)

# --- Refit best model ---
best_params = study.best_trial.params
best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KernelRidge(**best_params))
])

weights = 1 / (2022 - train_year_new + 1)
best_model.fit(X_train_new, y_train_new, model__sample_weight=weights)

with open('wr_te_kernel_ridge.pkl', 'wb') as file:
    pickle.dump(best_model, file)