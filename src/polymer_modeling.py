"""Shared modelling definitions for the polymer notebooks 07 and 08.

The notebooks stay the place where the analysis is explained and run. This module
only holds definitions that both notebooks must use identically:

- the five targets and the two descriptor sets (9 base, 13 extended),
- per-target datasets that keep the original row index of train.csv,
- the 80/20 split of notebooks 01 and 05 (`train_test_split`, `random_state=42`);
  the 20 % part is called the *validation* set because it has already been used
  for model comparisons in notebook 05,
- the 5-fold cross-validation folds on the remaining 80 % (development set),
- the documented model configurations and the limited search spaces.

Only descriptor columns can become model features. Identifiers, SMILES text and
the other targets are never part of `X`.
"""

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.descriptors import ALL_DESCRIPTORS, BASE_DESCRIPTORS

TARGETS = ("Tg", "FFV", "Tc", "Density", "Rg")
FEATURE_SETS = {
    "base9": list(BASE_DESCRIPTORS),
    "extended13": list(ALL_DESCRIPTORS),
}

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20
N_FOLDS = 5

MODEL_NAMES = ("median", "linear", "random_forest", "xgboost")


def target_dataset(table: pd.DataFrame, target: str, feature_set: str):
    """Return X and y for one target, restricted to rows where its label exists.

    The index of `table` (the original train.csv row index) is kept, so every
    prediction can be traced back to its original row. Missing labels are dropped,
    never imputed.
    """
    if target not in TARGETS:
        raise KeyError(f"Unknown target: {target}")
    features = FEATURE_SETS[feature_set]
    labelled = table[target].notna()
    X = table.loc[labelled, features].copy()
    y = table.loc[labelled, target].copy()
    if not np.isfinite(y).all():
        raise ValueError(f"{target} contains non-finite labels")
    if X.isna().any().any():
        raise ValueError(f"{target}: descriptor table contains missing values")
    return X, y


def validation_split(X: pd.DataFrame, y: pd.Series):
    """Split into development (80 %) and validation (20 %) as in notebooks 01 and 05."""
    return train_test_split(
        X, y, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE, shuffle=True
    )


def cv_folds(n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the fixed 5-fold split (positions) used for every model comparison."""
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(n_rows)))


def make_model(name: str, n_jobs: int = -1):
    """Return an unfitted model with the documented fixed configuration.

    - median: predicts the training median (baseline).
    - linear: ordinary least squares after standardisation; the scaler is part
      of the pipeline, so it is fitted on the training fold only.
    - random_forest: configuration of notebook 05 (200 trees).
    - xgboost: Suhmi's initial XGBoost configuration as documented in notebook 05.
    """
    if name == "median":
        return DummyRegressor(strategy="median")
    if name == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=n_jobs)
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )
    raise KeyError(f"Unknown model: {name}")


def search_space(name: str) -> dict:
    """Return the limited random-search space for a tree model.

    XGBoost uses Suhmi's search ranges as documented in notebook 05. The random
    forest space is a small, added space so that both tree models are tuned.
    """
    if name == "xgboost":
        return {
            "n_estimators": randint(200, 1501),
            "learning_rate": loguniform(0.01, 0.2),
            "max_depth": randint(2, 9),
            "min_child_weight": randint(1, 11),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_lambda": loguniform(0.1, 10),
        }
    if name == "random_forest":
        return {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, 0.5, "sqrt"],
        }
    raise KeyError(f"No search space for model: {name}")


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
