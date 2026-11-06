"""Train and save the Polymind v2 bundle (models/polymind_v2).

Training data: the shared training partition only (train_test_split 0.2, random_state 42),
as for v1. The validation partition is not read for fitting or scoring.
Tg also uses the cleaned external Tg pool (weight 0.5, source flag), restricted to
structures marked eligible in Mohamed's exclusion ledger (no exact matches to original
Tg structures, Tanimoto < 0.95 to the outer Tg holdout).

Run from the project root:
    uv run python tools/build_polymind_v2.py
"""

import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.polymer_features import SHAPE_COLUMNS
from src.polymind_predict_v2 import build_v2_features, feature_matrix

OUT = ROOT / "models" / "polymind_v2"
LEDGER = ROOT / "results/tg_shared_candidate_review_local_02/external_exclusion_ledger.csv"
FULL = ["B+", "chain", "shape", "D"]

# Hyperparameters from notebooks 09-12 (tuned with cross-validation, no new search here).
SETUPS = {
    "Tg": (FULL, dict(colsample_bytree=0.7669644012595116, learning_rate=0.019452208847287395, max_depth=4,
                      min_child_weight=10, n_estimators=883, reg_lambda=0.307787547057142,
                      subsample=0.6673164168691722)),
    "FFV": (FULL, dict(learning_rate=0.073, max_depth=6, min_child_weight=2, n_estimators=1077,
                       reg_lambda=1.25, subsample=0.877, colsample_bytree=0.641)),
    "Tc": (["B+", "D"], dict(n_estimators=708, learning_rate=0.013399060561509796, max_depth=5,
                             min_child_weight=7, subsample=0.6693458614031088,
                             colsample_bytree=0.7218455076693483, reg_lambda=4.638630972397279)),
    "Density": (FULL, dict(n_estimators=1464, learning_rate=0.013815607382950859, max_depth=8,
                           min_child_weight=7, subsample=0.8034282764658811,
                           colsample_bytree=0.610167650697638, reg_lambda=0.4253162363790871)),
    "Rg": (["B+", "chain"], dict(n_estimators=1031, learning_rate=0.018187859051288217, max_depth=8,
                                 min_child_weight=4, subsample=0.9439761626945282,
                                 colsample_bytree=0.9140703845572055, reg_lambda=0.85807606199213)),
}

# Five-fold CV on the shared training partition (tools/compare: Tg from
# results/tg_shared_candidate_review_local_02, others from the v1-vs-v2 comparison).
CV_MAE = {"Tg": {"v1": 53.139, "v2": 48.159},
          "FFV": {"v1": 0.00923, "v2": 0.00609},
          "Tc": {"v1": 0.03374, "v2": 0.02510},
          "Density": {"v1": 0.04322, "v2": 0.02435},
          "Rg": {"v1": 1.7733, "v2": 1.4287}}

raw = pd.read_csv(ROOT / "data/raw/train.csv")
train_rows, _ = train_test_split(raw.index.to_numpy(), test_size=0.2, random_state=42, shuffle=True)
shape = pd.read_csv(ROOT / "data/processed/shape_descriptors.csv", index_col=0)
blocks = build_v2_features(raw.SMILES.loc[train_rows], shape=shape.loc[train_rows])

ext = pd.read_csv(ROOT / "data/processed/external_tg_clean.csv")
eligible = pd.read_csv(LEDGER).set_index("external_row_index").eligible_before_cv.reindex(ext.index)
ext = ext[eligible.fillna(False).astype(bool).to_numpy()]
ext_blocks = build_v2_features(ext.SMILES, shape=ext[SHAPE_COLUMNS])

OUT.mkdir(parents=True, exist_ok=True)
targets = {}
for target, (keys, params) in SETUPS.items():
    labelled = raw.loc[train_rows, target].notna().to_numpy()
    rows = train_rows[labelled]
    flag = 0 if target == "Tg" else None
    X = feature_matrix(blocks, keys, flag).loc[rows]
    y = raw.loc[rows, target].to_numpy()
    weights = np.ones(len(rows))
    if target == "Tg":
        X_ext = feature_matrix(ext_blocks, keys, 1)
        X = pd.concat([X, X_ext.set_axis(X_ext.index + 10**6)])
        y = np.concatenate([y, ext.Tg.to_numpy()])
        weights = np.concatenate([weights, np.full(len(ext), 0.5)])
    model = make_pipeline(SimpleImputer(strategy="median"),
                          XGBRegressor(**params, random_state=42, n_jobs=-1))
    model.fit(X.to_numpy(dtype=float), y, xgbregressor__sample_weight=weights)
    joblib.dump(model, OUT / f"{target.lower()}.joblib", compress=3)
    targets[target] = {
        "model": "xgboost",
        "feature_blocks": keys,
        "external_source_flag": target == "Tg",
        "parameters": params,
        "file": f"{target.lower()}.joblib",
        "n_training_polymers": int(len(rows)),
        "n_external_polymers": int(len(ext)) if target == "Tg" else 0,
        "cv_mae_shared_split": CV_MAE[target],
        "feature_columns": list(X.columns),
    }
    print(target, len(rows), "polymers", flush=True)

manifest = {
    "version": 2,
    "training_scope": "Shared training partition; observed labels only. Tg adds the eligible external pool (weight 0.5).",
    "feature_blocks": {"B+": "chemistry and group descriptors", "chain": "chain flexibility and packing descriptors",
                       "shape": "3D trimer shape descriptors (one ETKDG conformer, NaN if embedding fails)",
                       "D": "Morgan fingerprint, radius 2, 1024 bits"},
    "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "rdkit", "xgboost", "joblib"]},
    "targets": targets,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("saved", OUT)
