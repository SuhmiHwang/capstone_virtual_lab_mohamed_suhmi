"""FFV/Tc/Density/Rg: Polymind v1 settings vs Suhmi's full-feature XGBoost on Mohamed's shared split.
Run: uv run python tools/compare_v1_v2_shared_split.py .
Training partition only (train_test_split 0.2, rs=42), KFold(5, shuffle, rs=42) per target, as in compare_tg_shared.py."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

ROOT = Path(sys.argv[1]); sys.path.append(str(ROOT))
from src.polymind_predict import build_polymind_features
from src.polymer_features import SHAPE_COLUMNS, build_chain_features, build_features

raw = pd.read_csv(ROOT / "data/raw/train.csv")
tr, _ = train_test_split(raw.index.to_numpy(), test_size=.2, random_state=42, shuffle=True)  # holdout never touched
v1 = build_polymind_features(raw.SMILES)
bplus, fp = build_features(raw.SMILES)
mine = {"B+": bplus, "chain": build_chain_features(raw.SMILES),
        "shape": pd.read_csv(ROOT / "data/processed/shape_descriptors.csv", index_col=0)[SHAPE_COLUMNS].set_axis(bplus.index), "D": fp}
full = ["B+", "chain", "shape", "D"]
P = {
 "FFV": (full, dict(learning_rate=0.073, max_depth=6, min_child_weight=2, n_estimators=1077, reg_lambda=1.25, subsample=0.877, colsample_bytree=0.641)),
 "Tc": (["B+", "D"], dict(n_estimators=708, learning_rate=0.0134, max_depth=5, min_child_weight=7, subsample=0.669, colsample_bytree=0.722, reg_lambda=4.64)),
 "Density": (full, dict(n_estimators=1464, learning_rate=0.0138, max_depth=8, min_child_weight=7, subsample=0.803, colsample_bytree=0.610, reg_lambda=0.425)),
 "Rg": (["B+", "chain"], dict(n_estimators=1031, learning_rate=0.0182, max_depth=8, min_child_weight=4, subsample=0.944, colsample_bytree=0.914, reg_lambda=0.858)),
}
V1 = {"FFV": ("morgan1024", "rf"), "Tc": ("expanded21", "xgb"), "Density": ("expanded21", "xgb"), "Rg": ("morgan1024", "rf")}
rows = []
for t, (keys, params) in P.items():
    r = tr[raw.loc[tr, t].notna().to_numpy()]; y = raw.loc[r, t].to_numpy()
    Xa = v1[V1[t][0]].loc[r].to_numpy(float); Xb = pd.concat([mine[k] for k in keys], axis=1).loc[r].to_numpy(float)
    for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=42).split(r), 1):
        m1 = RandomForestRegressor(random_state=42, n_jobs=-1) if V1[t][1] == "rf" else XGBRegressor(random_state=42, n_jobs=-1)
        m1.fit(Xa[a], y[a])
        m2 = make_pipeline(SimpleImputer(strategy="median"), XGBRegressor(**params, random_state=42, n_jobs=-1)); m2.fit(Xb[a], y[a])
        med = np.median(y[a])
        rows.append(dict(target=t, fold=f, n=len(b), v1=np.abs(y[b]-m1.predict(Xa[b])).mean(),
                         new=np.abs(y[b]-m2.predict(Xb[b])).mean(), median=np.abs(y[b]-med).mean()))
    print(t, "done", flush=True)
d = pd.DataFrame(rows); d.to_csv(ROOT / "results/polymind_v2_shared_split/fold_mae_v1_vs_v2.csv", index=False)
s = d.groupby("target", sort=False).agg(n=("n", "sum"), median=("median", "mean"), v1=("v1", "mean"), new=("new", "mean"),
      folds_better=("v1", lambda x: int((d.loc[x.index, "new"] < x).sum())))
s["change %"] = 100 * (s.new / s.v1 - 1); s["v1 vs median %"] = 100 * (1 - s.v1 / s["median"]); s["new vs median %"] = 100 * (1 - s.new / s["median"])
pd.set_option("display.width", 200); print(s.round(5).to_string())
