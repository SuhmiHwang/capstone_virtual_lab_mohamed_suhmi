# Polymind

Predict five polymer properties from a repeat-unit SMILES using a shared
Python interface and an interactive Streamlit demo.

**Authors:** Mohamed Elbadry and Suhmi Hwang

**Dataset:** NeurIPS Open Polymer Prediction 2025

## Run the demo

From the repository root, install the locked project dependencies with uv:

```powershell
uv sync --locked --all-groups
```

Start the app:

```powershell
uv run streamlit run app.py
```

Open the Local URL shown in the terminal.

1. Enter a polymer repeat-unit SMILES, such as `*CC*` or `*CCO*`.
2. Click **Predict properties**.
3. View the structure and the five predictions with their units.
4. Download the results as CSV.

The `*` symbols represent polymer attachment points.
The displayed structure is the represented repeat unit.

The demo loads the saved reference models from `models/polymind_v1/`.
It does not retrain models and does not require the raw dataset for inference.
Stop the server with **Ctrl+C** in the terminal.

## Python interface

Run Python from the repository root:

```python
from src.polymind_predict import predict_polymind

predictions = predict_polymind(["*CC*", "*CCO*"])
print(predictions)
```

The function returns a pandas DataFrame with columns
`Tg`, `FFV`, `Tc`, `Density` and `Rg`.

## Reference models

Each property has a separate model and feature configuration behind the
common prediction interface.

| Target | Property | Unit | Model | Features |
|---|---|---|---|---|
| Tg | Glass transition temperature | °C | Random Forest | Morgan fingerprints + 21 descriptors |
| FFV | Fractional free volume | dimensionless | Random Forest | Morgan fingerprints |
| Tc | Thermal conductivity | W/(m·K) | XGBoost | 21 descriptors |
| Density | Density | g/cm³ | XGBoost | 21 descriptors |
| Rg | Radius of gyration | Å | Random Forest | Morgan fingerprints |

Morgan fingerprints use 1,024 bits, radius 2 and chirality.
The saved manifest records feature order, model parameters, training-row
indices and library versions.

## Validation results

The reference experiment uses one shared random 80/20 training–validation
split with seed 42. Missing target labels are excluded separately for each
property; they are not replaced with model predictions for this experiment.

Model selection uses development cross-validation within the training
partition. Selected configurations were fixed before their holdout evaluation.
Saved models were fitted on the training partition only.

| Target | Training rows | Validation rows | MAE | RMSE | R² | Median baseline MAE |
|---|---:|---:|---:|---:|---:|---:|
| Tg | 424 | 87 | 45.91843 | 58.82817 | 0.66721 | 79.57905 |
| FFV | 5,611 | 1,419 | 0.00787 | 0.01213 | 0.79593 | 0.01991 |
| Tc | 592 | 145 | 0.03418 | 0.04888 | 0.69204 | 0.07404 |
| Density | 490 | 123 | 0.03423 | 0.07449 | 0.71944 | 0.09198 |
| Rg | 490 | 124 | 1.84297 | 2.81449 | 0.67466 | 4.11703 |

MAE, RMSE and baseline MAE use each target's unit listed above.
R² is dimensionless. Lower MAE and RMSE are better.
The baseline predicts the median of the corresponding training labels.

### Tg development findings

- Adding 21 descriptors to Morgan fingerprints reduced mean five-fold
  random-CV MAE from 56.04 to 53.88 °C.
- Targeted Random Forest tuning further reduced mean CV MAE to 53.14 °C.
- In an exploratory cluster-grouped comparison, Morgan + 21 descriptors
  achieved 54.37 °C mean MAE versus 56.36 °C for Morgan alone.
- These CV comparisons are development results, separate from the holdout
  metrics above.

## Checks performed

- Feature compatibility was verified for all 424 Tg training structures.
- Saved models reproduced the recorded holdout predictions for all five targets.
- Module predictions matched notebook predictions and also worked in a
  separate Python process.
- Empty input and empty-SMILES checks passed for the prediction interface.
- The demo was manually checked with `*CC*`, `*CCO*`, invalid SMILES and CSV export.
- Python compilation and Ruff checks passed for the demo.

## Limitations and ongoing experiments

Predictions are model estimates, not experimental measurements.
Large individual errors remain, and results concern one shared random holdout.
Performance on unfamiliar polymer families has not been established.


The inputs describe the represented repeat unit. They do not explicitly
provide processing conditions, chain molecular weight or formulation details.

The existing holdout has already been examined and must not guide further
model selection or tuning.

Additional descriptors, Uni-Mol features and external Tg data are separate
experiments and are not included in this reference bundle. Changes to feature
representations require the corresponding feature calculation and trained
model to be integrated together.

## Key project files

| Path | Purpose |
|---|---|
| `app.py` | Streamlit demo |
| `src/polymind_predict.py` | Feature calculation and five-property prediction interface |
| `src/descriptors.py` | Shared descriptor implementation |
| `models/polymind_v1/` | Five saved reference models and manifest |
| `notebooks/06_mohamed_shared_split_and_representations.ipynb` | Shared split and representation comparisons |
| `notebooks/07_mohamed_tg_error_analysis.ipynb` | Tg analysis, model comparisons and integration checks |
| `results/07_tg_validation/` | Fixed Tg candidate holdout results |
| `results/08_selected_model_validation/` | Recorded validation results for all five targets |
| `pyproject.toml` and `uv.lock` | Project dependencies and locked environment |

## Development

```powershell
uv run pytest
uv run ruff check app.py
uv run ruff format --check app.py
```

Use a separate branch and pull request for changes. Commit both
`pyproject.toml` and `uv.lock` when updating dependencies.

For tooling and environment help, see [TOOLS.md](TOOLS.md) and
[UV_SETUP.md](UV_SETUP.md).
