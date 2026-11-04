# Tg candidate comparison on the shared training split

## Evaluation

Four fixed configurations were compared on the same 424 polymers
with observed Tg labels in the shared training partition.
The original row order and five cross-validation folds were preserved.

The 87 outer Tg holdout structures were used only for structure
exclusion. Their target values were not used for fitting or scoring.

## Local results

MAE is the mean of the five fold MAEs, in degrees Celsius.

| Configuration | CV MAE | Mean fold R2 |
|---|---:|---:|
| Selected Random Forest reference | 53.1388 | 0.6121 |
| Default XGBoost, original data only | 54.4760 | 0.5899 |
| Default XGBoost with external data and source flag | 49.0792 | 0.6610 |
| Suhmi's fixed candidate with external data and source flag | 48.1591 | 0.6765 |

The fixed candidate reduced MAE by 4.98 degrees Celsius
(9.37%) relative to the Random Forest reference and improved
all five folds. No new hyperparameter search was performed.

## External data

The supplied cleaned pool contained 7,194 structures.
After additional global structure exclusions, 7,167 remained.
Further exclusions were applied within each fold to remove
external structures with Morgan Tanimoto similarity >= 0.95
to that fold's evaluation structures.

External samples received weight 0.5.
The feature implementation was supplied by Suhmi.
Exact settings and inclusion decisions are recorded in the
accompanying JSON and CSV files.

## Limitations

This is an exploratory development comparison, not independent
validation. Both candidates were previously selected using
overlapping training data.

The external pool had already been filtered using Suhmi's
different split. It was not rebuilt from the original source.
Canonical SMILES and fingerprint similarity do not guarantee
equivalence across different polymer repeat-unit boundaries.
Shape descriptors reuse supplied single-trimer conformer caches.

These results use a different split and stricter exclusions than
Suhmi's original experiment, so its reported MAE is not directly
comparable.

No replacement models were exported. The existing reference
bundle and Streamlit demo were not changed.

## Reproduce

Run from the project root, using a new output directory:

```powershell
uv run python tools/tg_shared_comparison/compare_tg_shared.py --data-root . --output results/tg_shared_candidate_review_rerun --jobs 2
```

Required inputs:
- data/raw/train.csv
- data/processed/external_tg_clean.csv
- data/processed/shape_descriptors.csv

The run manifest records package versions and input/source hashes.
