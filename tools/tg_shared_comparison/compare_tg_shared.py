"""Fixed-candidate Tg comparison on Mohamed's shared split; never score holdout.

Run: uv run python compare_tg_shared.py --data-root PATH --output NEW_DIRECTORY
Requires numpy, pandas, scipy, scikit-learn, rdkit, xgboost.
Companion feature modules must be next to this script.
"""
import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler, train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from descriptors_reference import calculate_descriptor_table
from polymer_features_suhmi import SHAPE_COLUMNS, build_chain_features, build_features


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    root = args.data_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
    RDLogger.DisableLog('rdApp.*')
    start = time.time()
    paths = {'train': root / 'data/raw/train.csv',
             'external': root / 'data/processed/external_tg_clean.csv',
             'shape': root / 'data/processed/shape_descriptors.csv'}
    if not paths['train'].is_file():
        paths['train'] = root / 'train.csv'
    raw = pd.read_csv(paths['train'])
    assert len(raw) == 7973 and raw.id.is_unique
    tr, va = train_test_split(raw.index.to_numpy(), test_size=.2, random_state=42,
                             shuffle=True)
    rows = tr[raw.loc[tr, 'Tg'].notna().to_numpy()]
    # Retain holdout structures only. No holdout targets are read below.
    holdout_rows = va[raw.loc[va, 'Tg'].notna().to_numpy()]
    assert len(rows) == 424 and len(holdout_rows) == 87
    y = raw.loc[rows, 'Tg'].copy()
    smiles = raw.SMILES.copy()
    ids = raw.id.copy()
    del raw
    folds = list(KFold(5, shuffle=True, random_state=42).split(rows))
    assignment = pd.DataFrame({'row_index': rows, 'id': ids.loc[rows], 'fold': 0})
    for f, (_, te) in enumerate(folds, 1):
        assignment.loc[rows[te], 'fold'] = f
    assignment.to_csv(out / 'training_fold_assignment.csv', index=False)
    print('Verified shared split: 424 Tg training rows, 87 untouched holdout rows.', flush=True)

    ext = pd.read_csv(paths['external'])
    assert len(ext) == 7194 and ext.canonical.is_unique
    original_ext_count = len(ext)
    mol_ours = {r: Chem.MolFromSmiles(smiles.loc[r]) for r in np.r_[rows, holdout_rows]}
    assert all(m is not None for m in mol_ours.values())
    canon_ours = {r: Chem.MolToSmiles(m) for r, m in mol_ours.items()}
    mol_ext = [Chem.MolFromSmiles(s) for s in ext.SMILES]
    assert all(m is not None for m in mol_ext)
    canon_ext = np.array([Chem.MolToSmiles(m) for m in mol_ext])
    assert (canon_ext == ext.canonical.to_numpy()).all(), 'Canonical cache/version mismatch'
    achiral = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024,
                                                        includeChirality=False)
    fp_ext = [achiral.GetFingerprint(m) for m in mol_ext]
    fp_ours = {r: achiral.GetFingerprint(m) for r, m in mol_ours.items()}
    # Exact duplicates of any labelled original Tg structure are never augmented.
    exact = np.isin(canon_ext, list(canon_ours.values()))
    # Holdout structures only: preserve the notebook's 0.95 exclusion rule on OUR split.
    max_holdout = np.zeros(len(ext))
    for r in holdout_rows:
        max_holdout = np.maximum(max_holdout,
                                DataStructs.BulkTanimotoSimilarity(fp_ours[r], fp_ext))
    eligible = ~exact & (max_holdout < .95)
    ledger = pd.DataFrame({'external_row_index': ext.index, 'canonical': canon_ext,
                           'exact_original_tg_match': exact,
                           'max_similarity_to_outer_tg_structures': max_holdout,
                           'eligible_before_cv': eligible})
    print(f'External pool: {eligible.sum()} / {len(ext)} eligible before fold exclusions.', flush=True)
    # Calculate features only for labelled training structures; no holdout predictions.
    shape = pd.read_csv(paths['shape'], index_col=0)
    assert shape.index.equals(smiles.index), 'Shape cache must follow raw train row indices'

    def candidate_features(ss, shape_frame):
        desc, fp = build_features(ss)
        chain = build_chain_features(ss)
        x = pd.concat([desc, chain, shape_frame[SHAPE_COLUMNS], fp], axis=1)
        assert x.shape[1] == 1097 and x.index.equals(ss.index)
        return x.to_numpy(dtype=float)

    print('Building Suhmi features from supplied shape caches...', flush=True)
    x_ours = candidate_features(smiles.loc[rows], shape.loc[rows])
    x_ext = candidate_features(ext.SMILES, ext)
    assert not np.isinf(x_ours).any() and not np.isinf(x_ext).any()
    np.savez_compressed(out / 'candidate_features.npz', original=x_ours, external=x_ext,
                        training_rows=rows, external_rows=ext.index.to_numpy())
    x_flag = np.column_stack([x_ours, np.zeros(len(rows))])
    xe_flag = np.column_stack([x_ext, np.ones(len(ext))])

    # Recreate the 21 descriptors exactly as printed in notebook 07, cell 53.
    d13 = calculate_descriptor_table(smiles.loc[rows], extended=True)
    additional = ['FractionCSP3', 'NumRotatableBonds', 'NumAliphaticRings',
                  'NumSaturatedRings', 'NumHeteroatoms', 'LabuteASA', 'BertzCT', 'MolMR']
    d8 = np.array([[float(getattr(Descriptors, k)(mol_ours[r])) for k in additional]
                   for r in rows])
    chiral = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024,
                                                       includeChirality=True,
                                                       countSimulation=False)
    ref = np.column_stack([np.stack([chiral.GetFingerprintAsNumPy(mol_ours[r]) for r in rows]),
                           d13.to_numpy(), d8])
    assert ref.shape == (424, 1045) and np.isfinite(ref).all()
    defaults = dict(n_estimators=500, learning_rate=.05, max_depth=4,
                    subsample=.8, colsample_bytree=.8)
    search_space = dict(n_estimators=randint(200, 1501), learning_rate=loguniform(.01, .2),
                        max_depth=randint(2, 9), min_child_weight=randint(1, 11),
                        subsample=uniform(.6, .4), colsample_bytree=uniform(.6, .4),
                        reg_lambda=loguniform(.1, 10))
    # Zero-based candidate 21, selected previously in Suhmi notebook 12; no new search.
    tuned = list(ParameterSampler(search_space, n_iter=25, random_state=42))[21]
    params = {'default_xgb': defaults, 'suhmi_candidate21': tuned}
    (out / 'fixed_parameters.json').write_text(json.dumps(params, indent=2, default=int))
    metrics, oof = [], []
    for f, (fit, test) in enumerate(folds, 1):
        max_cv = np.zeros(len(ext))
        for r in rows[test]:
            max_cv = np.maximum(max_cv, DataStructs.BulkTanimotoSimilarity(fp_ours[r], fp_ext))
        keep = eligible & (max_cv < .95)
        ledger[f'max_similarity_fold_{f}'] = max_cv
        ledger[f'included_fold_{f}'] = keep
        ledger.to_csv(out / 'external_exclusion_ledger.csv', index=False)
        # All variants retain the exact training-row order used by Mohamed's CV.
        for variant in ['reference_rf', 'xgb_default_original', 'xgb_default_external',
                        'xgb_fixed21_external']:
            if variant == 'reference_rf':
                model = RandomForestRegressor(n_estimators=200, random_state=42,
                                               max_features=.6, min_samples_leaf=2,
                                               n_jobs=args.jobs)
                model.fit(ref[fit], y.iloc[fit])
                pred = model.predict(ref[test])
            else:
                p = tuned if variant == 'xgb_fixed21_external' else defaults
                model = make_pipeline(SimpleImputer(strategy='median'),
                                      XGBRegressor(**p, random_state=42, n_jobs=args.jobs))
                if variant == 'xgb_default_original':
                    model.fit(x_ours[fit], y.iloc[fit])
                    pred = model.predict(x_ours[test])
                else:
                    xx = np.vstack([x_flag[fit], xe_flag[keep]])
                    yy = np.r_[y.iloc[fit].to_numpy(), ext.loc[keep, 'Tg'].to_numpy()]
                    weights = np.r_[np.ones(len(fit)), np.full(keep.sum(), .5)]
                    model.fit(xx, yy, xgbregressor__sample_weight=weights)
                    pred = model.predict(x_flag[test])
            baseline = mean_absolute_error(y.iloc[test], np.full(len(test), y.iloc[fit].median()))
            mae = mean_absolute_error(y.iloc[test], pred)
            metrics.append(dict(variant=variant, fold=f, n_test=len(test), mae=mae,
                                r2=r2_score(y.iloc[test], pred), baseline_mae=baseline,
                                error_reduction_pct=100*(1-mae/baseline),
                                external_rows=int(keep.sum()) if variant.endswith('external') else 0))
            oof.extend(dict(variant=variant, row_index=int(r), fold=f,
                            observed=float(y.loc[r]), predicted=float(p))
                       for r, p in zip(rows[test], pred))
            pd.DataFrame(metrics).to_csv(out / 'fold_metrics.csv', index=False)
            pd.DataFrame(oof).to_csv(out / 'oof_predictions.csv', index=False)
            print(f'Fold {f}, {variant}: MAE {mae:.4f} C ({time.time()-start:.0f}s)', flush=True)
    m = pd.DataFrame(metrics)
    summary = m.groupby('variant', sort=False).agg(mean_fold_mae=('mae','mean'),
                    sd_fold_mae=('mae','std'), mean_fold_r2=('r2','mean'),
                    mean_error_reduction_pct=('error_reduction_pct','mean'))
    predictions = pd.DataFrame(oof)
    for v in summary.index:
        p = predictions[predictions.variant.eq(v)]
        assert len(p) == 424 and p.row_index.is_unique
        summary.loc[v, 'pooled_oof_mae'] = mean_absolute_error(p.observed, p.predicted)
    summary.to_csv(out / 'summary.csv')
    pairs = m.pivot(index='fold', columns='variant', values='mae')
    for v in summary.index:
        pairs[f'{v}_minus_reference'] = pairs[v] - pairs.reference_rf
    pairs.to_csv(out / 'paired_fold_comparison.csv')
    manifest = dict(python=platform.python_version(),
                    versions={k: version(k) for k in ['rdkit','xgboost','scikit-learn','numpy','scipy','pandas']},
                    input_sha256={k:digest(v) for k,v in paths.items()},
                    source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                    training_rows=424, outer_tg_rows_not_scored=87,
                    original_external_pool=original_ext_count,
                    external_eligible_before_cv=int(eligible.sum()),
                    similarity_threshold=.95, external_weight=.5,
                    elapsed_seconds=time.time()-start,
                    limitations=[
                        'Exploratory comparison, not independent validation: candidate parameters previously selected on overlapping training data.',
                        'External input is the supplied 7194-row pool, already filtered using Suhmis different split; original 7225-row pool not rebuilt.',
                        'Additional exact/all-labelled and near/evaluation-fold exclusions differ from Suhmis published CV protocol.',
                        'RDKit canonical SMILES plus achiral Morgan similarity do not guarantee polymer-repeat-boundary equivalence.',
                        'Shape cache alignment verified by row indices; source cache lacks per-row SMILES checksums.',
                        'No holdout scores generated, no fitted models exported, no reference/demo files modified.'
                    ])
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(summary.round(4).to_string(), flush=True)


if __name__ == '__main__':
    main()
