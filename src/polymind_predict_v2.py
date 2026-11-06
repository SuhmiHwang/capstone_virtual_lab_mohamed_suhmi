"""Polymind v2: five XGBoost models on the extended chemistry, chain and shape descriptors.

The models are trained on the shared training partition only (see tools/build_polymind_v2.py).
The v1 bundle and src/polymind_predict.py are unchanged.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

from src.polymer_features import (
    SHAPE_COLUMNS,
    build_chain_features,
    build_features,
    calculate_shape_descriptors,
)

POLYMIND_V2_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "polymind_v2"
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]


def build_v2_features(smiles, shape=None):
    """Return the four feature blocks used by the v2 models.

    smiles: one SMILES, an iterable of SMILES or a pandas Series.
    shape: optional precomputed shape descriptors with the same index (used for training).
    Shape descriptors need a 3D conformer; if none is found they are NaN and the models impute them.
    """
    if isinstance(smiles, str):
        smiles = pd.Series([smiles], name="SMILES")
    elif not isinstance(smiles, pd.Series):
        smiles = pd.Series(list(smiles), name="SMILES")

    for row_index, value in smiles.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or non-string SMILES at row {row_index}")
        molecule = Chem.MolFromSmiles(value)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"Invalid or empty structure at row {row_index}")

    descriptors, fingerprints = build_features(smiles)
    if shape is None:
        shape = pd.DataFrame([calculate_shape_descriptors(s) for s in smiles], index=smiles.index)
    return {
        "B+": descriptors,
        "chain": build_chain_features(smiles),
        "shape": shape.loc[:, SHAPE_COLUMNS].set_axis(descriptors.index),
        "D": fingerprints,
    }


def feature_matrix(blocks, keys, external_flag=None):
    """Join the blocks in manifest order; Tg adds a data-source column (0 = original data)."""
    matrix = pd.concat([blocks[k] for k in keys], axis=1)
    if external_flag is not None:
        matrix = matrix.assign(external_source=float(external_flag))
    return matrix


def predict_polymind_v2(smiles, model_dir=POLYMIND_V2_MODEL_DIR):
    """Predict all five properties with the v2 models."""
    model_dir = Path(model_dir)
    with (model_dir / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    blocks = build_v2_features(smiles)
    predictions = pd.DataFrame(index=blocks["B+"].index)
    for target in TARGETS:
        spec = manifest["targets"][target]
        flag = 0 if spec.get("external_source_flag") else None
        matrix = feature_matrix(blocks, spec["feature_blocks"], flag)
        if list(matrix.columns) != spec["feature_columns"]:
            raise ValueError(f"{target}: feature columns do not match the saved model.")
        model = joblib.load(model_dir / spec["file"])
        predictions[target] = model.predict(matrix.to_numpy(dtype=float))
        del model
    return predictions.loc[:, TARGETS]
