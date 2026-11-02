"""Predict five polymer properties using the fixed v1 models.

Models were fitted on the shared training partition only.
Predictions are estimates for the represented polymer repeat unit.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from src.descriptors import calculate_descriptor_table, ALL_DESCRIPTORS

POLYMIND_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "polymind_v1"
)

ADDITIONAL_DESCRIPTOR_NAMES = ['FractionCSP3', 'NumRotatableBonds', 'NumAliphaticRings', 'NumSaturatedRings', 'NumHeteroatoms', 'LabuteASA', 'BertzCT', 'MolMR']

POLYMIND_DESCRIPTOR_COLUMNS = ['MolWt', 'HeavyAtomCount', 'NumAtoms', 'NumRings', 'NumAromaticRings', 'NumHDonors', 'NumHAcceptors', 'MolLogP', 'TPSA', 'NetFormalCharge', 'PositiveAtomCount', 'NegativeAtomCount', 'DummyAtomCount', 'FractionCSP3', 'NumRotatableBonds', 'NumAliphaticRings', 'NumSaturatedRings', 'NumHeteroatoms', 'LabuteASA', 'BertzCT', 'MolMR']

POLYMIND_MORGAN_SETTINGS = {'radius': 2, 'fp_size': 1024, 'include_chirality': True, 'count_simulation': False}

def build_polymind_features(smiles):
    """Return the three selected representations in fixed column order.

    Accept one SMILES, an iterable of SMILES, or a pandas Series.
    Preserve Series indices and reject missing or invalid structures.
    """
    if isinstance(smiles, str):
        smiles = pd.Series([smiles], name="SMILES")
    elif isinstance(smiles, pd.Series):
        smiles = smiles.copy()
    else:
        smiles = pd.Series(list(smiles), name="SMILES")

    molecules = []
    for row_index, value in smiles.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or non-string SMILES at row {row_index}")

        molecule = Chem.MolFromSmiles(value)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"Invalid or empty structure at row {row_index}")

        molecules.append(molecule)

    # Reuse the original implementation for the first 13 descriptors.
    descriptors13 = calculate_descriptor_table(smiles, extended=True)

    additional = pd.DataFrame(
        [
            {
                name: float(getattr(Descriptors, name)(molecule))
                for name in ADDITIONAL_DESCRIPTOR_NAMES
            }
            for molecule in molecules
        ],
        index=smiles.index,
        columns=ADDITIONAL_DESCRIPTOR_NAMES,
    )

    descriptors21 = pd.concat(
        [descriptors13, additional], axis=1
    ).loc[:, POLYMIND_DESCRIPTOR_COLUMNS]

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=POLYMIND_MORGAN_SETTINGS["radius"],
        fpSize=POLYMIND_MORGAN_SETTINGS["fp_size"],
        includeChirality=POLYMIND_MORGAN_SETTINGS["include_chirality"],
        countSimulation=POLYMIND_MORGAN_SETTINGS["count_simulation"],
    )

    fingerprint_values = (
        np.stack([
            generator.GetFingerprintAsNumPy(molecule)
            for molecule in molecules
        ])
        if molecules
        else np.empty((0, POLYMIND_MORGAN_SETTINGS["fp_size"]), dtype=np.uint8)
    )

    fingerprints = pd.DataFrame(
        fingerprint_values,
        index=smiles.index,
        columns=[
            f"morgan_{position}"
            for position in range(POLYMIND_MORGAN_SETTINGS["fp_size"])
        ],
        dtype=np.uint8,
    )

    combined = pd.concat(
        [fingerprints, descriptors21.add_prefix("descriptor_")],
        axis=1,
    )

    representations = {
        "expanded21": descriptors21,
        "morgan1024": fingerprints,
        "morgan_plus21": combined,
    }

    for name, features in representations.items():
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {name}")

    return representations


def predict_polymind(smiles, model_dir=POLYMIND_MODEL_DIR):
    """Predict all five properties with the fixed training-partition models."""
    model_dir = Path(model_dir)

    with (model_dir / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest["morgan_settings"] != POLYMIND_MORGAN_SETTINGS:
        raise ValueError("Fingerprint settings do not match the saved models.")
    if manifest["descriptor_columns"] != POLYMIND_DESCRIPTOR_COLUMNS:
        raise ValueError("Descriptor definitions do not match the saved models.")

    features = build_polymind_features(smiles)
    predictions = pd.DataFrame(index=features["expanded21"].index)

    for target, specification in manifest["targets"].items():
        matrix = features[specification["feature_set"]].loc[
            :, specification["feature_columns"]
        ]

        if matrix.empty:
            predictions[target] = pd.Series(
                index=matrix.index, dtype=float
            )
            continue

        model = joblib.load(model_dir / specification["file"])
        predictions[target] = model.predict(matrix)
        del model

    return predictions.loc[:, ["Tg", "FFV", "Tc", "Density", "Rg"]]
