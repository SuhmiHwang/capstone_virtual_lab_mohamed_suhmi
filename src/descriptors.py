"""RDKit descriptors for polymer repeat units given as SMILES.

Reusable version of the descriptor calculation in the notebooks, defined in
specs/002-polymer-descriptors.md:

- The nine base descriptors follow Suhmi's implementation in
  notebooks/02_Suhmi_Featureengineering.ipynb (cell 41), as reused in
  notebooks/01_mohamed_initial_eda.ipynb (cell 111) and
  notebooks/05_mohamed_model_diagnostics_and_feature_comparison.ipynb (cell 122).
- The four additional charge and attachment-point descriptors follow
  notebook 05, cell 229.

The descriptors describe the represented repeat unit, including its `*`
attachment points. MolWt is not the molecular weight of a polymer chain.
NumAtoms counts explicit graph atoms including `*`, HeavyAtomCount excludes
them, and neither counts implicit hydrogens. Charges and `*` dummy atoms are
used exactly as given in the SMILES; nothing is neutralised, removed or capped.

This module is independent of the template pipeline (features, train, predict).
"""

import math
from collections.abc import Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

BASE_DESCRIPTORS = (
    "MolWt",
    "HeavyAtomCount",
    "NumAtoms",
    "NumRings",
    "NumAromaticRings",
    "NumHDonors",
    "NumHAcceptors",
    "MolLogP",
    "TPSA",
)
EXTENDED_DESCRIPTORS = (
    "NetFormalCharge",
    "PositiveAtomCount",
    "NegativeAtomCount",
    "DummyAtomCount",
)
ALL_DESCRIPTORS = BASE_DESCRIPTORS + EXTENDED_DESCRIPTORS


def _is_missing(smiles) -> bool:
    """True for None, NaN, pd.NA and empty or whitespace-only strings."""
    if smiles is None or smiles is pd.NA:
        return True
    if isinstance(smiles, float):
        if math.isnan(smiles):
            return True
    elif isinstance(smiles, str):
        return smiles.strip() == ""
    raise TypeError(f"SMILES must be a string, None or NaN, not {type(smiles).__name__}")


def _parse(smiles) -> tuple[Chem.Mol | None, str]:
    """Parse with RDKit default settings and return the molecule and its status."""
    if _is_missing(smiles):
        return None, "missing"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid"
    return mol, "valid"


def smiles_status(smiles) -> str:
    """Return 'missing', 'invalid' or 'valid' for one SMILES."""
    return _parse(smiles)[1]


def _base_descriptors(mol: Chem.Mol) -> dict[str, int | float]:
    return {
        "MolWt": Descriptors.MolWt(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NumAtoms": mol.GetNumAtoms(),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
    }


def _additional_descriptors(mol: Chem.Mol) -> dict[str, int]:
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    return {
        "NetFormalCharge": sum(charges),
        "PositiveAtomCount": sum(charge > 0 for charge in charges),
        "NegativeAtomCount": sum(charge < 0 for charge in charges),
        "DummyAtomCount": sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()),
    }


def calculate_descriptors(smiles, *, extended: bool = False) -> dict[str, int | float]:
    """Return the descriptors of one SMILES, keys in fixed order.

    With `extended=True` the four charge and attachment-point descriptors are
    appended. Missing, empty or unparsable input gives every name with NaN;
    use `smiles_status` to tell these cases apart.
    """
    names = ALL_DESCRIPTORS if extended else BASE_DESCRIPTORS
    mol, _ = _parse(smiles)
    if mol is None:
        return dict.fromkeys(names, math.nan)

    values = _base_descriptors(mol)
    if extended:
        values.update(_additional_descriptors(mol))
    return values


def calculate_descriptor_table(smiles: Iterable, *, extended: bool = False) -> pd.DataFrame:
    """Return one row of descriptors per SMILES, columns in fixed order.

    A pandas Series keeps its index. An empty input gives 0 rows with all columns.
    """
    if isinstance(smiles, str):
        raise TypeError("Pass an iterable of SMILES, not a single string")

    names = list(ALL_DESCRIPTORS if extended else BASE_DESCRIPTORS)
    index = smiles.index if isinstance(smiles, pd.Series) else None
    rows = [calculate_descriptors(value, extended=extended) for value in smiles]
    return pd.DataFrame(rows, index=index, columns=names)
