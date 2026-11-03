"""Calculate Uni-Mol embeddings for the repeat units and save them to a CSV file.

Uni-Mol (MIT licence, DP Technology) is a model pretrained on millions of 3D molecular
structures. Each repeat unit is turned into a normal molecule by capping the two open
ends (`*`), and the model's 512-number summary vector (CLS embedding) is saved.

This script needs the separate Uni-Mol environment, because `unimol-tools` requires
older package versions than the project:

    uv venv --python 3.12 %USERPROFILE%\\.venvs\\unimol
    uv pip install --python %USERPROFILE%\\.venvs\\unimol\\Scripts\\python.exe unimol-tools torch ^
        --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu

Usage (from the project root):
    %USERPROFILE%\\.venvs\\unimol\\Scripts\\python.exe -m src.unimol_cache data/raw/train.csv data/processed/unimol_embeddings.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

BATCH_SIZE = 500


def cap_repeat_unit(smiles):
    """Replace each `*` by H (single bond) or C (double/triple bond); None if impossible."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 0:
            single = all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())
            atom.SetAtomicNum(1 if single else 6)
            atom.SetNoImplicit(False)
    try:
        capped = editable.GetMol()
        Chem.SanitizeMol(capped)
        return Chem.MolToSmiles(Chem.RemoveHs(capped))
    except (Chem.rdchem.MolSanitizeException, RuntimeError, ValueError):
        return None


def compute_embeddings(smiles_file, out_file, limit=None):
    from unimol_tools import UniMolRepr

    smiles = pd.read_csv(smiles_file)["SMILES"]
    if limit:
        smiles = smiles.iloc[:limit]
    capped = smiles.apply(cap_repeat_unit)
    print(f"{capped.isna().sum()} repeat units could not be capped")

    model = UniMolRepr(data_type="molecule", remove_hs=False, use_gpu=False)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    valid = capped.dropna()

    # Resume: keep batches that are already saved
    done = pd.read_csv(out_file, index_col=0) if out_file.exists() else None
    todo = valid if done is None else valid.drop(done.index, errors="ignore")

    start = time.time()
    for begin in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[begin:begin + BATCH_SIZE]
        result = model.get_repr(batch.tolist(), return_atomic_reprs=False)
        # Older unimol-tools versions return a dict, newer ones the list of CLS vectors
        vectors = np.asarray(result["cls_repr"] if isinstance(result, dict) else result)
        part = pd.DataFrame(vectors, index=batch.index).add_prefix("unimol_")
        part.to_csv(out_file, mode="a", header=not out_file.exists())
        print(f"{begin + len(batch)} of {len(todo)} done ({time.time() - start:.0f} s)", flush=True)

    embeddings = pd.read_csv(out_file, index_col=0).sort_index().reindex(smiles.index)
    embeddings.to_csv(out_file)
    return embeddings


if __name__ == "__main__":
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    result = compute_embeddings(sys.argv[1], sys.argv[2], limit)
    print(result.shape, "| rows without embedding:", int(result.isna().any(axis=1).sum()))
