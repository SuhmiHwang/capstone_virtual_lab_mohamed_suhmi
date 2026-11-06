"""Turn a drawn or typed structure into a repeat-unit SMILES with two * attachment points."""

import re

from rdkit import Chem

# Ketcher and other editors write attachment points as *, [*], [*:1], R, [R1], [R#] ...
_R_GROUP = re.compile(r"\[R\d*#?\]|\bR\d*#?(?![a-z])")


def to_repeat_unit(text):
    """Return (smiles, message). smiles is None when the structure cannot be used."""
    block = text or ""
    text = block.strip()
    if not text:
        return None, "Draw a repeat unit or enter a SMILES first."
    if "M  END" in text:  # MOL block from the structure editor; * / A / R atoms become dummy atoms
        mol = Chem.MolFromMolBlock(block, sanitize=True)  # unstripped: the first lines are positional
        if mol is None:
            return None, ("This structure is not chemically valid (for example, an atom has too many bonds). "
                          "Please check the drawing.")
        if mol.GetNumAtoms() == 0:
            return None, "Draw a repeat unit or enter a SMILES first."
        else:
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0 or atom.GetSymbol() in ("*", "A", "R", "R#"):
                    atom.SetAtomicNum(0)
            text = Chem.MolToSmiles(mol)
    text = _R_GROUP.sub("*", text)
    mol = Chem.MolFromSmiles(text)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, "This structure could not be read. Please check it."
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "Please draw one connected repeat unit (the drawing has separate parts)."
    dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None, (f"A repeat unit needs exactly two attachment points (*), found {len(dummies)}. "
                      "Add a * atom where the chain continues on each side.")
    for atom in dummies:  # plain * without map numbers or isotopes
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
        atom.SetNoImplicit(True)
        if atom.GetDegree() != 1:
            return None, "Each attachment point (*) must be bonded to exactly one atom."
    return Chem.MolToSmiles(mol), None
