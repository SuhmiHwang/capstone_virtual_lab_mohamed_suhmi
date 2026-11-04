"""Polymer features calculated from SMILES.

The same descriptors as in notebook 04 (sections 17, 21 and 23), collected in
one module so that other notebooks and the prediction pipeline can reuse them:

- the nine original RDKit descriptors (feature set A),
- the twelve chemistry-motivated descriptors (feature set B),
- the functional-group and side-group descriptors (feature set B+),
- 1,024-bit Morgan fingerprints (feature set D).
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_COLUMNS = [
    "MolWt",
    "HeavyAtomCount",
    "NumAtoms",
    "NumRings",
    "NumAromaticRings",
    "NumHDonors",
    "NumHAcceptors",
    "MolLogP",
    "TPSA",
]

HALOGEN_ATOMIC_NUMBERS = {9, 17, 35, 53}

# Polymer functional groups. The atom labelled :1 is the anchor atom
# that decides where the group sits (backbone or side chain).
FUNCTIONAL_GROUPS = {
    "Ester": "[#6][CX3:1](=O)[OX2][#6]",
    "Carbonate": "[OX2][CX3:1](=O)[OX2]",
    "Amide": "[#6][CX3:1](=O)[NX3;!$(N([CX3]=O)[CX3]=O)]",
    "Imide": "[CX3](=O)[NX3:1][CX3](=O)",
    "Urethane": "[OX2][CX3:1](=O)[NX3]",
    "Ether": "[OX2;!$(O[#6]=O):1]([#6])[#6]",
    "Siloxane": "[Si][OX2:1][Si]",
    "Sulfone": "[SX4:1](=O)(=O)",
    "Hydroxyl": "[OX2H:1][CX4,c]",
    "Nitrile": "[CX2:1]#[NX1]",
}

# Hydroxyl and nitrile groups are terminal, so they never occur in the backbone
CONSTANT_GROUP_COLUMNS = ["HydroxylBackbone", "NitrileBackbone"]

GROUP_PATTERNS = {}
for _name, _smarts in FUNCTIONAL_GROUPS.items():
    _pattern = Chem.MolFromSmarts(_smarts)
    _anchor = next(atom.GetIdx() for atom in _pattern.GetAtoms() if atom.GetAtomMapNum() == 1)
    GROUP_PATTERNS[_name] = (_pattern, _anchor)

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def calculate_descriptors(mol):
    """The nine original descriptors (feature set A)."""
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


def calculate_chemistry_descriptors(mol):
    """The twelve chemistry-motivated descriptors added in feature set B."""
    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    n_heavy = len(heavy_atoms)
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]

    descriptors = {
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "AromaticAtomFraction": sum(atom.GetIsAromatic() for atom in heavy_atoms) / n_heavy,
        "HalogenCount": sum(
            atom.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS for atom in heavy_atoms
        ),
        "SulfurCount": sum(atom.GetAtomicNum() == 16 for atom in heavy_atoms),
        "SiliconCount": sum(atom.GetAtomicNum() == 14 for atom in heavy_atoms),
        "MolWtPerHeavyAtom": Descriptors.MolWt(mol) / n_heavy,
        "MolLogPPerHeavyAtom": Descriptors.MolLogP(mol) / n_heavy,
        "TPSAPerHeavyAtom": Descriptors.TPSA(mol) / n_heavy,
    }

    # Backbone = shortest path between the two attachment points
    if len(dummy_indices) == 2:
        path = Chem.GetShortestPath(mol, dummy_indices[0], dummy_indices[1])
        backbone = [idx for idx in path if idx not in dummy_indices]
        ring_info = mol.GetRingInfo()
        backbone_in_ring = sum(ring_info.NumAtomRings(idx) > 0 for idx in backbone)

        descriptors["BackboneLength"] = len(backbone)
        descriptors["BackboneRingFraction"] = (
            backbone_in_ring / len(backbone) if backbone else np.nan
        )
        descriptors["SideChainFraction"] = 1 - len(backbone) / n_heavy
    else:
        descriptors["BackboneLength"] = np.nan
        descriptors["BackboneRingFraction"] = np.nan
        descriptors["SideChainFraction"] = np.nan

    return descriptors


def build_trimer(mol, dummy_indices):
    """Join three copies of the repeat unit, so that groups crossing the
    attachment points (e.g. the carbonate in polycarbonate) are complete."""
    n_atoms = mol.GetNumAtoms()
    first, last = dummy_indices
    first_neighbor = mol.GetAtomWithIdx(first).GetNeighbors()[0].GetIdx()
    last_neighbor = mol.GetAtomWithIdx(last).GetNeighbors()[0].GetIdx()
    bond_type = mol.GetBondBetweenAtoms(last, last_neighbor).GetBondType()

    labelled = Chem.Mol(mol)
    for atom in labelled.GetAtoms():
        atom.SetIntProp("repeat_index", atom.GetIdx())

    trimer = Chem.RWMol(Chem.CombineMols(Chem.CombineMols(labelled, labelled), labelled))
    for copy in range(3):
        for atom_idx in range(n_atoms):
            trimer.GetAtomWithIdx(copy * n_atoms + atom_idx).SetIntProp("copy", copy)

    for copy in range(2):
        trimer.AddBond(
            copy * n_atoms + last_neighbor, (copy + 1) * n_atoms + first_neighbor, bond_type
        )

    internal_dummies = [last, n_atoms + first, n_atoms + last, 2 * n_atoms + first]
    for idx in sorted(internal_dummies, reverse=True):
        trimer.RemoveAtom(idx)

    trimer = trimer.GetMol()
    Chem.SanitizeMol(trimer)
    return trimer


def calculate_group_descriptors(mol):
    """Functional-group and side-group descriptors added in feature set B+."""
    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    n_heavy = len(heavy_atoms)
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]

    descriptors = {
        "HDonorsPerHeavyAtom": Lipinski.NumHDonors(mol) / n_heavy,
        "HAcceptorsPerHeavyAtom": Lipinski.NumHAcceptors(mol) / n_heavy,
    }
    location_columns = [
        f"{name}{place}" for name in GROUP_PATTERNS for place in ["Backbone", "SideChain"]
    ]

    # Backbone-based descriptors require exactly two attachment points
    if len(dummy_indices) != 2:
        descriptors.update({col: np.nan for col in location_columns})
        descriptors["LargestSideGroup"] = np.nan
        descriptors["BackboneUnsaturation"] = np.nan
        return descriptors

    path = Chem.GetShortestPath(mol, dummy_indices[0], dummy_indices[1])
    backbone = set(path) - set(dummy_indices)
    trimer = build_trimer(mol, dummy_indices)

    # Count each group once, using the anchor atoms in the middle copy
    for name, (pattern, anchor) in GROUP_PATTERNS.items():
        in_backbone = in_side_chain = 0
        for match in trimer.GetSubstructMatches(pattern):
            anchor_atom = trimer.GetAtomWithIdx(match[anchor])
            if anchor_atom.GetIntProp("copy") != 1:
                continue
            if anchor_atom.GetIntProp("repeat_index") in backbone:
                in_backbone += 1
            else:
                in_side_chain += 1
        descriptors[f"{name}Backbone"] = in_backbone
        descriptors[f"{name}SideChain"] = in_side_chain

    # Size of the largest connected group of atoms outside the backbone
    side_atoms = {atom.GetIdx() for atom in heavy_atoms} - backbone
    largest, visited = 0, set()
    for start in side_atoms:
        if start in visited:
            continue
        stack, size = [start], 0
        visited.add(start)
        while stack:
            idx = stack.pop()
            size += 1
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                if neighbor.GetIdx() in side_atoms and neighbor.GetIdx() not in visited:
                    visited.add(neighbor.GetIdx())
                    stack.append(neighbor.GetIdx())
        largest = max(largest, size)
    descriptors["LargestSideGroup"] = largest

    # Share of non-aromatic double and triple bonds along the backbone
    path_bonds = [mol.GetBondBetweenAtoms(a, b) for a, b in zip(path, path[1:])]
    unsaturated = sum(
        not bond.GetIsAromatic() and bond.GetBondTypeAsDouble() >= 2 for bond in path_bonds
    )
    descriptors["BackboneUnsaturation"] = unsaturated / len(path_bonds)

    return descriptors


def build_features(smiles):
    """Return feature set B+ and the Morgan fingerprints for a list of SMILES.

    Both DataFrames use the same index as `smiles` (a pandas Series).
    """
    mols = smiles.apply(Chem.MolFromSmiles)

    descriptor_df = pd.DataFrame(
        [calculate_descriptors(mol) for mol in mols], index=smiles.index
    )[DESCRIPTOR_COLUMNS]
    chemistry_df = pd.DataFrame(
        [calculate_chemistry_descriptors(mol) for mol in mols], index=smiles.index
    )
    group_df = pd.DataFrame(
        [calculate_group_descriptors(mol) for mol in mols], index=smiles.index
    ).drop(columns=CONSTANT_GROUP_COLUMNS)
    fingerprint_df = pd.DataFrame(
        [MORGAN_GENERATOR.GetFingerprintAsNumPy(mol) for mol in mols], index=smiles.index
    ).add_prefix("Morgan_")

    bplus_df = pd.concat([descriptor_df, chemistry_df, group_df], axis=1)
    return bplus_df, fingerprint_df


# ---------------------------------------------------------------------------
# Chain descriptors (notebook 06)
# ---------------------------------------------------------------------------

# Atomic van der Waals volumes in Å³ (Zhao, Abraham and Zissimos, J. Org. Chem. 2003)
VDW_ATOM_VOLUMES = {
    1: 7.24, 5: 40.48, 6: 20.58, 7: 15.60, 8: 14.71, 9: 13.31, 14: 38.79, 15: 24.43,
    16: 24.43, 17: 22.45, 33: 26.52, 34: 28.73, 35: 26.52, 52: 36.62, 53: 32.52,
}
AVOGADRO_CM3 = 0.6022  # 1 Å³ per molecule = 0.6022 cm³ per mol

AMIDE_BOND = Chem.MolFromSmarts("[CX3](=O)-[NX3]")
QUATERNARY_CARBON = Chem.MolFromSmarts("[CX4;H0]([#6])([#6])([#6])[#6]")

CHAIN_COLUMNS = [
    "BackboneRotatableBonds", "BackboneRotatableFraction", "LongestFlexibleRun",
    "FullyFlexibleBackbone", "BackboneAromaticFraction", "BackboneRingCount",
    "ParaLinkages", "MetaLinkages", "OrthoLinkages", "ParaLinkageFraction",
    "NumSubstituents", "NumSideChains", "LongestSideChain", "MeanSubstituentSize",
    "SideChainRotatableBonds", "SideChainBranchPoints", "PendantAromaticRings",
    "QuaternaryCarbons", "VdWVolume", "SpecificVdWVolume", "VdWVolumePerHeavyAtom",
    "LabuteASAPerVolume", "KierFlexibility",
]
SHAPE_COLUMNS = [
    "TrimerRadiusOfGyration", "TrimerRgPerBackboneAtom", "TrimerEndToEndRatio",
    "TrimerAsphericity", "TrimerEccentricity", "TrimerNPR1", "TrimerNPR2",
]


def _is_flexible(mol, bond, amide_bonds):
    """Single, non-ring bond that is not a (partly double) amide C–N bond."""
    if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
        return False
    return frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) not in amide_bonds


def calculate_vdw_volume(mol):
    """Van der Waals volume of one repeat unit (Zhao et al. 2003), in Å³.

    V = sum of atom volumes − 5.92·bonds − 14.7·aromatic rings − 3.8·other rings.
    The two bonds to the attachment points form one bond of the chain.
    """
    heavy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0]
    n_hydrogens = sum(atom.GetTotalNumHs() for atom in heavy)
    volume = sum(VDW_ATOM_VOLUMES.get(atom.GetAtomicNum(), 30.0) for atom in heavy)
    volume += VDW_ATOM_VOLUMES[1] * n_hydrogens

    n_dummy_bonds = sum(
        bond.GetBeginAtom().GetAtomicNum() == 0 or bond.GetEndAtom().GetAtomicNum() == 0
        for bond in mol.GetBonds()
    )
    n_bonds = mol.GetNumBonds() - n_dummy_bonds + min(n_dummy_bonds, 1) + n_hydrogens
    n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_other = rdMolDescriptors.CalcNumRings(mol) - n_aromatic
    return volume - 5.92 * n_bonds - 14.7 * n_aromatic - 3.8 * n_other


def calculate_chain_descriptors(mol):
    """Backbone flexibility, ring linkages, side chains and van der Waals volume."""
    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    n_heavy = len(heavy_atoms)
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    amide_bonds = {frozenset((m[0], m[2])) for m in mol.GetSubstructMatches(AMIDE_BOND)}

    volume = calculate_vdw_volume(mol)
    descriptors = {
        "QuaternaryCarbons": len(mol.GetSubstructMatches(QUATERNARY_CARBON)),
        "VdWVolume": volume,
        "SpecificVdWVolume": volume * AVOGADRO_CM3 / Descriptors.MolWt(mol),
        "VdWVolumePerHeavyAtom": volume / n_heavy,
        "LabuteASAPerVolume": rdMolDescriptors.CalcLabuteASA(mol) / volume,
        "KierFlexibility": rdMolDescriptors.CalcPhi(mol),
    }
    if len(dummy_indices) != 2:
        descriptors.update({col: np.nan for col in CHAIN_COLUMNS if col not in descriptors})
        return descriptors

    path = list(Chem.GetShortestPath(mol, dummy_indices[0], dummy_indices[1]))
    backbone = path[1:-1]
    backbone_set = set(backbone)
    ring_info = mol.GetRingInfo()

    # Backbone bonds of one repeat unit; the last bond (to *) is the link to the next unit
    bonds = [mol.GetBondBetweenAtoms(a, b) for a, b in zip(path[1:], path[2:])]
    flexible = [_is_flexible(mol, bond, amide_bonds) for bond in bonds]
    n_flexible = sum(flexible)

    # Longest run of consecutive flexible bonds; the chain repeats, so the run wraps around
    if all(flexible):
        longest_run = len(flexible)
    else:
        run, longest_run = 0, 0
        for value in flexible + flexible:
            run = run + 1 if value else 0
            longest_run = max(longest_run, run)

    descriptors.update({
        "BackboneRotatableBonds": n_flexible,
        "BackboneRotatableFraction": n_flexible / len(bonds),
        "LongestFlexibleRun": longest_run,
        "FullyFlexibleBackbone": int(all(flexible)),
        "BackboneAromaticFraction": sum(mol.GetAtomWithIdx(i).GetIsAromatic() for i in backbone)
        / len(backbone),
    })

    # Rings crossed by the backbone; for benzene rings, the distance between the
    # entry and exit atoms gives the linkage: 1 = ortho, 2 = meta, 3 = para
    backbone_rings = [ring for ring in ring_info.AtomRings() if backbone_set & set(ring)]
    linkages = {1: 0, 2: 0, 3: 0}
    for ring in backbone_rings:
        if len(ring) != 6 or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        on_path = [i for i in path if i in ring]
        entry, exit_ = ring.index(on_path[0]), ring.index(on_path[-1])
        distance = min(abs(entry - exit_), 6 - abs(entry - exit_))
        if distance in linkages:
            linkages[distance] += 1
    n_linkages = sum(linkages.values())
    descriptors.update({
        "BackboneRingCount": len(backbone_rings),
        "OrthoLinkages": linkages[1],
        "MetaLinkages": linkages[2],
        "ParaLinkages": linkages[3],
        "ParaLinkageFraction": linkages[3] / n_linkages if n_linkages else 0.0,
    })

    # Substituents: heavy atoms outside the backbone and outside rings that contain backbone atoms
    in_backbone_ring = set().union(*map(set, backbone_rings)) if backbone_rings else set()
    chain_atoms = backbone_set | in_backbone_ring
    pendant = {atom.GetIdx() for atom in heavy_atoms} - chain_atoms

    components, visited = [], set()
    for start in pendant:
        if start in visited:
            continue
        stack, members = [start], [start]
        visited.add(start)
        while stack:
            for neighbor in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                idx = neighbor.GetIdx()
                if idx in pendant and idx not in visited:
                    visited.add(idx)
                    stack.append(idx)
                    members.append(idx)
        components.append(members)

    # Longest substituent, counted in atoms from its attachment point outwards
    distances = Chem.GetDistanceMatrix(mol)
    longest = 0
    for members in components:
        roots = [i for i in members
                 if any(n.GetIdx() in chain_atoms for n in mol.GetAtomWithIdx(i).GetNeighbors())]
        if roots:
            longest = max(longest, int(max(distances[r][m] for r in roots for m in members)) + 1)

    side_bonds = [bond for bond in mol.GetBonds()
                  if bond.GetBeginAtomIdx() in pendant or bond.GetEndAtomIdx() in pendant]
    side_rotatable = sum(
        _is_flexible(mol, bond, amide_bonds)
        and bond.GetBeginAtom().GetDegree() > 1 and bond.GetEndAtom().GetDegree() > 1
        and bond.GetBeginAtom().GetAtomicNum() > 0 and bond.GetEndAtom().GetAtomicNum() > 0
        for bond in side_bonds
    )
    pendant_rings = [ring for ring in ring_info.AtomRings() if set(ring) <= pendant]

    descriptors.update({
        "NumSubstituents": len(components),
        "NumSideChains": sum(len(members) >= 2 for members in components),
        "LongestSideChain": longest,
        "MeanSubstituentSize": np.mean([len(m) for m in components]) if components else 0.0,
        "SideChainRotatableBonds": side_rotatable,
        "SideChainBranchPoints": sum(
            sum(n.GetAtomicNum() > 1 for n in mol.GetAtomWithIdx(i).GetNeighbors()) >= 3
            for i in pendant
        ),
        "PendantAromaticRings": sum(
            all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) for ring in pendant_rings
        ),
    })
    return descriptors


def calculate_shape_descriptors(smiles, seed=42):
    """3D shape of one conformer of the trimer (three joined repeat units).

    The two open ends are capped with hydrogen. Returns NaN if no conformer is found.
    """
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_indices) != 2:
        return {col: np.nan for col in SHAPE_COLUMNS}

    trimer = Chem.RWMol(build_trimer(mol, dummy_indices))
    ends = [atom.GetIdx() for atom in trimer.GetAtoms() if atom.GetAtomicNum() == 0]
    end_atoms = [trimer.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() for i in ends]
    for idx in ends:
        # Cap a single-bonded end with hydrogen, a double-bonded end with carbon
        end = trimer.GetAtomWithIdx(idx)
        single = end.GetBonds()[0].GetBondType() == Chem.BondType.SINGLE
        end.SetAtomicNum(1 if single else 6)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.maxIterations = 200
    try:
        trimer = trimer.GetMol()
        Chem.SanitizeMol(trimer)
        trimer = Chem.AddHs(trimer)
        if AllChem.EmbedMolecule(trimer, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(trimer, params) != 0:
                return {col: np.nan for col in SHAPE_COLUMNS}
    except (Chem.rdchem.MolSanitizeException, RuntimeError, ValueError):
        return {col: np.nan for col in SHAPE_COLUMNS}

    heavy = Chem.RemoveHs(trimer)
    position = trimer.GetConformer().GetAtomPosition
    end_to_end = (position(end_atoms[0]) - position(end_atoms[1])).Length()
    backbone_atoms = 3 * (len(Chem.GetShortestPath(mol, *dummy_indices)) - 2)
    rg = rdMolDescriptors.CalcRadiusOfGyration(heavy)

    return {
        "TrimerRadiusOfGyration": rg,
        "TrimerRgPerBackboneAtom": rg / backbone_atoms,
        # Distance between the chain ends relative to a stretched chain (~1.25 Å per backbone bond)
        "TrimerEndToEndRatio": end_to_end / (1.25 * max(backbone_atoms - 1, 1)),
        "TrimerAsphericity": rdMolDescriptors.CalcAsphericity(heavy),
        "TrimerEccentricity": rdMolDescriptors.CalcEccentricity(heavy),
        "TrimerNPR1": rdMolDescriptors.CalcNPR1(heavy),
        "TrimerNPR2": rdMolDescriptors.CalcNPR2(heavy),
    }


# ---------------------------------------------------------------------------
# Interaction and steric descriptors for Tg (notebook 11)
# ---------------------------------------------------------------------------

INTERACTION_COLUMNS = [
    "BackboneSubstitutedFraction", "BackboneDisubstitutedAtoms", "StericBulkPerBackboneAtom",
    "SymmetricDisubstitution", "HDonorsPerBackboneAtom", "BackboneHDonorsPerAtom",
    "HBondGroupsPerBackboneAtom", "PolarGroupsPerHeavyAtom", "PolarGroupsPerBackboneAtom",
    "SideChainAtomsPerBackboneAtom", "FlexibleSideBondsPerBackboneAtom",
]

# Groups that form hydrogen bonds along the chain: amide, urethane, urea, imide
HBOND_GROUP = Chem.MolFromSmarts("[CX3](=O)[NX3]")
# Dipolar groups: C=O, C≡N, S=O, C–halogen, N–O
POLAR_GROUPS = [Chem.MolFromSmarts(s) for s in
                ["[CX3]=O", "C#N", "[SX4]=O", "[#6][F,Cl,Br,I]", "[NX3+](=O)[O-]"]]


def calculate_interaction_descriptors(mol):
    """Hydrogen bonding, dipoles, steric hindrance at the backbone and internal plasticisation.

    Values are given per backbone atom, so that they describe the density along the chain.
    """
    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    n_heavy = len(heavy_atoms)
    n_polar = sum(len(mol.GetSubstructMatches(pattern)) for pattern in POLAR_GROUPS)
    descriptors = {"PolarGroupsPerHeavyAtom": n_polar / n_heavy}

    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_indices) != 2:
        descriptors.update({col: np.nan for col in INTERACTION_COLUMNS if col not in descriptors})
        return descriptors

    # Backbone, backbone rings and pendant atoms, as in calculate_chain_descriptors
    path = list(Chem.GetShortestPath(mol, dummy_indices[0], dummy_indices[1]))
    backbone = path[1:-1]
    backbone_set = set(backbone)
    backbone_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if backbone_set & set(ring)]
    chain_atoms = backbone_set.union(*backbone_rings)
    pendant = {atom.GetIdx() for atom in heavy_atoms} - chain_atoms
    n_backbone = len(backbone)
    amide_bonds = {frozenset((m[0], m[2])) for m in mol.GetSubstructMatches(AMIDE_BOND)}

    # Steric hindrance: substituents on backbone atoms outside rings, and the pendant
    # atoms within two bonds of the chain (bulk close to the backbone)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    open_backbone = [i for i in backbone if not mol.GetAtomWithIdx(i).IsInRing()]
    substituted, disubstituted, symmetric = 0, 0, 0
    for i in open_backbone:
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in pendant]
        substituted += len(neighbors) >= 1
        disubstituted += len(neighbors) >= 2
        symmetric += len(neighbors) == 2 and ranks[neighbors[0]] == ranks[neighbors[1]]

    distances = Chem.GetDistanceMatrix(mol)
    near_chain = sum(min(distances[p][c] for c in chain_atoms) <= 2 for p in pendant)

    # Hydrogen-bond donors (N–H, O–H) on the whole unit and on the chain itself
    donors = [atom.GetIdx() for atom in heavy_atoms
              if atom.GetAtomicNum() in (7, 8) and atom.GetTotalNumHs() > 0]

    side_flexible = sum(
        _is_flexible(mol, bond, amide_bonds)
        and bond.GetBeginAtom().GetDegree() > 1 and bond.GetEndAtom().GetDegree() > 1
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in pendant or bond.GetEndAtomIdx() in pendant
    )

    descriptors.update({
        "BackboneSubstitutedFraction": substituted / len(open_backbone) if open_backbone else 0.0,
        "BackboneDisubstitutedAtoms": disubstituted,
        "StericBulkPerBackboneAtom": near_chain / n_backbone,
        "SymmetricDisubstitution": symmetric,
        "HDonorsPerBackboneAtom": len(donors) / n_backbone,
        "BackboneHDonorsPerAtom": sum(i in chain_atoms for i in donors) / n_backbone,
        "HBondGroupsPerBackboneAtom": len(mol.GetSubstructMatches(HBOND_GROUP)) / n_backbone,
        "PolarGroupsPerBackboneAtom": n_polar / n_backbone,
        "SideChainAtomsPerBackboneAtom": len(pendant) / n_backbone,
        "FlexibleSideBondsPerBackboneAtom": side_flexible / n_backbone,
    })
    return descriptors


def build_interaction_features(smiles):
    """Return the interaction and steric descriptors for a pandas Series of SMILES."""
    mols = smiles.apply(Chem.MolFromSmiles)
    return pd.DataFrame(
        [calculate_interaction_descriptors(mol) for mol in mols], index=smiles.index
    )[INTERACTION_COLUMNS]


def build_chain_features(smiles):
    """Return the 2D chain descriptors for a pandas Series of SMILES."""
    mols = smiles.apply(Chem.MolFromSmiles)
    return pd.DataFrame(
        [calculate_chain_descriptors(mol) for mol in mols], index=smiles.index
    )[CHAIN_COLUMNS]
