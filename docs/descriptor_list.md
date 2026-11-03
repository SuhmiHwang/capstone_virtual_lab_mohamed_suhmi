# Polymer descriptors

All descriptors are calculated from the SMILES of the repeat unit with RDKit
(`src/polymer_features.py`). The **backbone** is the shortest path between the two
attachment points (`*`); everything else is side chain.

| Feature set | Descriptors | Notebook |
|---|---:|---|
| A: original | 9 | 04, section 17 |
| B: + chemistry | 12 (21 in total) | 04, section 21 |
| B+: + functional groups | 22 (43 in total) | 04, section 23 |
| Chain: backbone, side chains, volume | 23 | 06 |
| Shape: 3D trimer | 7 | 06 |
| D: Morgan fingerprints | 1,024 bits | 04, section 21 |
| **All** | **73 descriptors + 1,024 fingerprint bits** | |

## A: original descriptors (9)

| Descriptor | Meaning |
|---|---|
| `MolWt` | molar mass of the repeat unit |
| `HeavyAtomCount` | number of non-hydrogen atoms |
| `NumAtoms` | number of atoms (incl. the two `*`) |
| `NumRings` | number of rings |
| `NumAromaticRings` | number of aromatic rings |
| `NumHDonors` | hydrogen-bond donors (OH, NH) |
| `NumHAcceptors` | hydrogen-bond acceptors (O, N) |
| `MolLogP` | hydrophobicity (octanol/water partition coefficient) |
| `TPSA` | topological polar surface area |

## B: chemistry descriptors (12)

| Descriptor | Meaning |
|---|---|
| `NumRotatableBonds` | flexible bonds in the whole repeat unit |
| `FractionCSP3` | share of sp³ carbons (aliphatic vs. aromatic/unsaturated) |
| `AromaticAtomFraction` | share of heavy atoms that are aromatic |
| `HalogenCount` | F, Cl, Br, I atoms |
| `SulfurCount` | S atoms |
| `SiliconCount` | Si atoms (siloxanes) |
| `MolWtPerHeavyAtom` | average mass per heavy atom |
| `MolLogPPerHeavyAtom` | hydrophobicity per heavy atom |
| `TPSAPerHeavyAtom` | polar surface per heavy atom |
| `BackboneLength` | atoms in the backbone per repeat unit |
| `BackboneRingFraction` | share of backbone atoms that are in a ring |
| `SideChainFraction` | share of heavy atoms outside the backbone |

## B+: functional-group descriptors (22)

Groups are counted on three joined repeat units (trimer), so groups that cross
the attachment points (e.g. the carbonate of polycarbonate) are found.

| Descriptor | Meaning |
|---|---|
| `EsterBackbone`, `EsterSideChain` | esters, e.g. PET (backbone) vs. PMMA (side chain) |
| `CarbonateBackbone`, `CarbonateSideChain` | carbonates |
| `AmideBackbone`, `AmideSideChain` | amides, e.g. nylons |
| `ImideBackbone`, `ImideSideChain` | imides, e.g. polyimides |
| `UrethaneBackbone`, `UrethaneSideChain` | urethanes |
| `EtherBackbone`, `EtherSideChain` | ethers |
| `SiloxaneBackbone`, `SiloxaneSideChain` | Si–O–Si |
| `SulfoneBackbone`, `SulfoneSideChain` | sulfones |
| `HydroxylSideChain` | OH groups |
| `NitrileSideChain` | C≡N groups |
| `LargestSideGroup` | heavy atoms in the largest side group |
| `BackboneUnsaturation` | share of non-aromatic double/triple bonds in the backbone |
| `HDonorsPerHeavyAtom` | H-bond donors per heavy atom |
| `HAcceptorsPerHeavyAtom` | H-bond acceptors per heavy atom |

## Chain descriptors (23)

A bond is **flexible** if it is single, not in a ring, and not an amide C–N bond.

| Descriptor | Meaning |
|---|---|
| `BackboneRotatableBonds` | flexible bonds in the backbone |
| `BackboneRotatableFraction` | share of backbone bonds that are flexible |
| `LongestFlexibleRun` | longest run of consecutive flexible backbone bonds (spacer length) |
| `FullyFlexibleBackbone` | 1 if all backbone bonds are flexible |
| `BackboneAromaticFraction` | share of backbone atoms that are aromatic |
| `BackboneRingCount` | rings the backbone passes through |
| `ParaLinkages` | para-linked benzene rings in the backbone (straight, stiff) |
| `MetaLinkages` | meta-linked benzene rings in the backbone (kinked) |
| `OrthoLinkages` | ortho-linked benzene rings in the backbone |
| `ParaLinkageFraction` | share of backbone benzene rings that are para-linked |
| `NumSubstituents` | groups attached to the backbone (incl. single atoms) |
| `NumSideChains` | substituents with at least two heavy atoms |
| `LongestSideChain` | atoms in the longest substituent, from the backbone outwards |
| `MeanSubstituentSize` | average heavy atoms per substituent |
| `SideChainRotatableBonds` | flexible bonds in the substituents |
| `SideChainBranchPoints` | substituent atoms with ≥ 3 heavy neighbours (branching) |
| `PendantAromaticRings` | aromatic rings in substituents (e.g. polystyrene) |
| `QuaternaryCarbons` | carbons with four carbon neighbours (tert-butyl, bisphenol A) |
| `VdWVolume` | van der Waals volume of the repeat unit, Å³ (Zhao et al. 2003) |
| `SpecificVdWVolume` | van der Waals volume per mass, cm³/g |
| `VdWVolumePerHeavyAtom` | van der Waals volume per heavy atom |
| `LabuteASAPerVolume` | surface area per volume |
| `KierFlexibility` | Kier flexibility index Φ |

## Shape descriptors (7)

One 3D conformer of three joined repeat units (ends capped with H, RDKit ETKDG).
Cached in `data/processed/shape_descriptors.csv`.

| Descriptor | Meaning |
|---|---|
| `TrimerRadiusOfGyration` | radius of gyration, Å |
| `TrimerRgPerBackboneAtom` | radius of gyration per backbone atom |
| `TrimerEndToEndRatio` | end-to-end distance relative to a stretched chain |
| `TrimerAsphericity` | deviation from a sphere (0 = sphere, 1 = rod) |
| `TrimerEccentricity` | elongation of the shape |
| `TrimerNPR1`, `TrimerNPR2` | normalised principal moments of inertia (rod / disc / sphere) |

## D: Morgan fingerprints (1,024 bits)

Each bit marks whether a particular atom neighbourhood (up to two bonds around an
atom) occurs in the repeat unit. Radius 2, 1,024 bits.

## Not available from SMILES

Chain length (molecular weight), polydispersity, tacticity, crystallinity and
processing history are not encoded in the repeat-unit SMILES.
