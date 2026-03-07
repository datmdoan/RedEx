
from __future__ import annotations
import torch
from rdkit import Chem
from torch_geometric.data import Data


def one_hot(value, choices):
    """One-hot encoding with an extra *unknown* bucket."""
    vec = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    vec[idx] = 1.0
    return vec


class MoleculeFeaturiser:
    """Convert an RDKit ``Mol`` into a PyG ``Data`` object.

    Atom features (74-dim by default):
      - Atomic number, degree, hybridisation, chirality, hydrogen count
        (all one-hot)
      - Aromaticity, ring membership, ring sizes 3–8 (binary)
      - Formal charge, mass, total valence, radical electrons (continuous)

    Bond features (13-dim):
      - Bond type (single/double/triple/aromatic), conjugation, ring (binary)
      - Stereo (one-hot)
    """

    ATOMS = [
        1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15,
        16, 17, 19, 20, 23, 24, 25, 26, 27, 28,
        29, 30, 34, 35, 42, 53,
    ]
    DEGREE = list(range(6))
    NUM_HS = list(range(5))
    RING_SIZES = [3, 4, 5, 6, 7, 8]

    def __init__(self, use_bond_features: bool = True):
        self.use_bond_features = use_bond_features

        self.hybrid_choices = sorted(
            int(getattr(Chem.HybridizationType, k))
            for k in Chem.HybridizationType.names
        )
        self.chiral_choices = sorted(
            int(getattr(Chem.ChiralType, k)) for k in Chem.ChiralType.names
        )
        self.stereo_choices = sorted(
            int(getattr(Chem.BondStereo, k)) for k in Chem.BondStereo.names
        )
        self.edge_feat_len = 4 + 1 + 1 + (len(self.stereo_choices) + 1)

        self.atom_dim: int | None = None
        self.edge_dim: int | None = None
        self.atom_cont_idx: list[int] | None = None
        self.atom_cont_names = ["charge", "mass", "valence", "radicals"]

    # ── atom features ─────────────────────────────────────────────────────
    def atom_features(self, atom) -> list[float]:
        f: list[float] = []
        f.extend(one_hot(atom.GetAtomicNum(), self.ATOMS))
        f.extend(one_hot(atom.GetTotalDegree(), self.DEGREE))
        f.extend(one_hot(int(atom.GetHybridization()), self.hybrid_choices))
        f.extend(one_hot(int(atom.GetChiralTag()), self.chiral_choices))
        f.extend(one_hot(atom.GetTotalNumHs(), self.NUM_HS))
        f.append(1.0 if atom.GetIsAromatic() else 0.0)
        f.append(1.0 if atom.IsInRing() else 0.0)
        for n in self.RING_SIZES:
            f.append(1.0 if atom.IsInRingSize(n) else 0.0)

        cont = [
            float(atom.GetFormalCharge()),
            float(atom.GetMass()),
            float(atom.GetTotalValence()),
            float(atom.GetNumRadicalElectrons()),
        ]
        if self.atom_cont_idx is None:
            start = len(f)
            self.atom_cont_idx = list(range(start, start + len(cont)))
        f.extend(cont)
        return f

    # ── bond features ─────────────────────────────────────────────────────
    def bond_features(self, bond) -> list[float]:
        v = [
            1.0 if bond.GetBondType() == Chem.BondType.SINGLE else 0.0,
            1.0 if bond.GetBondType() == Chem.BondType.DOUBLE else 0.0,
            1.0 if bond.GetBondType() == Chem.BondType.TRIPLE else 0.0,
            1.0 if bond.GetBondType() == Chem.BondType.AROMATIC else 0.0,
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
        ]
        v.extend(one_hot(int(bond.GetStereo()), self.stereo_choices))
        return v

    # ── __call__ ──────────────────────────────────────────────────────────
    def __call__(self, mol):
        x = torch.tensor(
            [self.atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32
        )
        src, dst, e_attr = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([i, j])
            dst.extend([j, i])
            if self.use_bond_features:
                bf = self.bond_features(bond)
                e_attr.extend([bf, bf])

        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = (
                torch.tensor(e_attr, dtype=torch.float32)
                if self.use_bond_features
                else torch.zeros((len(src), 0))
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.edge_feat_len if self.use_bond_features else 0))

        self.atom_dim = x.size(-1)
        self.edge_dim = edge_attr.size(-1)
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


def smiles_to_data(smiles: str, featurizer: MoleculeFeaturiser, y=None) -> Data | None:
    """Convert a SMILES string to a PyG ``Data`` object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    feats = featurizer(mol)
    data = Data(
        x=feats["x"],
        edge_index=feats["edge_index"],
        edge_attr=feats["edge_attr"],
        smiles=smiles,
    )
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data
