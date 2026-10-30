#!/usr/bin/env python3
# =====================================================================================
# dmpnn.py  —  LANCE's local CatPred-derived D-MPNN substrate encoder.
# -------------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
#   LANCE must be self-contained. This file vendors the compact D-MPNN
#   wrapper needed by LANCE while documenting exactly where the
#   CatPred-derived code came from.
#
# WHAT WAS CONSOLIDATED FROM CATPRED (so this prototype does not depend on four files):
#   * catpred/features/featurization.py  -> Featurization_parameters constants,
#                                           onek_encoding_unk, atom_features,
#                                           bond_features, MolGraph, BatchMolGraph,
#                                           mol2graph, get_atom_fdim, get_bond_fdim
#   * catpred/nn_utils.py                -> index_select_ND, initialize_weights
#   * catpred/models/mpn.py              -> MPNEncoder / MPN-style molecule encoder
#
# WHAT WAS REMOVED FROM CATPRED (optional flags unused for kcat/Km/Ki single-molecule
# tasks; each removal is also commented near the relevant code):
#   * reaction / reaction_solvent / reaction_substrate modes
#   * atom_messages (CatPred default is bond-centered messages)
#   * undirected message symmetrization
#   * atom_descriptors / bond_descriptors / *_features_extra
#   * features_only / use_input_features
#   * is_atom_bond_targets per-atom/bond readout
#   * mpn_shared, number_of_molecules>1, solvent encoder
#
# WHAT WAS MODIFIED VS CATPRED:
#   * CatPred's catpred/models/mpn.py::MPNEncoder.forward ultimately pools atoms
#     into one molecule vector.  This local version returns per-atom hidden states
#     plus per-molecule atom scopes, then atom_hiddens_to_tokens pads them into
#     (B, max_atoms, hidden) tokens.  LANCE needs atom-level tokens so the
#     molecule state can act as a latent reactant/product state.
#
# CatPred defaults retained:
#   hidden_size=300, depth=3, bias=False, activation=ReLU, aggregation=mean.
#   Atom feature dim = 133, bond feature dim = 147.
# =====================================================================================

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn


# -------------------------------------------------------------------------------------
# 1) FEATURE CONFIG
# -------------------------------------------------------------------------------------
# Slightly modified from catpred/features/featurization.py::Featurization_parameters:
# CatPred stores these in a mutable global PARAMS object because reaction modes can
# widen feature dimensions.  LANCE uses only the default single-molecule
# kcat/Km/Ki path, so constants are hard-coded here.
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def get_atom_fdim() -> int:
    """Return CatPred's default atom feature width.

    CatPred provenance:
        Slightly modified from catpred/features/featurization.py::get_atom_fdim.
        Removed reaction/overwrite branches because LANCE uses default
        molecule features only.
    """

    return ATOM_FDIM


def get_bond_fdim() -> int:
    """Return CatPred's default directed-bond feature width.

    CatPred provenance:
        Slightly modified from catpred/features/featurization.py::get_bond_fdim.
        CatPred's bond-centered D-MPNN prepends origin-atom features to bond
        features, so the directed-bond vector is 133 + 14 = 147 dimensions.
        Removed atom_messages/reaction/overwrite branches.
    """

    return BOND_FDIM + get_atom_fdim()


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """One-hot encode a categorical value with one final unknown slot.

    CatPred provenance:
        Exact logic from catpred/features/featurization.py::onek_encoding_unk.
    """

    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """Return CatPred's 133-dimensional atom feature vector.

    CatPred provenance:
        Slightly modified from catpred/features/featurization.py::atom_features.
        The core feature list is the same: element, degree, formal charge,
        chirality, hydrogens, hybridization, aromaticity, and scaled mass.
        Removed functional_groups because CatPred does not pass it for the
        kcat/Km/Ki molecule-level settings mirrored by this prototype.
    """

    if atom is None:
        return [0] * ATOM_FDIM
    return (
        onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES["hybridization"])
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """Return CatPred's 14-dimensional bond feature vector.

    CatPred provenance:
        Exact feature logic from catpred/features/featurization.py::bond_features:
        no-bond flag, bond type one-hots, conjugation, ring membership, and
        stereo one-hot.  The surrounding reaction-mode code is not included.
    """

    if bond is None:
        return [1] + [0] * (BOND_FDIM - 1)
    bt = bond.GetBondType()
    fbond = [
        0,
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0),
    ]
    fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """One molecule as CatPred-style directed bonds.

    CatPred provenance:
        Trimmed from catpred/features/featurization.py::MolGraph.  The default
        single-molecule graph construction is retained.  Reaction-mode branches,
        atom/bond descriptors, overwrite_default_* feature paths, and atom/bond
        target bookkeeping are removed.
    """

    def __init__(self, mol: Union[str, Chem.Mol]):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        if mol is None:
            raise ValueError("MolGraph received an invalid SMILES/RDKit Mol")

        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms: List[List[float]] = []
        self.f_bonds: List[List[float]] = []
        self.a2b: List[List[int]] = []
        self.b2a: List[int] = []
        self.b2revb: List[int] = []

        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
        self.n_atoms = len(self.f_atoms)
        for _ in range(self.n_atoms):
            self.a2b.append([])

        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is None:
                    continue

                f_bond = bond_features(bond)

                # CatPred's bond-centered D-MPNN stores two directed bonds per
                # chemical bond.  Each directed bond carries [origin atom feats
                # || bond feats].
                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)
                self.b2a.append(a1)
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class BatchMolGraph:
    """A padded batch of CatPred-style MolGraph objects.

    CatPred provenance:
        Trimmed from catpred/features/featurization.py::BatchMolGraph.
        Retained: flat atom/bond feature tensors, a2b, b2a, b2revb, a_scope,
        b_scope, and the reserved zero row.  Removed: protein_record_list,
        embed_feature_list, reaction handling, b2b/a2a/b2br helper caches, and
        atom_messages branches.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()
        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope: List[Tuple[int, int]] = []
        self.b_scope: List[Tuple[int, int]] = []

        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        a2b: List[List[int]] = [[]]
        b2a = [0]
        b2revb = [0]

        for mg in mol_graphs:
            f_atoms.extend(mg.f_atoms)
            f_bonds.extend(mg.f_bonds)
            for atom_idx in range(mg.n_atoms):
                a2b.append([bond_idx + self.n_bonds for bond_idx in mg.a2b[atom_idx]])
            for bond_idx in range(mg.n_bonds):
                b2a.append(self.n_atoms + mg.b2a[bond_idx])
                b2revb.append(self.n_bonds + mg.b2revb[bond_idx])
            self.a_scope.append((self.n_atoms, mg.n_atoms))
            self.b_scope.append((self.n_bonds, mg.n_bonds))
            self.n_atoms += mg.n_atoms
            self.n_bonds += mg.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))
        self.f_atoms = torch.tensor(f_atoms, dtype=torch.float)
        self.f_bonds = torch.tensor(f_bonds, dtype=torch.float)
        self.a2b = torch.tensor(
            [a2b[atom_idx] + [0] * (self.max_num_bonds - len(a2b[atom_idx])) for atom_idx in range(self.n_atoms)],
            dtype=torch.long,
        )
        self.b2a = torch.tensor(b2a, dtype=torch.long)
        self.b2revb = torch.tensor(b2revb, dtype=torch.long)

    def get_components(self):
        """Return the tensors consumed by MPNEncoder.forward.

        CatPred provenance:
            Slightly modified from catpred/features/featurization.py::
            BatchMolGraph.get_components.  The atom_messages branch is removed;
            this always returns bond-centered directed-bond features.
        """

        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def to(self, device: torch.device) -> "BatchMolGraph":
        """Move graph tensors to a device.

        CatPred provenance:
            New convenience helper for LANCE.  CatPred typically moves
            graph tensors inside the model/data pipeline; this local method
            keeps prototype batch movement explicit.
        """

        self.f_atoms = self.f_atoms.to(device)
        self.f_bonds = self.f_bonds.to(device)
        self.a2b = self.a2b.to(device)
        self.b2a = self.b2a.to(device)
        self.b2revb = self.b2revb.to(device)
        return self


def mol2graph(smiles: List[str]) -> BatchMolGraph:
    """Convert SMILES strings to a BatchMolGraph.

    CatPred provenance:
        Slightly modified from catpred/features/featurization.py::mol2graph.
        Removed atom_features_batch, bond_features_batch, overwrite flags, and
        other optional Chemprop/CatPred feature paths.  LANCE only needs
        the default molecule graph.
    """

    return BatchMolGraph([MolGraph(s) for s in smiles])


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather neighbor vectors into an N-D tensor.

    CatPred provenance:
        Exact logic from catpred/nn_utils.py::index_select_ND.
    """

    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim=0, index=index.view(-1))
    return target.view(final_size)


def initialize_weights(model: nn.Module) -> None:
    """Initialize trainable D-MPNN weights.

    CatPred provenance:
        Exact policy from catpred/nn_utils.py::initialize_weights for the
        D-MPNN subset used here: 1-D parameters to zero, matrices to Xavier
        normal.
    """

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


class MPNEncoder(nn.Module):
    """Bond-centered CatPred D-MPNN encoder returning atom tokens.

    CatPred provenance:
        Slightly modified from catpred/models/mpn.py::MPNEncoder.
        Retained: W_i/W_h/W_o update equations, ReLU activation, dropout, and
        directed-message subtraction of reverse bonds.
        Removed: atom_messages, undirected messages, descriptors,
        is_atom_bond_targets, cached_zero_vector for empty molecules, and
        CatPred's final molecule-level mean aggregation.
        Modified: forward returns flat atom hidden states plus a_scope so
        atom_hiddens_to_tokens can expose per-atom tokens to LANCE.
    """

    def __init__(self, hidden_size: int = 300, depth: int = 3, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.act_func = nn.ReLU()
        self.W_i = nn.Linear(self.bond_fdim, self.hidden_size, bias=bias)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        initialize_weights(self)

    def forward(self, mol_graph: BatchMolGraph) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        device = self.W_i.weight.device
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, _ = mol_graph.get_components()
        f_atoms, f_bonds = f_atoms.to(device), f_bonds.to(device)
        a2b, b2a, b2revb = a2b.to(device), b2a.to(device), b2revb.to(device)

        # Initial directed-bond messages.
        initial_message = self.W_i(f_bonds)
        message = self.act_func(initial_message)

        # CatPred's bond-centered message passing:
        #   m(a1->a2) = sum incoming messages to a1 - m(a2->a1)
        for _ in range(self.depth - 1):
            nei_a_message = index_select_ND(message, a2b)
            a_message = nei_a_message.sum(dim=1)
            rev_message = message[b2revb]
            message = a_message[b2a] - rev_message
            message = self.W_h(message)
            message = self.act_func(initial_message + message)
            message = self.dropout(message)

        # Atom readout from atom features and summed incoming bond messages.
        nei_a_message = index_select_ND(message, a2b)
        a_message = nei_a_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout(atom_hiddens)
        return atom_hiddens, a_scope


def atom_hiddens_to_tokens(
    atom_hiddens: torch.Tensor,
    a_scope: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad flat atom hidden states into batch-major atom tokens.

    CatPred provenance:
        New LANCE helper replacing CatPred's molecule-level mean-pool
        loop in catpred/models/mpn.py::MPNEncoder.forward.  The hidden states
        are CatPred D-MPNN atom states; only the final packing/pooling behavior
        differs.
    """

    hidden = atom_hiddens.shape[-1]
    batch = len(a_scope)
    max_atoms = max((size for _, size in a_scope), default=1)
    tokens = atom_hiddens.new_zeros((batch, max_atoms, hidden))
    mask = atom_hiddens.new_zeros((batch, max_atoms), dtype=torch.bool)
    for row, (start, size) in enumerate(a_scope):
        if size == 0:
            continue
        tokens[row, :size] = atom_hiddens.narrow(0, start, size)
        mask[row, :size] = True
    return tokens, mask


class DMPNN(nn.Module):
    """Thin CatPred D-MPNN wrapper returning padded atom tokens.

    CatPred provenance:
        Plays the single-molecule role of catpred/models/mpn.py::MPN, but the
        output is modified from pooled molecule vectors to (atom_tokens, mask).
    """

    def __init__(self, hidden_size: int = 300, depth: int = 3, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = MPNEncoder(hidden_size, depth, bias, dropout)

    def forward(self, bmg: BatchMolGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_hiddens, a_scope = self.encoder(bmg)
        return atom_hiddens_to_tokens(atom_hiddens, a_scope)
