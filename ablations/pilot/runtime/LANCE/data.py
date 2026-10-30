#!/usr/bin/env python3
"""LANCE CatPred-DB data utilities.

This file is intentionally self-contained inside ``LANCE``.

CatPred provenance:
    Several small pieces are copied or slightly modified from CatPred-style
    utilities already used in this repository:

      - target scaling follows CatPred's ``catpred/data/scaler.py::StandardScaler``
        logic exactly for mean/std normalization;
      - split names follow CatPred-DB's ``scripts/analyze_reproduce.py``
        cluster convention;
      - invalid-SMILES filtering follows the one-molecule case of
        ``catpred/data/utils.py::filter_invalid_smiles``;
      - graph construction uses the local vendored
        ``LANCE.dmpnn.mol2graph`` implementation, whose file documents
        the exact CatPred source classes/functions it consolidates.

No code in this file imports from ``New_approach``.  That is deliberate so the
cluster jobs can run LANCE as its own project folder.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from torch.utils.data import ConcatDataset, Dataset

# Running ``python LANCE/train.py`` puts LANCE/ on sys.path, not the
# repository root.  Add the root so absolute package imports resolve the same
# way locally and under Slurm.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# CatPred-derived graph builder vendored locally for LANCE.  See
# LANCE/dmpnn.py for exact source-file/function provenance.
from LANCE.dmpnn import mol2graph


# Slightly modified from the CatPred reproduction scripts: CatPred wires the
# SMILES and target column names per kinetic parameter.  This table preserves
# those CatPred-DB column choices while keeping the prototype CLI compact.
PARAM_SPECS = {
    "kcat": {
        "smiles": "reactant_smiles",
        "target": "log10kcat_max",
        "processed": "kcat_max_wt_singleSeqs_wpdbs.csv",
    },
    "km": {
        "smiles": "substrate_smiles",
        "target": "log10km_mean",
        "processed": "km_mean_wt_singleSeqs_wpdbs.csv",
    },
    "ki": {
        "smiles": "substrate_smiles",
        "target": "log10ki_mean",
        "processed": "ki_mean_wt_singleSeqs_wpdbs.csv",
    },
}

PARAM_ORDER = ("kcat", "km", "ki")
TASK_TO_INDEX = {name: index for index, name in enumerate(PARAM_ORDER)}

# Exact list from CatPred's reproduce-analysis convention, plus the kcat-only
# 20% cluster split present in the local CatPred-DB layout.
CATPRED_CLUSTER_LEVELS = [99, 80, 60, 40]
OOD_CLUSTER_LEVELS = tuple(CATPRED_CLUSTER_LEVELS + [20])

# Product/reaction-state columns are not guaranteed in CatPred-DB.  If one is
# present for a row, the collate function builds product_bmg and enables JEPA
# loss for that row.  If absent, the row still trains kinetics normally and the
# JEPA mask is false.
PRODUCT_COLUMN_CANDIDATES = (
    "product_smiles",
    "products_smiles",
    "product",
    "products",
    "reaction_product_smiles",
    "rxn_product_smiles",
)

# Product-free substrate descriptors used by the LANCE2 binding auxiliary.
# These are intentionally cheap RDKit scalars available for every valid row;
# they make the binding state carry molecule chemistry even when no product
# JEPA target exists.  The values are coarsely normalized to order-one ranges so
# SmoothL1 does not get dominated by molecular weight or TPSA.
SUBSTRATE_DESCRIPTOR_NAMES = (
    "mol_wt_500",
    "logp_5",
    "tpsa_150",
    "hbd_8",
    "hba_12",
    "formal_charge_3",
    "ring_count_8",
    "rotatable_bonds_12",
)

# Measurement-condition features available in the CatPred-DB split CSVs.  These
# are inference-time covariates, not target leakage: kinetic constants are
# condition-dependent, and CatPred-DB carries temperature/pH for kcat, Km, and Ki.
CONDITION_FEATURE_NAMES = (
    "temperature_100",
    "ph_14",
    "temperature_present",
    "ph_present",
)

EC_LEVEL1_CLASSES = 8  # 0=unknown/partial, 1..7=top-level EC class.


def clean(value: Any) -> str:
    """Normalize a CSV cell into a stable string."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).strip().split())


def safe_float(value: Any, default: float = 0.0) -> float:
    """Parse a finite float without letting bad CSV cells crash collation."""

    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def stable_key(text: Any) -> str:
    """SHA1 key used by the existing ESM++ cache files."""

    return hashlib.sha1(clean(text).encode("utf-8")).hexdigest()


def stable_group_id(text: Any, missing: int = -1) -> int:
    """Map a non-empty text key to a deterministic signed-int64-safe id."""

    value = clean(text)
    if not value:
        return int(missing)
    return int(stable_key(value)[:15], 16)


def fallback_cluster_id(sequence: str, modulus: int = 1_000_003) -> int:
    """Deterministic fallback cluster id when a split lacks cluster columns."""

    return int(stable_key(sequence)[:12], 16) % int(modulus)


def row_sequence_cluster_id(row: Any, sequence: str, cluster_level: int) -> int:
    """Read a CatPred-DB sequence cluster id from a split row.

    CatPred provenance:
        CatPred-DB split files include columns such as ``sequence_60cluster``.
        This helper reads those existing columns; it does not copy CatPred code.
        The fallback hash path only exists so smoke tests and future derived
        files can still run when cluster columns are absent.
    """

    column = f"sequence_{int(cluster_level)}cluster"
    if column in row.index:
        value = row.get(column)
        try:
            if value is not None and math.isfinite(float(value)):
                return int(float(value))
        except (TypeError, ValueError):
            pass
    return fallback_cluster_id(sequence)


def cluster_fold(seed: int, cluster_id: int, n_folds: int) -> int:
    """Per-seed deterministic cluster-fold rotation.

    This is the seed-diversity mechanism from the LANCE2 review, implemented in
    data loading rather than as a post-hoc blend.  Different seeds hold out
    different sequence clusters for checkpoint selection.
    """

    return int((int(cluster_id) + int(seed)) % int(n_folds))


def normalized_substrate_descriptors(smiles: str) -> list[float]:
    """Return eight cheap, normalized RDKit descriptors for one molecule."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return [0.0 for _ in SUBSTRATE_DESCRIPTOR_NAMES]
    return [
        float(Descriptors.MolWt(molecule) / 500.0),
        float(Crippen.MolLogP(molecule) / 5.0),
        float(rdMolDescriptors.CalcTPSA(molecule) / 150.0),
        float(Lipinski.NumHDonors(molecule) / 8.0),
        float(Lipinski.NumHAcceptors(molecule) / 12.0),
        float(Chem.GetFormalCharge(molecule) / 3.0),
        float(rdMolDescriptors.CalcNumRings(molecule) / 8.0),
        float(Lipinski.NumRotatableBonds(molecule) / 12.0),
    ]


def normalized_condition_features(row: Any) -> list[float]:
    """Return normalized pH/temperature features plus missingness indicators.

    CatPred provenance:
        This is new LANCE code.  CatPred-DB already stores ``temperature`` and
        ``ph`` columns in the split files; this helper only turns those columns
        into stable numeric covariates for the kinetic heads.
    """

    temp_raw = safe_float(row.get("temperature", float("nan")), float("nan"))
    ph_raw = safe_float(row.get("ph", float("nan")), float("nan"))
    temp_present = float(math.isfinite(temp_raw))
    ph_present = float(math.isfinite(ph_raw))
    temp = min(max(temp_raw / 100.0, -0.5), 1.5) if temp_present else 0.0
    ph = min(max(ph_raw / 14.0, 0.0), 1.0) if ph_present else 0.0
    return [float(temp), float(ph), temp_present, ph_present]


def ec_level1_target(ec_value: Any) -> tuple[int, bool]:
    """Return a coarse EC class target for auxiliary catalytic supervision.

    The target is deliberately coarse: EC level 1 captures broad reaction
    chemistry while avoiding a huge sparse classifier.  We use it as a training
    auxiliary, not as a kinetic-head input.
    """

    ec = clean(ec_value)
    if not ec or ec.startswith("-"):
        return 0, False
    first = ec.split(".", 1)[0]
    try:
        value = int(first)
    except ValueError:
        return 0, False
    if 1 <= value <= 7:
        return value, True
    return 0, False


def split_path(data_root: Path, parameter: str, split: str) -> Path:
    """Return the CatPred-DB split file path for one parameter/split."""

    if split in {"train", "val", "test", "trainval", "trainvaltest"}:
        name = f"{parameter}-random_{split}.csv"
    elif split.startswith("ood"):
        name = f"{parameter}-seq_test_sequence_{split.removeprefix('ood')}cluster.csv"
    else:
        raise ValueError(f"Unknown split: {split}")
    return Path(data_root) / "datasets" / "splits" / name


def available_splits(data_root: Path, parameter: str) -> list[str]:
    """List the heldout/OOD splits that exist for a parameter."""

    splits = ["train", "val", "test"]
    for level in OOD_CLUSTER_LEVELS:
        split = f"ood{level}"
        if split_path(data_root, parameter, split).exists():
            splits.append(split)
    return splits


def expected_columns(parameter: str) -> set[str]:
    """Required CatPred-DB columns for kinetic supervision."""

    spec = PARAM_SPECS[parameter]
    return {"sequence", spec["smiles"], spec["target"]}


def validate_data_root(
    data_root: Path,
    parameters: Iterable[str] = PARAM_ORDER,
    required_splits: Iterable[str] = ("train", "val", "test"),
) -> None:
    """Read-only validation before expensive cluster training starts."""

    data_root = Path(data_root)
    errors: list[str] = []
    processed_dir = data_root / "datasets" / "processed"
    splits_dir = data_root / "datasets" / "splits"
    if not processed_dir.is_dir():
        errors.append(f"missing processed directory: {processed_dir}")
    if not splits_dir.is_dir():
        errors.append(f"missing splits directory: {splits_dir}")

    for parameter in parameters:
        processed_path = processed_dir / PARAM_SPECS[parameter]["processed"]
        if processed_dir.is_dir() and not processed_path.is_file():
            errors.append(f"missing processed file: {processed_path}")
        for split in required_splits:
            path = split_path(data_root, parameter, split)
            if not path.is_file():
                errors.append(f"missing split file: {path}")
                continue
            try:
                columns = set(pd.read_csv(path, nrows=0).columns)
            except Exception as exc:
                errors.append(f"unreadable split file: {path} ({exc})")
                continue
            missing = sorted(expected_columns(parameter) - columns)
            if missing:
                errors.append(f"{path} missing columns: {missing}")

    if errors:
        joined = "\n  - ".join(errors)
        raise FileNotFoundError(f"CatPred-DB validation failed:\n  - {joined}")


def load_split_frame(data_root: Path, parameter: str, split: str) -> pd.DataFrame:
    """Load one split CSV with required-column validation."""

    path = split_path(Path(data_root), parameter, split)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(expected_columns(parameter) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return frame


# Exact logic from CatPred's catpred/data/scaler.py::StandardScaler, copied so
# LANCE checkpoints do not depend on CatPred package imports.
class StandardScaler:
    """Mean/std scaler for log10 kinetic targets."""

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: list[list[Optional[float]]]) -> "StandardScaler":
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)
        return self

    def transform(self, X: list[list[Optional[float]]]) -> np.ndarray:
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        return np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

    def inverse_transform(self, X: list[list[Optional[float]]]) -> np.ndarray:
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        return np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)


class TargetScaler(StandardScaler):
    """Single-target convenience wrapper around the CatPred scaler."""

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "TargetScaler":
        rows = [[float(value)] for value in values if math.isfinite(float(value))]
        if not rows:
            raise ValueError("cannot fit TargetScaler with no finite targets")
        scaler = cls()
        scaler.fit(rows)
        return scaler

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "TargetScaler":
        return cls(means=np.asarray(state["means"], dtype=float), stds=np.asarray(state["stds"], dtype=float))

    def transform_value(self, value: float) -> float:
        return float(self.transform([[float(value)]])[0][0])

    def inverse_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value * float(self.stds[0]) + float(self.means[0])

    def state(self) -> dict[str, list[float]]:
        return {"means": self.means.tolist(), "stds": self.stds.tolist()}


def load_torch(path: Path) -> Any:
    """Compatibility wrapper for older/newer PyTorch ``torch.load`` kwargs."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class EmbeddingCache:
    """Read frozen per-residue ESM++ tensors keyed by sequence SHA1."""

    def __init__(self, path: Path, *, kind: str) -> None:
        self.path = Path(path)
        self.kind = kind
        if not self.path.is_file():
            raise FileNotFoundError(f"missing {kind} cache: {self.path}")
        payload = load_torch(self.path)
        if isinstance(payload, dict) and "embeddings" in payload:
            self.metadata = {key: value for key, value in payload.items() if key != "embeddings"}
            self.embeddings = payload["embeddings"]
        elif isinstance(payload, dict):
            self.metadata = {}
            self.embeddings = payload
        else:
            raise TypeError(f"{self.path} must contain a dict of embeddings")
        if not self.embeddings:
            raise ValueError(f"{self.path} contains no embeddings")
        sample = torch.as_tensor(next(iter(self.embeddings.values())))
        self.dim = int(sample.shape[-1]) if sample.dim() > 0 else 1

    def has_text(self, text: str) -> bool:
        return stable_key(text) in self.embeddings

    def get_text(self, text: str, *, max_tokens: int | None = None) -> torch.Tensor:
        key = stable_key(text)
        if key not in self.embeddings:
            raise KeyError(f"{self.kind} cache miss for key={key}")
        value = torch.as_tensor(self.embeddings[key]).float()
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if value.dim() != 2:
            raise ValueError(f"{self.kind} cache entry {key} must be 1D or 2D, got shape={tuple(value.shape)}")
        if max_tokens is not None:
            value = value[:max_tokens]
        return value.contiguous()


def is_valid_single_smiles(smiles: str) -> bool:
    """CatPred's one-molecule invalid-SMILES predicate."""

    if not smiles:
        return False
    molecule = Chem.MolFromSmiles(smiles)
    return molecule is not None and molecule.GetNumHeavyAtoms() > 0


def filter_invalid_smiles_frame(frame: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
    """Drop rows CatPred would skip before graph featurization."""

    valid_rows: list[int] = []
    for row_index, value in enumerate(frame[smiles_column].tolist()):
        if is_valid_single_smiles(clean(value)):
            valid_rows.append(row_index)
    filtered = frame.iloc[valid_rows].copy()
    filtered.insert(0, "_source_row_index", filtered.index.astype(int))
    return filtered.reset_index(drop=True)


def validate_cache_coverage(
    data_root: Path,
    esm_cache: EmbeddingCache,
    parameters: Iterable[str] = PARAM_ORDER,
    splits: Iterable[str] | None = ("train", "val", "test"),
) -> None:
    """Ensure every valid row has a frozen ESM++ sequence embedding."""

    missing_esm: set[str] = set()
    for parameter in parameters:
        spec = PARAM_SPECS[parameter]
        parameter_splits = available_splits(data_root, parameter) if splits is None else list(splits)
        for split in parameter_splits:
            frame = load_split_frame(data_root, parameter, split)
            for _, row in frame.iterrows():
                if not is_valid_single_smiles(clean(row[spec["smiles"]])):
                    continue
                sequence = clean(row["sequence"])
                if not esm_cache.has_text(sequence):
                    missing_esm.add(stable_key(sequence))
    if missing_esm:
        msg = {"missing_esm": len(missing_esm), "example_missing_esm": sorted(missing_esm)[:5]}
        raise KeyError(f"frozen ESM++ cache coverage failed: {json.dumps(msg, sort_keys=True)}")


def first_existing_product_column(columns: Iterable[str]) -> str | None:
    """Return the first recognized product column in a split frame, if any."""

    column_set = set(columns)
    for candidate in PRODUCT_COLUMN_CANDIDATES:
        if candidate in column_set:
            return candidate
    return None


class KineticsDataset(Dataset):
    """CatPred-DB kinetic rows plus optional JEPA product-state targets."""

    def __init__(
        self,
        data_root: Path,
        parameter: str,
        split: str,
        scaler: TargetScaler,
        esm_cache: EmbeddingCache,
        *,
        max_residues: int = 1024,
        cluster_level: int = 60,
        cluster_fold_seed: int = 0,
        cluster_fold_count: int = 0,
        cluster_fold_index: int = 0,
        cluster_fold_role: str = "all",
    ) -> None:
        self.data_root = Path(data_root)
        self.parameter = parameter
        self.task_idx = TASK_TO_INDEX[parameter]
        self.split = split
        self.spec = PARAM_SPECS[parameter]
        self.scaler = scaler
        self.esm_cache = esm_cache
        self.max_residues = int(max_residues)
        self.cluster_level = int(cluster_level)
        self.frame = load_split_frame(self.data_root, parameter, split).reset_index(drop=True)
        self.product_column = first_existing_product_column(self.frame.columns)
        self.frame = filter_invalid_smiles_frame(self.frame, self.spec["smiles"])
        if cluster_fold_role not in {"all", "train", "holdout"}:
            raise ValueError(f"cluster_fold_role must be all/train/holdout, got {cluster_fold_role}")
        if split == "train" and int(cluster_fold_count) > 1 and cluster_fold_role != "all":
            fold_index = int(cluster_fold_index)
            if fold_index < 0:
                fold_index = int(cluster_fold_seed) % int(cluster_fold_count)
            keep_rows: list[int] = []
            for row_index, row in self.frame.iterrows():
                sequence = clean(row["sequence"])
                cid = row_sequence_cluster_id(row, sequence, self.cluster_level)
                fold = cluster_fold(cluster_fold_seed, cid, cluster_fold_count)
                if (cluster_fold_role == "train" and fold != fold_index) or (
                    cluster_fold_role == "holdout" and fold == fold_index
                ):
                    keep_rows.append(row_index)
            self.frame = self.frame.iloc[keep_rows].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        sequence = clean(row["sequence"])
        smiles = clean(row[self.spec["smiles"]])
        cluster_id = row_sequence_cluster_id(row, sequence, self.cluster_level)
        raw_target = row[self.spec["target"]]
        target_valid = math.isfinite(safe_float(raw_target, float("nan")))
        target = safe_float(raw_target, 0.0)
        target_z = self.scaler.transform_value(target) if target_valid else 0.0
        reaction_group_id = stable_group_id(row.get("reaction_smiles", "")) if self.parameter == "kcat" else -1

        product_smiles = ""
        jepa_mask = False
        if self.product_column is not None:
            candidate = clean(row.get(self.product_column, ""))
            if is_valid_single_smiles(candidate):
                product_smiles = candidate
                jepa_mask = True

        return {
            "row_index": int(idx),
            "source_row_index": int(row.get("_source_row_index", idx)),
            "parameter": self.parameter,
            "split": self.split,
            "task_idx": self.task_idx,
            "sequence_key": stable_key(sequence),
            "smiles_key": stable_key(smiles),
            "cluster_id": cluster_id,
            "reaction_group_id": reaction_group_id,
            "substrate_desc": normalized_substrate_descriptors(smiles),
            "condition": normalized_condition_features(row),
            "ec_level1": ec_level1_target(row.get("ec", ""))[0],
            "ec_mask": ec_level1_target(row.get("ec", ""))[1],
            "protein": self.esm_cache.get_text(sequence, max_tokens=self.max_residues),
            "smiles": smiles,
            # Use the reactant as a graph-building placeholder when no product
            # target exists.  The loss mask is false, so this never contributes
            # to the JEPA objective.
            "product_smiles": product_smiles if jepa_mask else smiles,
            "jepa_mask": jepa_mask,
            "target": target,
            "target_z": target_z,
            "target_scale": float(self.scaler.stds[0]),
            "target_mask": target_valid,
        }


def normalize_parameters(parameters: Iterable[str] | None = None) -> tuple[str, ...]:
    """Validate and canonicalize the kinetic parameters selected for a run.

    CatPred provenance:
        CatPred trains/evaluates kinetic parameters separately in its
        reproduction scripts.  This helper is new LANCE glue that lets us run
        either the original multitask LANCE surface or a task-specialist
        ablation without importing CatPred code.
    """

    if parameters is None:
        return PARAM_ORDER
    selected: list[str] = []
    for parameter in parameters:
        name = clean(parameter).lower()
        if name == "km":
            name = "km"
        if name not in PARAM_SPECS:
            raise ValueError(f"unknown kinetic parameter {parameter!r}; expected one of {PARAM_ORDER}")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("at least one kinetic parameter must be selected")
    return tuple(selected)


def build_scalers(
    data_root: Path,
    train_split: str = "train",
    parameters: Iterable[str] | None = None,
) -> dict[str, TargetScaler]:
    """Fit CatPred-style target scalers on valid training rows only."""

    scalers: dict[str, TargetScaler] = {}
    for parameter in normalize_parameters(parameters):
        frame = load_split_frame(data_root, parameter, train_split)
        frame = filter_invalid_smiles_frame(frame, PARAM_SPECS[parameter]["smiles"])
        scalers[parameter] = TargetScaler.from_values(frame[PARAM_SPECS[parameter]["target"]].to_numpy(dtype=float))
    return scalers


def make_multitask_dataset(
    data_root: Path,
    split: str,
    scalers: dict[str, TargetScaler],
    esm_cache: EmbeddingCache,
    *,
    max_residues: int = 1024,
    cluster_level: int = 60,
    cluster_fold_seed: int = 0,
    cluster_fold_count: int = 0,
    cluster_fold_index: int = 0,
    cluster_fold_role: str = "all",
    parameters: Iterable[str] | None = None,
) -> ConcatDataset:
    """Concatenate selected kinetic tasks into one training/evaluation surface.

    The original LANCE1 runs used all three tasks.  The 18/18 recovery plan also
    needs kcat-only specialists, so the task list is now explicit while keeping
    the default behavior identical to the old multitask path.
    """

    return ConcatDataset(
        [
            KineticsDataset(
                data_root,
                parameter,
                split,
                scalers[parameter],
                esm_cache,
                max_residues=max_residues,
                cluster_level=cluster_level,
                cluster_fold_seed=cluster_fold_seed,
                cluster_fold_count=cluster_fold_count,
                cluster_fold_index=cluster_fold_index,
                cluster_fold_role=cluster_fold_role,
            )
            for parameter in normalize_parameters(parameters)
        ]
    )


def _pad_embeddings(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length residue tensors into a dense batch."""

    batch = len(values)
    length = max(int(value.shape[0]) for value in values)
    dim = int(values[0].shape[-1])
    out = torch.zeros((batch, length, dim), dtype=torch.float32)
    mask = torch.zeros((batch, length), dtype=torch.bool)
    for index, value in enumerate(values):
        if value.shape[-1] != dim:
            raise ValueError(f"embedding dim mismatch: expected {dim}, got {value.shape[-1]}")
        n = min(int(value.shape[0]), length)
        out[index, :n] = value[:n].float()
        mask[index, :n] = True
    return out, mask


def collate_kinetics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch protein tensors and CatPred-derived molecule graphs."""

    protein, protein_mask = _pad_embeddings([item["protein"] for item in items])
    has_jepa_target = any(bool(item["jepa_mask"]) for item in items)
    return {
        "row_index": torch.tensor([item["row_index"] for item in items], dtype=torch.long),
        "source_row_index": torch.tensor([item["source_row_index"] for item in items], dtype=torch.long),
        "task_idx": torch.tensor([item["task_idx"] for item in items], dtype=torch.long),
        "cluster_id": torch.tensor([item["cluster_id"] for item in items], dtype=torch.long),
        "reaction_group_id": torch.tensor([item["reaction_group_id"] for item in items], dtype=torch.long),
        "protein": protein,
        "protein_mask": protein_mask,
        "bmg": mol2graph([item["smiles"] for item in items]),
        # Avoid a second D-MPNN pass when a batch has no real product targets.
        # If at least one row has a product target, rows without one use their
        # reactant graph as a harmless placeholder and jepa_mask=False.
        "product_bmg": mol2graph([item["product_smiles"] for item in items]) if has_jepa_target else None,
        "jepa_mask": torch.tensor([item["jepa_mask"] for item in items], dtype=torch.bool),
        "target": torch.tensor([item["target"] for item in items], dtype=torch.float32),
        "target_z": torch.tensor([item["target_z"] for item in items], dtype=torch.float32),
        "target_scale": torch.tensor([item["target_scale"] for item in items], dtype=torch.float32),
        "target_mask": torch.tensor([item["target_mask"] for item in items], dtype=torch.bool),
        "substrate_desc": torch.tensor([item["substrate_desc"] for item in items], dtype=torch.float32),
        "condition": torch.tensor([item["condition"] for item in items], dtype=torch.float32),
        "ec_level1": torch.tensor([item["ec_level1"] for item in items], dtype=torch.long),
        "ec_mask": torch.tensor([item["ec_mask"] for item in items], dtype=torch.bool),
        "parameter": [item["parameter"] for item in items],
        "split": [item["split"] for item in items],
        "sequence_key": [item["sequence_key"] for item in items],
        "smiles_key": [item["smiles_key"] for item in items],
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move dense tensors and BatchMolGraph objects onto the same device."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        elif key in {"bmg", "product_bmg"} and value is not None:
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    """Append one machine-readable training row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
