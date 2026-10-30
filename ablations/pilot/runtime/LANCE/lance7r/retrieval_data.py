"""Product-free retrieval features and memory construction for LANCE-7R.

The kinetic model still receives the exact LANCE-4R inputs.  This module adds a
small, deterministic key for label retrieval:

* a normalized random projection of mean-pooled frozen ESM++ residue features;
* a Morgan fingerprint of the reactant;
* exact-reactant and sequence-cluster identifiers used for routing and leakage
  prevention.

Products, reaction SMILES, assay source, and evaluation labels are never used
to build a query key.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from LANCE.lance7r.data import (
    EmbeddingCache,
    KineticsDataset,
    TargetScaler,
    clean,
    collate_kinetics,
    row_sequence_cluster_id,
    stable_group_id,
)


_PROJECTION_CACHE: dict[tuple[int, int, int], torch.Tensor] = {}
_FINGERPRINT_GENERATORS: dict[int, Any] = {}


def fixed_random_projection(input_dim: int, output_dim: int, seed: int = 713) -> torch.Tensor:
    """Return a deterministic Rademacher projection matrix.

    The matrix is fixed rather than learned so memory and query keys remain
    identical across training, checkpoint reloads, and ensemble members.
    """

    cache_key = (int(input_dim), int(output_dim), int(seed))
    projection = _PROJECTION_CACHE.get(cache_key)
    if projection is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        signs = torch.randint(0, 2, (input_dim, output_dim), generator=generator, dtype=torch.int8)
        projection = signs.to(torch.float32).mul_(2.0).sub_(1.0)
        projection.div_(math.sqrt(float(output_dim)))
        _PROJECTION_CACHE[cache_key] = projection
    return projection


def projected_protein_key(residue_features: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Compress frozen residue features into a normalized global retrieval key."""

    if residue_features.ndim != 2 or residue_features.shape[0] == 0:
        raise ValueError("residue_features must have shape (residues, channels)")
    pooled = residue_features.float().mean(dim=0)
    projection = fixed_random_projection(int(pooled.numel()), int(output_dim))
    return F.normalize(pooled @ projection, dim=0)


def morgan_fingerprint(smiles: str, output_dim: int) -> torch.Tensor:
    """Return a binary radius-two Morgan fingerprint for a valid reactant."""

    molecule = Chem.MolFromSmiles(clean(smiles))
    if molecule is None:
        return torch.zeros(output_dim, dtype=torch.float32)
    generator = _FINGERPRINT_GENERATORS.get(int(output_dim))
    if generator is None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=int(output_dim))
        _FINGERPRINT_GENERATORS[int(output_dim)] = generator
    values = generator.GetFingerprintAsNumPy(molecule)
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


class RetrievalKineticsDataset(KineticsDataset):
    """Kinetics rows augmented with deterministic product-free retrieval keys."""

    def __init__(
        self,
        data_root: Path,
        parameter: str,
        split: str,
        scaler: TargetScaler,
        esm_cache: EmbeddingCache,
        *,
        protein_key_dim: int = 64,
        fingerprint_dim: int = 128,
        retrieval_cluster_level: int = 99,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_root, parameter, split, scaler, esm_cache, **kwargs)
        self.protein_key_dim = int(protein_key_dim)
        self.fingerprint_dim = int(fingerprint_dim)
        self.retrieval_cluster_level = int(retrieval_cluster_level)
        if self.protein_key_dim <= 0 or self.fingerprint_dim <= 0:
            raise ValueError("retrieval key dimensions must be positive")
        self._retrieval_key_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None
        ] * len(self)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        row = self.frame.iloc[idx]
        sequence = clean(row["sequence"])
        smiles = clean(row[self.spec["smiles"]])
        is_kcat = self.parameter == "kcat"
        cached_keys = self._retrieval_key_cache[idx]
        if is_kcat and cached_keys is None:
            cached_keys = (
                projected_protein_key(item["protein"], self.protein_key_dim),
                morgan_fingerprint(smiles, self.fingerprint_dim),
            )
            self._retrieval_key_cache[idx] = cached_keys
        if cached_keys is None:
            cached_keys = (
                torch.zeros(self.protein_key_dim, dtype=torch.float32),
                torch.zeros(self.fingerprint_dim, dtype=torch.float32),
            )
        item.update(
            {
                "retrieval_protein_key": cached_keys[0],
                "retrieval_chemical_key": cached_keys[1],
                "retrieval_reactant_id": stable_group_id(smiles) if is_kcat else -1,
                "retrieval_cluster_id": (
                    row_sequence_cluster_id(row, sequence, self.retrieval_cluster_level)
                    if is_kcat
                    else -1
                ),
            }
        )
        return item


def collate_lance7r(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate ordinary LANCE-4R inputs plus retrieval keys."""

    product_free_items = [dict(item, jepa_mask=False) for item in items]
    batch = collate_kinetics(product_free_items)
    # LANCE-4R's kinetic prediction is already product-free; product graphs are
    # used only for its training auxiliary. LANCE-7R never constructs or moves
    # that privileged branch during training or inference.
    batch["product_bmg"] = None
    batch.update(
        {
            "retrieval_protein_key": torch.stack(
                [item["retrieval_protein_key"] for item in items]
            ).float(),
            "retrieval_chemical_key": torch.stack(
                [item["retrieval_chemical_key"] for item in items]
            ).float(),
            "retrieval_reactant_id": torch.tensor(
                [item["retrieval_reactant_id"] for item in items], dtype=torch.long
            ),
            "retrieval_cluster_id": torch.tensor(
                [item["retrieval_cluster_id"] for item in items], dtype=torch.long
            ),
        }
    )
    return batch


def collate_retrieval_training(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate only the tensors required to train label retrieval.

    The frozen LANCE-4R incumbent is intentionally not executed for these
    batches. This keeps retrieval training inexpensive and avoids fitting a
    correction to in-sample incumbent residuals.
    """

    return {
        "task_idx": torch.tensor([item["task_idx"] for item in items], dtype=torch.long),
        "target_z": torch.tensor([item["target_z"] for item in items], dtype=torch.float32),
        "target_mask": torch.tensor([item["target_mask"] for item in items], dtype=torch.bool),
        "retrieval_protein_key": torch.stack(
            [item["retrieval_protein_key"] for item in items]
        ).float(),
        "retrieval_chemical_key": torch.stack(
            [item["retrieval_chemical_key"] for item in items]
        ).float(),
        "retrieval_reactant_id": torch.tensor(
            [item["retrieval_reactant_id"] for item in items], dtype=torch.long
        ),
        "retrieval_cluster_id": torch.tensor(
            [item["retrieval_cluster_id"] for item in items], dtype=torch.long
        ),
    }


@dataclass(frozen=True)
class RetrievalMemory:
    """Fixed training-label memory serialized inside every LANCE-7R checkpoint."""

    protein_keys: torch.Tensor
    chemical_keys: torch.Tensor
    target_z: torch.Tensor
    reactant_ids: torch.Tensor
    cluster_ids: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.target_z.numel())

    def validate(self) -> None:
        if self.protein_keys.ndim != 2 or self.chemical_keys.ndim != 2:
            raise ValueError("memory keys must be rank-two tensors")
        expected = self.size
        tensors = {
            "protein_keys": self.protein_keys,
            "chemical_keys": self.chemical_keys,
            "reactant_ids": self.reactant_ids,
            "cluster_ids": self.cluster_ids,
        }
        for name, tensor in tensors.items():
            if int(tensor.shape[0]) != expected:
                raise ValueError(f"{name} has {tensor.shape[0]} rows; expected {expected}")
        if expected == 0:
            raise ValueError("retrieval memory is empty")
        if not torch.isfinite(self.target_z).all():
            raise ValueError("retrieval memory contains non-finite targets")


def build_retrieval_memory(dataset: RetrievalKineticsDataset) -> RetrievalMemory:
    """Materialize valid kcat rows from one declared training split."""

    if dataset.parameter != "kcat":
        raise ValueError("label memory must be built from the kcat dataset")

    protein_keys: list[torch.Tensor] = []
    chemical_keys: list[torch.Tensor] = []
    targets: list[float] = []
    reactant_ids: list[int] = []
    cluster_ids: list[int] = []
    for index in range(len(dataset)):
        item = dataset[index]
        if not bool(item["target_mask"]):
            continue
        protein_keys.append(item["retrieval_protein_key"])
        chemical_keys.append(item["retrieval_chemical_key"])
        targets.append(float(item["target_z"]))
        reactant_ids.append(int(item["retrieval_reactant_id"]))
        cluster_ids.append(int(item["retrieval_cluster_id"]))

    memory = RetrievalMemory(
        protein_keys=torch.stack(protein_keys).to(torch.float16),
        chemical_keys=torch.stack(chemical_keys).to(torch.uint8),
        target_z=torch.tensor(targets, dtype=torch.float32),
        reactant_ids=torch.tensor(reactant_ids, dtype=torch.long),
        cluster_ids=torch.tensor(cluster_ids, dtype=torch.long),
    )
    memory.validate()
    return memory
