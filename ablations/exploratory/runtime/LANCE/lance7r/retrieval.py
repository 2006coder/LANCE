"""Leakage-safe label retrieval for the LANCE-7R kcat residual.

The retriever has two complementary channels:

1. exact-reactant retrieval ranks enzymes acting on the same reactant by their
   frozen ESM++ protein key;
2. approximate retrieval combines protein cosine similarity with reactant
   Morgan similarity.

Rows from the query's sequence-99 cluster are always masked before neighbors
are selected. This makes every training query behave like a sequence-OOD query
and prevents the memory from copying its own label.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from LANCE.lance7r.retrieval_data import RetrievalMemory


@dataclass
class RetrievalConfig:
    """Architecture and fixed inference policy for label memory."""

    mode: str = "dual"
    protein_key_dim: int = 64
    fingerprint_dim: int = 128
    topk_exact: int = 8
    topk_approx: int = 16
    temperature: float = 0.20
    residual_limit: float = 0.50
    learn_metric: bool = True

    def validate(self) -> None:
        if self.mode not in {"exact", "dual"}:
            raise ValueError(f"retrieval mode must be exact or dual, got {self.mode!r}")
        if self.protein_key_dim <= 0 or self.fingerprint_dim <= 0:
            raise ValueError("retrieval key dimensions must be positive")
        if self.topk_exact <= 0 or self.topk_approx <= 0:
            raise ValueError("retrieval top-k values must be positive")
        if self.temperature <= 0:
            raise ValueError("retrieval temperature must be positive")
        if self.residual_limit < 0:
            raise ValueError("retrieval residual limit must be non-negative")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalConfig":
        allowed = set(cls.__dataclass_fields__)
        config = cls(**{key: payload[key] for key in allowed if key in payload})
        config.validate()
        return config

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


class LabelAnchoredRetriever(nn.Module):
    """Differentiable nearest-neighbor regression over a fixed label memory."""

    def __init__(
        self,
        config: RetrievalConfig,
        memory: RetrievalMemory | None = None,
        *,
        memory_size: int | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config

        if memory is not None:
            memory.validate()
            size = memory.size
            if memory.protein_keys.shape[1] != config.protein_key_dim:
                raise ValueError("protein memory width does not match RetrievalConfig")
            if memory.chemical_keys.shape[1] != config.fingerprint_dim:
                raise ValueError("chemical memory width does not match RetrievalConfig")
            protein_keys = memory.protein_keys
            chemical_keys = memory.chemical_keys
            target_z = memory.target_z
            reactant_ids = memory.reactant_ids
            cluster_ids = memory.cluster_ids
        else:
            size = int(memory_size or 0)
            if size <= 0:
                raise ValueError("memory_size must be positive when memory tensors are not supplied")
            protein_keys = torch.zeros(size, config.protein_key_dim, dtype=torch.float16)
            chemical_keys = torch.zeros(size, config.fingerprint_dim, dtype=torch.uint8)
            target_z = torch.zeros(size, dtype=torch.float32)
            reactant_ids = torch.zeros(size, dtype=torch.long)
            cluster_ids = torch.zeros(size, dtype=torch.long)

        self.memory_size = int(size)
        self.register_buffer("memory_protein_keys", protein_keys.contiguous())
        self.register_buffer("memory_chemical_keys", chemical_keys.contiguous())
        self.register_buffer("memory_target_z", target_z.float().contiguous())
        self.register_buffer("memory_reactant_ids", reactant_ids.long().contiguous())
        self.register_buffer("memory_cluster_ids", cluster_ids.long().contiguous())
        self.register_buffer("memory_target_mean", target_z.float().mean().reshape(()))

        # Diagonal NCA is deliberately small. It can reshape the frozen ESM++
        # metric without repeatedly projecting every memory row through an MLP.
        initial_scale = _inverse_softplus(1.0)
        self.protein_scale_logits = nn.Parameter(
            torch.full((config.protein_key_dim,), initial_scale)
        )
        self.similarity_logits = nn.Parameter(torch.zeros(2))
        self.channel_logits = nn.Parameter(torch.tensor([-1.0, 0.8, 0.0]))

        # The fixed convex neighborhood estimate remains the main prediction.
        # This zero-initialized bounded residual only learns systematic errors
        # that repeat under cross-cluster retrieval.
        self.fusion_residual = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.fusion_residual[-1].weight)
        nn.init.zeros_(self.fusion_residual[-1].bias)

        if not config.learn_metric:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _topk_regression(
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        memory_target: torch.Tensor,
        topk: int,
        temperature: float,
    ) -> dict[str, torch.Tensor]:
        """Return weighted label statistics from one masked neighbor channel."""

        k = min(int(topk), int(scores.shape[1]))
        masked_scores = scores.masked_fill(~valid_mask, -1.0e4)
        top_scores, top_indices = torch.topk(masked_scores, k=k, dim=1)
        top_valid = valid_mask.gather(1, top_indices)
        logits = (top_scores / float(temperature)).masked_fill(~top_valid, -1.0e4)
        weights = torch.softmax(logits, dim=1) * top_valid.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        labels = memory_target[top_indices]
        mean = (weights * labels).sum(dim=1)
        variance = (weights * (labels - mean.unsqueeze(1)).square()).sum(dim=1)
        available = top_valid.any(dim=1)
        maximum = torch.where(
            available,
            top_scores.masked_fill(~top_valid, -1.0e4).max(dim=1).values,
            torch.full_like(mean, -1.0),
        )
        return {
            "mean": mean,
            # Singleton neighborhoods have exactly zero variance. Clamping
            # before sqrt keeps the forward value negligible while avoiding
            # the infinite derivative of sqrt at zero during NCA training.
            "std": variance.clamp_min(1.0e-8).sqrt(),
            "max_similarity": maximum,
            "available": available,
            "valid_neighbors": top_valid.sum(dim=1),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Retrieve product-free kcat labels for the kcat rows in ``batch``."""

        query_protein = batch["retrieval_protein_key"].float()
        query_chemical = batch["retrieval_chemical_key"].float()
        query_reactant = batch["retrieval_reactant_id"].long()
        query_cluster = batch["retrieval_cluster_id"].long()

        scale = F.softplus(self.protein_scale_logits).clamp_min(1.0e-4)
        query_protein = F.normalize(query_protein * scale, dim=1)
        memory_protein = F.normalize(self.memory_protein_keys.float() * scale, dim=1)
        protein_similarity = query_protein @ memory_protein.transpose(0, 1)

        memory_chemical = self.memory_chemical_keys.float()
        intersection = query_chemical @ memory_chemical.transpose(0, 1)
        query_count = query_chemical.sum(dim=1, keepdim=True)
        memory_count = memory_chemical.sum(dim=1).unsqueeze(0)
        union = query_count + memory_count - intersection
        chemical_similarity = intersection / union.clamp_min(1.0)

        same_cluster = query_cluster.unsqueeze(1).eq(self.memory_cluster_ids.unsqueeze(0))
        same_reactant = query_reactant.unsqueeze(1).eq(self.memory_reactant_ids.unsqueeze(0))
        cluster_safe = ~same_cluster

        exact = self._topk_regression(
            protein_similarity,
            cluster_safe & same_reactant,
            self.memory_target_z,
            self.config.topk_exact,
            self.config.temperature,
        )

        similarity_weights = torch.softmax(self.similarity_logits, dim=0)
        joint_similarity = (
            similarity_weights[0] * protein_similarity
            + similarity_weights[1] * chemical_similarity
        )
        approximate = self._topk_regression(
            joint_similarity,
            cluster_safe & ~same_reactant,
            self.memory_target_z,
            self.config.topk_approx,
            self.config.temperature,
        )

        exact_confidence = (
            exact["available"].to(query_protein.dtype)
            * torch.sigmoid(4.0 * (exact["max_similarity"] - 0.10))
            * torch.exp(-0.35 * exact["std"])
        )
        approximate_confidence = (
            approximate["available"].to(query_protein.dtype)
            * torch.sigmoid(5.0 * (approximate["max_similarity"] - 0.20))
            * torch.exp(-0.50 * approximate["std"])
        )
        if self.config.mode == "exact":
            approximate_confidence = torch.zeros_like(approximate_confidence)

        channel_strength = F.softplus(self.channel_logits)
        global_weight = channel_strength[0].expand_as(exact_confidence)
        exact_weight = channel_strength[1] * exact_confidence
        approximate_weight = channel_strength[2] * approximate_confidence
        denominator = (global_weight + exact_weight + approximate_weight).clamp_min(1.0e-8)
        heuristic = (
            global_weight * self.memory_target_mean
            + exact_weight * exact["mean"]
            + approximate_weight * approximate["mean"]
        ) / denominator

        features = torch.stack(
            [
                exact["mean"],
                approximate["mean"],
                exact["std"],
                approximate["std"],
                exact["max_similarity"],
                approximate["max_similarity"],
                exact["available"].to(query_protein.dtype),
                approximate["available"].to(query_protein.dtype),
            ],
            dim=1,
        )
        residual = self.config.residual_limit * torch.tanh(
            self.fusion_residual(features).squeeze(1)
        )
        retrieval_mu = heuristic + residual
        available = (
            exact["available"]
            if self.config.mode == "exact"
            else exact["available"] | approximate["available"]
        )

        return {
            "retrieval_mu": retrieval_mu,
            "retrieval_available": available,
            "retrieval_cluster_seen": same_cluster.any(dim=1),
            "retrieval_reactant_seen": same_reactant.any(dim=1),
            "retrieval_exact_mu": exact["mean"],
            "retrieval_approx_mu": approximate["mean"],
            "retrieval_exact_std": exact["std"],
            "retrieval_approx_std": approximate["std"],
            "retrieval_exact_max_similarity": exact["max_similarity"],
            "retrieval_approx_max_similarity": approximate["max_similarity"],
            "retrieval_exact_neighbors": exact["valid_neighbors"],
            "retrieval_approx_neighbors": approximate["valid_neighbors"],
            "retrieval_protein_weight": similarity_weights[0].expand_as(retrieval_mu),
            "retrieval_chemical_weight": similarity_weights[1].expand_as(retrieval_mu),
        }
