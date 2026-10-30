"""One-checkpoint LANCE-7R model.

LANCE-7R is composition rather than a rewrite:

* ``incumbent`` is the byte-for-byte LANCE-4R architecture;
* ``retriever`` is a product-free kcat label-memory expert;
* Km and Ki always return the incumbent prediction exactly;
* kcat receives a bounded convex movement toward the retrieval estimate.

The separation makes the scientific claim and the checkpoint contract easy to
audit. There is no runtime architecture flag and no external task checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from LANCE.lance7r.incumbent import LANCEConfig as LANCE4RConfig
from LANCE.lance7r.incumbent import LANCEModel as LANCE4RModel
from LANCE.lance7r.retrieval import LabelAnchoredRetriever, RetrievalConfig
from LANCE.lance7r.retrieval_data import RetrievalMemory


@dataclass
class LANCE7RConfig:
    """Serializable contract for an incumbent plus one retrieval expert."""

    incumbent_config: dict[str, Any]
    retrieval_config: dict[str, Any]
    memory_size: int
    blend_alpha: float = 0.0
    unseen_blend_alpha: float | None = None
    architecture: str = "lance7r_label_anchored_residual"

    def validate(self) -> None:
        if self.architecture != "lance7r_label_anchored_residual":
            raise ValueError(f"unsupported LANCE-7R architecture {self.architecture!r}")
        if self.memory_size <= 0:
            raise ValueError("LANCE-7R memory_size must be positive")
        if not 0.0 <= self.blend_alpha <= 1.0:
            raise ValueError("LANCE-7R blend_alpha must be in [0, 1]")
        if (
            self.unseen_blend_alpha is not None
            and not 0.0 <= self.unseen_blend_alpha <= 1.0
        ):
            raise ValueError("LANCE-7R unseen_blend_alpha must be in [0, 1]")
        LANCE4RConfig.from_dict(self.incumbent_config)
        RetrievalConfig.from_dict(self.retrieval_config)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LANCE7RConfig":
        config = cls(
            incumbent_config=dict(payload["incumbent_config"]),
            retrieval_config=dict(payload["retrieval_config"]),
            memory_size=int(payload["memory_size"]),
            blend_alpha=float(payload.get("blend_alpha", 0.0)),
            unseen_blend_alpha=(
                None
                if payload.get("unseen_blend_alpha") is None
                else float(payload["unseen_blend_alpha"])
            ),
            architecture=str(
                payload.get("architecture", "lance7r_label_anchored_residual")
            ),
        )
        config.validate()
        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "incumbent_config": dict(self.incumbent_config),
            "retrieval_config": dict(self.retrieval_config),
            "memory_size": int(self.memory_size),
            "blend_alpha": float(self.blend_alpha),
            "unseen_blend_alpha": (
                None
                if self.unseen_blend_alpha is None
                else float(self.unseen_blend_alpha)
            ),
        }


class LANCE7RModel(nn.Module):
    """Frozen LANCE-4R with a trainable, sequence-safe kcat memory residual."""

    def __init__(
        self,
        config: LANCE7RConfig,
        memory: RetrievalMemory | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.incumbent = LANCE4RModel(LANCE4RConfig.from_dict(config.incumbent_config))
        retrieval_config = RetrievalConfig.from_dict(config.retrieval_config)
        self.retriever = LabelAnchoredRetriever(
            retrieval_config,
            memory,
            memory_size=config.memory_size,
        )
        self.register_buffer(
            "blend_alpha",
            torch.tensor(float(config.blend_alpha), dtype=torch.float32),
        )
        self.register_buffer(
            "unseen_blend_alpha",
            (
                None
                if config.unseen_blend_alpha is None
                else torch.tensor(
                    float(config.unseen_blend_alpha),
                    dtype=torch.float32,
                )
            ),
        )
        self.freeze_incumbent()

    def freeze_incumbent(self) -> None:
        """Make preservation of all LANCE-4R paths an exact constraint."""

        for parameter in self.incumbent.parameters():
            parameter.requires_grad_(False)
        self.incumbent.eval()

    def load_incumbent_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load an exact archived LANCE-4R state before retrieval training."""

        self.incumbent.load_state_dict(state_dict, strict=True)
        self.freeze_incumbent()

    def set_blend_alpha(self, value: float) -> None:
        """Set the validation-calibrated convex blend coefficient."""

        alpha = float(value)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("blend alpha must be in [0, 1]")
        self.blend_alpha.fill_(alpha)
        self.config.blend_alpha = alpha

    def set_blend_alphas(self, seen: float, unseen: float) -> None:
        """Freeze separate blend strengths for familiar and novel clusters."""

        seen_alpha = float(seen)
        unseen_alpha = float(unseen)
        if not 0.0 <= seen_alpha <= 1.0 or not 0.0 <= unseen_alpha <= 1.0:
            raise ValueError("blend alphas must be in [0, 1]")
        self.blend_alpha.fill_(seen_alpha)
        if self.unseen_blend_alpha is None:
            self.unseen_blend_alpha = torch.tensor(
                unseen_alpha,
                dtype=self.blend_alpha.dtype,
                device=self.blend_alpha.device,
            )
        else:
            self.unseen_blend_alpha.fill_(unseen_alpha)
        self.config.blend_alpha = seen_alpha
        self.config.unseen_blend_alpha = unseen_alpha

    def train(self, mode: bool = True) -> "LANCE7RModel":
        """Train only retrieval while keeping the incumbent deterministic."""

        super().train(mode)
        self.incumbent.eval()
        self.retriever.train(mode)
        return self

    def retrieve(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expose the memory predictor for its cross-cluster training objective."""

        return self.retriever(batch)

    @staticmethod
    def _product_free_batch(batch: dict[str, Any]) -> dict[str, Any]:
        if batch.get("product_bmg") is None:
            return batch
        product_free = dict(batch)
        product_free["product_bmg"] = None
        return product_free

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return one vector containing kcat, Km, and Ki predictions."""

        with torch.no_grad():
            incumbent_out = self.incumbent(self._product_free_batch(batch))

        base_mu = incumbent_out["mu"]
        final_mu = base_mu.clone()
        kcat_mask = batch["task_idx"].eq(0)

        retrieval_mu_full = base_mu.clone()
        retrieval_available_full = torch.zeros_like(kcat_mask)
        retrieval_alpha_full = self.blend_alpha.expand_as(base_mu).clone()
        if bool(kcat_mask.any()):
            retrieval_batch = {
                key: batch[key][kcat_mask]
                for key in (
                    "retrieval_protein_key",
                    "retrieval_chemical_key",
                    "retrieval_reactant_id",
                    "retrieval_cluster_id",
                )
            }
            retrieval_out = self.retriever(retrieval_batch)
            retrieval_mu = retrieval_out["retrieval_mu"].to(base_mu.dtype)
            available = retrieval_out["retrieval_available"]
            base_kcat = base_mu[kcat_mask]
            if self.unseen_blend_alpha is None:
                row_alpha = self.blend_alpha.expand_as(base_kcat)
            else:
                row_alpha = torch.where(
                    retrieval_out["retrieval_cluster_seen"],
                    self.blend_alpha,
                    self.unseen_blend_alpha,
                ).to(base_mu.dtype)
            movement = (
                row_alpha
                * available.to(base_mu.dtype)
                * (retrieval_mu - base_kcat)
            )
            final_mu[kcat_mask] = base_kcat + movement
            retrieval_mu_full[kcat_mask] = retrieval_mu
            retrieval_available_full[kcat_mask] = available
            retrieval_alpha_full[kcat_mask] = row_alpha

            for name, value in retrieval_out.items():
                full_name = name
                if value.ndim == 0:
                    incumbent_out[full_name] = value
                    continue
                fill = torch.zeros(
                    (base_mu.shape[0], *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                fill[kcat_mask] = value
                incumbent_out[full_name] = fill

        incumbent_out["mu_base"] = base_mu
        incumbent_out["mu"] = final_mu
        incumbent_out["retrieval_mu"] = retrieval_mu_full
        incumbent_out["retrieval_available"] = retrieval_available_full
        incumbent_out["retrieval_blend_alpha"] = retrieval_alpha_full
        return incumbent_out

    def config_dict(self) -> dict[str, Any]:
        """Return the complete architecture contract stored in checkpoints."""

        self.config.blend_alpha = float(self.blend_alpha.detach().cpu())
        self.config.unseen_blend_alpha = (
            None
            if self.unseen_blend_alpha is None
            else float(self.unseen_blend_alpha.detach().cpu())
        )
        return self.config.as_dict()
