import common  # noqa: F401
import torch
from LANCE.lance7r.retrieval import LabelAnchoredRetriever, RetrievalConfig
from LANCE.lance7r.retrieval_data import RetrievalMemory


class AblatedRetriever(LabelAnchoredRetriever):
    def __init__(self, variant_config, memory, seed):
        self.variant = variant_config
        if variant_config.get("memory_fraction"):
            n = max(1, round(memory.size * variant_config["memory_fraction"]))
            ix = torch.randperm(
                memory.size, generator=torch.Generator().manual_seed(seed + 781)
            )[:n]
            memory = RetrievalMemory(
                **{k: getattr(memory, k)[ix] for k in memory.__dataclass_fields__}
            )
        if variant_config.get("shuffle_labels"):
            order = torch.randperm(
                memory.size, generator=torch.Generator().manual_seed(seed + 971)
            )
            memory = RetrievalMemory(
                **{
                    k: (
                        memory.target_z[order]
                        if k == "target_z"
                        else getattr(memory, k)
                    )
                    for k in memory.__dataclass_fields__
                }
            )
        config = RetrievalConfig(
            mode=variant_config.get("mode", "exact"),
            learn_metric=True,
            topk_exact=variant_config.get("topk_exact", 8),
            temperature=variant_config.get("temperature", 0.2),
        )
        super().__init__(config, memory)
        learn = variant_config.get("learn", "all")
        for name, p in self.named_parameters():
            active = (
                learn == "all"
                or (learn == "metric" and name == "protein_scale_logits")
                or (learn == "residual" and name.startswith("fusion_residual."))
                or (learn == "channel" and name == "channel_logits")
                or (
                    learn == "metric_channels"
                    and not name.startswith("fusion_residual.")
                )
            )
            p.requires_grad_(active)
        if variant_config.get("strict_exact"):
            self.fusion_residual.register_forward_pre_hook(self.exact_features)

    def exact_features(self, module, args):
        x = args[0].clone()
        x[:, [1, 3, 7]] = 0
        x[:, 5] = -1
        return (x,)

    def forward(self, batch):
        out = super().forward(batch)
        if self.variant.get("estimate") == "mean":
            eligible = (
                batch["retrieval_reactant_id"][:, None]
                == self.memory_reactant_ids[None, :]
            ) & (
                batch["retrieval_cluster_id"][:, None]
                != self.memory_cluster_ids[None, :]
            )
            out["retrieval_mu"] = (eligible * self.memory_target_z[None, :]).sum(
                1
            ) / eligible.sum(1).clamp_min(1)
        if self.variant.get("no_shrinkage"):
            e = (
                torch.sigmoid(4 * (out["retrieval_exact_max_similarity"] - 0.10))
                * torch.exp(-0.35 * out["retrieval_exact_std"])
                * (out["retrieval_exact_neighbors"] > 0)
            )
            strengths = torch.nn.functional.softplus(self.channel_logits)
            original = (
                strengths[0] * self.memory_target_mean
                + strengths[1] * e * out["retrieval_exact_mu"]
            ) / (strengths[0] + strengths[1] * e).clamp_min(1e-8)
            out["retrieval_mu"] = (
                out["retrieval_mu"] - original + out["retrieval_exact_mu"]
            )
        return out


def alpha_fit(y, base, retr, available, maximum=0.5):
    import numpy as np

    mask = available.astype(bool)
    d = (retr - base)[mask]
    r = (y - base)[mask]
    if len(d) < 2 or float(d @ d) <= 1e-12:
        return 0.0
    return float(np.clip((r @ d) / (d @ d), 0, maximum))
