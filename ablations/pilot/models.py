import common  # noqa: F401
import torch
from LANCE.model import (
    MLP,
    CatalyticInterfaceEncoder,
    ConditionEncoder,
    LANCEConfig,
    LANCEModel,
    MoleculeStateEncoder,
    ProteinContextEncoder,
    masked_mean,
)
from torch import nn


class Pool(nn.Module):
    def forward(self, x, mask):
        return masked_mean(x, mask), mask.to(x.dtype) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)


class Interface(nn.Module):
    def __init__(self, config, mode, budget):
        super().__init__()
        d = config.model_dim
        self.mode = mode
        self.norm = nn.LayerNorm(d)
        if mode == "matched_mlp":
            # Count includes both LayerNorms and biases.
            h = max(1, round((budget - 2 * d - 8 * d - d) / (5 * d + 1)))
            self.fusion = MLP(4 * d, h, d, config.dropout)

    def forward(self, e, p, mask, m, t):
        u = (
            self.fusion(torch.cat([e, m, e * m, (e - m).abs()], -1))
            if self.mode == "matched_mlp"
            else torch.zeros_like(e)
        )
        return dict(
            interface_context=self.norm(e + u),
            interface_attention=mask.to(e.dtype)
            / mask.sum(1, keepdim=True).clamp_min(1),
            interface_gate=e.new_zeros(len(e)),
        )


class AblatedModel(LANCEModel):
    def __init__(self, config, intervention):
        super().__init__(config)
        self.intervention = intervention
        if intervention.get("interface"):
            budget = sum(p.numel() for p in self.interface_encoder.parameters())
            self.interface_encoder = Interface(
                config, intervention["interface"], budget
            )
        if intervention.get("pool") == "mean":
            self.protein_encoder.pool = Pool()
            if self.protein_encoder.raw_rate_pool is not None:
                self.protein_encoder.raw_rate_pool = Pool()
        if intervention.get("molecule_pool") == "mean":
            self.molecule_encoder.pool = Pool()
        if intervention.get("action"):
            self.action_encoder.register_forward_hook(self.action_hook)

    def action_hook(self, module, args, out):
        out = dict(out)
        mode = self.intervention["action"]
        if mode == "tied":
            out["z_cat"] = out["z_bind"]
            out["z_cat_raw"] = out["z_bind_raw"]
        else:
            for route in ("bind", "cat"):
                if mode == "zero" or mode == "no_" + route:
                    for key in ("z_" + route, "z_" + route + "_raw"):
                        out[key] = torch.zeros_like(out[key])
        return out

    def forward(self, batch):
        batch = dict(batch)
        mode = self.intervention.get("conditions")
        if mode:
            batch["condition"] = batch["condition"].clone()
            if mode == "no_temperature":
                batch["condition"][:, [0, 2]] = 0
            elif mode == "no_ph":
                batch["condition"][:, [1, 3]] = 0
            else:
                batch["condition"][:, :2] = 0
                if mode == "zero":
                    batch["condition"].zero_()
        if self.intervention.get("protein") == "pooled":
            x = masked_mean(batch["protein"], batch["protein_mask"])
            batch["protein"] = (
                x[:, None, :].expand_as(batch["protein"])
                * batch["protein_mask"][..., None]
            )
        if self.intervention.get("products") == "none":
            batch["product_bmg"] = None
        return super().forward(batch)


class LateFusion(nn.Module):
    def __init__(self, config, budget, tasks, cross=False):
        super().__init__()
        self.config = config
        self.protein_encoder = ProteinContextEncoder(config)
        self.molecule_encoder = MoleculeStateEncoder(config)
        self.interface_encoder = CatalyticInterfaceEncoder(config) if cross else None
        self.condition_encoder = ConditionEncoder(config)
        self.task_embedding = nn.Embedding(3, config.model_dim)
        d = config.model_dim
        dims = {"kcat": 5 * d, "km": 4 * d, "ki": 4 * d}
        overhead = sum(p.numel() for p in self.parameters())
        fixed = sum(2 * dims[t] + 2 for t in tasks)
        slope = sum(dims[t] + 3 for t in tasks)
        h = max(1, round((budget - overhead - fixed) / slope))
        self.heads = nn.ModuleDict(
            {t: MLP(dims[t], h, 2, config.dropout) for t in tasks}
        )

    def forward(self, batch):
        p = self.protein_encoder(batch["protein"], batch["protein_mask"])
        m = self.molecule_encoder(batch["bmg"])["molecule_state"]
        cond = self.condition_encoder(batch["condition"])
        task = self.task_embedding(batch["task_idx"])
        if self.interface_encoder is not None:
            p["enzyme_context"] = self.interface_encoder(
                p["enzyme_context"],
                p["protein_tokens"],
                batch["protein_mask"],
                m,
                batch["task_idx"],
            )["interface_context"]
        mu = m.new_zeros(len(m))
        lv = mu.clone()
        for i, t in enumerate(("kcat", "km", "ki")):
            ix = batch["task_idx"] == i
            if not ix.any():
                continue
            parts = [p["enzyme_context"][ix], m[ix], cond[ix], task[ix]]
            if t == "kcat":
                parts.append(p["global_rate_context"][ix])
            y = self.heads[t](torch.cat(parts, -1))
            mu[ix] = y[:, 0]
            lv[ix] = y[:, 1]
        return dict(mu=mu, log_var=lv, molecule_state=m)


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prune_heads(model, tasks):
    for t in ("kcat", "km", "ki"):
        if t not in tasks:
            setattr(model.kinetic_heads, t + "_head", nn.Identity())


def make_model(variant_config, seed, protein_dim=1152):
    from LANCE.train import seed_all

    seed_all(seed)
    config = LANCEConfig.from_dict(
        {**variant_config["model"], "protein_dim": protein_dim}
    )
    reference = LANCEModel(config)
    reference_state = reference.state_dict()
    budget = count(reference)
    if variant_config.get("capacity_fraction"):
        # Match the combined independent-model budget, with only active task heads.
        target = budget * variant_config["capacity_fraction"]
        best = None
        del reference
        reference = None
        for width in range(128, 513, 8):
            trial = LANCEConfig.from_dict(
                {
                    **config.__dict__,
                    "model_dim": width,
                    "hidden_dim": 2 * width,
                    "task_action_adapter_dim": max(32, width // 2),
                }
            )
            m = LANCEModel(trial)
            prune_heads(m, variant_config["tasks"])
            n = count(m)
            del m
            candidate = (abs(n - target), width, n)
            if best is None or candidate < best:
                best = candidate
        config = LANCEConfig.from_dict(
            {
                **config.__dict__,
                "model_dim": best[1],
                "hidden_dim": 2 * best[1],
                "task_action_adapter_dim": max(32, best[1] // 2),
            }
        )
        low = LANCEModel(config)
        prune_heads(low, variant_config["tasks"])
        n0 = count(low)
        del low
        config.hidden_dim += 1
        high = LANCEModel(config)
        prune_heads(high, variant_config["tasks"])
        slope = count(high) - n0
        del high
        config.hidden_dim = max(1, round(config.hidden_dim - 1 + (target - n0) / slope))
        probe = LANCEModel(config)
        prune_heads(probe, variant_config["tasks"])
        actual = count(probe)
        del probe
        if abs(actual - target) / target > 0.01:
            raise RuntimeError("Capacity match exceeds 1%")
    seed_all(seed)
    if variant_config["intervention"].get("architecture") in (
        "late_fusion",
        "cross_attention_fusion",
    ):
        model = LateFusion(
            config,
            budget,
            variant_config["tasks"],
            variant_config["intervention"]["architecture"] == "cross_attention_fusion",
        )
    else:
        model = AblatedModel(config, variant_config["intervention"])
        if len(variant_config["tasks"]) < 3:
            prune_heads(model, variant_config["tasks"])
    state = model.state_dict()
    for k, x in reference_state.items():
        if k in state and state[k].shape == x.shape:
            state[k] = x
    model.load_state_dict(state)
    if (
        variant_config["intervention"].get("architecture")
        in ("late_fusion", "cross_attention_fusion")
        and abs(count(model) - budget) / budget > 0.01
    ):
        raise RuntimeError("Late-fusion capacity mismatch")
    return model
