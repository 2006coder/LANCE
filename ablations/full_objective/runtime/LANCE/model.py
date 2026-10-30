#!/usr/bin/env python3
"""LANCE model components.

The core idea is a LeCun-world-model-style JEPA for enzyme kinetics:

    current state  = substrate/reactant graph embedding
    action         = enzyme-conditioned catalytic action vector z_cat
    next state     = product/reaction-state embedding

The model predicts the next-state embedding in latent space, not product SMILES.
The kinetic heads then use enzyme context, molecule state, binding action, and
catalytic action to predict log-scaled kcat/Km/Ki.

CatPred provenance:
    This file DOES NOT copy CatPred training loops or CatPred model classes.
    It uses the local LANCE.dmpnn module, which vendors a consolidated,
    heavily commented version of CatPred code from:

      - catpred/features/featurization.py:
        atom_features, bond_features, MolGraph, BatchMolGraph, mol2graph
      - catpred/models/mpn.py:
        MPNEncoder
      - catpred/nn_utils.py:
        index_select_ND and initialize_weights

    LANCE.dmpnn modifies CatPred's MPNEncoder so it returns per-atom
    tokens instead of CatPred's original per-molecule pooled vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# CatPred-derived dependency, vendored locally so LANCE is self-contained.
# See LANCE/dmpnn.py for line-by-line CatPred provenance comments.
from LANCE.dmpnn import BatchMolGraph, DMPNN


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average real tokens while ignoring padded positions.

    CatPred provenance:
        No direct CatPred equivalent is copied here.  CatPred's original D-MPNN
        pools atoms inside catpred/models/mpn.py::MPNEncoder.forward.  The
        The local LANCE D-MPNN wrapper moves pooling out of D-MPNN so
        cross-modal heads can see atom tokens.  This helper is new glue for
        token sequences.
    """

    # Convert the boolean mask to the same dtype as tokens so multiplication is
    # cheap and works under autocast.
    mask_f = mask.unsqueeze(-1).to(tokens.dtype)

    # The clamp handles pathological empty sequences without creating NaNs.
    denom = mask_f.sum(dim=1).clamp_min(1.0)

    # Padded tokens contribute zero to the numerator.
    return (tokens * mask_f).sum(dim=1) / denom


class AttentivePooling(nn.Module):
    """Masked tanh-score-softmax pooling over a variable-length token sequence.

    CatPred provenance:
        Slightly modified from catpred/models/model.py::AttentivePooling.
        CatPred uses the same broad pattern:

            Linear -> tanh -> Linear(score) -> softmax -> batch matrix multiply

        Modification in this file:
          - accepts an explicit boolean mask,
          - returns attention weights for router/interpretability diagnostics,
          - is used for protein/product/substrate state summaries rather than
            CatPred's original molecule/protein branch.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Score each real token independently.
        scores = self.linear2(self.tanh(self.linear1(tokens)))

        # Make padded positions impossible to attend to.  -1e4 is finite and
        # behaves well under bf16/fp16 autocast.
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1.0e4)

        # Normalize scores across the token dimension.
        weights = self.softmax(scores)

        # Weighted sum: (B, 1, L) x (B, L, D) -> (B, 1, D).
        pooled = torch.bmm(weights.transpose(1, 2), tokens).squeeze(1)
        return pooled, weights.squeeze(-1)


class MLP(nn.Module):
    """Small LayerNorm-first MLP used for action and readout projections."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LANCEConfig:
    """Configuration for the LANCE prototype.

    Defaults are sized to match the current ESM++/D-MPNN research stack:
      - ESM++ large residue features: 1152
      - CatPred D-MPNN hidden size: 300
      - Shared model width: 512
    """

    protein_dim: int = 1152
    dmpnn_hidden: int = 300
    dmpnn_depth: int = 3
    dmpnn_dropout: float = 0.0
    model_dim: int = 512
    action_dim: int = 64
    hidden_dim: int = 1024
    num_tasks: int = 3
    dropout: float = 0.15
    # A small number of molecule-conditioned queries attend over residue tokens
    # before the action vector is formed.  This keeps the protein tower frozen
    # and cheap, but prevents LANCE from degenerating into two independently
    # pooled vectors followed by a product/difference MLP.
    num_interface_queries: int = 4
    interface_heads: int = 8
    # Default to stop-gradient targets.  This keeps the optional product-state
    # JEPA branch from finding the trivial route where reactant/product encoders
    # co-adapt into the same constant embedding.  LeCun-style SIGReg still
    # regularizes collapse, but stop-grad is cheap insurance for this supervised
    # kinetics setting.
    stop_jepa_target: bool = True
    # LANCE2-style switches are disabled by default for checkpoint compatibility.
    # Older LANCE checkpoints did not contain these modules, so from_dict() must
    # rebuild the old architecture unless a new training run explicitly opts in.
    use_soft_catalytic_focus: bool = False
    focus_heads: int = 4
    focus_temperature: float = 1.0
    use_task_film: bool = False
    use_task_interface_gate: bool = False
    use_task_specific_actions: bool = False
    # A single shared action encoder with small task-conditioned residuals.  It
    # is deliberately different from ``use_task_specific_actions``: all tasks
    # retain one common catalytic/binding basis and only learn the part of an
    # action that genuinely differs between turnover, substrate binding, and
    # inhibitor binding.
    use_task_residual_actions: bool = False
    task_action_adapter_dim: int = 256
    use_gaussian_heads: bool = False
    n_substrate_desc: int = 8
    use_condition_features: bool = False
    condition_dim: int = 4
    use_ec_aux: bool = False
    ec_num_classes: int = 8
    # CatPred's strongest sequence path pools the high-dimensional pretrained
    # residue representation before the final readout.  The original LANCE
    # encoder projected every residue to model_dim first, which can discard a
    # global family/rate signal before pooling.  This optional kcat-only bypass
    # pools frozen ESM++ in its native width and projects only the two pooled
    # summaries.  Km/Ki keep their established LANCE-3S feature surfaces.
    use_kcat_global_rate_context: bool = False
    # Product-free rows cannot supervise a reaction product.  This optional
    # branch asks z_bind to predict a cross-modal bound-state target instead.
    use_binding_jepa: bool = False
    # The LANCE-3S audit found that the self-derived binding target did not add
    # independent rate information.  New runs can keep it for Km/Ki while
    # removing both its feature and auxiliary pressure from the kcat pathway.
    binding_jepa_exclude_kcat: bool = False
    # A conservative kcat-only residual encodes pH shape, temperature response,
    # and a self-predicted EC-level rate prior.  It starts close to zero so the
    # ordinary regression path remains the initial model.
    use_mechanistic_kcat_prior: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LANCEConfig":
        """Rebuild a config saved inside a checkpoint.

        CatPred provenance:
            CatPred checkpoints rebuild models from TrainArgs.  This tiny
            dataclass loader is new LANCE code that keeps the prototype
            checkpoint format lightweight.
        """

        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: payload[key] for key in allowed if key in payload})


class ProteinContextEncoder(nn.Module):
    """Project frozen/live protein residue tokens into one enzyme context vector.

    The current prototype expects precomputed protein tokens in the batch:

        batch["protein"]      -> (B, L, protein_dim)
        batch["protein_mask"] -> (B, L)

    That keeps the JEPA prototype independent from ESM download/loading logic.
    A later training script can feed frozen ESM++ cache tensors or live LoRA ESM
    outputs into the same interface.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        self.use_kcat_global_rate_context = bool(config.use_kcat_global_rate_context)
        self.project = nn.Sequential(
            nn.LayerNorm(config.protein_dim),
            nn.Linear(config.protein_dim, config.model_dim),
            nn.GELU(),
        )
        self.pool = AttentivePooling(config.model_dim, config.model_dim)
        if self.use_kcat_global_rate_context:
            self.raw_rate_pool = AttentivePooling(config.protein_dim, config.action_dim)
            self.raw_rate_project = nn.Sequential(
                nn.LayerNorm(2 * config.protein_dim),
                nn.Linear(2 * config.protein_dim, config.model_dim),
                nn.GELU(),
            )
        else:
            self.raw_rate_pool = None
            self.raw_rate_project = None

    def forward(self, protein: torch.Tensor, protein_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        # Project each residue independently.  This is intentionally light:
        # JEPA/action learning should carry novelty, not a large random protein
        # tower that could erode the pretrained ESM prior.
        tokens = self.project(protein) * protein_mask.unsqueeze(-1).to(protein.dtype)

        # Keep both mean and attentive summaries.  The mean is stable; the
        # attentive pool lets the model emphasize putative catalytic regions.
        mean_context = masked_mean(tokens, protein_mask)
        attn_context, attn_weights = self.pool(tokens, protein_mask)

        # A residual average of the two summaries is less redundant than
        # concatenating both into a huge fusion vector.
        context = 0.5 * (mean_context + attn_context)
        output = {
            "enzyme_context": context,
            "protein_tokens": tokens,
            "protein_attention": attn_weights,
        }
        if self.raw_rate_pool is not None and self.raw_rate_project is not None:
            # Keep the pretrained coordinates intact until after pooling.  Mean
            # and attentive summaries carry complementary global/family signal;
            # their post-pooling projection is cheap and avoids an O(B*L*D^2)
            # trainable protein tower.
            raw_mean = masked_mean(protein, protein_mask)
            raw_attn, raw_weights = self.raw_rate_pool(protein, protein_mask)
            output["global_rate_context"] = self.raw_rate_project(torch.cat([raw_mean, raw_attn], dim=-1))
            output["global_rate_attention"] = raw_weights
        return output


class MoleculeStateEncoder(nn.Module):
    """Encode substrate/inhibitor/product graphs into latent molecular states."""

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()

        # CatPred-derived D-MPNN wrapper vendored in LANCE/dmpnn.py.
        # It returns per-atom tokens and a mask. Original CatPred MPNEncoder
        # returned a pooled molecule vector inside catpred/models/mpn.py.
        self.dmpnn = DMPNN(
            hidden_size=config.dmpnn_hidden,
            depth=config.dmpnn_depth,
            dropout=config.dmpnn_dropout,
        )

        self.project = nn.Sequential(
            nn.LayerNorm(config.dmpnn_hidden),
            nn.Linear(config.dmpnn_hidden, config.model_dim),
            nn.GELU(),
        )
        self.pool = AttentivePooling(config.model_dim, config.model_dim)

    def forward(self, bmg: BatchMolGraph) -> dict[str, torch.Tensor]:
        # atom_tokens: (B, max_atoms, dmpnn_hidden)
        # atom_mask:   (B, max_atoms), True for real atoms.
        atom_tokens, atom_mask = self.dmpnn(bmg)

        # Project CatPred-style atom states into the shared JEPA/action width.
        tokens = self.project(atom_tokens) * atom_mask.unsqueeze(-1).to(atom_tokens.dtype)

        # Mean and attentive pools are both useful for molecules: mean is a
        # whole-graph fingerprint; attention can emphasize reactive groups.
        mean_state = masked_mean(tokens, atom_mask)
        attn_state, attn_weights = self.pool(tokens, atom_mask)
        state = 0.5 * (mean_state + attn_state)
        return {
            "molecule_state": state,
            "molecule_tokens": tokens,
            "molecule_mask": atom_mask,
            "molecule_attention": attn_weights,
        }


class CatalyticInterfaceEncoder(nn.Module):
    """Let molecule-conditioned queries inspect frozen protein residue tokens.

    This is new LANCE code, not CatPred code.  The intent is deliberately
    conservative: use one lightweight cross-attention read, not a trainable
    protein tower.  The frozen ESM++ residue geometry remains intact, while the
    action encoder receives a substrate-aware protein context instead of a
    substrate-agnostic pooled enzyme vector.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        if config.model_dim % config.interface_heads != 0:
            raise ValueError("model_dim must be divisible by interface_heads")
        self.num_queries = int(config.num_interface_queries)
        self.use_task_gate = bool(config.use_task_interface_gate)
        self.query_from_molecule = nn.Linear(config.model_dim, self.num_queries * config.model_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.model_dim,
            num_heads=config.interface_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.query_norm = nn.LayerNorm(config.model_dim)
        self.context_norm = nn.LayerNorm(config.model_dim)
        if self.use_task_gate:
            # New LANCE2 code, not CatPred code.  The prior is deliberately
            # skeptical for kcat because the first LANCE run showed the learned
            # interface can overfit sequence-cluster shortcuts there; Ki starts
            # more trusting because the interface helped inhibition-like rows.
            gate_init = torch.tensor([-2.0, -1.0, 1.0], dtype=torch.float32)
            if config.num_tasks != 3:
                gate_init = torch.zeros(config.num_tasks, dtype=torch.float32)
            self.task_gate_logit = nn.Parameter(gate_init[: config.num_tasks].clone())
        else:
            self.register_parameter("task_gate_logit", None)

    def forward(
        self,
        enzyme_context: torch.Tensor,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        molecule_state: torch.Tensor,
        task_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, _, width = protein_tokens.shape

        # The queries are generated from the molecule state, so the residues
        # read from ESM++ can change with the substrate/inhibitor.  Only a few
        # queries are used to keep attention cost tiny: O(B * Q * residues).
        queries = self.query_from_molecule(molecule_state).view(batch_size, self.num_queries, width)
        attended, weights = self.cross_attention(
            query=queries,
            key=protein_tokens,
            value=protein_tokens,
            key_padding_mask=~protein_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        # Residual-normalize the query states, then fold them back into one
        # substrate-conditioned interface context.  The residual with the
        # original enzyme context preserves the stable global ESM++ prior.
        query_states = self.query_norm(queries + self.dropout(attended))
        interface_update = query_states.mean(dim=1)
        if self.use_task_gate:
            if task_idx is None:
                raise ValueError("task_idx is required when use_task_interface_gate=True")
            gate = torch.sigmoid(self.task_gate_logit[task_idx]).unsqueeze(-1)
            interface_update = gate * interface_update
        else:
            gate = enzyme_context.new_ones((enzyme_context.shape[0], 1))
        interface_context = self.context_norm(enzyme_context + interface_update)

        # Average over heads and molecule-conditioned queries for an interpretable
        # per-residue interface attention map.
        residue_attention = weights.mean(dim=(1, 2))
        return {
            "interface_context": interface_context,
            "interface_query_states": query_states,
            "interface_attention": residue_attention,
            "interface_gate": gate.squeeze(-1),
        }


class SoftCatalyticFocus(nn.Module):
    """Differentiable multi-head catalytic residue focus.

    CatPred provenance:
        This is new LANCE code.  It deliberately avoids the hard top-k focus
        from the Claude sketch because hard index selection sends no useful
        gradient to residues that were not selected.  Here every residue keeps a
        soft attention path, so the focus scorer can learn from the kinetic
        loss on every step.

    Why this exists:
        Global protein pooling diluted the kcat signal on distant clusters.
        This module lets the substrate/inhibitor state ask several soft
        catalytic "questions" of the frozen ESM++ residue tokens, then merges
        those pocket-like summaries into one context update.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        if config.focus_heads <= 0:
            raise ValueError("focus_heads must be positive when soft focus is enabled")
        self.num_heads = int(config.focus_heads)
        self.temperature = float(config.focus_temperature)
        self.query_from_molecule = nn.Linear(config.model_dim, self.num_heads * config.model_dim)
        self.task_query = nn.Embedding(config.num_tasks, self.num_heads * config.model_dim)
        self.key = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.value = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.merge = nn.Sequential(
            nn.LayerNorm(self.num_heads * config.model_dim),
            nn.Linear(self.num_heads * config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.out_norm = nn.LayerNorm(config.model_dim)
        nn.init.zeros_(self.task_query.weight)

    def forward(
        self,
        enzyme_context: torch.Tensor,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        molecule_state: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, _, width = protein_tokens.shape
        queries = self.query_from_molecule(molecule_state) + self.task_query(task_idx)
        queries = queries.view(batch_size, self.num_heads, width)
        keys = self.key(protein_tokens)
        values = self.value(protein_tokens)

        # Dot-product scoring is enough here because the source tokens are
        # frozen ESM++ geometry projected into model_dim.  The temperature lets
        # us anneal between broad enzyme context and sharper pocket focus.
        denom = math.sqrt(float(width)) * max(self.temperature, 1.0e-4)
        scores = torch.einsum("bhd,bld->bhl", queries, keys) / denom
        scores = scores.masked_fill(~protein_mask.unsqueeze(1), -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        focused = torch.einsum("bhl,bld->bhd", weights, values)
        update = self.merge(focused.reshape(batch_size, self.num_heads * width))
        context = self.out_norm(enzyme_context + update)
        return {
            "focus_context": context,
            "focus_update": update,
            "focus_attention": weights,
            "focus_attention_mean": weights.mean(dim=1),
        }


class TaskFiLM(nn.Module):
    """Identity-initialized per-task modulation for the shared latent context."""

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        self.film = nn.Embedding(config.num_tasks, 2 * config.model_dim)
        nn.init.zeros_(self.film.weight)

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(task_idx).chunk(2, dim=-1)
        return (1.0 + gamma) * x + beta


class ConditionEncoder(nn.Module):
    """Encode assay pH/temperature covariates into model space.

    CatPred provenance:
        This is new LANCE code.  The CatPred-DB split files contain pH and
        temperature columns, but CatPred model code is not copied here.  The
        intent is to remove condition-driven label noise from kcat/Km without
        changing the frozen protein prior.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.model_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim // 2, config.model_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition.float())


class CatalyticActionEncoder(nn.Module):
    """Infer binding and catalytic action vectors from enzyme+molecule state."""

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()

        # Pair features include symmetric interaction terms.  This is cheap and
        # avoids a full all-residue/all-atom cross-attention block in this first
        # JEPA prototype.
        pair_dim = config.model_dim * 4
        self.bind_action = MLP(pair_dim, config.hidden_dim, config.action_dim, config.dropout)
        self.catalytic_action = MLP(pair_dim, config.hidden_dim, config.action_dim, config.dropout)

        # Bring action vectors back to model_dim when they are concatenated with
        # enzyme/molecule states.  Keeping action_dim configurable lets us test
        # whether z should be a tight bottleneck later.
        self.bind_to_model = nn.Linear(config.action_dim, config.model_dim)
        self.cat_to_model = nn.Linear(config.action_dim, config.model_dim)
        self.bind_norm = nn.LayerNorm(config.model_dim)
        self.cat_norm = nn.LayerNorm(config.model_dim)

    def project_raw_actions(
        self,
        z_bind_raw: torch.Tensor,
        z_cat_raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Project raw bottlenecks into the common acted-state coordinate frame."""

        z_bind = self.bind_norm(self.bind_to_model(z_bind_raw))
        z_cat = self.cat_norm(self.cat_to_model(z_cat_raw))
        return {
            "z_bind": z_bind,
            "z_cat": z_cat,
            "z_bind_raw": z_bind_raw,
            "z_cat_raw": z_cat_raw,
        }

    def forward(self, enzyme_context: torch.Tensor, molecule_state: torch.Tensor) -> dict[str, torch.Tensor]:
        # Multiplicative and absolute-difference terms are a compact interaction
        # surface.  They are common in matching networks, but the latent action
        # interpretation here is new project code, not CatPred code.
        pair = torch.cat(
            [
                enzyme_context,
                molecule_state,
                enzyme_context * molecule_state,
                torch.abs(enzyme_context - molecule_state),
            ],
            dim=-1,
        )

        z_bind_raw = self.bind_action(pair)
        z_cat_raw = self.catalytic_action(pair)

        # Normalized action vectors are easier for the JEPA predictor and
        # kinetic heads to consume.  The raw vectors are still returned for
        # diagnostics, regularization, or future probing.
        return self.project_raw_actions(z_bind_raw, z_cat_raw)


class TaskSpecificCatalyticActionEncoder(nn.Module):
    """Separate latent-action encoders per kinetic task inside one model.

    CatPred provenance:
        This is new LANCE code.  It does not create separate CatPred-style
        models.  Instead, the frozen protein and D-MPNN substrate encoders stay
        shared, while the task-specific action bottlenecks can learn different
        chemistry for turnover, binding affinity, and inhibition.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([CatalyticActionEncoder(config) for _ in range(config.num_tasks)])

    def forward(self, enzyme_context: torch.Tensor, molecule_state: torch.Tensor, task_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = enzyme_context.shape[0]
        action_dim = self.encoders[0].bind_action.net[-1].out_features
        model_dim = enzyme_context.shape[-1]
        z_bind = enzyme_context.new_zeros((batch_size, model_dim))
        z_cat = enzyme_context.new_zeros((batch_size, model_dim))
        z_bind_raw = enzyme_context.new_zeros((batch_size, action_dim))
        z_cat_raw = enzyme_context.new_zeros((batch_size, action_dim))
        for task_id in torch.unique(task_idx):
            task_value = int(task_id.item())
            mask = task_idx == task_id
            out = self.encoders[task_value](enzyme_context[mask], molecule_state[mask])
            z_bind[mask] = out["z_bind"].to(z_bind.dtype)
            z_cat[mask] = out["z_cat"].to(z_cat.dtype)
            z_bind_raw[mask] = out["z_bind_raw"].to(z_bind_raw.dtype)
            z_cat_raw[mask] = out["z_cat_raw"].to(z_cat_raw.dtype)
        return {
            "z_bind": z_bind,
            "z_cat": z_cat,
            "z_bind_raw": z_bind_raw,
            "z_cat_raw": z_cat_raw,
        }


class TaskResidualActionEncoder(nn.Module):
    """Keep one action basis while allowing controlled kinetic-task residuals.

    Separate full action encoders made the specialist runs strong, but they also
    remove the transfer that makes a single LANCE checkpoint attractive.  This
    module starts exactly as one shared ``CatalyticActionEncoder``.  A
    zero-initialized residual adapter then learns how kcat, Km, and Ki need to
    bend that common action basis differently.  It is an intentionally small
    compromise between an all-shared bottleneck and three disconnected models.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        self.shared = CatalyticActionEncoder(config)
        self.task_embedding = nn.Embedding(config.num_tasks, config.model_dim)
        adapter_in = 2 * config.action_dim + config.model_dim
        adapter_hidden = max(32, int(config.task_action_adapter_dim))
        self.adapter = nn.Sequential(
            nn.LayerNorm(adapter_in),
            nn.Linear(adapter_in, adapter_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(adapter_hidden, 2 * config.action_dim),
        )
        # Preserve the well-tested shared LANCE action path at initialization.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(
        self,
        enzyme_context: torch.Tensor,
        molecule_state: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base = self.shared(enzyme_context, molecule_state)
        task = self.task_embedding(task_idx)
        residual = self.adapter(torch.cat([base["z_bind_raw"], base["z_cat_raw"], task], dim=-1))
        bind_delta, cat_delta = residual.chunk(2, dim=-1)
        return self.shared.project_raw_actions(
            base["z_bind_raw"] + bind_delta,
            base["z_cat_raw"] + cat_delta,
        )


class ReactionStateJEPA(nn.Module):
    """Predict product/reaction-state embeddings from reactant state + z_cat."""

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()

        # The predictor is the world-model part:
        #     catalytic action -> reaction delta.
        #
        # Earlier versions concatenated current state and action directly.  That
        # let the predictor reconstruct product-like states from the molecule
        # state while ignoring z_cat.  Predicting a delta from z_cat makes the
        # action load-bearing: z_cat ~= 0 leaves the model near the reactant.
        self.predictor = MLP(config.model_dim, config.hidden_dim, config.model_dim, config.dropout)

        # The target is not raw product SMILES.  It is the normalized product
        # state.  Earlier notes considered product-plus-delta targets, but that
        # can look like extrapolating beyond the product.  Product-state
        # prediction is the cleanest first target and easiest to ablate.

    def delta(self, z_cat: torch.Tensor) -> torch.Tensor:
        """Predict the latent reaction delta supplied by the catalytic action."""

        return self.predictor(z_cat)

    def predict(self, reactant_state: torch.Tensor, z_cat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the reaction-state prediction in the normalized latent frame.

        The base state is the stop-gradient reactant/molecule state after
        parameter-free LayerNorm.  That puts the predicted delta in the same
        coordinate frame as ``reaction_delta_target``:

            LayerNorm(stopgrad(molecule_state)) + delta(z_cat)
                -> reaction_state

        If z_cat becomes inert, the prediction stays near the reactant state and
        the product-state JEPA loss remains visible.
        """

        delta = self.delta(z_cat)
        reactant_base = F.layer_norm(reactant_state.detach(), reactant_state.shape[-1:])
        return reactant_base + delta, delta

    def target(self, reactant_state: torch.Tensor, product_state: torch.Tensor, stop_target: bool) -> torch.Tensor:
        target = F.layer_norm(product_state, product_state.shape[-1:])

        # LeWorldModel-style training can be end-to-end if an anti-collapse
        # regularizer is active.  stop_target=True is included as a conservative
        # BYOL/I-JEPA-style fallback for ablations.
        return target.detach() if stop_target else target


class BindingStateJEPA(nn.Module):
    """Predict a product-free enzyme-bound molecular state from ``z_bind``.

    Km and Ki lack product graphs, so product-state JEPA is biologically the
    wrong missing target for those rows.  We instead define a cross-modal target
    from the detached molecule state plus its substrate-conditioned enzyme
    interface.  The predictor only sees the molecule base and z_bind, making
    the binding action load-bearing.  This is latent binding consistency, not a
    claim that we observed an experimental enzyme-substrate complex.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        self.predictor = MLP(config.model_dim, config.hidden_dim, config.model_dim, config.dropout)

    def predict(self, molecule_state: torch.Tensor, z_bind: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.predictor(z_bind)
        molecule_base = F.layer_norm(molecule_state.detach(), molecule_state.shape[-1:])
        return molecule_base + delta, delta

    def target(self, molecule_state: torch.Tensor, interface_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        molecule_base = F.layer_norm(molecule_state.detach(), molecule_state.shape[-1:])
        target = F.layer_norm(
            molecule_state.detach() + interface_context.detach(),
            molecule_state.shape[-1:],
        )
        return target, target - molecule_base


class MechanisticKcatPrior(nn.Module):
    """Small, identity-starting rate correction for the kcat OOD failure mode.

    The ordinary kcat head estimates a broad sequence/chemistry residual.  This
    branch adds only effects that are routinely responsible for turnover spread:
    a pH optimum, local temperature slope/curvature, and a rate prior weighted
    by the model's own EC prediction.  EC remains an auxiliary target rather
    than an inference-time input, so this cannot rely on a supplied annotation.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        state_dim = 3 * config.model_dim
        self.condition_parameters = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, 4),
        )
        # Outputs are temperature slope, temperature curvature, pH optimum
        # logit, and pH penalty logit.  Start with a broad/weak pH effect.
        nn.init.zeros_(self.condition_parameters[-1].weight)
        with torch.no_grad():
            self.condition_parameters[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.0, -3.0]))
        self.ec_rate_offsets = nn.Parameter(torch.zeros(config.ec_num_classes))
        self.condition_gate_logit = nn.Parameter(torch.tensor(-3.0))
        self.ec_gate_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        interface_context: torch.Tensor,
        binding_state: torch.Tensor,
        catalytic_state: torch.Tensor,
        condition: torch.Tensor,
        ec_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        params = self.condition_parameters(torch.cat([interface_context, binding_state, catalytic_state], dim=-1))
        temp_slope = 0.75 * torch.tanh(params[:, 0])
        temp_curve = 0.40 * torch.tanh(params[:, 1])
        ph_optimum = torch.sigmoid(params[:, 2])
        ph_penalty = F.softplus(params[:, 3])

        temp = condition[:, 0].float()
        ph = condition[:, 1].float()
        temp_present = condition[:, 2].float()
        ph_present = condition[:, 3].float()
        # Center in the existing temperature/100 representation.  The raw assay
        # convention is not assumed beyond what the data loader already uses.
        temp_centered = temp - 0.25
        temp_effect = temp_slope * temp_centered + temp_curve * temp_centered.pow(2)
        ph_effect = -ph_penalty * (ph - ph_optimum).pow(2)
        condition_effect = temp_present * temp_effect + ph_present * ph_effect
        condition_delta = torch.sigmoid(self.condition_gate_logit) * condition_effect

        ec_probabilities = torch.softmax(ec_logits.float(), dim=-1)
        ec_rate_prior = ec_probabilities @ self.ec_rate_offsets
        ec_delta = torch.sigmoid(self.ec_gate_logit) * ec_rate_prior
        return {
            "mechanistic_kcat_delta": condition_delta + ec_delta,
            "mechanistic_condition_delta": condition_delta,
            "mechanistic_ec_delta": ec_delta,
            "mechanistic_ph_optimum": ph_optimum,
            "mechanistic_temperature_slope": temp_slope,
            "ec_rate_prior": ec_rate_prior,
        }


class KineticHeads(nn.Module):
    """Task-specific kinetic readouts for kcat, Km, and Ki.

    Task order follows the local CatPred kinetic convention:
        0 = kcat
        1 = km
        2 = ki

    The heads are separate because the biology is different:
      - kcat should lean on z_cat and predicted reaction state,
      - Km should use binding and catalytic information,
      - Ki should mostly use binding/inhibition information.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(config.num_tasks, config.model_dim)
        self.use_condition_features = bool(config.use_condition_features)
        self.use_binding_jepa = bool(config.use_binding_jepa)
        self.binding_jepa_exclude_kcat = bool(config.binding_jepa_exclude_kcat)
        self.use_mechanistic_kcat_prior = bool(config.use_mechanistic_kcat_prior)
        self.use_kcat_global_rate_context = bool(config.use_kcat_global_rate_context)
        condition_dim = config.model_dim if self.use_condition_features else 0
        binding_jepa_dim = config.model_dim if self.use_binding_jepa else 0

        full_dim = config.model_dim * 5 + condition_dim + binding_jepa_dim
        binding_dim = config.model_dim * 3 + condition_dim + binding_jepa_dim
        kcat_dim = (
            full_dim
            - (binding_jepa_dim if self.binding_jepa_exclude_kcat else 0)
            + (config.model_dim if self.use_kcat_global_rate_context else 0)
        )

        def make_head(feature_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        # kcat and Km can use the catalytic/reaction-state pathway.  Ki is a
        # binding/inhibition task in this dataset, so it gets a lean binding
        # head instead of being handed a fictional product-state feature.
        self.kcat_head = make_head(kcat_dim)
        self.km_head = make_head(full_dim)
        self.ki_head = make_head(binding_dim)
        self.mechanistic_kcat_prior = MechanisticKcatPrior(config) if self.use_mechanistic_kcat_prior else None

    def forward(
        self,
        task_idx: torch.Tensor,
        enzyme_context: torch.Tensor,
        binding_state: torch.Tensor,
        catalytic_state: torch.Tensor,
        reaction_state_pred: torch.Tensor,
        condition_context: torch.Tensor | None = None,
        binding_state_pred: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        ec_logits: torch.Tensor | None = None,
        global_rate_context: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        task = self.task_embedding(task_idx)
        full_parts = [enzyme_context, binding_state, catalytic_state, reaction_state_pred, task]
        kcat_parts = list(full_parts)
        binding_parts = [enzyme_context, binding_state, task]
        if self.use_binding_jepa:
            if binding_state_pred is None:
                raise ValueError("binding_state_pred is required when use_binding_jepa=True")
            full_parts.append(binding_state_pred)
            binding_parts.append(binding_state_pred)
            if not self.binding_jepa_exclude_kcat:
                kcat_parts.append(binding_state_pred)
        if self.use_condition_features:
            if condition_context is None:
                raise ValueError("condition_context is required when use_condition_features=True")
            full_parts.append(condition_context)
            kcat_parts.append(condition_context)
            binding_parts.append(condition_context)
        full_features = torch.cat(full_parts, dim=-1)
        kcat_features = torch.cat(kcat_parts, dim=-1)
        binding_features = torch.cat(binding_parts, dim=-1)
        if self.use_kcat_global_rate_context:
            if global_rate_context is None:
                raise ValueError("global_rate_context is required when use_kcat_global_rate_context=True")
            kcat_features = torch.cat([kcat_features, global_rate_context], dim=-1)

        # Dispatch rows to the correct task head while preserving batch order.
        # Under bf16 autocast, Linear layers can return bf16 while embeddings or
        # concatenated features remain fp32.  Cast each head output back to
        # mu.dtype before indexed assignment; this was the runtime failure in
        # the first cluster launch.
        mu = full_features.new_zeros((full_features.shape[0],))
        for task_id in torch.unique(task_idx):
            task_value = int(task_id.item())
            mask = task_idx == task_id
            if task_value == 0:
                pred = self.kcat_head(kcat_features[mask]).squeeze(-1)
            elif task_value == 1:
                pred = self.km_head(full_features[mask]).squeeze(-1)
            elif task_value == 2:
                pred = self.ki_head(binding_features[mask]).squeeze(-1)
            else:
                raise ValueError(f"task_idx contains unsupported task id {task_value}")
            mu[mask] = pred.to(mu.dtype)
        if self.mechanistic_kcat_prior is None:
            return mu
        if condition is None or ec_logits is None:
            raise ValueError("condition and ec_logits are required for the mechanistic kcat prior")
        prior_out = self.mechanistic_kcat_prior(
            enzyme_context,
            binding_state,
            catalytic_state,
            condition,
            ec_logits,
        )
        kcat_mask = task_idx == 0
        mu = mu + prior_out["mechanistic_kcat_delta"].to(mu.dtype) * kcat_mask.to(mu.dtype)
        return {"mu": mu, **prior_out}


class GaussianKineticHeads(nn.Module):
    """Task-specific heteroscedastic kinetic heads.

    CatPred provenance:
        This is new LANCE code.  CatPred uses MVE in its own model family, but
        no CatPred head implementation is copied here.  The motivation is from
        the LANCE diagnostic: ordinary R2 and Pearson r2 were close, so the head
        alone cannot solve OOD, but a Gaussian head still helps noisy kcat rows
        stop dominating the shared latent action space.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(config.num_tasks, config.model_dim)
        self.use_condition_features = bool(config.use_condition_features)
        self.use_binding_jepa = bool(config.use_binding_jepa)
        self.binding_jepa_exclude_kcat = bool(config.binding_jepa_exclude_kcat)
        self.use_mechanistic_kcat_prior = bool(config.use_mechanistic_kcat_prior)
        self.use_kcat_global_rate_context = bool(config.use_kcat_global_rate_context)
        condition_dim = config.model_dim if self.use_condition_features else 0
        binding_jepa_dim = config.model_dim if self.use_binding_jepa else 0

        full_dim = config.model_dim * 5 + condition_dim + binding_jepa_dim
        binding_dim = config.model_dim * 3 + condition_dim + binding_jepa_dim
        kcat_dim = (
            full_dim
            - (binding_jepa_dim if self.binding_jepa_exclude_kcat else 0)
            + (config.model_dim if self.use_kcat_global_rate_context else 0)
        )

        def make_head(feature_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                # Output order: mean, log variance.  The loss clamps log_var,
                # not the head, so diagnostics can still show if it saturates.
                nn.Linear(config.hidden_dim, 2),
            )

        self.kcat_head = make_head(kcat_dim)
        self.km_head = make_head(full_dim)
        self.ki_head = make_head(binding_dim)
        self.substrate_descriptor_head = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim // 2, config.n_substrate_desc),
        )
        self.mechanistic_kcat_prior = MechanisticKcatPrior(config) if self.use_mechanistic_kcat_prior else None

    def forward(
        self,
        task_idx: torch.Tensor,
        enzyme_context: torch.Tensor,
        binding_state: torch.Tensor,
        catalytic_state: torch.Tensor,
        reaction_state_pred: torch.Tensor,
        condition_context: torch.Tensor | None = None,
        binding_state_pred: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        ec_logits: torch.Tensor | None = None,
        global_rate_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        task = self.task_embedding(task_idx)
        full_parts = [enzyme_context, binding_state, catalytic_state, reaction_state_pred, task]
        kcat_parts = list(full_parts)
        binding_parts = [enzyme_context, binding_state, task]
        if self.use_binding_jepa:
            if binding_state_pred is None:
                raise ValueError("binding_state_pred is required when use_binding_jepa=True")
            full_parts.append(binding_state_pred)
            binding_parts.append(binding_state_pred)
            if not self.binding_jepa_exclude_kcat:
                kcat_parts.append(binding_state_pred)
        if self.use_condition_features:
            if condition_context is None:
                raise ValueError("condition_context is required when use_condition_features=True")
            full_parts.append(condition_context)
            kcat_parts.append(condition_context)
            binding_parts.append(condition_context)
        full_features = torch.cat(full_parts, dim=-1)
        kcat_features = torch.cat(kcat_parts, dim=-1)
        binding_features = torch.cat(binding_parts, dim=-1)
        if self.use_kcat_global_rate_context:
            if global_rate_context is None:
                raise ValueError("global_rate_context is required when use_kcat_global_rate_context=True")
            kcat_features = torch.cat([kcat_features, global_rate_context], dim=-1)

        mu = full_features.new_zeros((full_features.shape[0],))
        log_var = full_features.new_zeros((full_features.shape[0],))
        for task_id in torch.unique(task_idx):
            task_value = int(task_id.item())
            mask = task_idx == task_id
            if task_value == 0:
                pred = self.kcat_head(kcat_features[mask])
            elif task_value == 1:
                pred = self.km_head(full_features[mask])
            elif task_value == 2:
                pred = self.ki_head(binding_features[mask])
            else:
                raise ValueError(f"task_idx contains unsupported task id {task_value}")
            mu[mask] = pred[:, 0].to(mu.dtype)
            log_var[mask] = pred[:, 1].to(log_var.dtype)

        output = {
            "mu": mu,
            "log_var": log_var,
            "substrate_desc_pred": self.substrate_descriptor_head(binding_state),
        }
        if self.mechanistic_kcat_prior is not None:
            if condition is None or ec_logits is None:
                raise ValueError("condition and ec_logits are required for the mechanistic kcat prior")
            prior_out = self.mechanistic_kcat_prior(
                enzyme_context,
                binding_state,
                catalytic_state,
                condition,
                ec_logits,
            )
            kcat_mask = task_idx == 0
            output["mu"] = output["mu"] + prior_out["mechanistic_kcat_delta"].to(mu.dtype) * kcat_mask.to(mu.dtype)
            output.update(prior_out)
        return output


class LANCEModel(nn.Module):
    """End-to-end catalytic-action JEPA prototype.

    Expected batch keys:
        protein:      (B, L, protein_dim) precomputed residue features
        protein_mask: (B, L) True for real residues
        bmg:          BatchMolGraph for substrate/inhibitor/reactant
        task_idx:     (B,) where 0=kcat, 1=km, 2=ki

    Optional JEPA pretraining/fine-tuning key:
        product_bmg:  BatchMolGraph for product/reaction outcome

    If product_bmg is absent, the model still predicts kinetics.  It simply
    cannot compute the JEPA target for that batch.
    """

    def __init__(self, config: LANCEConfig) -> None:
        super().__init__()
        if config.use_task_specific_actions and config.use_task_residual_actions:
            raise ValueError("task-specific and task-residual actions are mutually exclusive")
        if config.use_mechanistic_kcat_prior and not config.use_condition_features:
            raise ValueError("the mechanistic kcat prior requires condition features")
        if config.binding_jepa_exclude_kcat and not config.use_binding_jepa:
            raise ValueError("binding_jepa_exclude_kcat requires use_binding_jepa")
        self.config = config
        self.protein_encoder = ProteinContextEncoder(config)
        self.molecule_encoder = MoleculeStateEncoder(config)
        self.interface_encoder = CatalyticInterfaceEncoder(config)
        self.condition_encoder = ConditionEncoder(config) if config.use_condition_features else None
        self.soft_focus = SoftCatalyticFocus(config) if config.use_soft_catalytic_focus else None
        self.task_film = TaskFiLM(config) if config.use_task_film else None
        self.action_encoder = (
            TaskSpecificCatalyticActionEncoder(config)
            if config.use_task_specific_actions
            else TaskResidualActionEncoder(config)
            if config.use_task_residual_actions
            else CatalyticActionEncoder(config)
        )
        self.reaction_jepa = ReactionStateJEPA(config)
        self.binding_jepa = BindingStateJEPA(config) if config.use_binding_jepa else None
        self.kinetic_heads = GaussianKineticHeads(config) if config.use_gaussian_heads else KineticHeads(config)
        self.ec_aux_head = (
            nn.Sequential(
                nn.LayerNorm(config.model_dim),
                nn.Linear(config.model_dim, config.model_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.model_dim // 2, config.ec_num_classes),
            )
            if config.use_ec_aux or config.use_mechanistic_kcat_prior
            else None
        )
        self.binding_state_norm = nn.LayerNorm(config.model_dim)
        self.binding_jepa_state_norm = nn.LayerNorm(config.model_dim)
        self.catalytic_state_norm = nn.LayerNorm(config.model_dim)
        self.reaction_state_norm = nn.LayerNorm(config.model_dim)
        # Normalized model-space actions have norm around sqrt(model_dim).  These
        # learned global scales start near 0.1 so the model can calibrate action
        # strength instead of having LayerNorm impose a full-scale shift on day
        # one.  The sigmoid keeps scales positive and bounded.
        self.bind_action_scale_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.cat_action_scale_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.sigreg_projector = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        protein_out = self.protein_encoder(batch["protein"], batch["protein_mask"])
        reactant_out = self.molecule_encoder(batch["bmg"])

        enzyme_context = protein_out["enzyme_context"]
        molecule_state = reactant_out["molecule_state"]
        condition_context = None
        if self.condition_encoder is not None:
            if "condition" not in batch:
                raise ValueError("batch is missing condition features for condition-aware LANCE")
            condition_context = self.condition_encoder(batch["condition"])
        interface_out = self.interface_encoder(
            enzyme_context,
            protein_out["protein_tokens"],
            batch["protein_mask"],
            molecule_state,
            batch["task_idx"],
        )
        interface_context = interface_out["interface_context"]
        focus_out: dict[str, torch.Tensor] = {}
        if self.soft_focus is not None:
            focus_out = self.soft_focus(
                interface_context,
                protein_out["protein_tokens"],
                batch["protein_mask"],
                molecule_state,
                batch["task_idx"],
            )
            interface_context = focus_out["focus_context"]
        if self.task_film is not None:
            interface_context = self.task_film(interface_context, batch["task_idx"])
        if self.config.use_task_specific_actions or self.config.use_task_residual_actions:
            actions = self.action_encoder(interface_context, molecule_state, batch["task_idx"])
        else:
            actions = self.action_encoder(interface_context, molecule_state)

        # Make the word "action" true by construction: the action vectors are
        # not merely concatenated into a readout.  They explicitly shift the
        # molecule state into binding and catalytic acted states.  The kinetic
        # heads consume these acted states rather than raw free-floating action
        # vectors.
        bind_action_scale = torch.sigmoid(self.bind_action_scale_logit)
        cat_action_scale = torch.sigmoid(self.cat_action_scale_logit)
        binding_state = self.binding_state_norm(molecule_state + bind_action_scale * actions["z_bind"])
        catalytic_state = self.catalytic_state_norm(molecule_state + cat_action_scale * actions["z_cat"])
        binding_jepa_out: dict[str, torch.Tensor] = {}
        binding_state_pred = None
        if self.binding_jepa is not None:
            binding_state_pre, binding_delta = self.binding_jepa.predict(molecule_state, actions["z_bind"])
            binding_state_pred = self.binding_jepa_state_norm(binding_state_pre)
            binding_state_target, binding_delta_target = self.binding_jepa.target(molecule_state, interface_context)
            binding_jepa_out = {
                "binding_state_pred": binding_state_pred,
                "binding_state_target": binding_state_target,
                "binding_delta": binding_delta,
                "binding_delta_target": binding_delta_target,
            }
        reaction_state_pre, reaction_delta = self.reaction_jepa.predict(molecule_state, actions["z_cat"])
        reaction_state_pred = self.reaction_state_norm(reaction_state_pre)
        ec_logits = self.ec_aux_head(actions["z_cat"]) if self.ec_aux_head is not None else None
        head_out = self.kinetic_heads(
            batch["task_idx"],
            interface_context,
            binding_state,
            catalytic_state,
            reaction_state_pred,
            condition_context=condition_context,
            binding_state_pred=binding_state_pred,
            condition=batch.get("condition"),
            ec_logits=ec_logits,
            global_rate_context=protein_out.get("global_rate_context"),
        )
        if isinstance(head_out, dict):
            mu = head_out["mu"]
            head_extra = {key: value for key, value in head_out.items() if key != "mu"}
        else:
            mu = head_out
            head_extra = {}

        out: dict[str, torch.Tensor] = {
            "mu": mu,
            "enzyme_context": enzyme_context,
            "interface_context": interface_context,
            "molecule_state": molecule_state,
            "binding_state": binding_state,
            "catalytic_state": catalytic_state,
            "reaction_delta": reaction_delta,
            "reaction_state_pred": reaction_state_pred,
            "bind_action_scale": bind_action_scale.expand(molecule_state.shape[0]),
            "cat_action_scale": cat_action_scale.expand(molecule_state.shape[0]),
            "sigreg_molecule_state": self.sigreg_projector(molecule_state),
            "sigreg_binding_state": self.sigreg_projector(binding_state),
            "sigreg_catalytic_state": self.sigreg_projector(catalytic_state),
            "sigreg_reaction_delta": self.sigreg_projector(reaction_delta),
            "sigreg_reaction_state_pred": self.sigreg_projector(reaction_state_pred),
            **actions,
            **binding_jepa_out,
            **head_extra,
            **protein_out,
            **interface_out,
            **focus_out,
            **{f"reactant_{k}": v for k, v in reactant_out.items()},
        }
        if condition_context is not None:
            out["condition_context"] = condition_context
        if binding_state_pred is not None:
            out["sigreg_binding_state_pred"] = self.sigreg_projector(binding_state_pred)
            out["sigreg_binding_state_target"] = self.sigreg_projector(binding_jepa_out["binding_state_target"])
            out["sigreg_binding_delta"] = self.sigreg_projector(binding_jepa_out["binding_delta"])
            out["sigreg_binding_delta_target"] = self.sigreg_projector(binding_jepa_out["binding_delta_target"])
        if ec_logits is not None:
            out["ec_logits"] = ec_logits

        if "product_bmg" in batch and batch["product_bmg"] is not None:
            product_out = self.molecule_encoder(batch["product_bmg"])
            product_state = product_out["molecule_state"]
            reaction_state_target = self.reaction_jepa.target(
                molecule_state,
                product_state,
                stop_target=bool(self.config.stop_jepa_target),
            )
            # Directly expose the latent reaction displacement target.  The
            # whole product embedding can be dominated by an unchanged scaffold;
            # this target lets the loss and diagnostics ask whether z_cat moves
            # the reactant state in the product direction.
            reactant_base = F.layer_norm(molecule_state.detach(), molecule_state.shape[-1:])
            reaction_delta_target = reaction_state_target - reactant_base
            out["product_state"] = product_state
            out["reaction_state_target"] = reaction_state_target
            out["reaction_delta_target"] = reaction_delta_target
            out["sigreg_product_state"] = self.sigreg_projector(product_state)
            out["sigreg_reaction_state_target"] = self.sigreg_projector(reaction_state_target)
            out.update({f"product_{k}": v for k, v in product_out.items()})

        return out

    def config_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable checkpoint config payload."""

        return dict(self.config.__dict__)
