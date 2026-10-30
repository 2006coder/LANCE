#!/usr/bin/env python3
"""Loss functions for LANCE.

The loss has three separable parts:

  1. kinetic regression loss on log-scaled/standardized kcat/Km/Ki labels,
  2. JEPA latent prediction loss for reaction/product-state embeddings,
  3. a SIGReg-style anti-collapse regularizer for latent embeddings.

CatPred provenance:
    This file does not copy CatPred loss code.  CatPred's molecule-level
    training losses live in catpred/train/loss_functions.py.  The helpers below
    are new prototype losses for the catalytic-action JEPA idea.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class LANCELossWeights:
    """Weights for the prototype objective.

    The defaults are deliberately conservative: kinetic regression remains the
    anchor, while JEPA and anti-collapse terms provide representation pressure.
    """

    kinetic: float = 1.0
    jepa: float = 0.2
    jepa_delta: float = 0.1
    jepa_contrastive: float = 0.05
    action_decorrelation: float = 0.01
    sigreg: float = 0.02
    vrex: float = 0.0
    binding_aux: float = 0.0
    binding_jepa: float = 0.0
    binding_jepa_delta: float = 0.0
    binding_jepa_contrastive: float = 0.0
    binding_jepa_exclude_kcat: bool = False
    # kcat is the remaining OOD wall.  These losses counter the observed
    # regression-to-the-mean without perturbing the already strong Km/Ki paths.
    kcat_tail: float = 0.0
    kcat_tail_strength: float = 0.75
    kcat_tail_center: float = 0.5
    kcat_mechanism_ranking: float = 0.0
    kcat_reaction_ranking: float = 0.0
    kcat_ranking_temperature: float = 0.35
    kcat_ranking_min_gap: float = 0.25
    km_p1: float = 0.0
    km_p1_temperature: float = 0.15
    # Gaussian-head controls.  Defaults preserve the LANCE2 run for exact
    # reproducibility; new ablations can opt into beta-NLL or a decoupled head.
    gaussian_loss: str = "nll"
    beta_nll_beta: float = 0.5
    decoupled_sigma: float = 0.05
    vrex_per_task: bool = False
    ec_aux: float = 0.0


def masked_mse(mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared error over valid rows.

    This is new code, not CatPred code.  It mirrors the masking convention used
    in the local kinetics prototypes, where target_mask=False rows are ignored.
    """

    per_row = (mu.float() - target.float()).pow(2)
    if mask is None:
        return per_row.mean()
    if not mask.any():
        return per_row.sum() * 0.0
    return per_row[mask].mean()


def gaussian_nll_per_sample(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row Gaussian negative log likelihood for standardized targets."""

    log_var = log_var.float().clamp(-7.0, 7.0)
    err2 = (mu.float() - target.float()).pow(2)
    return 0.5 * (torch.exp(-log_var) * err2 + log_var)


def beta_gaussian_nll_per_sample(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Beta-NLL with detached variance weighting.

    This is new LANCE code.  Naive Gaussian NLL scales the mean-gradient by
    ``1 / variance``; on noisy tail rows the sigma head can inflate variance
    and starve the mean of signal.  Beta-NLL multiplies the NLL by
    ``variance ** beta`` using a detached variance, restoring much of the
    mean-gradient while still letting the sigma head model aleatoric noise.
    """

    log_var_clamped = log_var.float().clamp(-7.0, 7.0)
    nll = gaussian_nll_per_sample(mu, log_var_clamped, target)
    beta_value = float(beta)
    if beta_value <= 0.0:
        return nll
    return nll * torch.exp(beta_value * log_var_clamped.detach())


def squared_residual_per_sample(mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain squared residual used for V-REx risk grouping."""

    return (mu.float() - target.float()).pow(2)


def masked_gaussian_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    mode: str = "nll",
    beta: float = 0.5,
    decoupled_sigma: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Gaussian-head training loss, valid per-row V-REx risk, and mask."""

    mode = str(mode)
    if mode == "nll":
        per_row = gaussian_nll_per_sample(mu, log_var, target)
    elif mode == "beta_nll":
        per_row = beta_gaussian_nll_per_sample(mu, log_var, target, beta)
    elif mode == "decoupled":
        # Train the point estimate directly.  The sigma head still learns from
        # NLL, but mu is detached inside that term so high sigma cannot silence
        # the mean-gradient on difficult rows.
        mu_loss = F.smooth_l1_loss(mu.float(), target.float(), reduction="none")
        sigma_loss = gaussian_nll_per_sample(mu.detach(), log_var, target)
        per_row = mu_loss + float(decoupled_sigma) * sigma_loss
    else:
        raise ValueError(f"unknown gaussian loss mode: {mode}")
    valid = mask.bool() if mask is not None else torch.ones_like(per_row, dtype=torch.bool)
    if not valid.any():
        return per_row.sum() * 0.0, per_row[:0], valid
    # V-REx should see mean-prediction risk, not NLL.  Otherwise the sigma head
    # can game the robust objective by changing uncertainty instead of ranking.
    residual_risk = squared_residual_per_sample(mu, target)
    return per_row[valid].mean(), residual_risk[valid], valid


def vrex_penalty(per_sample_loss: torch.Tensor, cluster_ids: torch.Tensor | None) -> torch.Tensor:
    """Variance of per-cluster risk means.

    This is new LANCE code, not CatPred code.  It penalizes shortcuts that only
    work in some sequence clusters, which is the failure pattern exposed by the
    LANCE OOD99 diagnostic.
    """

    if cluster_ids is None or per_sample_loss.numel() < 2:
        return per_sample_loss.sum() * 0.0
    risks: list[torch.Tensor] = []
    for cluster_id in torch.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        if mask.any():
            risks.append(per_sample_loss[mask].mean())
    if len(risks) < 2:
        return per_sample_loss.sum() * 0.0
    return torch.stack(risks).var(unbiased=False)


def per_task_vrex_penalty(
    per_sample_loss: torch.Tensor,
    cluster_ids: torch.Tensor | None,
    task_idx: torch.Tensor | None,
) -> torch.Tensor:
    """V-REx scoped within each kinetic task, then averaged.

    Mixing kcat/Km/Ki in one V-REx group penalizes task composition differences
    between clusters.  That is not sequence-cluster invariance; it is task-mix
    regularization.  This helper only compares clusters inside the same kinetic
    task, preserving the intended robust-risk signal.
    """

    if cluster_ids is None or task_idx is None or per_sample_loss.numel() < 2:
        return per_sample_loss.sum() * 0.0
    penalties: list[torch.Tensor] = []
    for task in torch.unique(task_idx):
        task_mask = task_idx == task
        if task_mask.sum() < 2:
            continue
        penalty = vrex_penalty(per_sample_loss[task_mask], cluster_ids[task_mask])
        if torch.isfinite(penalty):
            penalties.append(penalty)
    if not penalties:
        return per_sample_loss.sum() * 0.0
    return torch.stack(penalties).mean()


def binding_aux_loss(predicted: torch.Tensor | None, target: torch.Tensor | None) -> torch.Tensor:
    """SmoothL1 substrate-descriptor loss for the binding state."""

    if predicted is None or target is None:
        reference = predicted if predicted is not None else target
        if reference is None:
            return torch.tensor(0.0)
        return reference.sum() * 0.0
    return F.smooth_l1_loss(predicted.float(), target.float())


def ec_aux_loss(
    logits: torch.Tensor | None,
    target: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Cross-entropy auxiliary loss for coarse EC reaction class.

    EC is used as a train-time catalytic target, not as an input feature.  That
    keeps the kinetic prediction path usable without EC at inference while
    encouraging ``z_cat`` to encode transferable reaction chemistry.
    """

    if logits is None or target is None or mask is None:
        reference = logits if logits is not None else target
        if reference is None:
            return torch.tensor(0.0)
        return reference.sum() * 0.0
    valid = mask.bool()
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid].float(), target[valid].long())


def kcat_tail_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor | None,
    task_idx: torch.Tensor | None,
    *,
    center: float,
    strength: float,
) -> torch.Tensor:
    """Tail-aware kcat MSE that counteracts the measured OOD range compression.

    Targets are standardized independently per kinetic quantity.  For kcat rows
    only, samples farther than ``center`` standard deviations from the training
    mean receive smoothly higher weight.  This preserves the base likelihood for
    all rows while making slow and fast enzymes less cheap to collapse inward.
    """

    if task_idx is None:
        return prediction.sum() * 0.0
    valid = task_idx == 0
    if target_mask is not None:
        valid = valid & target_mask.bool()
    if not valid.any():
        return prediction.sum() * 0.0
    error = (prediction[valid].float() - target[valid].float()).pow(2)
    tail_distance = (target[valid].float().abs() - float(center)).clamp_min(0.0)
    row_weight = 1.0 + float(strength) * tail_distance.clamp_max(3.0)
    return (row_weight * error).sum() / row_weight.sum().clamp_min(1.0)


def kcat_mechanism_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor | None,
    task_idx: torch.Tensor | None,
    ec_target: torch.Tensor | None,
    ec_mask: torch.Tensor | None,
    *,
    temperature: float,
    min_gap: float,
) -> torch.Tensor:
    """Pairwise kcat ranking inside coarse EC mechanism groups.

    A global pairwise loss would compare unrelated chemistry.  EC level 1 is
    coarse but available for nearly every row, so pairs are formed only within a
    predicted biochemical class.  This directly targets the OOD kcat ranking
    gap while leaving absolute-scale MSE/NLL as the dominant objective.
    """

    if task_idx is None or ec_target is None or ec_mask is None:
        return prediction.sum() * 0.0
    valid = (task_idx == 0) & ec_mask.bool()
    if target_mask is not None:
        valid = valid & target_mask.bool()
    if valid.sum() < 2:
        return prediction.sum() * 0.0

    pred = prediction[valid].float()
    tgt = target[valid].float()
    ec = ec_target[valid].long()
    losses: list[torch.Tensor] = []
    pair_temperature = max(float(temperature), 1.0e-4)
    for class_id in torch.unique(ec):
        class_mask = ec == class_id
        if class_mask.sum() < 2:
            continue
        class_pred = pred[class_mask]
        class_tgt = tgt[class_mask]
        target_delta = class_tgt.unsqueeze(1) - class_tgt.unsqueeze(0)
        prediction_delta = class_pred.unsqueeze(1) - class_pred.unsqueeze(0)
        upper = torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
        informative = upper & (target_delta.abs() >= float(min_gap))
        if not informative.any():
            continue
        direction = target_delta[informative].sign()
        margin = direction * prediction_delta[informative]
        # Larger label separations are more reliable ordering constraints, but
        # cap their influence so a handful of extreme measurements do not win.
        separation = target_delta[informative].abs().clamp(max=2.0)
        losses.append((F.softplus(-margin / pair_temperature) * separation).sum() / separation.sum().clamp_min(1.0))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def kcat_reaction_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor | None,
    task_idx: torch.Tensor | None,
    reaction_group_id: torch.Tensor | None,
    *,
    temperature: float,
    min_gap: float,
) -> torch.Tensor:
    """Rank kcat values only among rows describing the exact same reaction.

    Unlike EC-level ranking, an exact reaction cohort holds reaction chemistry
    fixed.  Its remaining label variation is therefore a much cleaner signal
    for enzyme sequence, assay condition, and catalytic-efficiency differences.
    """

    if task_idx is None or reaction_group_id is None:
        return prediction.sum() * 0.0
    valid = (task_idx == 0) & (reaction_group_id >= 0)
    if target_mask is not None:
        valid = valid & target_mask.bool()
    if valid.sum() < 2:
        return prediction.sum() * 0.0

    pred = prediction[valid].float()
    tgt = target[valid].float()
    groups = reaction_group_id[valid].long()
    losses: list[torch.Tensor] = []
    pair_temperature = max(float(temperature), 1.0e-4)
    for group_id in torch.unique(groups):
        group_mask = groups == group_id
        if group_mask.sum() < 2:
            continue
        group_pred = pred[group_mask]
        group_tgt = tgt[group_mask]
        target_delta = group_tgt.unsqueeze(1) - group_tgt.unsqueeze(0)
        prediction_delta = group_pred.unsqueeze(1) - group_pred.unsqueeze(0)
        upper = torch.triu(torch.ones_like(target_delta, dtype=torch.bool), diagonal=1)
        informative = upper & (target_delta.abs() >= float(min_gap))
        if not informative.any():
            continue
        direction = target_delta[informative].sign()
        margin = direction * prediction_delta[informative]
        separation = target_delta[informative].abs().clamp(max=2.0)
        losses.append((F.softplus(-margin / pair_temperature) * separation).sum() / separation.sum().clamp_min(1.0))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def km_p1_boundary_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_scale: torch.Tensor | None,
    target_mask: torch.Tensor | None,
    task_idx: torch.Tensor | None,
    *,
    temperature: float,
) -> torch.Tensor:
    """Smoothly penalize Km errors that cross the one-log-unit p1 boundary."""

    if task_idx is None or target_scale is None:
        return prediction.sum() * 0.0
    valid = task_idx == 1
    if target_mask is not None:
        valid = valid & target_mask.bool()
    if not valid.any():
        return prediction.sum() * 0.0
    raw_error = (prediction[valid].float() - target[valid].float()).abs() * target_scale[valid].float()
    boundary_temperature = max(float(temperature), 1.0e-4)
    return (boundary_temperature * F.softplus((raw_error - 1.0) / boundary_temperature)).mean()


def jepa_prediction_loss(
    predicted: torch.Tensor | None,
    target: torch.Tensor | None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Huber loss between predicted and target reaction-state embeddings.

    The target is an embedding, not a decoded product molecule.  This is the
    core JEPA principle: predict useful latent structure instead of raw details.
    """

    if predicted is None or target is None:
        # Return a graph-friendly zero if the caller passed no JEPA target.
        reference = predicted if predicted is not None else target
        if reference is None:
            return torch.tensor(0.0)
        return reference.sum() * 0.0

    per_dim = F.smooth_l1_loss(predicted.float(), target.float(), reduction="none")
    per_row = per_dim.mean(dim=-1)
    if mask is None:
        return per_row.mean()
    if not mask.any():
        return per_row.sum() * 0.0
    return per_row[mask].mean()


def jepa_contrastive_loss(
    predicted: torch.Tensor | None,
    target: torch.Tensor | None,
    mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """In-batch product matching loss for latent JEPA states.

    This is not CatPred code.  It addresses a failure mode of whole-product
    regression: if product and reactant embeddings are nearly identical, a
    plain Huber loss can reward a near-zero reaction delta.  In-batch matching
    still asks the predicted reaction state to identify its own product target
    among other product targets.
    """

    if predicted is None or target is None:
        reference = predicted if predicted is not None else target
        if reference is None:
            return torch.tensor(0.0)
        return reference.sum() * 0.0

    pred = predicted.float()
    tgt = target.float()
    if mask is not None:
        if not mask.any():
            return pred.sum() * 0.0
        pred = pred[mask]
        tgt = tgt[mask]
    if pred.shape[0] < 2:
        return pred.sum() * 0.0

    pred = F.normalize(pred, dim=-1)
    tgt = F.normalize(tgt, dim=-1)
    logits = pred @ tgt.transpose(0, 1) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)

    # Symmetric matching gives gradients both to the prediction geometry and,
    # when stop_jepa_target=False in ablations, to the target geometry.
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def action_decorrelation_loss(z_bind_raw: torch.Tensor | None, z_cat_raw: torch.Tensor | None) -> torch.Tensor:
    """Discourage the binding and catalytic bottlenecks from becoming copies."""

    if z_bind_raw is None or z_cat_raw is None:
        reference = z_bind_raw if z_bind_raw is not None else z_cat_raw
        if reference is None:
            return torch.tensor(0.0)
        return reference.sum() * 0.0
    cosine = F.cosine_similarity(z_bind_raw.float(), z_cat_raw.float(), dim=-1)
    return cosine.pow(2).mean()


def _masked_rows(latent: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor | None:
    """Return only rows with a true JEPA target, preserving graph-friendly zeros."""

    if latent is None or mask is None:
        return latent
    if not mask.any():
        return None
    return latent[mask]


def _flatten_latents(latents: Iterable[torch.Tensor]) -> torch.Tensor | None:
    """Collect 2-D or 3-D latent tensors into one (N, D) matrix."""

    rows: list[torch.Tensor] = []
    for latent in latents:
        if latent is None:
            continue
        if latent.dim() == 2:
            rows.append(latent)
        elif latent.dim() == 3:
            rows.append(latent.reshape(-1, latent.shape[-1]))
        else:
            raise ValueError(f"expected latent tensor with rank 2 or 3, got shape {tuple(latent.shape)}")
    if not rows:
        return None
    return torch.cat(rows, dim=0)


def sigreg_gaussian_loss(
    latents: Iterable[torch.Tensor],
    num_slices: int = 64,
    num_frequencies: int = 17,
    max_frequency: float = 4.0,
) -> torch.Tensor:
    """A differentiable Gaussian signal regularizer for JEPA anti-collapse.

    Why this exists:
        End-to-end JEPA/world-model training can collapse if all embeddings
        become constant.  The LeWorldModel family uses SIGReg-style signal
        regularization to keep latent states informative while still allowing
        next-embedding prediction without an EMA teacher.

    What this implementation does:
        This is an implementation-level, SIGReg-style regularizer, not copied
        from any released code.  It projects embeddings onto random directions
        and compares each one-dimensional empirical characteristic function to
        the characteristic function of N(0, 1):

            phi_normal(t) = exp(-0.5 * t^2)

        A constant embedding has a very different characteristic function, so
        the loss penalizes collapse.  The operation is cheap relative to ESM and
        D-MPNN and is used only during training.
    """

    z = _flatten_latents(latents)
    if z is None:
        return torch.tensor(0.0)

    z = z.float()
    n, d = z.shape
    if n < 2:
        return z.sum() * 0.0

    # Random unit directions.  The randomness is intentionally local to the
    # training step; it estimates many one-dimensional projections over time.
    directions = torch.randn(d, int(num_slices), device=z.device, dtype=z.dtype)
    directions = F.normalize(directions, dim=0)
    projected = z @ directions

    # Frequencies for the empirical characteristic function.  Avoid t=0 because
    # every distribution has phi(0)=1, which carries no regularization signal.
    t = torch.linspace(
        0.2,
        float(max_frequency),
        int(num_frequencies),
        device=z.device,
        dtype=z.dtype,
    )
    phase = projected.unsqueeze(-1) * t.view(1, 1, -1)

    # Empirical characteristic function over batch rows for each slice.
    real = torch.cos(phase).mean(dim=0)
    imag = torch.sin(phase).mean(dim=0)

    # Target characteristic function for a standard Gaussian.
    target_real = torch.exp(-0.5 * t.pow(2)).view(1, -1)

    # Squared complex distance: |(real + i imag) - target|^2.
    sq_dist = (real - target_real).pow(2) + imag.pow(2)

    # Trapezoid integration over frequency, averaged over random slices.
    if sq_dist.shape[-1] > 1:
        dt = t[1:] - t[:-1]
        integral = (0.5 * (sq_dist[:, :-1] + sq_dist[:, 1:]) * dt.view(1, -1)).sum(dim=-1)
    else:
        integral = sq_dist.squeeze(-1)
    return integral.mean()


def lance_total_loss(
    outputs: dict[str, torch.Tensor],
    target_z: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    weights: LANCELossWeights | None = None,
    jepa_mask: torch.Tensor | None = None,
    cluster_id: torch.Tensor | None = None,
    task_idx: torch.Tensor | None = None,
    substrate_desc_target: torch.Tensor | None = None,
    ec_target: torch.Tensor | None = None,
    ec_mask: torch.Tensor | None = None,
    reaction_group_id: torch.Tensor | None = None,
    target_scale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the complete LANCE objective.

    Expected output keys from LANCEModel:
        mu
        reaction_state_pred
        optional reaction_state_target
        molecule_state, z_bind, z_cat, optional product_state
    """

    weights = weights or LANCELossWeights()

    if "log_var" in outputs:
        kinetic, per_sample_kinetic, valid_mask = masked_gaussian_nll(
            outputs["mu"],
            outputs["log_var"],
            target_z,
            target_mask,
            mode=weights.gaussian_loss,
            beta=weights.beta_nll_beta,
            decoupled_sigma=weights.decoupled_sigma,
        )
    else:
        kinetic = masked_mse(outputs["mu"], target_z, target_mask)
        per_row = (outputs["mu"].float() - target_z.float()).pow(2)
        valid_mask = target_mask.bool() if target_mask is not None else torch.ones_like(per_row, dtype=torch.bool)
        per_sample_kinetic = per_row[valid_mask] if valid_mask.any() else per_row[:0]
    valid_cluster_id = cluster_id[valid_mask] if cluster_id is not None and valid_mask.any() else None
    valid_task_idx = task_idx[valid_mask] if task_idx is not None and valid_mask.any() else None
    if weights.vrex_per_task:
        vrex = per_task_vrex_penalty(per_sample_kinetic, valid_cluster_id, valid_task_idx)
    else:
        vrex = vrex_penalty(per_sample_kinetic, valid_cluster_id)
    # JEPA targets are not guaranteed for every CatPred-DB row.  kcat rows may
    # have a product/reaction-state column in future prepared data, while Km/Ki
    # binding rows usually do not.  The dedicated mask keeps kinetic supervision
    # independent from target availability for the latent world-model branch.
    jepa = jepa_prediction_loss(
        outputs.get("reaction_state_pred"),
        outputs.get("reaction_state_target"),
        jepa_mask,
    )
    jepa_delta = jepa_prediction_loss(
        outputs.get("reaction_delta"),
        outputs.get("reaction_delta_target"),
        jepa_mask,
    )
    jepa_contrastive = jepa_contrastive_loss(
        outputs.get("reaction_state_pred"),
        outputs.get("reaction_state_target"),
        jepa_mask,
    )
    action_decorrelation = action_decorrelation_loss(outputs.get("z_bind_raw"), outputs.get("z_cat_raw"))
    binding_aux = binding_aux_loss(outputs.get("substrate_desc_pred"), substrate_desc_target)
    ec_aux = ec_aux_loss(outputs.get("ec_logits"), ec_target, ec_mask)
    # Product-free binding supervision is defined for every labeled kinetic row;
    # it therefore lets Km/Ki examples improve the same shared action space that
    # kcat uses, instead of disappearing whenever a product graph is absent.
    binding_jepa_mask = (
        target_mask.bool()
        if target_mask is not None
        else torch.ones_like(outputs["mu"], dtype=torch.bool)
    )
    if weights.binding_jepa_exclude_kcat and task_idx is not None:
        binding_jepa_mask = binding_jepa_mask & (task_idx != 0)
    binding_jepa = jepa_prediction_loss(
        outputs.get("binding_state_pred"),
        outputs.get("binding_state_target"),
        binding_jepa_mask,
    )
    binding_jepa_delta = jepa_prediction_loss(
        outputs.get("binding_delta"),
        outputs.get("binding_delta_target"),
        binding_jepa_mask,
    )
    binding_jepa_contrastive = jepa_contrastive_loss(
        outputs.get("binding_state_pred"),
        outputs.get("binding_state_target"),
        binding_jepa_mask,
    )
    kcat_tail = kcat_tail_loss(
        outputs["mu"],
        target_z,
        target_mask,
        task_idx,
        center=weights.kcat_tail_center,
        strength=weights.kcat_tail_strength,
    )
    kcat_mechanism_ranking = kcat_mechanism_ranking_loss(
        outputs["mu"],
        target_z,
        target_mask,
        task_idx,
        ec_target,
        ec_mask,
        temperature=weights.kcat_ranking_temperature,
        min_gap=weights.kcat_ranking_min_gap,
    )
    kcat_reaction_ranking = kcat_reaction_ranking_loss(
        outputs["mu"],
        target_z,
        target_mask,
        task_idx,
        reaction_group_id,
        temperature=weights.kcat_ranking_temperature,
        min_gap=weights.kcat_ranking_min_gap,
    )
    km_p1 = km_p1_boundary_loss(
        outputs["mu"],
        target_z,
        target_scale,
        target_mask,
        task_idx,
        temperature=weights.km_p1_temperature,
    )

    # Regularize projection-head copies of the latent state family most exposed
    # to collapse.  This follows the SSL convention where the regularizer sees
    # a projected view, reducing direct pressure to make the exact regression
    # features isotropic.
    sigreg = sigreg_gaussian_loss(
        [
            outputs.get("sigreg_molecule_state"),
            outputs.get("sigreg_binding_state"),
            outputs.get("sigreg_catalytic_state"),
            _masked_rows(outputs.get("sigreg_reaction_delta"), jepa_mask),
            outputs.get("sigreg_reaction_state_pred"),
            _masked_rows(outputs.get("sigreg_product_state"), jepa_mask),
            _masked_rows(outputs.get("sigreg_reaction_state_target"), jepa_mask),
            _masked_rows(outputs.get("sigreg_binding_state_pred"), binding_jepa_mask),
            _masked_rows(outputs.get("sigreg_binding_state_target"), binding_jepa_mask),
            _masked_rows(outputs.get("sigreg_binding_delta"), binding_jepa_mask),
            _masked_rows(outputs.get("sigreg_binding_delta_target"), binding_jepa_mask),
        ]
    )

    total = (
        weights.kinetic * kinetic
        + weights.jepa * jepa
        + weights.jepa_delta * jepa_delta
        + weights.jepa_contrastive * jepa_contrastive
        + weights.action_decorrelation * action_decorrelation
        + weights.sigreg * sigreg
        + weights.vrex * vrex
        + weights.binding_aux * binding_aux
        + weights.binding_jepa * binding_jepa
        + weights.binding_jepa_delta * binding_jepa_delta
        + weights.binding_jepa_contrastive * binding_jepa_contrastive
        + weights.kcat_tail * kcat_tail
        + weights.kcat_mechanism_ranking * kcat_mechanism_ranking
        + weights.kcat_reaction_ranking * kcat_reaction_ranking
        + weights.km_p1 * km_p1
        + weights.ec_aux * ec_aux
    )
    return {
        "loss": total,
        "kinetic": kinetic.detach(),
        "jepa": jepa.detach(),
        "jepa_delta": jepa_delta.detach(),
        "jepa_contrastive": jepa_contrastive.detach(),
        "action_decorrelation": action_decorrelation.detach(),
        "sigreg": sigreg.detach(),
        "vrex": vrex.detach(),
        "binding_aux": binding_aux.detach(),
        "binding_jepa": binding_jepa.detach(),
        "binding_jepa_delta": binding_jepa_delta.detach(),
        "binding_jepa_contrastive": binding_jepa_contrastive.detach(),
        "kcat_tail": kcat_tail.detach(),
        "kcat_mechanism_ranking": kcat_mechanism_ranking.detach(),
        "kcat_reaction_ranking": kcat_reaction_ranking.detach(),
        "km_p1": km_p1.detach(),
        "ec_aux": ec_aux.detach(),
    }
