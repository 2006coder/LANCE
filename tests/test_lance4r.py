#!/usr/bin/env python3
"""CPU smoke tests for LANCE.

These tests intentionally avoid loading ESM.  They feed fake protein residue
features into the model and use the existing D-MPNN path for molecule graphs.
That validates the new catalytic-action/JEPA logic without needing a GPU or a
Hugging Face model download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset

# Running this file directly sets sys.path[0] to LANCE/.  Add the repo
# root so package imports like LANCE.dmpnn resolve from the workspace.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LANCE.dmpnn import mol2graph
from LANCE.data import KineticsDataset, TargetScaler

from LANCE import LANCEConfig, LANCEModel, LANCELossWeights, lance_total_loss
from LANCE.losses import (
    beta_gaussian_nll_per_sample,
    gaussian_nll_per_sample,
    kcat_reaction_ranking_loss,
    km_p1_boundary_loss,
    per_task_vrex_penalty,
    vrex_penalty,
)
from LANCE.train import TaskReactionCohortBatchSampler


def _fake_batch(with_products: bool = True) -> dict[str, object]:
    """Build a tiny mixed-task batch with valid RDKit SMILES."""

    batch_size = 4
    protein_len = 9
    protein_dim = 16

    protein = torch.randn(batch_size, protein_len, protein_dim)
    protein_mask = torch.ones(batch_size, protein_len, dtype=torch.bool)
    protein_mask[1, -2:] = False
    protein_mask[3, -1:] = False

    batch: dict[str, object] = {
        "protein": protein,
        "protein_mask": protein_mask,
        "bmg": mol2graph(["CCO", "CC(=O)O", "CCN", "c1ccccc1O"]),
        "task_idx": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        "cluster_id": torch.tensor([3, 3, 7, 11], dtype=torch.long),
        "reaction_group_id": torch.tensor([101, -1, -1, 101], dtype=torch.long),
        "substrate_desc": torch.randn(batch_size, 8),
        "condition": torch.tensor(
            [
                [0.37, 0.50, 1.0, 1.0],
                [0.25, 0.48, 1.0, 1.0],
                [0.00, 0.00, 0.0, 0.0],
                [0.55, 0.57, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "ec_level1": torch.tensor([1, 3, 4, 2], dtype=torch.long),
        "ec_mask": torch.tensor([True, True, True, True], dtype=torch.bool),
        "target_z": torch.tensor([0.2, -0.7, 1.1, 0.4], dtype=torch.float32),
        "target_scale": torch.tensor([1.5, 1.2, 2.0, 1.5], dtype=torch.float32),
        "target_mask": torch.tensor([True, True, True, True], dtype=torch.bool),
    }
    if with_products:
        # Products can be disconnected molecules in one SMILES string.  RDKit
        # and the CatPred-derived D-MPNN wrapper treat the disconnected graph as
        # one molecule-level state, which is enough for this JEPA prototype.
        batch["product_bmg"] = mol2graph(["CC=O", "CCO", "CCN", "c1ccccc1"])
    return batch


def test_forward_and_loss_with_product_targets() -> None:
    """The model should produce kinetics plus a JEPA target when products exist."""

    torch.manual_seed(7)
    cfg = LANCEConfig(
        protein_dim=16,
        dmpnn_hidden=32,
        dmpnn_depth=2,
        model_dim=48,
        action_dim=24,
        hidden_dim=96,
        dropout=0.0,
    )
    model = LANCEModel(cfg)
    batch = _fake_batch(with_products=True)

    outputs = model(batch)
    assert outputs["mu"].shape == (4,)
    assert outputs["z_bind"].shape == (4, 48)
    assert outputs["z_cat"].shape == (4, 48)
    assert outputs["z_bind_raw"].shape == (4, 24)
    assert outputs["z_cat_raw"].shape == (4, 24)
    assert outputs["interface_context"].shape == (4, 48)
    assert outputs["interface_attention"].shape == (4, 9)
    assert outputs["reaction_delta"].shape == (4, 48)
    assert outputs["reaction_state_pred"].shape == (4, 48)
    assert outputs["reaction_state_target"].shape == (4, 48)
    assert outputs["reaction_delta_target"].shape == (4, 48)
    assert 0.0 < float(outputs["cat_action_scale"].mean()) < 1.0

    loss_parts = lance_total_loss(
        outputs,
        batch["target_z"],
        batch["target_mask"],
        LANCELossWeights(kinetic=1.0, jepa=0.2, sigreg=0.01),
    )
    assert torch.isfinite(loss_parts["loss"])

    loss_parts["loss"].backward()
    grad_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_norm += float(parameter.grad.detach().abs().sum().item())
    assert grad_norm > 0.0

    # The first cluster launch exposed an autocast-only dtype issue in the
    # task-head indexed assignment.  CPU bf16 autocast catches the same class of
    # bug without requiring a GPU in the local smoke test.
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_outputs = model(batch)
        amp_loss_parts = lance_total_loss(
            amp_outputs,
            batch["target_z"],
            batch["target_mask"],
            LANCELossWeights(kinetic=1.0, jepa=0.2, jepa_delta=0.1, jepa_contrastive=0.05, sigreg=0.01),
        )
    assert torch.isfinite(amp_loss_parts["loss"])


def test_forward_without_product_targets() -> None:
    """Kinetic prediction should still work on ordinary CatPred rows."""

    torch.manual_seed(11)
    cfg = LANCEConfig(
        protein_dim=16,
        dmpnn_hidden=24,
        dmpnn_depth=2,
        model_dim=40,
        action_dim=20,
        hidden_dim=80,
        dropout=0.0,
        use_kcat_global_rate_context=True,
    )
    model = LANCEModel(cfg)
    batch = _fake_batch(with_products=False)

    outputs = model(batch)
    assert outputs["mu"].shape == (4,)
    assert outputs["reaction_delta"].shape == (4, 40)
    assert outputs["global_rate_context"].shape == (4, 40)
    assert "reaction_state_target" not in outputs

    loss_parts = lance_total_loss(outputs, batch["target_z"], batch["target_mask"])
    assert torch.isfinite(loss_parts["loss"])


def test_lance2_soft_focus_cluster_loss_path() -> None:
    """New LANCE2 switches should train without hard top-k or CatPred imports."""

    torch.manual_seed(13)
    cfg = LANCEConfig(
        protein_dim=16,
        dmpnn_hidden=24,
        dmpnn_depth=2,
        model_dim=40,
        action_dim=20,
        hidden_dim=80,
        dropout=0.0,
        use_soft_catalytic_focus=True,
        focus_heads=3,
        use_task_film=True,
        use_task_interface_gate=True,
        use_task_specific_actions=True,
        use_gaussian_heads=True,
    )
    model = LANCEModel(cfg)
    batch = _fake_batch(with_products=True)

    outputs = model(batch)
    assert outputs["mu"].shape == (4,)
    assert outputs["log_var"].shape == (4,)
    assert outputs["substrate_desc_pred"].shape == (4, 8)
    assert outputs["focus_attention"].shape == (4, 3, 9)
    assert outputs["focus_attention_mean"].shape == (4, 9)
    assert outputs["interface_gate"].shape == (4,)

    loss_parts = lance_total_loss(
        outputs,
        batch["target_z"],
        batch["target_mask"],
        LANCELossWeights(
            kinetic=1.0,
            jepa=0.2,
            jepa_delta=0.1,
            jepa_contrastive=0.05,
            sigreg=0.01,
            vrex=0.5,
            binding_aux=0.1,
        ),
        jepa_mask=torch.tensor([True, True, True, True], dtype=torch.bool),
        cluster_id=batch["cluster_id"],
        task_idx=batch["task_idx"],
        substrate_desc_target=batch["substrate_desc"],
    )
    assert torch.isfinite(loss_parts["loss"])
    assert "vrex" in loss_parts and "binding_aux" in loss_parts
    loss_parts["loss"].backward()
    assert model.soft_focus is not None
    assert model.soft_focus.query_from_molecule.weight.grad is not None
    assert model.interface_encoder.task_gate_logit is not None
    assert model.interface_encoder.task_gate_logit.grad is not None


def test_beta_nll_restores_noisy_tail_mean_gradient() -> None:
    """Beta-NLL should starve the mean less than naive NLL on high-sigma rows."""

    mu_clean = torch.tensor([0.0], requires_grad=True)
    log_var_clean = torch.tensor([0.0])
    target_clean = torch.tensor([1.0])
    clean_loss = gaussian_nll_per_sample(mu_clean, log_var_clean, target_clean).mean()
    clean_loss.backward()
    clean_grad = abs(float(mu_clean.grad.item()))

    mu_noisy = torch.tensor([0.0], requires_grad=True)
    log_var_noisy = torch.tensor([4.0])
    target_noisy = torch.tensor([1.0])
    naive_loss = gaussian_nll_per_sample(mu_noisy, log_var_noisy, target_noisy).mean()
    naive_loss.backward()
    naive_ratio = abs(float(mu_noisy.grad.item())) / clean_grad

    mu_beta = torch.tensor([0.0], requires_grad=True)
    beta_loss = beta_gaussian_nll_per_sample(mu_beta, log_var_noisy, target_noisy, beta=0.5).mean()
    beta_loss.backward()
    beta_ratio = abs(float(mu_beta.grad.item())) / clean_grad

    assert beta_ratio > naive_ratio * 5.0
    assert beta_ratio < 1.0


def test_per_task_vrex_ignores_task_mix_but_detects_cluster_shift() -> None:
    """Mixed-task V-REx should not fire when clusters only differ by task mix."""

    # Cluster 0 has more kcat-like rows, cluster 1 has more Km-like rows.  Within
    # each task, risk is constant across clusters, so per-task V-REx should be 0.
    risk = torch.tensor([1.0, 1.0, 4.0, 4.0])
    cluster = torch.tensor([0, 1, 0, 1])
    task = torch.tensor([0, 0, 1, 1])
    assert float(per_task_vrex_penalty(risk, cluster, task)) == 0.0

    mixed_penalty = float(vrex_penalty(torch.tensor([1.0, 1.0, 4.0, 4.0]), torch.tensor([0, 0, 1, 1])))
    assert mixed_penalty > 0.0

    shifted_risk = torch.tensor([1.0, 3.0, 4.0, 8.0])
    shifted_penalty = float(per_task_vrex_penalty(shifted_risk, cluster, task))
    assert shifted_penalty > 0.0


def test_condition_and_ec_aux_path() -> None:
    """Condition-aware heads and EC auxiliary supervision should be trainable."""

    torch.manual_seed(17)
    cfg = LANCEConfig(
        protein_dim=16,
        dmpnn_hidden=24,
        dmpnn_depth=2,
        model_dim=40,
        action_dim=20,
        hidden_dim=80,
        dropout=0.0,
        use_condition_features=True,
        use_ec_aux=True,
        use_gaussian_heads=True,
    )
    model = LANCEModel(cfg)
    batch = _fake_batch(with_products=True)
    outputs = model(batch)
    assert outputs["mu"].shape == (4,)
    assert outputs["log_var"].shape == (4,)
    assert outputs["condition_context"].shape == (4, 40)
    assert outputs["ec_logits"].shape == (4, 8)

    loss_parts = lance_total_loss(
        outputs,
        batch["target_z"],
        batch["target_mask"],
        LANCELossWeights(
            kinetic=1.0,
            jepa=0.2,
            binding_aux=0.1,
            ec_aux=0.2,
            gaussian_loss="beta_nll",
        ),
        jepa_mask=torch.ones(4, dtype=torch.bool),
        substrate_desc_target=batch["substrate_desc"],
        ec_target=batch["ec_level1"],
        ec_mask=batch["ec_mask"],
    )
    assert torch.isfinite(loss_parts["loss"])
    assert "ec_aux" in loss_parts
    loss_parts["loss"].backward()


def test_single_model_three_state_path() -> None:
    """One checkpoint should support task-aware JEPA paths for all three targets."""

    torch.manual_seed(23)
    cfg = LANCEConfig(
        protein_dim=16,
        dmpnn_hidden=24,
        dmpnn_depth=2,
        model_dim=40,
        action_dim=20,
        hidden_dim=80,
        dropout=0.0,
        use_task_interface_gate=True,
        use_task_residual_actions=True,
        task_action_adapter_dim=32,
        use_gaussian_heads=True,
        use_condition_features=True,
        use_ec_aux=True,
        use_binding_jepa=True,
        binding_jepa_exclude_kcat=True,
        use_mechanistic_kcat_prior=True,
        use_kcat_global_rate_context=True,
    )
    model = LANCEModel(cfg)
    batch = _fake_batch(with_products=True)
    outputs = model(batch)
    assert outputs["mu"].shape == (4,)
    assert outputs["binding_state_pred"].shape == (4, 40)
    assert outputs["binding_state_target"].shape == (4, 40)
    assert outputs["binding_delta"].shape == (4, 40)
    assert outputs["mechanistic_kcat_delta"].shape == (4,)
    assert outputs["ec_rate_prior"].shape == (4,)
    assert outputs["global_rate_context"].shape == (4, 40)
    assert outputs["global_rate_attention"].shape == (4, 9)

    loss_parts = lance_total_loss(
        outputs,
        batch["target_z"],
        batch["target_mask"],
        LANCELossWeights(
            kinetic=1.0,
            jepa=0.2,
            jepa_delta=0.1,
            jepa_contrastive=0.05,
            binding_aux=0.1,
            binding_jepa=0.2,
            binding_jepa_delta=0.1,
            binding_jepa_contrastive=0.05,
            binding_jepa_exclude_kcat=True,
            kcat_tail=0.15,
            kcat_mechanism_ranking=0.02,
            kcat_reaction_ranking=0.03,
            km_p1=0.03,
            ec_aux=0.1,
        ),
        jepa_mask=torch.ones(4, dtype=torch.bool),
        task_idx=batch["task_idx"],
        substrate_desc_target=batch["substrate_desc"],
        ec_target=batch["ec_level1"],
        ec_mask=batch["ec_mask"],
        reaction_group_id=batch["reaction_group_id"],
        target_scale=batch["target_scale"],
    )
    assert torch.isfinite(loss_parts["loss"])
    assert "binding_jepa" in loss_parts and "kcat_reaction_ranking" in loss_parts
    loss_parts["loss"].backward()
    assert model.binding_jepa is not None
    assert model.action_encoder.adapter[-1].weight.grad is not None
    assert model.protein_encoder.raw_rate_project[1].weight.grad is not None

    # The H100 recipe trains under bf16 autocast.  Exercise the new residual,
    # binding-JEPA, and rate-prior dispatch under the same dtype pressure.
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp_outputs = model(batch)
        amp_loss_parts = lance_total_loss(
            amp_outputs,
            batch["target_z"],
            batch["target_mask"],
            LANCELossWeights(
                binding_jepa=0.2,
                binding_jepa_exclude_kcat=True,
                kcat_tail=0.1,
                kcat_mechanism_ranking=0.02,
                kcat_reaction_ranking=0.03,
                km_p1=0.03,
                ec_aux=0.1,
                gaussian_loss="beta_nll",
            ),
            task_idx=batch["task_idx"],
            ec_target=batch["ec_level1"],
            ec_mask=batch["ec_mask"],
            reaction_group_id=batch["reaction_group_id"],
            target_scale=batch["target_scale"],
        )
    assert torch.isfinite(amp_loss_parts["loss"])


def test_exact_reaction_ranking_and_km_boundary_losses() -> None:
    """New targeted losses should reward ordering and one-log-unit precision."""

    target = torch.tensor([-1.0, 0.0, 1.0, 0.2])
    task = torch.tensor([0, 0, 0, 1])
    groups = torch.tensor([17, 17, 17, -1])
    mask = torch.ones(4, dtype=torch.bool)
    correct = kcat_reaction_ranking_loss(
        target,
        target,
        mask,
        task,
        groups,
        temperature=0.2,
        min_gap=0.25,
    )
    reversed_order = kcat_reaction_ranking_loss(
        -target,
        target,
        mask,
        task,
        groups,
        temperature=0.2,
        min_gap=0.25,
    )
    assert correct < reversed_order

    km_task = torch.tensor([1, 1, 0])
    km_target = torch.zeros(3)
    scales = torch.tensor([2.0, 2.0, 1.0])
    inside = km_p1_boundary_loss(
        torch.tensor([0.20, 0.30, 9.0]),
        km_target,
        scales,
        torch.ones(3, dtype=torch.bool),
        km_task,
        temperature=0.15,
    )
    crossing = km_p1_boundary_loss(
        torch.tensor([0.20, 0.65, 9.0]),
        km_target,
        scales,
        torch.ones(3, dtype=torch.bool),
        km_task,
        temperature=0.15,
    )
    assert inside < crossing


def test_reaction_cohort_sampler_materializes_pairs() -> None:
    """A cohort batch must contain an exact-reaction kcat pair by construction."""

    def fake_dataset(parameter: str, task_idx: int, frame: pd.DataFrame) -> KineticsDataset:
        dataset = KineticsDataset.__new__(KineticsDataset)
        dataset.parameter = parameter
        dataset.task_idx = task_idx
        dataset.frame = frame.reset_index(drop=True)
        dataset.spec = {"target": "target"}
        dataset.scaler = TargetScaler.from_values(frame["target"].tolist())
        return dataset

    kcat_frame = pd.DataFrame(
        {
            "reaction_smiles": ["A>>B", "A>>B", "C>>D", "C>>D", "E>>F", "E>>F"],
            "target": [-2.0, 1.0, -1.0, 2.0, -0.5, 1.5],
        }
    )
    binding_frame = pd.DataFrame({"target": list(range(6))})
    dataset = ConcatDataset(
        [
            fake_dataset("kcat", 0, kcat_frame),
            fake_dataset("km", 1, binding_frame),
            fake_dataset("ki", 2, binding_frame),
        ]
    )
    sampler = TaskReactionCohortBatchSampler(
        dataset,
        batch_size=12,
        seed=5,
        cohort_fraction=0.5,
        min_gap=0.25,
    )
    batch = next(iter(sampler))
    kcat_indices = [index for index in batch if index < len(kcat_frame)]
    reactions = [kcat_frame.iloc[index]["reaction_smiles"] for index in kcat_indices]
    assert len(kcat_indices) == 4
    assert len(set(reactions)) < len(reactions)


if __name__ == "__main__":
    test_forward_and_loss_with_product_targets()
    test_forward_without_product_targets()
    test_lance2_soft_focus_cluster_loss_path()
    test_beta_nll_restores_noisy_tail_mean_gradient()
    test_per_task_vrex_ignores_task_mix_but_detects_cluster_shift()
    test_condition_and_ec_aux_path()
    test_single_model_three_state_path()
    test_exact_reaction_ranking_and_km_boundary_losses()
    test_reaction_cohort_sampler_materializes_pairs()
    print("LANCE tests passed")
