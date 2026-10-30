#!/usr/bin/env python3
"""Train LANCE on CatPred-DB splits.

This trainer is intentionally compact and cluster-facing.  One Slurm allocation
can launch multiple independent Python processes, each with its own seed, run
directory, optimizer, and checkpoint files.

CatPred provenance:
    - ``NoamLR`` is copied from ``catpred/nn_utils.py::NoamLR`` through the
      existing CatPred training pattern in this repository, with comments kept
      explicit so the scheduler behavior is auditable.
    - Target scaling, split loading, invalid-SMILES filtering, and D-MPNN graph
      construction live in ``LANCE/data.py`` and are commented there.
    - The model itself does not copy CatPred's MoleculeModel.  It uses the local
      CatPred-derived D-MPNN substrate encoder and new catalytic-action JEPA
      components from ``LANCE/model.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import ConcatDataset, DataLoader, Sampler, WeightedRandomSampler

# See LANCE/data.py for the same launch-mode explanation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LANCE.data import (
    PARAM_ORDER,
    EmbeddingCache,
    KineticsDataset,
    build_scalers,
    clean,
    collate_kinetics,
    make_multitask_dataset,
    move_batch,
    normalize_parameters,
    row_sequence_cluster_id,
    safe_float,
    validate_cache_coverage,
    validate_data_root,
    write_jsonl_row,
)
from LANCE.losses import LANCELossWeights, lance_total_loss
from LANCE.model import LANCEConfig, LANCEModel


# Exact scheduler structure from CatPred's NoamLR.  CatPred warms up linearly
# then exponentially decays toward a final learning rate over total epochs.
class NoamLR(_LRScheduler):
    """Noam learning rate scheduler with linear warmup and exponential decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: list[float | int],
        total_epochs: list[int],
        steps_per_epoch: int,
        init_lr: list[float],
        max_lr: list[float],
        final_lr: list[float],
    ) -> None:
        if not (
            len(optimizer.param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        ):
            raise ValueError("NoamLR list lengths must match optimizer parameter groups")

        self.num_lrs = len(optimizer.param_groups)
        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)
        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        return list(self.lr)

    def step(self, current_step: int | None = None) -> None:
        self.current_step = int(current_step) if current_step is not None else self.current_step + 1
        for index in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[index]:
                self.lr[index] = self.init_lr[index] + self.current_step * self.linear_increment[index]
            elif self.current_step <= self.total_steps[index]:
                self.lr[index] = self.max_lr[index] * (
                    self.exponential_gamma[index] ** (self.current_step - self.warmup_steps[index])
                )
            else:
                self.lr[index] = self.final_lr[index]
            self.optimizer.param_groups[index]["lr"] = self.lr[index]


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for independent seed reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_peak_memory(device: torch.device, prefix: str) -> dict[str, float]:
    """Report CUDA allocator peaks for VRAM calibration logs."""

    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    scale = 1024 ** 3
    return {
        f"{prefix}_cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / scale, 3),
        f"{prefix}_cuda_peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / scale, 3),
    }


def reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def add_vram_status(row: dict[str, Any], args: argparse.Namespace, prefix: str) -> None:
    """Annotate logs with under/within/over relative to requested VRAM target."""

    target = float(args.target_vram_gb)
    if target <= 0:
        return
    peak = row.get(f"{prefix}_cuda_peak_reserved_gb")
    if peak is None:
        return
    low = target - float(args.target_vram_tolerance_gb)
    high = target + float(args.target_vram_tolerance_gb)
    row[f"{prefix}_target_vram_gb"] = target
    row[f"{prefix}_target_vram_low_gb"] = low
    row[f"{prefix}_target_vram_high_gb"] = high
    row[f"{prefix}_target_vram_status"] = "within" if low <= float(peak) <= high else ("under" if float(peak) < low else "over")


def regression_metrics(targets: list[float], preds: list[float]) -> dict[str, float]:
    """CatPred-style MAE/RMSE/R2/p1mag computed without extra dependencies."""

    y = np.asarray(targets, dtype=float)
    p = np.asarray(preds, dtype=float)
    if y.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "p1mag": float("nan")}
    mae = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((p - y) ** 2) / denom) if denom > 0 else float("nan")
    p1mag = float(np.mean(np.abs(y - p) < 1.0) * 100.0)
    return {"mae": mae, "rmse": rmse, "r2": r2, "p1mag": p1mag}


def latent_diagnostics(outputs: dict[str, torch.Tensor], jepa_mask: torch.Tensor | None = None) -> dict[str, float]:
    """Cheap collapse diagnostics for the latent action/state family.

    The loss can stay finite even if a JEPA-style representation collapses.
    These metrics make collapse visible in metrics.jsonl:
      - mean per-dimension standard deviation,
      - effective rank from singular values, normalized by max possible rank.
    """

    diagnostics: dict[str, float] = {}
    keys = ("molecule_state", "binding_state", "catalytic_state", "reaction_state_pred", "z_bind", "z_cat")
    for key in keys:
        value = outputs.get(key)
        if value is None or not torch.is_tensor(value):
            continue
        x = value.detach().float()
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        if x.dim() != 2 or x.shape[0] < 2:
            continue
        centered = x - x.mean(dim=0, keepdim=True)
        per_dim_std = centered.std(dim=0, unbiased=False)
        diagnostics[f"latent_{key}_std_mean"] = float(per_dim_std.mean().cpu())
        try:
            singular = torch.linalg.svdvals(centered)
            energy = singular.pow(2)
            denom = energy.sum().clamp_min(1.0e-12)
            probs = energy / denom
            entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
            effective_rank = torch.exp(entropy)
            max_rank = float(min(centered.shape))
            diagnostics[f"latent_{key}_effective_rank"] = float(effective_rank.cpu())
            diagnostics[f"latent_{key}_effective_rank_frac"] = float((effective_rank / max_rank).cpu())
        except RuntimeError:
            # SVD diagnostics should never break training; std is still useful.
            diagnostics[f"latent_{key}_effective_rank"] = float("nan")
            diagnostics[f"latent_{key}_effective_rank_frac"] = float("nan")

    molecule = outputs.get("molecule_state")
    z_bind = outputs.get("z_bind")
    z_cat = outputs.get("z_cat")
    binding_state = outputs.get("binding_state")
    catalytic_state = outputs.get("catalytic_state")
    reaction_delta = outputs.get("reaction_delta")
    reaction_delta_target = outputs.get("reaction_delta_target")
    reaction_state_pred = outputs.get("reaction_state_pred")
    reaction_state_target = outputs.get("reaction_state_target")
    z_bind_raw = outputs.get("z_bind_raw")
    z_cat_raw = outputs.get("z_cat_raw")
    bind_action_scale = outputs.get("bind_action_scale")
    cat_action_scale = outputs.get("cat_action_scale")
    if all(torch.is_tensor(x) for x in (molecule, z_bind, z_cat, binding_state, catalytic_state, reaction_delta, reaction_state_pred)):
        mol = molecule.detach().float()
        zb = z_bind.detach().float()
        zc = z_cat.detach().float()
        bind = binding_state.detach().float()
        cat = catalytic_state.detach().float()
        delta = reaction_delta.detach().float()
        pred = reaction_state_pred.detach().float()
        mol_norm = mol.norm(dim=-1).clamp_min(1.0e-6)
        mol_ln = F.layer_norm(mol, mol.shape[-1:])
        mol_ln_norm = mol_ln.norm(dim=-1).clamp_min(1.0e-6)
        diagnostics.update(
            {
                "action_z_bind_norm": float(zb.norm(dim=-1).mean().cpu()),
                "action_z_cat_norm": float(zc.norm(dim=-1).mean().cpu()),
                "action_reaction_delta_norm": float(delta.norm(dim=-1).mean().cpu()),
                "action_molecule_norm": float(mol_norm.mean().cpu()),
                "action_z_bind_to_molecule_ratio": float((zb.norm(dim=-1) / mol_norm).mean().cpu()),
                "action_z_cat_to_molecule_ratio": float((zc.norm(dim=-1) / mol_norm).mean().cpu()),
                "action_reaction_delta_to_molecule_ratio": float((delta.norm(dim=-1) / mol_ln_norm).mean().cpu()),
                "action_binding_displacement_norm": float((bind - mol_ln).norm(dim=-1).mean().cpu()),
                "action_catalytic_displacement_norm": float((cat - mol_ln).norm(dim=-1).mean().cpu()),
                "action_cos_binding_state_molecule": float(F.cosine_similarity(bind, mol_ln, dim=-1).mean().cpu()),
                "action_cos_catalytic_state_molecule": float(F.cosine_similarity(cat, mol_ln, dim=-1).mean().cpu()),
                "action_cos_reaction_pred_molecule": float(F.cosine_similarity(pred, mol_ln, dim=-1).mean().cpu()),
            }
        )
        if torch.is_tensor(z_bind_raw):
            diagnostics["action_z_bind_raw_norm"] = float(z_bind_raw.detach().float().norm(dim=-1).mean().cpu())
        if torch.is_tensor(z_cat_raw):
            diagnostics["action_z_cat_raw_norm"] = float(z_cat_raw.detach().float().norm(dim=-1).mean().cpu())
        if torch.is_tensor(bind_action_scale):
            diagnostics["action_bind_scale"] = float(bind_action_scale.detach().float().mean().cpu())
        if torch.is_tensor(cat_action_scale):
            diagnostics["action_cat_scale"] = float(cat_action_scale.detach().float().mean().cpu())
        if torch.is_tensor(reaction_state_target):
            target = reaction_state_target.detach().float()
            row_mask = jepa_mask.detach().bool() if torch.is_tensor(jepa_mask) else torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
            if row_mask.any():
                target = target[row_mask]
                pred_for_target = pred[row_mask]
                mol_for_target = mol_ln[row_mask]
            else:
                target = target[:0]
                pred_for_target = pred[:0]
                mol_for_target = mol_ln[:0]
            if target.shape[0] > 0:
                molecule_target_cos = F.cosine_similarity(mol_for_target, target, dim=-1)
                pred_target_cos = F.cosine_similarity(pred_for_target, target, dim=-1)
            else:
                molecule_target_cos = pred.new_tensor([float("nan")])
                pred_target_cos = pred.new_tensor([float("nan")])
            diagnostics["action_cos_molecule_target"] = float(molecule_target_cos.mean().cpu())
            diagnostics["action_cos_reaction_pred_target"] = float(pred_target_cos.mean().cpu())
            diagnostics["action_pred_target_minus_molecule_target_cos"] = float((pred_target_cos - molecule_target_cos).mean().cpu())
            if torch.is_tensor(reaction_delta_target):
                delta_target = reaction_delta_target.detach().float()
                if row_mask.any():
                    delta_target = delta_target[row_mask]
                    delta_for_target = delta[row_mask]
                else:
                    delta_target = delta_target[:0]
                    delta_for_target = delta[:0]
                if delta_target.shape[0] > 0:
                    diagnostics["action_true_reaction_delta_norm"] = float(delta_target.norm(dim=-1).mean().cpu())
                    diagnostics["action_true_reaction_delta_to_molecule_ratio"] = float(
                        (delta_target.norm(dim=-1) / mol_ln_norm[row_mask]).mean().cpu()
                    )
                    diagnostics["action_cos_reaction_delta_target"] = float(
                        F.cosine_similarity(delta_for_target, delta_target, dim=-1).mean().cpu()
                    )
    return diagnostics


@torch.no_grad()
def evaluate(
    model: LANCEModel,
    loaders: dict[str, tuple[str, DataLoader]],
    scalers: dict[str, Any],
    device: torch.device,
    *,
    score_loader_keys: set[str] | None = None,
    score_r2_weight: float = 0.0,
    score_p1mag_weight: float = 0.0,
) -> dict[str, float]:
    """Evaluate validation/checkpoint splits and return one selection score.

    The old LANCE trainer checkpointed on random-val MAE.  New LANCE2 runs can
    pass only per-seed cluster-holdout loaders in ``score_loader_keys`` and use
    a composite score ``MAE - a*R2 - b*p1mag``.  That makes checkpointing care
    about OOD ranking instead of only interpolation error.
    """

    model.eval()
    metrics: dict[str, float] = {}
    val_scores: list[float] = []
    for loader_key, (parameter, loader) in loaders.items():
        targets: list[float] = []
        preds: list[float] = []
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch)
            pred = scalers[parameter].inverse_tensor(out["mu"]).detach().cpu().tolist()
            target = batch["target"].detach().cpu().tolist()
            mask = batch["target_mask"].detach().cpu().tolist()
            for is_valid, y, p in zip(mask, target, pred):
                if is_valid:
                    targets.append(float(y))
                    preds.append(float(p))
        param_metrics = regression_metrics(targets, preds)
        prefix = loader_key.replace("/", "_")
        for name, value in param_metrics.items():
            metrics[f"{prefix}_{name}"] = value
            # Preserve the old metric names for scripts that inspect random val.
            if loader_key == f"{parameter}/val":
                metrics[f"{parameter}_{name}"] = value
        if score_loader_keys is None or loader_key in score_loader_keys:
            score = (
                param_metrics["mae"]
                - float(score_r2_weight) * param_metrics["r2"]
                - float(score_p1mag_weight) * param_metrics["p1mag"]
            )
            if math.isfinite(score):
                val_scores.append(score)
    metrics["score"] = float(np.nanmean(val_scores))
    return metrics


def save_checkpoint(
    path: Path,
    model: LANCEModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    scalers: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Save enough state for exact ensemble reconstruction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "scalers": {name: scaler.state() for name, scaler in scalers.items()},
            "args": vars(args),
            "model_config": model.config_dict(),
            "experiment_family": "LANCE",
        },
        path,
    )


def _dataset_cluster_ids(dataset: KineticsDataset) -> list[int]:
    """Read cluster ids from a KineticsDataset without materializing ESM tensors."""

    ids: list[int] = []
    for _, row in dataset.frame.iterrows():
        sequence = clean(row["sequence"])
        ids.append(row_sequence_cluster_id(row, sequence, dataset.cluster_level))
    return ids


def multitask_sampling_weights(
    dataset: ConcatDataset,
    *,
    task_balanced: bool,
    cluster_balanced: bool,
) -> torch.Tensor | None:
    """Build task/cluster-balanced weights for a concatenated multitask set.

    CatPred provenance:
        This is new LANCE training glue.  It changes which rows are sampled; it
        does not copy CatPred data-loader code.
    """

    if not task_balanced and not cluster_balanced:
        return None
    if not isinstance(dataset, ConcatDataset):
        raise TypeError("balanced sampling expects the ConcatDataset returned by make_multitask_dataset")

    all_weights: list[float] = []
    task_count = max(1, len(dataset.datasets))
    for sub_dataset in dataset.datasets:
        if not isinstance(sub_dataset, KineticsDataset):
            raise TypeError("expected KineticsDataset children inside ConcatDataset")
        n_rows = len(sub_dataset)
        if n_rows == 0:
            continue
        if cluster_balanced:
            cluster_ids = _dataset_cluster_ids(sub_dataset)
            counts: dict[int, int] = {}
            for cluster_id in cluster_ids:
                counts[cluster_id] = counts.get(cluster_id, 0) + 1
            n_clusters = max(1, len(counts))
            weights = [1.0 / (n_clusters * counts[cluster_id]) for cluster_id in cluster_ids]
        else:
            weights = [1.0 / n_rows for _ in range(n_rows)]
        if task_balanced:
            weights = [weight / task_count for weight in weights]
        all_weights.extend(weights)
    if not all_weights:
        return None
    return torch.tensor(all_weights, dtype=torch.double)


class TaskReactionCohortBatchSampler(Sampler[list[int]]):
    """Task-balanced batches with informative exact-reaction kcat pairs.

    Random minibatches almost never contain two kcat measurements for the same
    reaction, making an exact-reaction ranking objective inert.  This sampler
    reserves part of each batch's kcat quota for such pairs while drawing all
    remaining rows with replacement, matching the effective sampling behavior
    of the existing weighted multitask loader.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        *,
        batch_size: int,
        seed: int,
        cohort_fraction: float,
        min_gap: float,
    ) -> None:
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("reaction cohort sampling expects a ConcatDataset")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= cohort_fraction <= 1.0:
            raise ValueError("cohort_fraction must be between 0 and 1")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.cohort_fraction = float(cohort_fraction)
        self.min_gap = float(min_gap)
        self.epoch = 0
        self.task_pools: dict[int, list[int]] = {}
        kcat_groups: dict[str, list[tuple[int, float]]] = {}

        offset = 0
        for sub_dataset in dataset.datasets:
            if not isinstance(sub_dataset, KineticsDataset):
                raise TypeError("expected KineticsDataset children inside ConcatDataset")
            global_indices = list(range(offset, offset + len(sub_dataset)))
            self.task_pools.setdefault(int(sub_dataset.task_idx), []).extend(global_indices)
            if sub_dataset.parameter == "kcat":
                target_column = sub_dataset.spec["target"]
                for local_index, (_, row) in enumerate(sub_dataset.frame.iterrows()):
                    reaction = clean(row.get("reaction_smiles", ""))
                    raw_target = safe_float(row.get(target_column), float("nan"))
                    if not reaction or not math.isfinite(raw_target):
                        continue
                    target_z = sub_dataset.scaler.transform_value(raw_target)
                    kcat_groups.setdefault(reaction, []).append((offset + local_index, target_z))
            offset += len(sub_dataset)

        self.kcat_groups = []
        for rows in kcat_groups.values():
            ordered = sorted(rows, key=lambda item: item[1])
            if len(ordered) >= 2 and ordered[-1][1] - ordered[0][1] >= self.min_gap:
                self.kcat_groups.append(ordered)
        if 0 in self.task_pools and self.cohort_fraction > 0.0 and not self.kcat_groups:
            raise ValueError("no informative exact-reaction kcat cohorts were found")

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

    @staticmethod
    def _draw(pool: list[int], count: int, rng: np.random.Generator) -> list[int]:
        if count <= 0:
            return []
        choices = rng.integers(0, len(pool), size=count)
        return [pool[int(index)] for index in choices]

    def _draw_kcat_pair(self, rng: np.random.Generator) -> tuple[int, int]:
        group = self.kcat_groups[int(rng.integers(0, len(self.kcat_groups)))]
        for _ in range(8):
            selected = rng.choice(len(group), size=2, replace=False)
            left, right = group[int(selected[0])], group[int(selected[1])]
            if abs(left[1] - right[1]) >= self.min_gap:
                return left[0], right[0]
        return group[0][0], group[-1][0]

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + 104_729 * self.epoch)
        task_ids = sorted(self.task_pools)
        base, remainder = divmod(self.batch_size, len(task_ids))
        for batch_index in range(len(self)):
            allocations = {task_id: base for task_id in task_ids}
            for extra in range(remainder):
                allocations[task_ids[(batch_index + extra) % len(task_ids)]] += 1

            batch: list[int] = []
            for task_id in task_ids:
                count = allocations[task_id]
                if task_id == 0 and self.kcat_groups:
                    paired_rows = 2 * int((count * self.cohort_fraction) // 2)
                    for _ in range(paired_rows // 2):
                        batch.extend(self._draw_kcat_pair(rng))
                    count -= paired_rows
                batch.extend(self._draw(self.task_pools[task_id], count, rng))
            rng.shuffle(batch)
            yield batch
        self.epoch += 1


def build_loaders(
    args: argparse.Namespace,
    scalers: dict[str, Any],
    esm_cache: EmbeddingCache,
) -> tuple[DataLoader, dict[str, tuple[str, DataLoader]], set[str]]:
    """Construct train loader plus random-val and cluster-holdout eval loaders."""

    train_dataset = make_multitask_dataset(
        args.data_root,
        args.train_split,
        scalers,
        esm_cache,
        max_residues=args.max_residues,
        cluster_level=args.cluster_level,
        cluster_fold_seed=args.seed,
        cluster_fold_count=args.cluster_fold_count if args.cluster_holdout_selection else 0,
        cluster_fold_index=args.cluster_fold_index,
        cluster_fold_role="train" if args.cluster_holdout_selection else "all",
        parameters=args.tasks,
    )
    if args.reaction_cohort_sampler:
        batch_sampler = TaskReactionCohortBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            seed=args.seed + 17_029,
            cohort_fraction=args.kcat_cohort_fraction,
            min_gap=args.kcat_ranking_min_gap,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_kinetics,
            pin_memory=False,
            persistent_workers=args.num_workers > 0,
        )
    else:
        weights = multitask_sampling_weights(
            train_dataset,
            task_balanced=args.task_balanced_sampler,
            cluster_balanced=args.cluster_balanced_sampler,
        )
        sampler = None
        shuffle = True
        if weights is not None:
            generator = torch.Generator()
            generator.manual_seed(int(args.seed) + 17_029)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
            shuffle = False
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_kinetics,
            pin_memory=False,
            persistent_workers=args.num_workers > 0,
            drop_last=True,
        )
    eval_loaders: dict[str, tuple[str, DataLoader]] = {}
    score_loader_keys: set[str] = set()
    for parameter in normalize_parameters(args.tasks):
        random_val = KineticsDataset(
            args.data_root,
            parameter,
            "val",
            scalers[parameter],
            esm_cache,
            max_residues=args.max_residues,
            cluster_level=args.cluster_level,
        )
        eval_loaders[f"{parameter}/val"] = (
            parameter,
            DataLoader(
                random_val,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_kinetics,
                pin_memory=False,
            ),
        )
        if args.checkpoint_score_splits == "val":
            score_loader_keys.add(f"{parameter}/val")

        if args.cluster_holdout_selection:
            holdout = KineticsDataset(
                args.data_root,
                parameter,
                "train",
                scalers[parameter],
                esm_cache,
                max_residues=args.max_residues,
                cluster_level=args.cluster_level,
                cluster_fold_seed=args.seed,
                cluster_fold_count=args.cluster_fold_count,
                cluster_fold_index=args.cluster_fold_index,
                cluster_fold_role="holdout",
            )
            key = f"{parameter}/cluster_holdout"
            eval_loaders[key] = (
                parameter,
                DataLoader(
                    holdout,
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=collate_kinetics,
                    pin_memory=False,
                ),
            )
            if args.checkpoint_score_splits in {"cluster_holdout", "both"}:
                score_loader_keys.add(key)
        if args.checkpoint_score_splits == "both":
            score_loader_keys.add(f"{parameter}/val")
    return train_loader, eval_loaders, score_loader_keys


def train(args: argparse.Namespace) -> None:
    """Main LANCE training routine."""

    args.tasks = list(normalize_parameters(args.tasks))
    if args.warmup_epochs <= 0 or args.epochs <= args.warmup_epochs:
        raise ValueError("--warmup-epochs must be > 0 and smaller than --epochs")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if args.cluster_fold_count < 2 and args.cluster_holdout_selection:
        raise ValueError("--cluster-fold-count must be >= 2 when --cluster-holdout-selection is enabled")
    if args.checkpoint_score_splits in {"cluster_holdout", "both"} and not args.cluster_holdout_selection:
        raise ValueError("--checkpoint-score-splits cluster_holdout/both requires --cluster-holdout-selection")
    if args.focus_heads <= 0:
        raise ValueError("--focus-heads must be positive")
    if args.n_substrate_desc != 8:
        raise ValueError("current LANCE data pipeline emits exactly 8 substrate descriptors")
    if args.reaction_cohort_sampler and (args.task_balanced_sampler or args.cluster_balanced_sampler):
        raise ValueError("--reaction-cohort-sampler cannot be combined with weighted balanced samplers")
    if not 0.0 <= args.kcat_cohort_fraction <= 1.0:
        raise ValueError("--kcat-cohort-fraction must be between 0 and 1")
    if args.train_split != "train" and args.cluster_holdout_selection:
        raise ValueError("cluster holdout selection is only defined for --train-split train")
    if args.binding_jepa_exclude_kcat and not args.use_binding_jepa:
        raise ValueError("--binding-jepa-exclude-kcat requires --use-binding-jepa")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    seed_all(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    required_splits = (args.train_split, "val", "test")
    validate_data_root(args.data_root, parameters=args.tasks, required_splits=required_splits)
    esm_cache = EmbeddingCache(args.esm_cache, kind="ESM++")
    validate_cache_coverage(args.data_root, esm_cache, parameters=args.tasks)
    scalers = build_scalers(args.data_root, args.train_split, parameters=args.tasks)
    train_loader, val_loaders, score_loader_keys = build_loaders(args, scalers, esm_cache)
    if len(train_loader) == 0:
        raise ValueError("training loader has zero batches; lower --batch-size or check split files")

    config = LANCEConfig(
        protein_dim=esm_cache.dim,
        dmpnn_hidden=args.dmpnn_hidden,
        dmpnn_depth=args.dmpnn_depth,
        dmpnn_dropout=args.dmpnn_dropout,
        model_dim=args.model_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        stop_jepa_target=args.stop_jepa_target,
        use_soft_catalytic_focus=args.use_soft_catalytic_focus,
        focus_heads=args.focus_heads,
        focus_temperature=args.focus_temperature,
        use_task_film=args.use_task_film,
        use_task_interface_gate=args.use_task_interface_gate,
        use_task_specific_actions=args.use_task_specific_actions,
        use_task_residual_actions=args.use_task_residual_actions,
        task_action_adapter_dim=args.task_action_adapter_dim,
        use_gaussian_heads=args.use_gaussian_heads,
        n_substrate_desc=args.n_substrate_desc,
        use_condition_features=args.use_condition_features,
        use_ec_aux=args.use_ec_aux,
        ec_num_classes=args.ec_num_classes,
        use_kcat_global_rate_context=args.use_kcat_global_rate_context,
        use_binding_jepa=args.use_binding_jepa,
        binding_jepa_exclude_kcat=args.binding_jepa_exclude_kcat,
        use_mechanistic_kcat_prior=args.use_mechanistic_kcat_prior,
    )
    model = LANCEModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = NoamLR(
        optimizer,
        [args.warmup_epochs],
        [args.epochs],
        len(train_loader),
        [args.init_lr],
        [args.max_lr],
        [args.final_lr],
    )
    weights = LANCELossWeights(
        kinetic=args.kinetic_weight,
        jepa=args.jepa_weight,
        jepa_delta=args.jepa_delta_weight,
        jepa_contrastive=args.jepa_contrastive_weight,
        action_decorrelation=args.action_decorrelation_weight,
        sigreg=args.sigreg_weight,
        vrex=0.0,
        binding_aux=args.binding_aux_weight,
        gaussian_loss=args.gaussian_loss,
        beta_nll_beta=args.beta_nll_beta,
        decoupled_sigma=args.decoupled_sigma_weight,
        vrex_per_task=args.vrex_per_task,
        ec_aux=args.ec_aux_weight,
        binding_jepa=args.binding_jepa_weight,
        binding_jepa_delta=args.binding_jepa_delta_weight,
        binding_jepa_contrastive=args.binding_jepa_contrastive_weight,
        binding_jepa_exclude_kcat=args.binding_jepa_exclude_kcat,
        kcat_tail=args.kcat_tail_weight,
        kcat_tail_strength=args.kcat_tail_strength,
        kcat_tail_center=args.kcat_tail_center,
        kcat_mechanism_ranking=args.kcat_mechanism_ranking_weight,
        kcat_reaction_ranking=args.kcat_reaction_ranking_weight,
        kcat_ranking_temperature=args.kcat_ranking_temperature,
        kcat_ranking_min_gap=args.kcat_ranking_min_gap,
        km_p1=args.km_p1_weight,
        km_p1_temperature=args.km_p1_temperature,
    )

    run_dir = args.output_dir / args.run_name
    metrics_path = run_dir / "metrics.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "args": vars(args),
        "model_config": model.config_dict(),
        "scalers": {name: scaler.state() for name, scaler in scalers.items()},
        "experiment_family": "LANCE",
        "tasks": list(args.tasks),
        "checkpoint_score_loader_keys": sorted(score_loader_keys),
        "substrate_encoder": "LANCE.dmpnn.DMPNN (CatPred-derived atom-token path)",
    }
    if isinstance(train_loader.batch_sampler, TaskReactionCohortBatchSampler):
        config_payload["reaction_cohort_sampler"] = {
            "informative_groups": len(train_loader.batch_sampler.kcat_groups),
            "cohort_fraction": train_loader.batch_sampler.cohort_fraction,
            "min_standardized_gap": train_loader.batch_sampler.min_gap,
            "batches_per_epoch": len(train_loader.batch_sampler),
        }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"event": "start", "run_dir": str(run_dir), "device": str(device), "pid": os.getpid()}), flush=True)

    best = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    use_amp = device.type == "cuda"

    for epoch in range(args.epochs):
        model.train()
        if args.vrex_weight > 0:
            if epoch < args.vrex_ramp_start:
                weights.vrex = 0.0
            else:
                ramp_denom = max(1, int(args.vrex_ramp_epochs))
                ramp = min(1.0, float(epoch - args.vrex_ramp_start + 1) / ramp_denom)
                weights.vrex = float(args.vrex_weight) * ramp
        else:
            weights.vrex = 0.0
        if device.type == "cuda":
            torch.cuda.empty_cache()
        reset_cuda_peak_memory(device)
        sums = {
            "loss": 0.0,
            "kinetic": 0.0,
            "jepa": 0.0,
            "jepa_delta": 0.0,
            "jepa_contrastive": 0.0,
            "action_decorrelation": 0.0,
            "sigreg": 0.0,
            "vrex": 0.0,
            "binding_aux": 0.0,
            "binding_jepa": 0.0,
            "binding_jepa_delta": 0.0,
            "binding_jepa_contrastive": 0.0,
            "kcat_tail": 0.0,
            "kcat_mechanism_ranking": 0.0,
            "kcat_reaction_ranking": 0.0,
            "km_p1": 0.0,
            "ec_aux": 0.0,
            "jepa_rows": 0.0,
        }
        count = 0
        last_latent_diagnostics: dict[str, float] = {}

        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch)
                loss_parts = lance_total_loss(
                    out,
                    batch["target_z"],
                    target_mask=batch["target_mask"],
                    weights=weights,
                    jepa_mask=batch.get("jepa_mask"),
                    cluster_id=batch.get("cluster_id"),
                    task_idx=batch.get("task_idx"),
                    substrate_desc_target=batch.get("substrate_desc"),
                    ec_target=batch.get("ec_level1"),
                    ec_mask=batch.get("ec_mask"),
                    reaction_group_id=batch.get("reaction_group_id"),
                    target_scale=batch.get("target_scale"),
                )
                loss = loss_parts["loss"]

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            count += 1
            global_step += 1
            for name in sums:
                if name in loss_parts:
                    sums[name] += float(loss_parts[name].detach().cpu())
            sums["jepa_rows"] += float(batch.get("jepa_mask", torch.zeros_like(batch["target_mask"])).sum().detach().cpu())
            # Latent diagnostics include several SVDs and cosine probes.  They
            # are invaluable for validating the LANCE claim, but running them
            # on every optimization step wastes cluster time.  Compute them only
            # when the row will be logged, plus the final batch for the epoch
            # summary.
            log_due = args.log_every > 0 and global_step % args.log_every == 0
            epoch_last_batch = batch_index == len(train_loader) - 1
            if log_due or epoch_last_batch:
                last_latent_diagnostics = latent_diagnostics(out, batch.get("jepa_mask"))

            if log_due:
                row = {
                    "event": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "lr": scheduler.get_lr()[0],
                    "current_vrex_weight": weights.vrex,
                    **{name: value / max(1, count) for name, value in sums.items()},
                    "jepa_fraction": sums["jepa_rows"] / max(1, count * args.batch_size),
                    **last_latent_diagnostics,
                }
                row.update(cuda_peak_memory(device, "train"))
                add_vram_status(row, args, "train")
                write_jsonl_row(metrics_path, row)
                print(json.dumps(row, sort_keys=True), flush=True)

        train_memory = cuda_peak_memory(device, "train")
        reset_cuda_peak_memory(device)
        val_metrics = evaluate(
            model,
            val_loaders,
            scalers,
            device,
            score_loader_keys=score_loader_keys,
            score_r2_weight=args.score_r2_weight,
            score_p1mag_weight=args.score_p1mag_weight,
        )
        eval_memory = cuda_peak_memory(device, "eval")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        epoch_metrics = {
            "event": "epoch",
            "epoch": epoch,
            "step": global_step,
            "lr": scheduler.get_lr()[0],
            "current_vrex_weight": weights.vrex,
            **{f"train_{name}": value / max(1, count) for name, value in sums.items()},
            "train_jepa_fraction": sums["jepa_rows"] / max(1, count * args.batch_size),
            **{f"train_{key}": value for key, value in last_latent_diagnostics.items()},
            **train_memory,
        }
        epoch_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
        epoch_metrics.update(eval_memory)
        add_vram_status(epoch_metrics, args, "train")
        add_vram_status(epoch_metrics, args, "eval")
        write_jsonl_row(metrics_path, epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

        save_checkpoint(run_dir / "checkpoint_latest.pt", model, optimizer, epoch, epoch_metrics, scalers, args)
        score = float(epoch_metrics["val_score"])
        if score < best:
            best = score
            epochs_without_improvement = 0
            save_checkpoint(run_dir / "checkpoint_best.pt", model, optimizer, epoch, epoch_metrics, scalers, args)
        else:
            epochs_without_improvement += 1

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            row = {"event": "early_stop", "epoch": epoch, "best_val_score": best, "patience": args.early_stop_patience}
            write_jsonl_row(metrics_path, row)
            print(json.dumps(row, sort_keys=True), flush=True)
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/CatPred-DB"))
    parser.add_argument("--train-split", choices=("train", "trainval"), default="train")
    parser.add_argument("--esm-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lance"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", choices=PARAM_ORDER, default=list(PARAM_ORDER))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--init-lr", type=float, default=8.0e-5)
    parser.add_argument("--max-lr", type=float, default=5.0e-4)
    parser.add_argument("--final-lr", type=float, default=8.0e-6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=8.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kinetic-weight", type=float, default=1.0)
    parser.add_argument("--jepa-weight", type=float, default=0.2)
    parser.add_argument("--jepa-delta-weight", type=float, default=0.1)
    parser.add_argument("--jepa-contrastive-weight", type=float, default=0.05)
    parser.add_argument("--action-decorrelation-weight", type=float, default=0.01)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--vrex-weight", type=float, default=0.0)
    parser.add_argument("--vrex-ramp-start", type=int, default=5)
    parser.add_argument("--vrex-ramp-epochs", type=int, default=10)
    parser.add_argument("--vrex-per-task", action="store_true")
    parser.add_argument("--binding-aux-weight", type=float, default=0.0)
    parser.add_argument("--binding-jepa-weight", type=float, default=0.0)
    parser.add_argument("--binding-jepa-delta-weight", type=float, default=0.0)
    parser.add_argument("--binding-jepa-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--binding-jepa-exclude-kcat", action="store_true")
    parser.add_argument("--kcat-tail-weight", type=float, default=0.0)
    parser.add_argument("--kcat-tail-strength", type=float, default=0.75)
    parser.add_argument("--kcat-tail-center", type=float, default=0.5)
    parser.add_argument("--kcat-mechanism-ranking-weight", type=float, default=0.0)
    parser.add_argument("--kcat-reaction-ranking-weight", type=float, default=0.0)
    parser.add_argument("--kcat-ranking-temperature", type=float, default=0.35)
    parser.add_argument("--kcat-ranking-min-gap", type=float, default=0.25)
    parser.add_argument("--km-p1-weight", type=float, default=0.0)
    parser.add_argument("--km-p1-temperature", type=float, default=0.15)
    parser.add_argument("--gaussian-loss", choices=("nll", "beta_nll", "decoupled"), default="nll")
    parser.add_argument("--beta-nll-beta", type=float, default=0.5)
    parser.add_argument("--decoupled-sigma-weight", type=float, default=0.05)
    parser.add_argument("--max-residues", type=int, default=1024)
    parser.add_argument("--dmpnn-hidden", type=int, default=300)
    parser.add_argument("--dmpnn-depth", type=int, default=3)
    parser.add_argument("--dmpnn-dropout", type=float, default=0.0)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--action-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--use-soft-catalytic-focus", action="store_true")
    parser.add_argument("--focus-heads", type=int, default=4)
    parser.add_argument("--focus-temperature", type=float, default=1.0)
    parser.add_argument("--use-task-film", action="store_true")
    parser.add_argument("--use-task-interface-gate", action="store_true")
    parser.add_argument("--use-task-specific-actions", action="store_true")
    parser.add_argument("--use-task-residual-actions", action="store_true")
    parser.add_argument("--task-action-adapter-dim", type=int, default=256)
    parser.add_argument("--use-gaussian-heads", action="store_true")
    parser.add_argument("--n-substrate-desc", type=int, default=8)
    parser.add_argument("--use-condition-features", action="store_true")
    parser.add_argument("--use-ec-aux", action="store_true")
    parser.add_argument("--ec-num-classes", type=int, default=8)
    parser.add_argument("--ec-aux-weight", type=float, default=0.0)
    parser.add_argument("--use-kcat-global-rate-context", action="store_true")
    parser.add_argument("--use-binding-jepa", action="store_true")
    parser.add_argument("--use-mechanistic-kcat-prior", action="store_true")
    parser.add_argument("--cluster-level", type=int, default=60)
    parser.add_argument("--cluster-holdout-selection", action="store_true")
    parser.add_argument("--cluster-fold-count", type=int, default=5)
    parser.add_argument("--cluster-fold-index", type=int, default=-1)
    parser.add_argument("--checkpoint-score-splits", choices=("val", "cluster_holdout", "both"), default="val")
    parser.add_argument("--score-r2-weight", type=float, default=0.0)
    parser.add_argument("--score-p1mag-weight", type=float, default=0.0)
    parser.add_argument("--task-balanced-sampler", action="store_true")
    parser.add_argument("--cluster-balanced-sampler", action="store_true")
    parser.add_argument("--reaction-cohort-sampler", action="store_true")
    parser.add_argument("--kcat-cohort-fraction", type=float, default=0.5)
    parser.set_defaults(stop_jepa_target=True)
    parser.add_argument("--stop-jepa-target", dest="stop_jepa_target", action="store_true")
    parser.add_argument("--no-stop-jepa-target", dest="stop_jepa_target", action="store_false")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--target-vram-gb", type=float, default=0.0)
    parser.add_argument("--target-vram-tolerance-gb", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
