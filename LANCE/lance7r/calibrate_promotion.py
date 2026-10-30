#!/usr/bin/env python3
"""Freeze a benchmark-quarantined LANCE-7R trainval promotion recipe.

Variant selection and blend calibration have deliberately different jobs:

* the strict screen chooses the retrieval architecture on sequence-99-unseen
  random-validation rows;
* this script freezes one scalar blend for familiar clusters and one for
  unseen clusters using the selected three-seed validation ensemble;
* the resulting JSON is the only source of final-training hyperparameters.

No Heldout or OOD split is discovered, loaded, or scored here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LANCE.lance7r.data import (
    EmbeddingCache,
    TargetScaler,
    load_torch,
    move_batch,
    validate_cache_coverage,
    validate_data_root,
)
from LANCE.lance7r.model import LANCE7RConfig, LANCE7RModel
from LANCE.lance7r.retrieval_data import RetrievalKineticsDataset, collate_lance7r
from LANCE.lance7r.train import (
    PROMOTION_PROTOCOL,
    bounded_blend_alpha,
    regression_metrics,
    sha256_file,
)


def equal_payload(left: Any, right: Any) -> bool:
    """Compare JSON-like checkpoint configuration values exactly."""

    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right,
        sort_keys=True,
        default=str,
    )


def inverse_values(scaler: TargetScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


def subgroup_metrics(
    target: np.ndarray,
    base: np.ndarray,
    retrieval: np.ndarray,
    candidate: np.ndarray,
    available: np.ndarray,
    cluster_seen: np.ndarray,
) -> dict[str, Any]:
    """Report the exact evidence used to freeze the two-regime gate."""

    report: dict[str, Any] = {}
    for name, mask in (
        ("all", np.ones_like(cluster_seen, dtype=bool)),
        ("seen99", cluster_seen),
        ("unseen99", ~cluster_seen),
    ):
        base_metrics = regression_metrics(target[mask], base[mask])
        retrieval_metrics = regression_metrics(target[mask], retrieval[mask])
        candidate_metrics = regression_metrics(target[mask], candidate[mask])
        report[name] = {
            "base": base_metrics,
            "retrieval": retrieval_metrics,
            "candidate": candidate_metrics,
            "r2_gain": candidate_metrics["r2"] - base_metrics["r2"],
            "mae_gain": base_metrics["mae"] - candidate_metrics["mae"],
            "retrieval_available_fraction": float(available[mask].mean()),
        }
    return report


@torch.no_grad()
def collect_validation_predictions(
    models: list[LANCE7RModel],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Collect member and ensemble predictions on random validation only."""

    member_base_chunks: list[list[np.ndarray]] = [[] for _ in models]
    member_retrieval_chunks: list[list[np.ndarray]] = [[] for _ in models]
    target_chunks: list[np.ndarray] = []
    available_chunks: list[np.ndarray] = []
    cluster_seen_chunks: list[np.ndarray] = []

    use_amp = device.type == "cuda"
    for batch in loader:
        batch = move_batch(batch, device)
        batch_base: list[np.ndarray] = []
        batch_retrieval: list[np.ndarray] = []
        reference_available: np.ndarray | None = None
        reference_seen: np.ndarray | None = None
        for model_index, model in enumerate(models):
            with torch.amp.autocast(
                device_type="cuda" if use_amp else "cpu",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                output = model(batch)
            base = output["mu_base"].float().cpu().numpy()
            retrieval = output["retrieval_mu"].float().cpu().numpy()
            available = output["retrieval_available"].cpu().numpy().astype(bool)
            cluster_seen = output["retrieval_cluster_seen"].cpu().numpy().astype(bool)
            batch_base.append(base)
            batch_retrieval.append(retrieval)
            member_base_chunks[model_index].append(base)
            member_retrieval_chunks[model_index].append(retrieval)
            if reference_available is None:
                reference_available = available
                reference_seen = cluster_seen
            elif not np.array_equal(reference_available, available) or not np.array_equal(
                reference_seen,
                cluster_seen,
            ):
                raise RuntimeError(
                    "screen members disagree on retrieval availability or cluster status"
                )

        valid = batch["target_mask"].cpu().numpy().astype(bool)
        if not bool(valid.all()):
            raise RuntimeError("kcat validation unexpectedly contains missing targets")
        target_chunks.append(batch["target_z"].float().cpu().numpy())
        available_chunks.append(reference_available)
        cluster_seen_chunks.append(reference_seen)

    member_base = np.stack(
        [np.concatenate(chunks) for chunks in member_base_chunks],
        axis=0,
    )
    member_retrieval = np.stack(
        [np.concatenate(chunks) for chunks in member_retrieval_chunks],
        axis=0,
    )
    return {
        "target_z": np.concatenate(target_chunks),
        "member_base_z": member_base,
        "member_retrieval_z": member_retrieval,
        "base_z": member_base.mean(axis=0),
        "retrieval_z": member_retrieval.mean(axis=0),
        "available": np.concatenate(available_chunks),
        "cluster_seen": np.concatenate(cluster_seen_chunks),
    }


def freeze_recipe(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads(args.screen_summary.read_text(encoding="utf-8"))
    selected_variant = summary.get("selected_variant")
    if not bool(summary.get("promote_to_trainval", False)) or not selected_variant:
        raise ValueError("strict screen did not select a promotable variant")
    if bool(summary.get("uses_heldout_or_ood_labels", True)):
        raise ValueError("strict screen summary is not benchmark-quarantined")

    checkpoints = [load_torch(path) for path in args.checkpoint]
    selected = [
        (path, checkpoint)
        for path, checkpoint in zip(args.checkpoint, checkpoints)
        if checkpoint["args"].get("screen_variant") == selected_variant
    ]
    if len(selected) != 3:
        raise ValueError(
            f"expected three {selected_variant} checkpoints, found {len(selected)}"
        )
    seeds = sorted(int(checkpoint["args"]["seed"]) for _, checkpoint in selected)
    if len(set(seeds)) != 3:
        raise ValueError(f"strict screen seeds are not distinct: {seeds}")

    first_checkpoint = selected[0][1]
    first_retrieval_config = first_checkpoint["model_config"]["retrieval_config"]
    first_scaler_state = first_checkpoint["scalers"]["kcat"]
    consistency_fields = (
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "robust_loss_weight",
        "max_residues",
        "retrieval_cluster_level",
    )
    for _, checkpoint in selected[1:]:
        if not equal_payload(
            checkpoint["model_config"]["retrieval_config"],
            first_retrieval_config,
        ):
            raise ValueError("selected screen checkpoints use different retrievers")
        if not equal_payload(checkpoint["scalers"]["kcat"], first_scaler_state):
            raise ValueError("selected screen checkpoints use different kcat scalers")
        for field in consistency_fields:
            if checkpoint["args"].get(field) != first_checkpoint["args"].get(field):
                raise ValueError(f"selected screen checkpoints disagree on {field}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    models: list[LANCE7RModel] = []
    for _, checkpoint in selected:
        model = LANCE7RModel(
            LANCE7RConfig.from_dict(checkpoint["model_config"])
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models.append(model)

    scaler = TargetScaler.from_state(first_scaler_state)
    validate_data_root(
        args.data_root,
        parameters=("kcat",),
        required_splits=("val",),
    )
    esm_cache = EmbeddingCache(args.esm_cache, kind="ESM++")
    validate_cache_coverage(
        args.data_root,
        esm_cache,
        parameters=("kcat",),
        splits=("val",),
    )
    dataset = RetrievalKineticsDataset(
        args.data_root,
        "kcat",
        "val",
        scaler,
        esm_cache,
        max_residues=int(first_checkpoint["args"]["max_residues"]),
        protein_key_dim=int(first_retrieval_config["protein_key_dim"]),
        fingerprint_dim=int(first_retrieval_config["fingerprint_dim"]),
        retrieval_cluster_level=int(
            first_checkpoint["args"]["retrieval_cluster_level"]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_lance7r,
        pin_memory=False,
    )
    predictions = collect_validation_predictions(models, loader, device)

    target_z = predictions["target_z"]
    base_z = predictions["base_z"]
    retrieval_z = predictions["retrieval_z"]
    available = predictions["available"]
    cluster_seen = predictions["cluster_seen"]
    seen_alpha = bounded_blend_alpha(
        target_z,
        base_z,
        retrieval_z,
        available,
        calibration_mask=cluster_seen,
        maximum=args.max_blend_alpha,
    )
    unseen_alpha = bounded_blend_alpha(
        target_z,
        base_z,
        retrieval_z,
        available,
        calibration_mask=~cluster_seen,
        maximum=args.max_blend_alpha,
    )
    row_alpha = np.where(cluster_seen, seen_alpha, unseen_alpha)
    candidate_z = base_z + row_alpha * available.astype(float) * (
        retrieval_z - base_z
    )

    target = inverse_values(scaler, target_z)
    base = inverse_values(scaler, base_z)
    retrieval = inverse_values(scaler, retrieval_z)
    candidate = inverse_values(scaler, candidate_z)
    metrics = subgroup_metrics(
        target,
        base,
        retrieval,
        candidate,
        available,
        cluster_seen,
    )

    member_diagnostics: list[dict[str, Any]] = []
    for member_index, (path, checkpoint) in enumerate(selected):
        member_base = predictions["member_base_z"][member_index]
        member_retrieval = predictions["member_retrieval_z"][member_index]
        member_seen_alpha = bounded_blend_alpha(
            target_z,
            member_base,
            member_retrieval,
            available,
            calibration_mask=cluster_seen,
            maximum=args.max_blend_alpha,
        )
        member_unseen_alpha = bounded_blend_alpha(
            target_z,
            member_base,
            member_retrieval,
            available,
            calibration_mask=~cluster_seen,
            maximum=args.max_blend_alpha,
        )
        member_diagnostics.append(
            {
                "seed": int(checkpoint["args"]["seed"]),
                "checkpoint": str(path),
                "checkpoint_sha256": sha256_file(path),
                "selected_epoch": int(checkpoint["epoch"]),
                "seen_alpha": member_seen_alpha,
                "unseen_alpha": member_unseen_alpha,
            }
        )

    selected_epochs = [
        int(checkpoint["epoch"]) + 1 for _, checkpoint in selected
    ]
    training_args = first_checkpoint["args"]
    recipe = {
        "protocol": PROMOTION_PROTOCOL,
        "experiment_family": "LANCE-7R",
        "selected_variant": selected_variant,
        "promote_to_trainval": True,
        "uses_heldout_or_ood_labels": False,
        "screen_summary": str(args.screen_summary),
        "screen_summary_sha256": sha256_file(args.screen_summary),
        "selection_evidence": summary["variants"][selected_variant],
        "retrieval_config": dict(first_retrieval_config),
        "training_config": {
            "epochs": int(statistics.median(selected_epochs)),
            "epoch_rule": "median(best_screen_epoch_plus_one)",
            "selected_member_epochs": selected_epochs,
            "learning_rate": float(training_args["learning_rate"]),
            "weight_decay": float(training_args["weight_decay"]),
            "grad_clip": float(training_args["grad_clip"]),
            "robust_loss_weight": float(training_args["robust_loss_weight"]),
            "max_residues": int(training_args["max_residues"]),
            "retrieval_cluster_level": int(
                training_args["retrieval_cluster_level"]
            ),
        },
        "blend": {
            "seen_cluster": seen_alpha,
            "unseen_cluster": unseen_alpha,
            "maximum": float(args.max_blend_alpha),
            "rule": "least-squares convex blend on selected three-seed validation ensemble",
        },
        "calibration": {
            "dataset": "kcat-random_val.csv",
            "rows": int(target_z.size),
            "seen99_rows": int(cluster_seen.sum()),
            "unseen99_rows": int((~cluster_seen).sum()),
            "metrics": metrics,
            "members": member_diagnostics,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(recipe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/CatPred-DB"))
    parser.add_argument("--esm-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-blend-alpha", type=float, default=0.50)
    args = parser.parse_args()
    recipe = freeze_recipe(args)
    print(json.dumps(recipe, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
