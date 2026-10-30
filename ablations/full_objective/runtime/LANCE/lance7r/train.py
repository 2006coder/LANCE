#!/usr/bin/env python3
"""Train the LANCE-7R product-free kcat retrieval expert.

The archived LANCE-4R incumbent is loaded and frozen exactly. Retrieval learns
from kcat training rows under a sequence-99 exclusion mask, so a query can never
retrieve itself or another member of its near-identical sequence cluster.

For strict screening, the scalar incumbent/retrieval blend is calibrated on
random-validation rows whose sequence cluster exists in training and selected
on the disjoint rows whose sequence-99 cluster is absent from training. OOD
benchmark labels are never loaded by this trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
    write_jsonl_row,
)
from LANCE.lance7r.model import LANCE7RConfig, LANCE7RModel
from LANCE.lance7r.retrieval import RetrievalConfig
from LANCE.lance7r.retrieval_data import (
    RetrievalKineticsDataset,
    build_retrieval_memory,
    collate_lance7r,
    collate_retrieval_training,
)

PROMOTION_PROTOCOL = "lance7r_strict_promotion_v1"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        return {
            "n": 0,
            "r2": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "p1mag": float("nan"),
        }
    error = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "n": int(target.size),
        "r2": (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0
            else float("nan")
        ),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "p1mag": float(100.0 * np.mean(np.abs(error) < 1.0)),
    }


def validate_lance4r_checkpoint(checkpoint: dict[str, Any], expected_split: str) -> None:
    """Refuse to train on a checkpoint that is not the verified 4R recipe."""

    config = checkpoint.get("model_config", {})
    required_true = (
        "use_task_interface_gate",
        "use_task_residual_actions",
        "use_gaussian_heads",
        "use_condition_features",
        "use_ec_aux",
        "use_kcat_global_rate_context",
        "use_binding_jepa",
        "binding_jepa_exclude_kcat",
    )
    failures = [name for name in required_true if not bool(config.get(name, False))]
    if failures:
        raise ValueError(f"incumbent checkpoint is not LANCE-4R; false flags: {failures}")
    if bool(config.get("use_mechanistic_kcat_prior", False)):
        raise ValueError("LANCE-4R incumbent must not use the mechanistic kcat prior")
    tasks = [str(task).lower() for task in checkpoint.get("args", {}).get("tasks", [])]
    if tasks != ["kcat", "km", "ki"]:
        raise ValueError(f"incumbent checkpoint tasks are {tasks}, expected kcat/km/ki")
    actual_split = str(checkpoint.get("args", {}).get("train_split", ""))
    if actual_split != expected_split:
        raise ValueError(
            f"incumbent train split is {actual_split!r}, expected {expected_split!r}"
        )


def apply_promotion_recipe(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load one immutable strict-screen recipe into the final trainer."""

    if args.promotion_recipe is None:
        raise ValueError("--promotion-recipe is required for fixed-final training")
    recipe = json.loads(args.promotion_recipe.read_text(encoding="utf-8"))
    if recipe.get("protocol") != PROMOTION_PROTOCOL:
        raise ValueError("unsupported or missing LANCE-7R promotion protocol")
    if not bool(recipe.get("promote_to_trainval", False)):
        raise ValueError("promotion recipe did not pass the strict screen")
    if bool(recipe.get("uses_heldout_or_ood_labels", True)):
        raise ValueError("promotion recipe is not benchmark-quarantined")

    retrieval = dict(recipe["retrieval_config"])
    training = dict(recipe["training_config"])
    blend = dict(recipe["blend"])
    args.screen_variant = str(recipe["selected_variant"])
    args.retrieval_mode = str(retrieval["mode"])
    args.protein_key_dim = int(retrieval["protein_key_dim"])
    args.fingerprint_dim = int(retrieval["fingerprint_dim"])
    args.topk_exact = int(retrieval["topk_exact"])
    args.topk_approx = int(retrieval["topk_approx"])
    args.retrieval_temperature = float(retrieval["temperature"])
    args.retrieval_residual_limit = float(retrieval["residual_limit"])
    args.learn_metric = bool(retrieval["learn_metric"])
    args.epochs = int(training["epochs"])
    args.learning_rate = float(training["learning_rate"])
    args.weight_decay = float(training["weight_decay"])
    args.grad_clip = float(training["grad_clip"])
    args.robust_loss_weight = float(training["robust_loss_weight"])
    args.max_residues = int(training["max_residues"])
    args.retrieval_cluster_level = int(training["retrieval_cluster_level"])
    args.fixed_blend_alpha_seen = float(blend["seen_cluster"])
    args.fixed_blend_alpha_unseen = float(blend["unseen_cluster"])
    return recipe


def bounded_blend_alpha(
    target_z: np.ndarray,
    base_z: np.ndarray,
    retrieval_z: np.ndarray,
    available: np.ndarray,
    calibration_mask: np.ndarray,
    maximum: float,
) -> float:
    """Fit one convex coefficient on the declared calibration subset."""

    arrays = (target_z, base_z, retrieval_z)
    if not all(np.isfinite(values).all() for values in arrays):
        raise FloatingPointError("blend calibration received non-finite predictions")
    mask = available & calibration_mask
    if int(mask.sum()) < 2:
        return 0.0
    direction = retrieval_z[mask] - base_z[mask]
    denominator = float(np.dot(direction, direction))
    if denominator <= 1.0e-12:
        return 0.0
    numerator = float(np.dot(target_z[mask] - base_z[mask], direction))
    return float(np.clip(numerator / denominator, 0.0, maximum))


@torch.no_grad()
def evaluate_validation(
    model: LANCE7RModel,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
    max_blend_alpha: float,
) -> dict[str, float]:
    """Calibrate on seen-cluster validation and score unseen-cluster validation."""

    model.eval()
    target_z_rows: list[float] = []
    base_z_rows: list[float] = []
    retrieval_z_rows: list[float] = []
    available_rows: list[bool] = []
    cluster_seen_rows: list[bool] = []

    use_amp = device.type == "cuda"
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            device_type="cuda" if use_amp else "cpu",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            output = model(batch)
        valid = batch["target_mask"]
        target_z_rows.extend(batch["target_z"][valid].float().cpu().tolist())
        base_z_rows.extend(output["mu_base"][valid].float().cpu().tolist())
        retrieval_z_rows.extend(output["retrieval_mu"][valid].float().cpu().tolist())
        available_rows.extend(output["retrieval_available"][valid].cpu().tolist())
        cluster_seen_rows.extend(output["retrieval_cluster_seen"][valid].cpu().tolist())

    target_z = np.asarray(target_z_rows, dtype=float)
    base_z = np.asarray(base_z_rows, dtype=float)
    retrieval_z = np.asarray(retrieval_z_rows, dtype=float)
    available = np.asarray(available_rows, dtype=bool)
    cluster_seen = np.asarray(cluster_seen_rows, dtype=bool)

    alpha = bounded_blend_alpha(
        target_z,
        base_z,
        retrieval_z,
        available,
        calibration_mask=cluster_seen,
        maximum=max_blend_alpha,
    )
    model.set_blend_alpha(alpha)
    candidate_z = base_z + alpha * available.astype(float) * (retrieval_z - base_z)

    target = scaler.inverse_transform(target_z.reshape(-1, 1)).reshape(-1)
    base = scaler.inverse_transform(base_z.reshape(-1, 1)).reshape(-1)
    candidate = scaler.inverse_transform(candidate_z.reshape(-1, 1)).reshape(-1)
    retrieval = scaler.inverse_transform(retrieval_z.reshape(-1, 1)).reshape(-1)
    unseen = ~cluster_seen

    metrics: dict[str, float] = {
        "blend_alpha": alpha,
        "retrieval_available_fraction": float(available.mean()),
        "cluster_unseen_fraction": float(unseen.mean()),
        "cluster_unseen_available_fraction": (
            float(available[unseen].mean()) if unseen.any() else float("nan")
        ),
    }
    for prefix, mask in (
        ("all", np.ones_like(unseen, dtype=bool)),
        ("seen99", cluster_seen),
        ("unseen99", unseen),
    ):
        for model_name, prediction in (
            ("base", base),
            ("retrieval", retrieval),
            ("candidate", candidate),
        ):
            values = regression_metrics(target[mask], prediction[mask])
            metrics.update(
                {
                    f"{prefix}_{model_name}_{name}": value
                    for name, value in values.items()
                }
            )

    # R2 is the remaining benchmark wall. MAE is retained as a soft guardrail.
    metrics["selection_score"] = (
        -metrics["unseen99_candidate_r2"]
        + 0.10 * metrics["unseen99_candidate_mae"]
    )
    metrics["unseen99_r2_gain"] = (
        metrics["unseen99_candidate_r2"] - metrics["unseen99_base_r2"]
    )
    metrics["unseen99_mae_gain"] = (
        metrics["unseen99_base_mae"] - metrics["unseen99_candidate_mae"]
    )
    return metrics


def save_checkpoint(
    path: Path,
    model: LANCE7RModel,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    scalers: dict[str, TargetScaler],
    args: argparse.Namespace,
    incumbent_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "metrics": metrics,
            "scalers": {name: scaler.state() for name, scaler in scalers.items()},
            "args": vars(args),
            "model_config": model.config_dict(),
            "experiment_family": "LANCE-7R",
            "incumbent_checkpoint": str(args.incumbent_checkpoint),
            "incumbent_sha256": incumbent_sha256,
        },
        path,
    )


def train_retrieval_epoch(
    model: LANCE7RModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    device: torch.device,
    robust_loss_weight: float,
    grad_clip: float,
) -> tuple[float, int]:
    """Run one cross-cluster retrieval epoch without executing LANCE-4R."""

    model.train()
    loss_sum = 0.0
    step_count = 0
    training_rows = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        retrieval = model.retrieve(batch)
        valid = batch["target_mask"] & retrieval["retrieval_available"]
        if not bool(valid.any()):
            continue
        prediction = retrieval["retrieval_mu"][valid]
        target = batch["target_z"][valid]
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                "retrieval produced a non-finite training prediction"
            )
        mse = F.mse_loss(prediction, target)
        robust = F.smooth_l1_loss(prediction, target, beta=0.5)
        loss = mse + robust_loss_weight * robust
        if not torch.isfinite(loss):
            raise FloatingPointError("retrieval produced a non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("retrieval produced a non-finite gradient")
        optimizer.step()
        loss_sum += float(loss.detach().cpu())
        step_count += 1
        training_rows += int(valid.sum().detach().cpu())
    if step_count == 0:
        raise RuntimeError("retrieval epoch contained no valid neighbor batches")
    return loss_sum / step_count, training_rows


def train(args: argparse.Namespace) -> None:
    promotion_recipe: dict[str, Any] | None = None
    if args.protocol == "fixed-final":
        promotion_recipe = apply_promotion_recipe(args)
        if args.incumbent_split != "trainval" or args.memory_split != "trainval":
            raise ValueError("fixed-final promotion requires trainval incumbent and memory")
    elif args.promotion_recipe is not None:
        raise ValueError("--promotion-recipe is only valid with --protocol fixed-final")
    elif not args.screen_variant:
        raise ValueError("--screen-variant is required for strict-screen training")

    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    if not 0.0 <= args.max_blend_alpha <= 1.0:
        raise ValueError("--max-blend-alpha must be in [0, 1]")
    if args.incumbent_split != args.memory_split:
        raise ValueError("incumbent and retrieval memory must use the same split")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_all(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    required_splits = (
        (args.memory_split, "val")
        if args.protocol == "strict-screen"
        else (args.memory_split,)
    )
    validate_data_root(
        args.data_root,
        parameters=("kcat",),
        required_splits=required_splits,
    )
    esm_cache = EmbeddingCache(args.esm_cache, kind="ESM++")
    validate_cache_coverage(
        args.data_root,
        esm_cache,
        parameters=("kcat",),
        splits=required_splits,
    )

    incumbent_checkpoint = load_torch(args.incumbent_checkpoint)
    validate_lance4r_checkpoint(incumbent_checkpoint, args.incumbent_split)
    scalers = {
        name: TargetScaler.from_state(state)
        for name, state in incumbent_checkpoint["scalers"].items()
    }
    if "kcat" not in scalers:
        raise KeyError("incumbent checkpoint does not contain a kcat scaler")

    dataset_kwargs = {
        "max_residues": args.max_residues,
        "protein_key_dim": args.protein_key_dim,
        "fingerprint_dim": args.fingerprint_dim,
        "retrieval_cluster_level": args.retrieval_cluster_level,
    }
    memory_dataset = RetrievalKineticsDataset(
        args.data_root,
        "kcat",
        args.memory_split,
        scalers["kcat"],
        esm_cache,
        **dataset_kwargs,
    )
    validation_dataset = (
        RetrievalKineticsDataset(
            args.data_root,
            "kcat",
            "val",
            scalers["kcat"],
            esm_cache,
            **dataset_kwargs,
        )
        if args.protocol == "strict-screen"
        else None
    )
    memory = build_retrieval_memory(memory_dataset)

    retrieval_config = RetrievalConfig(
        mode=args.retrieval_mode,
        protein_key_dim=args.protein_key_dim,
        fingerprint_dim=args.fingerprint_dim,
        topk_exact=args.topk_exact,
        topk_approx=args.topk_approx,
        temperature=args.retrieval_temperature,
        residual_limit=args.retrieval_residual_limit,
        learn_metric=args.learn_metric,
    )
    model_config = LANCE7RConfig(
        incumbent_config=dict(incumbent_checkpoint["model_config"]),
        retrieval_config=retrieval_config.as_dict(),
        memory_size=memory.size,
        blend_alpha=float(getattr(args, "fixed_blend_alpha_seen", 0.0)),
        unseen_blend_alpha=getattr(args, "fixed_blend_alpha_unseen", None),
    )
    model = LANCE7RModel(model_config, memory)
    model.load_incumbent_state(incumbent_checkpoint["model"])
    model.to(device)

    trainable = [parameter for parameter in model.retriever.parameters() if parameter.requires_grad]
    train_loader = (
        DataLoader(
            memory_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_retrieval_training,
            pin_memory=False,
            drop_last=True,
            generator=torch.Generator().manual_seed(args.seed + 17_029),
        )
        if trainable and args.epochs > 0
        else None
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_lance7r,
            pin_memory=False,
        )
        if validation_dataset is not None
        else None
    )
    optimizer = (
        torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if trainable
        else None
    )

    run_dir = args.output_dir / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    incumbent_sha256 = sha256_file(args.incumbent_checkpoint)
    run_config = {
        "args": vars(args),
        "model_config": model.config_dict(),
        "memory": {
            "rows": memory.size,
            "protein_key_dim": args.protein_key_dim,
            "fingerprint_dim": args.fingerprint_dim,
            "cluster_exclusion_level": args.retrieval_cluster_level,
            "source_split": args.memory_split,
            "uses_products": False,
        },
        "incumbent_checkpoint": str(args.incumbent_checkpoint),
        "incumbent_sha256": incumbent_sha256,
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "experiment_family": "LANCE-7R",
        "protocol": args.protocol,
        "promotion_recipe": promotion_recipe,
        "promotion_recipe_sha256": (
            sha256_file(args.promotion_recipe)
            if args.promotion_recipe is not None
            else None
        ),
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "start",
                "run_dir": str(run_dir),
                "device": str(device),
                "pid": os.getpid(),
                "memory_rows": memory.size,
                "trainable_parameters": run_config["trainable_parameters"],
            }
        ),
        flush=True,
    )

    if args.protocol == "fixed-final":
        if bool(trainable) != bool(args.learn_metric):
            raise RuntimeError("retrieval trainability does not match the frozen recipe")
        if args.learn_metric and args.epochs <= 0:
            raise ValueError("learned-metric finalization requires positive fixed epochs")
        if not args.learn_metric and args.epochs != 0:
            raise ValueError("fixed-metric finalization requires zero epochs")

        final_row: dict[str, float | int | str] = {
            "event": "fixed_final",
            "epoch": max(args.epochs - 1, 0),
            "fixed_epochs": args.epochs,
            "blend_alpha_seen": float(model.blend_alpha.detach().cpu()),
            "blend_alpha_unseen": float(model.unseen_blend_alpha.detach().cpu()),
            "train_loss": float("nan"),
            "train_rows": 0,
        }
        for epoch in range(args.epochs):
            if optimizer is None or train_loader is None:
                raise RuntimeError("learned final recipe has no optimizer or loader")
            training_loss, training_rows = train_retrieval_epoch(
                model,
                train_loader,
                optimizer,
                trainable,
                device,
                args.robust_loss_weight,
                args.grad_clip,
            )
            final_row = {
                "event": "fixed_final",
                "epoch": epoch,
                "fixed_epochs": args.epochs,
                "blend_alpha_seen": float(model.blend_alpha.detach().cpu()),
                "blend_alpha_unseen": float(model.unseen_blend_alpha.detach().cpu()),
                "train_loss": training_loss,
                "train_rows": training_rows,
            }
            write_jsonl_row(metrics_path, final_row)
            print(json.dumps(final_row, sort_keys=True), flush=True)

        save_checkpoint(
            run_dir / "checkpoint_latest.pt",
            model,
            optimizer,
            int(final_row["epoch"]),
            final_row,
            scalers,
            args,
            incumbent_sha256,
        )
        save_checkpoint(
            run_dir / "checkpoint_best.pt",
            model,
            optimizer,
            int(final_row["epoch"]),
            final_row,
            scalers,
            args,
            incumbent_sha256,
        )
        return

    if validation_loader is None:
        raise RuntimeError("strict-screen training requires a validation loader")

    best_score = float("inf")
    epochs_without_improvement = 0
    total_epochs = max(1, args.epochs)
    for epoch in range(total_epochs):
        training_loss = float("nan")
        training_rows = 0
        if optimizer is not None and args.epochs > 0:
            if train_loader is None:
                raise RuntimeError("learned screen recipe has no training loader")
            training_loss, training_rows = train_retrieval_epoch(
                model,
                train_loader,
                optimizer,
                trainable,
                device,
                args.robust_loss_weight,
                args.grad_clip,
            )

        validation = evaluate_validation(
            model,
            validation_loader,
            scalers["kcat"],
            device,
            args.max_blend_alpha,
        )
        row = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": training_loss,
            "train_rows": training_rows,
            **validation,
        }
        write_jsonl_row(metrics_path, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        save_checkpoint(
            run_dir / "checkpoint_latest.pt",
            model,
            optimizer,
            epoch,
            row,
            scalers,
            args,
            incumbent_sha256,
        )

        score = float(validation["selection_score"])
        if math.isfinite(score) and score < best_score:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "checkpoint_best.pt",
                model,
                optimizer,
                epoch,
                row,
                scalers,
                args,
                incumbent_sha256,
            )
        else:
            epochs_without_improvement += 1
        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=("strict-screen", "fixed-final"),
        default="strict-screen",
    )
    parser.add_argument("--promotion-recipe", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/CatPred-DB"))
    parser.add_argument("--esm-cache", type=Path, required=True)
    parser.add_argument("--incumbent-checkpoint", type=Path, required=True)
    parser.add_argument("--incumbent-split", choices=("train", "trainval"), default="train")
    parser.add_argument("--memory-split", choices=("train", "trainval"), default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lance"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--screen-variant")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--robust-loss-weight", type=float, default=0.15)
    parser.add_argument("--max-residues", type=int, default=1024)
    parser.add_argument("--protein-key-dim", type=int, default=64)
    parser.add_argument("--fingerprint-dim", type=int, default=128)
    parser.add_argument("--retrieval-cluster-level", type=int, default=99)
    parser.add_argument("--retrieval-mode", choices=("exact", "dual"), default="dual")
    parser.add_argument("--topk-exact", type=int, default=8)
    parser.add_argument("--topk-approx", type=int, default=16)
    parser.add_argument("--retrieval-temperature", type=float, default=0.20)
    parser.add_argument("--retrieval-residual-limit", type=float, default=0.50)
    parser.add_argument("--max-blend-alpha", type=float, default=0.50)
    parser.set_defaults(learn_metric=True)
    parser.add_argument("--learn-metric", dest="learn_metric", action="store_true")
    parser.add_argument("--fixed-metric", dest="learn_metric", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
