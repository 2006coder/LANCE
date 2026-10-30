#!/usr/bin/env python3
"""Evaluate LANCE checkpoints as a seed ensemble.

CatPred provenance:
    CatPred's normal ensemble path sums per-model predictions and divides by
    ensemble size before scoring.  This evaluator uses the same point-prediction
    averaging idea, but reconstructs LANCE checkpoints and writes one
    combined JSON/CSV summary for all kinetic parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LANCE.data import (
    PARAM_ORDER,
    EmbeddingCache,
    KineticsDataset,
    TargetScaler,
    available_splits,
    collate_kinetics,
    load_torch,
    move_batch,
    normalize_parameters,
    validate_cache_coverage,
    validate_data_root,
)
from LANCE.model import LANCEConfig, LANCEModel


def cuda_peak_memory(device: torch.device) -> dict[str, float]:
    """Return CUDA peak memory in GiB for final evaluation logs."""

    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    scale = 1024 ** 3
    return {
        "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / scale, 3),
        "cuda_peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / scale, 3),
    }


def vram_status(peak_reserved_gb: float | None, target_gb: float, tolerance_gb: float) -> dict[str, float | str]:
    """Annotate memory status relative to the requested target."""

    if peak_reserved_gb is None or target_gb <= 0:
        return {}
    low = target_gb - tolerance_gb
    high = target_gb + tolerance_gb
    return {
        "target_vram_gb": target_gb,
        "target_vram_low_gb": low,
        "target_vram_high_gb": high,
        "target_vram_status": "within" if low <= peak_reserved_gb <= high else ("under" if peak_reserved_gb < low else "over"),
    }


def regression_metrics(targets: list[float], preds: list[float]) -> dict[str, float]:
    """CatPred-style MAE, R2, p1mag, RMSE, and sample count."""

    y = np.asarray(targets, dtype=float)
    p = np.asarray(preds, dtype=float)
    if y.size == 0:
        return {"MAE": float("nan"), "R2": float("nan"), "p1mag": float("nan"), "rmse": float("nan"), "n": 0}
    mae = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((p - y) ** 2) / denom) if denom > 0 else float("nan")
    p1mag = float(np.mean(np.abs(y - p) < 1.0) * 100.0)
    return {"MAE": mae, "R2": r2, "p1mag": p1mag, "rmse": rmse, "n": int(y.size)}


def prediction_diagnostics(targets: list[float], preds: list[float]) -> dict[str, float]:
    """Diagnose whether low R2 is bad ranking or compressed calibration.

    CatPred provenance:
        This is new LANCE diagnostic code.  CatPred reports aggregate R2/MAE
        but does not need the Pearson-r2-vs-R2 distinction because its standard
        reproduction path already saves prediction CSVs.  LANCE's first
        evaluator only wrote aggregate metrics, so this helper makes the
        row-level ranking/compression audit explicit.

    Interpretation:
        - Pearson r2 measures whether predictions rank examples correctly after
          allowing any linear rescaling.
        - Ordinary R2 measures whether predictions are already calibrated in
          the target scale.
        - If Pearson r2 is high but ordinary R2 is low, the model mostly has a
          compression/bias problem.  If both are low, representation/ranking is
          failing on that split.
    """

    y = np.asarray(targets, dtype=float)
    p = np.asarray(preds, dtype=float)
    if y.size == 0:
        return {
            "pearson_r": float("nan"),
            "pearson_r2": float("nan"),
            "target_std": float("nan"),
            "pred_std": float("nan"),
            "pred_target_std_ratio": float("nan"),
            "bias": float("nan"),
            "pred_vs_target_slope": float("nan"),
            "pred_vs_target_intercept": float("nan"),
            "target_vs_pred_slope": float("nan"),
            "target_vs_pred_intercept": float("nan"),
        }

    y_mean = float(y.mean())
    p_mean = float(p.mean())
    y_std = float(y.std(ddof=0))
    p_std = float(p.std(ddof=0))
    cov = float(np.mean((y - y_mean) * (p - p_mean)))
    pearson_r = cov / (y_std * p_std) if y_std > 0 and p_std > 0 else float("nan")
    pred_vs_target_slope = cov / (y_std ** 2) if y_std > 0 else float("nan")
    pred_vs_target_intercept = p_mean - pred_vs_target_slope * y_mean if np.isfinite(pred_vs_target_slope) else float("nan")
    target_vs_pred_slope = cov / (p_std ** 2) if p_std > 0 else float("nan")
    target_vs_pred_intercept = y_mean - target_vs_pred_slope * p_mean if np.isfinite(target_vs_pred_slope) else float("nan")
    return {
        "pearson_r": float(pearson_r),
        "pearson_r2": float(pearson_r ** 2) if np.isfinite(pearson_r) else float("nan"),
        "target_mean": y_mean,
        "pred_mean": p_mean,
        "target_std": y_std,
        "pred_std": p_std,
        "pred_target_std_ratio": float(p_std / y_std) if y_std > 0 else float("nan"),
        "bias": float(np.mean(p - y)),
        "pred_vs_target_slope": float(pred_vs_target_slope),
        "pred_vs_target_intercept": float(pred_vs_target_intercept),
        "target_vs_pred_slope": float(target_vs_pred_slope),
        "target_vs_pred_intercept": float(target_vs_pred_intercept),
    }


def catpred_split_label(split: str) -> str | None:
    """Map local split names to CatPred table labels."""

    if split == "test":
        return "Heldout"
    if split.startswith("ood"):
        return f"CLUSTER_{split.removeprefix('ood')}"
    return None


def write_catpred_results_csv(results_by_parameter: dict[str, dict[str, dict[str, float]]], output_file: Path) -> None:
    """Write a compact CatPred-style comparison table."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Parameter", "Test", "R2_mean", "R2_stderr", "MAE_mean", "MAE_stderr", "p1mag_mean", "p1mag_stderr", "rmse", "n"])
        for parameter in PARAM_ORDER:
            for test_name, metrics in results_by_parameter.get(parameter, {}).items():
                writer.writerow(
                    [
                        parameter,
                        test_name,
                        metrics.get("R2", float("nan")),
                        float("nan"),
                        metrics.get("MAE", float("nan")),
                        float("nan"),
                        metrics.get("p1mag", float("nan")),
                        float("nan"),
                        metrics.get("rmse", float("nan")),
                        metrics.get("n", 0),
                    ]
                )


def write_prediction_rows_csv(rows: list[dict[str, Any]], output_file: Path) -> None:
    """Write row-level ensemble predictions for ranking/compression audits."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parameter",
        "split",
        "catpred_test",
        "row_index",
        "source_row_index",
        "sequence_key",
        "smiles_key",
        "target",
        "pred",
        "error",
        "abs_error",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostics_csv(diagnostics: dict[str, dict[str, float]], output_file: Path) -> None:
    """Write per-split Pearson/compression diagnostics in a compact table."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parameter",
        "split",
        "ordinary_R2",
        "pearson_r",
        "pearson_r2",
        "target_mean",
        "pred_mean",
        "target_std",
        "pred_std",
        "pred_target_std_ratio",
        "bias",
        "pred_vs_target_slope",
        "pred_vs_target_intercept",
        "target_vs_pred_slope",
        "target_vs_pred_intercept",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for split_key, values in diagnostics.items():
            parameter, split = split_key.split("/", 1)
            writer.writerow({"parameter": parameter, "split": split, **values})


def load_eval_model(ckpt: dict[str, Any], device: torch.device) -> LANCEModel:
    """Reconstruct one LANCE seed model from checkpoint payload."""

    model = LANCEModel(LANCEConfig.from_dict(ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_checkpoints(
    checkpoints: list[Path],
    data_root: Path,
    esm_cache_path: Path | None,
    output: Path,
    batch_size: int,
    num_workers: int,
    device_arg: str,
    target_vram_gb: float,
    target_vram_tolerance_gb: float,
    predictions_output: Path | None = None,
    tasks: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Evaluate all available heldout/OOD splits for a checkpoint ensemble."""

    if not checkpoints:
        raise ValueError("at least one --checkpoint is required")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    device = torch.device(device_arg)
    ckpts = [load_torch(checkpoint) for checkpoint in checkpoints]
    first_args = ckpts[0]["args"]
    selected_tasks = normalize_parameters(tasks or first_args.get("tasks") or PARAM_ORDER)
    esm_cache = EmbeddingCache(esm_cache_path or Path(first_args["esm_cache"]), kind="ESM++")
    validate_data_root(data_root, parameters=selected_tasks)
    validate_cache_coverage(data_root, esm_cache, parameters=selected_tasks, splits=None)

    scalers = {name: TargetScaler.from_state(state) for name, state in ckpts[0]["scalers"].items()}
    missing_scalers = [parameter for parameter in selected_tasks if parameter not in scalers]
    if missing_scalers:
        raise KeyError(f"checkpoint is missing scalers for requested tasks: {missing_scalers}")
    models = [load_eval_model(ckpt, device) for ckpt in ckpts]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    summary: dict[str, Any] = {
        "checkpoints": [str(checkpoint) for checkpoint in checkpoints],
        "ensemble_size": len(checkpoints),
        "experiment_family": "LANCE",
        "splits": {},
        "catpred": {},
        "diagnostics": {},
    }
    prediction_rows: list[dict[str, Any]] = []

    for parameter in selected_tasks:
        summary["catpred"][parameter] = {}
        for split in available_splits(data_root, parameter):
            if split == "train":
                continue
            dataset = KineticsDataset(
                data_root,
                parameter,
                split,
                scalers[parameter],
                esm_cache,
                max_residues=int(first_args.get("max_residues", 1024)),
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_kinetics,
                pin_memory=False,
            )
            targets: list[float] = []
            preds: list[float] = []
            for batch in loader:
                batch = move_batch(batch, device)
                member_mu: list[torch.Tensor] = []
                for model in models:
                    out = model(batch)
                    member_mu.append(scalers[parameter].inverse_tensor(out["mu"]))
                pred_tensor = torch.stack(member_mu, dim=0).mean(dim=0)
                pred = pred_tensor.detach().cpu().tolist()
                target = batch["target"].detach().cpu().tolist()
                mask = batch["target_mask"].detach().cpu().tolist()
                row_indices = batch["row_index"].detach().cpu().tolist()
                source_row_indices = batch["source_row_index"].detach().cpu().tolist()
                sequence_keys = list(batch["sequence_key"])
                smiles_keys = list(batch["smiles_key"])
                catpred_label = catpred_split_label(split)
                for is_valid, y, p, row_index, source_row_index, sequence_key, smiles_key in zip(
                    mask,
                    target,
                    pred,
                    row_indices,
                    source_row_indices,
                    sequence_keys,
                    smiles_keys,
                ):
                    if is_valid:
                        targets.append(float(y))
                        preds.append(float(p))
                        prediction_rows.append(
                            {
                                "parameter": parameter,
                                "split": split,
                                "catpred_test": catpred_label or "",
                                "row_index": int(row_index),
                                "source_row_index": int(source_row_index),
                                "sequence_key": sequence_key,
                                "smiles_key": smiles_key,
                                "target": float(y),
                                "pred": float(p),
                                "error": float(p - y),
                                "abs_error": float(abs(p - y)),
                            }
                        )
            metrics = regression_metrics(targets, preds)
            diagnostics = prediction_diagnostics(targets, preds)
            diagnostics["ordinary_R2"] = metrics.get("R2", float("nan"))
            split_key = f"{parameter}/{split}"
            summary["splits"][split_key] = metrics
            summary["diagnostics"][split_key] = diagnostics
            catpred_label = catpred_split_label(split)
            if catpred_label is not None:
                summary["catpred"][parameter][catpred_label] = metrics

    memory = cuda_peak_memory(device)
    summary.update(memory)
    summary.update(vram_status(memory.get("cuda_peak_reserved_gb"), target_vram_gb, target_vram_tolerance_gb))
    catpred_csv = output.with_name(f"{output.stem}_catpred.csv")
    diagnostics_csv = output.with_name(f"{output.stem}_diagnostics.csv")
    write_catpred_results_csv(summary["catpred"], catpred_csv)
    write_diagnostics_csv(summary["diagnostics"], diagnostics_csv)
    summary["catpred_csv"] = str(catpred_csv)
    summary["diagnostics_csv"] = str(diagnostics_csv)
    if predictions_output is not None:
        write_prediction_rows_csv(prediction_rows, predictions_output)
        summary["predictions_csv"] = str(predictions_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/CatPred-DB"))
    parser.add_argument("--esm-cache", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/lance/evaluation_summary.json"))
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=PARAM_ORDER, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-vram-gb", type=float, default=0.0)
    parser.add_argument("--target-vram-tolerance-gb", type=float, default=0.5)
    args = parser.parse_args()
    evaluate_checkpoints(
        args.checkpoint,
        args.data_root,
        args.esm_cache,
        args.output,
        args.batch_size,
        args.num_workers,
        args.device,
        args.target_vram_gb,
        args.target_vram_tolerance_gb,
        args.predictions_output,
        args.tasks,
    )


if __name__ == "__main__":
    main()
