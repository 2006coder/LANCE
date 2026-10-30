#!/usr/bin/env python3
"""Validation-only calibration and CatPred scoreboard for LANCE predictions.

This tool consumes the row-level CSV written by ``LANCE/evaluate.py`` when
``--predictions-output`` is provided.  It fits a simple affine calibration per
task on the random validation split only:

    calibrated_pred = slope * raw_pred + intercept

Then it reports raw, affine, and validation-gated scoreboards for the CatPred
headline cells.  The gate never looks at Heldout or OOD99 labels when deciding
whether to apply calibration; this prevents the easy but invalid mistake of
choosing post-processing based on test performance.

CatPred provenance:
    No CatPred code is copied here.  The fixed CatPred reference values are the
    public comparison numbers already used throughout this repository's LANCE
    docs.  Metric definitions mirror ``LANCE/evaluate.py::regression_metrics``:
    R2 and p1mag are higher-is-better, MAE is lower-is-better.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# Fixed CatPred headline table used by the experiment history.  Parameter names
# are lowercase to match LANCE CSV output; display labels are handled separately.
CATPRED_REFERENCE: dict[tuple[str, str], dict[str, float]] = {
    ("kcat", "Heldout"): {"R2": 0.6023, "MAE": 0.7138, "p1mag": 75.91},
    ("kcat", "CLUSTER_99"): {"R2": 0.4042, "MAE": 0.9870, "p1mag": 61.43},
    ("km", "Heldout"): {"R2": 0.6422, "MAE": 0.5582, "p1mag": 83.91},
    ("km", "CLUSTER_99"): {"R2": 0.5332, "MAE": 0.6528, "p1mag": 79.04},
    ("ki", "Heldout"): {"R2": 0.6319, "MAE": 0.8758, "p1mag": 67.84},
    ("ki", "CLUSTER_99"): {"R2": 0.4462, "MAE": 0.9875, "p1mag": 60.50},
}

PARAM_DISPLAY = {"kcat": "kcat", "km": "Km", "ki": "Ki"}
METRIC_DIRECTIONS = {"R2": 1, "MAE": -1, "p1mag": 1}


@dataclass(frozen=True)
class PredictionRow:
    """One evaluated row from ``LANCE/evaluate.py``."""

    parameter: str
    split: str
    catpred_test: str
    target: float
    pred: float


@dataclass(frozen=True)
class AffineCalibration:
    """Per-task affine map fitted on validation predictions."""

    slope: float = 1.0
    intercept: float = 0.0
    fit_n: int = 0
    applied_by_gate: bool = False
    gate_reason: str = "identity"

    def apply(self, pred: np.ndarray) -> np.ndarray:
        return self.slope * pred + self.intercept


def canonical_parameter(value: str) -> str:
    """Normalize task labels from CSV or CLI."""

    name = str(value).strip().lower()
    if name not in {"kcat", "km", "ki"}:
        raise ValueError(f"unknown parameter {value!r}")
    return name


def catpred_label_from_split(split: str, explicit_label: str) -> str:
    """Recover the CatPred table label for rows written by older/newer evals."""

    explicit = str(explicit_label or "").strip()
    if explicit:
        return explicit
    split = str(split).strip()
    if split == "test":
        return "Heldout"
    if split.startswith("ood"):
        return f"CLUSTER_{split.removeprefix('ood')}"
    return split


def finite_float(value: str, *, column: str) -> float:
    """Parse a finite float with a useful error message."""

    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} must contain floats, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"column {column!r} must be finite, got {value!r}")
    return out


def read_prediction_rows(path: Path) -> list[PredictionRow]:
    """Read row-level predictions produced by ``LANCE/evaluate.py``."""

    rows: list[PredictionRow] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"parameter", "split", "target", "pred"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for raw in reader:
            parameter = canonical_parameter(raw["parameter"])
            split = str(raw["split"]).strip()
            rows.append(
                PredictionRow(
                    parameter=parameter,
                    split=split,
                    catpred_test=catpred_label_from_split(split, raw.get("catpred_test", "")),
                    target=finite_float(raw["target"], column="target"),
                    pred=finite_float(raw["pred"], column="pred"),
                )
            )
    if not rows:
        raise ValueError(f"{path} contained no prediction rows")
    return rows


def regression_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute the same headline metrics as ``LANCE/evaluate.py``."""

    if target.size == 0:
        return {"R2": float("nan"), "MAE": float("nan"), "p1mag": float("nan"), "n": 0}
    mae = float(np.mean(np.abs(pred - target)))
    denom = float(np.sum((target - target.mean()) ** 2))
    r2 = float(1.0 - np.sum((pred - target) ** 2) / denom) if denom > 0 else float("nan")
    p1mag = float(np.mean(np.abs(target - pred) < 1.0) * 100.0)
    return {"R2": r2, "MAE": mae, "p1mag": p1mag, "n": int(target.size)}


def pearson_r2(target: np.ndarray, pred: np.ndarray) -> float:
    """Return the affine-calibration R2 ceiling for one row group."""

    if target.size < 2:
        return float("nan")
    target_std = float(target.std(ddof=0))
    pred_std = float(pred.std(ddof=0))
    if target_std <= 0 or pred_std <= 0:
        return float("nan")
    cov = float(np.mean((target - target.mean()) * (pred - pred.mean())))
    r = cov / (target_std * pred_std)
    return float(r * r)


def fit_affine(rows: list[PredictionRow], *, min_n: int) -> AffineCalibration:
    """Fit target ~= slope * pred + intercept on validation rows."""

    if len(rows) < min_n:
        return AffineCalibration(fit_n=len(rows), gate_reason=f"too_few_val_rows<{min_n}")
    target = np.asarray([row.target for row in rows], dtype=float)
    pred = np.asarray([row.pred for row in rows], dtype=float)
    pred_var = float(np.var(pred))
    if pred_var <= 1.0e-12:
        return AffineCalibration(fit_n=len(rows), gate_reason="near_constant_val_predictions")
    cov = float(np.mean((pred - pred.mean()) * (target - target.mean())))
    slope = cov / pred_var
    intercept = float(target.mean() - slope * pred.mean())
    if not math.isfinite(slope) or not math.isfinite(intercept):
        return AffineCalibration(fit_n=len(rows), gate_reason="nonfinite_fit")
    return AffineCalibration(slope=float(slope), intercept=intercept, fit_n=len(rows), gate_reason="fit")


def metric_improved(raw_value: float, new_value: float, metric: str, eps: float = 1.0e-12) -> bool:
    """Direction-aware strict improvement test."""

    if not (math.isfinite(raw_value) and math.isfinite(new_value)):
        return False
    direction = METRIC_DIRECTIONS[metric]
    return direction * (new_value - raw_value) > eps


def metric_regressed(raw_value: float, new_value: float, metric: str, eps: float = 1.0e-12) -> bool:
    """Direction-aware strict regression test."""

    if not (math.isfinite(raw_value) and math.isfinite(new_value)):
        return False
    direction = METRIC_DIRECTIONS[metric]
    return direction * (new_value - raw_value) < -eps


def gate_calibration(
    calibration: AffineCalibration,
    val_rows: list[PredictionRow],
    *,
    gate_rule: str,
    min_improved_metrics: int,
) -> AffineCalibration:
    """Decide whether to apply calibration using validation rows only."""

    if gate_rule == "never":
        return AffineCalibration(calibration.slope, calibration.intercept, calibration.fit_n, False, "gate_never")
    if gate_rule == "always":
        return AffineCalibration(calibration.slope, calibration.intercept, calibration.fit_n, True, "gate_always")
    if not val_rows:
        return AffineCalibration(calibration.slope, calibration.intercept, calibration.fit_n, False, "no_val_rows")

    target = np.asarray([row.target for row in val_rows], dtype=float)
    raw_pred = np.asarray([row.pred for row in val_rows], dtype=float)
    cal_pred = calibration.apply(raw_pred)
    raw_metrics = regression_metrics(target, raw_pred)
    cal_metrics = regression_metrics(target, cal_pred)
    improved = sum(metric_improved(raw_metrics[m], cal_metrics[m], m) for m in METRIC_DIRECTIONS)
    regressed = sum(metric_regressed(raw_metrics[m], cal_metrics[m], m) for m in METRIC_DIRECTIONS)

    if gate_rule == "val_2of3":
        apply = improved >= min_improved_metrics
        reason = f"val_improved_{improved}_regressed_{regressed}"
    elif gate_rule == "val_2of3_no_r2_regress":
        apply = improved >= min_improved_metrics and not metric_regressed(raw_metrics["R2"], cal_metrics["R2"], "R2")
        reason = f"val_improved_{improved}_regressed_{regressed}_r2_guard"
    else:
        raise ValueError(f"unknown gate rule {gate_rule!r}")

    return AffineCalibration(calibration.slope, calibration.intercept, calibration.fit_n, bool(apply), reason)


def group_rows(rows: Iterable[PredictionRow]) -> dict[tuple[str, str], list[PredictionRow]]:
    """Group rows by parameter and CatPred test label."""

    grouped: dict[tuple[str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.parameter, row.catpred_test)].append(row)
    return dict(grouped)


def metrics_for_rows(rows: list[PredictionRow], calibration: AffineCalibration | None = None) -> dict[str, float]:
    """Compute metrics for raw or calibrated predictions."""

    target = np.asarray([row.target for row in rows], dtype=float)
    pred = np.asarray([row.pred for row in rows], dtype=float)
    if calibration is not None:
        pred = calibration.apply(pred)
    metrics = regression_metrics(target, pred)
    metrics["pearson_r2"] = pearson_r2(target, pred)
    return metrics


def beats_catpred(value: float, reference: float, metric: str) -> bool:
    """Return whether one metric beats the fixed CatPred reference."""

    if not math.isfinite(value):
        return False
    if metric == "MAE":
        return value < reference
    return value > reference


def build_scoreboard(
    rows: list[PredictionRow],
    calibrations: dict[str, AffineCalibration],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build per-cell rows plus compact win totals."""

    grouped = group_rows(rows)
    table: list[dict[str, object]] = []
    totals = {
        "raw_wins": 0,
        "affine_wins": 0,
        "gated_wins": 0,
        "total_cells": 0,
        "heldout_gated_wins": 0,
        "ood99_gated_wins": 0,
    }

    for key, reference in CATPRED_REFERENCE.items():
        parameter, test_name = key
        cell_rows = grouped.get(key, [])
        if not cell_rows:
            continue
        calibration = calibrations.get(parameter, AffineCalibration())
        gated_calibration = calibration if calibration.applied_by_gate else None
        raw_metrics = metrics_for_rows(cell_rows, None)
        affine_metrics = metrics_for_rows(cell_rows, calibration)
        gated_metrics = metrics_for_rows(cell_rows, gated_calibration)
        for metric in ("R2", "MAE", "p1mag"):
            raw_win = beats_catpred(raw_metrics[metric], reference[metric], metric)
            affine_win = beats_catpred(affine_metrics[metric], reference[metric], metric)
            gated_win = beats_catpred(gated_metrics[metric], reference[metric], metric)
            totals["raw_wins"] += int(raw_win)
            totals["affine_wins"] += int(affine_win)
            totals["gated_wins"] += int(gated_win)
            totals["total_cells"] += 1
            if gated_win and test_name == "Heldout":
                totals["heldout_gated_wins"] += 1
            if gated_win and test_name == "CLUSTER_99":
                totals["ood99_gated_wins"] += 1
            table.append(
                {
                    "parameter": PARAM_DISPLAY[parameter],
                    "test": test_name,
                    "metric": metric,
                    "catpred": reference[metric],
                    "raw": raw_metrics[metric],
                    "affine": affine_metrics[metric],
                    "gated": gated_metrics[metric],
                    "raw_win": raw_win,
                    "affine_win": affine_win,
                    "gated_win": gated_win,
                    "pearson_r2": raw_metrics["pearson_r2"],
                }
            )

    totals["raw_score"] = f"{totals['raw_wins']} / {totals['total_cells']}"
    totals["affine_score"] = f"{totals['affine_wins']} / {totals['total_cells']}"
    totals["gated_score"] = f"{totals['gated_wins']} / {totals['total_cells']}"
    totals["gated_heldout_score"] = f"{totals['heldout_gated_wins']} / 9"
    totals["gated_ood99_score"] = f"{totals['ood99_gated_wins']} / 9"
    return table, totals


def write_table_csv(table: list[dict[str, object]], path: Path) -> None:
    """Write the metric-level scoreboard."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "parameter",
        "test",
        "metric",
        "catpred",
        "raw",
        "affine",
        "gated",
        "raw_win",
        "affine_win",
        "gated_win",
        "pearson_r2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run validation-only calibration and return a JSON-serializable summary."""

    rows = read_prediction_rows(args.predictions_csv)
    selected = {canonical_parameter(parameter) for parameter in args.tasks} if args.tasks else None
    if selected is not None:
        rows = [row for row in rows if row.parameter in selected]
    if not rows:
        raise ValueError("no rows remain after task filtering")

    val_rows_by_task: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in rows:
        if row.split == args.validation_split:
            val_rows_by_task[row.parameter].append(row)

    calibrations: dict[str, AffineCalibration] = {}
    for parameter in sorted({row.parameter for row in rows}):
        fit = fit_affine(val_rows_by_task.get(parameter, []), min_n=args.min_val_rows)
        calibrations[parameter] = gate_calibration(
            fit,
            val_rows_by_task.get(parameter, []),
            gate_rule=args.gate_rule,
            min_improved_metrics=args.min_improved_metrics,
        )

    table, totals = build_scoreboard(rows, calibrations)
    summary: dict[str, object] = {
        "predictions_csv": str(args.predictions_csv),
        "validation_split": args.validation_split,
        "gate_rule": args.gate_rule,
        "totals": totals,
        "calibrations": {
            parameter: {
                "slope": calibration.slope,
                "intercept": calibration.intercept,
                "fit_n": calibration.fit_n,
                "applied_by_gate": calibration.applied_by_gate,
                "gate_reason": calibration.gate_reason,
            }
            for parameter, calibration in calibrations.items()
        },
        "table": table,
    }

    if args.output_csv:
        write_table_csv(table, args.output_csv)
        summary["output_csv"] = str(args.output_csv)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def self_test() -> None:
    """Small deterministic sanity test for metric math and gating."""

    rng = np.random.default_rng(7)
    target = np.linspace(-2.0, 2.0, 40)
    raw = 0.5 * target + 1.25 + rng.normal(0.0, 0.03, size=target.shape)
    val_rows = [
        PredictionRow("kcat", "val", "val", float(y), float(p))
        for y, p in zip(target[:20], raw[:20])
    ]
    test_rows = [
        PredictionRow("kcat", "test", "Heldout", float(y), float(p))
        for y, p in zip(target[20:], raw[20:])
    ]
    fit = fit_affine(val_rows, min_n=5)
    gated = gate_calibration(fit, val_rows, gate_rule="val_2of3", min_improved_metrics=2)
    raw_metrics = metrics_for_rows(test_rows)
    cal_metrics = metrics_for_rows(test_rows, gated)
    if not gated.applied_by_gate:
        raise AssertionError("self-test expected calibration gate to apply")
    if cal_metrics["MAE"] >= raw_metrics["MAE"]:
        raise AssertionError("self-test expected calibration to lower MAE")
    if cal_metrics["R2"] > raw_metrics["pearson_r2"] + 1.0e-9:
        raise AssertionError("affine calibrated R2 exceeded Pearson r2 ceiling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=Path, required=False)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=("kcat", "km", "ki"), default=None)
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--min-val-rows", type=int, default=20)
    parser.add_argument("--gate-rule", choices=("val_2of3", "val_2of3_no_r2_regress", "always", "never"), default="val_2of3")
    parser.add_argument("--min-improved-metrics", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.predictions_csv is None:
        parser.error("--predictions-csv is required unless --self-test is used")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("calibrate_and_score self-test passed")
        return
    summary = run(args)
    print(json.dumps({key: summary[key] for key in ("totals", "calibrations")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
