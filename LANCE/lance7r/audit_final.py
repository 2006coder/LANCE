#!/usr/bin/env python3
"""Independently audit a completed LANCE-7R prediction CSV.

This module intentionally does not import the evaluator or the standard
CatPred scoring script. It recomputes the public 18-cell scoreboard, protected
Km/Ki deltas, and the remaining kcat OOD99 error budget directly from saved
row-level predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CATPRED_REFERENCE = {
    ("kcat", "test"): {"R2": 0.6023, "MAE": 0.7138, "p1mag": 75.91},
    ("kcat", "ood99"): {"R2": 0.4042, "MAE": 0.9870, "p1mag": 61.43},
    ("km", "test"): {"R2": 0.6422, "MAE": 0.5582, "p1mag": 83.91},
    ("km", "ood99"): {"R2": 0.5332, "MAE": 0.6528, "p1mag": 79.04},
    ("ki", "test"): {"R2": 0.6319, "MAE": 0.8758, "p1mag": 67.84},
    ("ki", "ood99"): {"R2": 0.4462, "MAE": 0.9875, "p1mag": 60.50},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no predictions")
    required = {
        "parameter",
        "split",
        "target",
        "base_pred",
        "retrieval_pred",
        "retrieval_available_fraction",
        "blend_alpha",
        "pred",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def values(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=float)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "n": int(target.size),
        "R2": float(1.0 - np.sum(error**2) / denominator),
        "MAE": float(np.mean(np.abs(error))),
        "p1mag": float(100.0 * np.mean(np.abs(error) < 1.0)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "SSE": float(np.sum(error**2)),
        "TSS": denominator,
    }


def beats_reference(metric: str, value: float, reference: float) -> bool:
    return value < reference if metric == "MAE" else value > reference


def paired_bootstrap(
    target: np.ndarray,
    base: np.ndarray,
    final: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 713,
) -> dict[str, Any]:
    """Quantify paired row-sampling uncertainty for the kcat intervention."""

    generator = np.random.default_rng(seed)
    count = int(target.size)
    r2_gain = np.empty(samples, dtype=float)
    mae_gain = np.empty(samples, dtype=float)
    final_r2 = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        indices = generator.integers(0, count, count)
        sampled_target = target[indices]
        sampled_base = base[indices]
        sampled_final = final[indices]
        target_ss = float(
            np.sum((sampled_target - sampled_target.mean()) ** 2)
        )
        base_ss = float(np.sum((sampled_base - sampled_target) ** 2))
        final_ss = float(np.sum((sampled_final - sampled_target) ** 2))
        base_r2 = 1.0 - base_ss / target_ss
        final_r2[sample_index] = 1.0 - final_ss / target_ss
        r2_gain[sample_index] = final_r2[sample_index] - base_r2
        mae_gain[sample_index] = float(
            np.mean(np.abs(sampled_base - sampled_target))
            - np.mean(np.abs(sampled_final - sampled_target))
        )

    def interval(values: np.ndarray) -> dict[str, float]:
        low, median, high = np.quantile(values, (0.025, 0.5, 0.975))
        return {
            "p025": float(low),
            "median": float(median),
            "p975": float(high),
        }

    return {
        "samples": samples,
        "seed": seed,
        "r2_gain": interval(r2_gain),
        "mae_gain": interval(mae_gain),
        "probability_r2_gain_positive": float(np.mean(r2_gain > 0)),
        "probability_mae_gain_positive": float(np.mean(mae_gain > 0)),
        "probability_final_r2_above_fixed_catpred_reference": float(
            np.mean(final_r2 > CATPRED_REFERENCE[("kcat", "ood99")]["R2"])
        ),
    }


def raw_scoreboard(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["parameter"].lower(), row["split"])].append(row)

    table: list[dict[str, Any]] = []
    wins = 0
    for key, reference in CATPRED_REFERENCE.items():
        group = grouped.get(key)
        if not group:
            raise ValueError(f"prediction CSV has no rows for {key}")
        metrics = regression_metrics(values(group, "target"), values(group, "pred"))
        for metric in ("R2", "MAE", "p1mag"):
            won = beats_reference(metric, float(metrics[metric]), reference[metric])
            wins += int(won)
            table.append(
                {
                    "parameter": key[0],
                    "split": key[1],
                    "metric": metric,
                    "value": float(metrics[metric]),
                    "catpred": reference[metric],
                    "margin_in_favorable_direction": (
                        reference[metric] - float(metrics[metric])
                        if metric == "MAE"
                        else float(metrics[metric]) - reference[metric]
                    ),
                    "win": won,
                }
            )
    return table, wins


def kcat_ood_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    group = [
        row
        for row in rows
        if row["parameter"].lower() == "kcat" and row["split"] == "ood99"
    ]
    target = values(group, "target")
    base = values(group, "base_pred")
    retrieval = values(group, "retrieval_pred")
    final = values(group, "pred")
    available = values(group, "retrieval_available_fraction") > 0.0
    alpha = values(group, "blend_alpha")
    direction = available.astype(float) * (retrieval - base)
    reconstructed = base + alpha * direction

    denominator = float(np.dot(direction, direction))
    oracle_alpha = (
        float(np.dot(target - base, direction) / denominator)
        if denominator > 0
        else 0.0
    )
    oracle_alpha = float(np.clip(oracle_alpha, 0.0, 1.0))
    oracle = base + oracle_alpha * direction
    base_metrics = regression_metrics(target, base)
    final_metrics = regression_metrics(target, final)
    oracle_metrics = regression_metrics(target, oracle)
    catpred_sse_limit = (1.0 - CATPRED_REFERENCE[("kcat", "ood99")]["R2"]) * float(
        base_metrics["TSS"]
    )

    base_error = base - target
    final_error = final - target
    subsets = {}
    for name, mask in (("available", available), ("unavailable", ~available)):
        subsets[name] = {
            "base": regression_metrics(target[mask], base[mask]),
            "final": regression_metrics(target[mask], final[mask]),
        }

    quartiles = []
    for index, indices in enumerate(np.array_split(np.argsort(target), 4), start=1):
        quartiles.append(
            {
                "quartile": index,
                "n": int(indices.size),
                "target_min": float(target[indices].min()),
                "target_max": float(target[indices].max()),
                "base_bias": float(np.mean(base_error[indices])),
                "final_bias": float(np.mean(final_error[indices])),
                "base_sse": float(np.sum(base_error[indices] ** 2)),
                "final_sse": float(np.sum(final_error[indices] ** 2)),
            }
        )

    return {
        "base": base_metrics,
        "final": final_metrics,
        "r2_gain": float(final_metrics["R2"] - base_metrics["R2"]),
        "mae_gain": float(base_metrics["MAE"] - final_metrics["MAE"]),
        "retrieval_available_rows": int(available.sum()),
        "retrieval_unavailable_rows": int((~available).sum()),
        "max_prediction_reconstruction_error": float(
            np.max(np.abs(reconstructed - final))
        ),
        "least_squares_oracle_alpha_diagnostic_only": oracle_alpha,
        "oracle_alpha_metrics": oracle_metrics,
        "catpred_r2_sse_limit": catpred_sse_limit,
        "remaining_sse_reduction_needed": float(
            float(final_metrics["SSE"]) - catpred_sse_limit
        ),
        "rows_with_lower_squared_error": int(
            np.sum(final_error**2 < base_error**2)
        ),
        "rows_with_higher_squared_error": int(
            np.sum(final_error**2 > base_error**2)
        ),
        "rows_unchanged": int(np.sum(np.isclose(final_error, base_error))),
        "paired_bootstrap": paired_bootstrap(target, base, final),
        "subsets": subsets,
        "target_quartiles": quartiles,
    }


def novelty_curve(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    curve = []
    for split in ("ood20", "ood40", "ood60", "ood80", "ood99"):
        group = [
            row
            for row in rows
            if row["parameter"].lower() == "kcat" and row["split"] == split
        ]
        target = values(group, "target")
        base = regression_metrics(target, values(group, "base_pred"))
        final = regression_metrics(target, values(group, "pred"))
        curve.append(
            {
                "split": split,
                "n": int(target.size),
                "base_R2": base["R2"],
                "final_R2": final["R2"],
                "R2_gain": float(final["R2"] - base["R2"]),
                "base_MAE": base["MAE"],
                "final_MAE": final["MAE"],
                "MAE_gain": float(base["MAE"] - final["MAE"]),
            }
        )
    return curve


def audit(predictions_csv: Path) -> dict[str, Any]:
    rows = read_rows(predictions_csv)
    table, wins = raw_scoreboard(rows)
    protected = {}
    for parameter in ("km", "ki"):
        group = [row for row in rows if row["parameter"].lower() == parameter]
        protected[parameter] = float(
            np.max(np.abs(values(group, "pred") - values(group, "base_pred")))
        )
    return {
        "predictions_csv": str(predictions_csv),
        "raw_wins": wins,
        "raw_score": f"{wins} / 18",
        "scoreboard": table,
        "protected_task_max_abs_delta": protected,
        "kcat_ood99": kcat_ood_audit(rows),
        "kcat_novelty_curve": novelty_curve(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.predictions_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
