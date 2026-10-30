#!/usr/bin/env python3
"""Summarize strict LANCE-7R screens without opening benchmark splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LANCE.lance7r.data import load_torch


def checkpoint_row(path: Path) -> dict[str, Any]:
    checkpoint = load_torch(path)
    args = checkpoint["args"]
    metrics = checkpoint["metrics"]
    return {
        "run": path.parent.name,
        "variant": str(args["screen_variant"]),
        "seed": int(args["seed"]),
        "epoch": int(checkpoint["epoch"]),
        "blend_alpha": float(metrics["blend_alpha"]),
        "unseen99_base_r2": float(metrics["unseen99_base_r2"]),
        "unseen99_candidate_r2": float(metrics["unseen99_candidate_r2"]),
        "unseen99_r2_gain": float(metrics["unseen99_r2_gain"]),
        "unseen99_base_mae": float(metrics["unseen99_base_mae"]),
        "unseen99_candidate_mae": float(metrics["unseen99_candidate_mae"]),
        "unseen99_mae_gain": float(metrics["unseen99_mae_gain"]),
        "unseen99_available_fraction": float(
            metrics["cluster_unseen_available_fraction"]
        ),
        "selection_score": float(metrics["selection_score"]),
        "incumbent_sha256": str(checkpoint["incumbent_sha256"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    variants: dict[str, Any] = {}
    for variant, members in sorted(grouped.items()):
        r2_gains = np.asarray([row["unseen99_r2_gain"] for row in members])
        mae_gains = np.asarray([row["unseen99_mae_gain"] for row in members])
        candidate_r2 = np.asarray([row["unseen99_candidate_r2"] for row in members])
        aggregate = {
            "seeds": len(members),
            "mean_candidate_r2": float(candidate_r2.mean()),
            "mean_r2_gain": float(r2_gains.mean()),
            "min_r2_gain": float(r2_gains.min()),
            "positive_r2_seeds": int(np.sum(r2_gains > 0)),
            "mean_mae_gain": float(mae_gains.mean()),
            "mean_blend_alpha": float(
                np.mean([row["blend_alpha"] for row in members])
            ),
            "mean_unseen99_available_fraction": float(
                np.mean([row["unseen99_available_fraction"] for row in members])
            ),
        }
        aggregate["passes_screen"] = bool(
            aggregate["seeds"] >= 3
            and aggregate["mean_r2_gain"] >= 0.02
            and aggregate["positive_r2_seeds"] >= 2
            and aggregate["mean_mae_gain"] >= -0.005
        )
        variants[variant] = aggregate

    eligible = [
        (name, values)
        for name, values in variants.items()
        if values["passes_screen"]
    ]
    selected = (
        max(eligible, key=lambda item: item[1]["mean_r2_gain"])[0]
        if eligible
        else None
    )
    return {
        "protocol": "random-val sequence-99 unseen-cluster screen",
        "uses_heldout_or_ood_labels": False,
        "selection_rule": {
            "minimum_seeds": 3,
            "minimum_mean_r2_gain": 0.02,
            "minimum_positive_r2_seeds": 2,
            "minimum_mean_mae_gain": -0.005,
        },
        "variants": variants,
        "selected_variant": selected,
        "promote_to_trainval": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    missing = [path for path in args.checkpoint if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing screen checkpoints: {missing}")
    rows = [checkpoint_row(path) for path in args.checkpoint]
    report = summarize(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
