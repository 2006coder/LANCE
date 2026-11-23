"""Reproduce the paired benchmark intervals from the archived log-scale predictions."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEEDS = range(0, 100, 10)
RESAMPLES = 20_000


def cluster_totals(rows):
    y = rows["observed_log10"].to_numpy()
    lance_error = rows["lance7r_log10"].to_numpy() - y
    catpred = rows[[f"catpred_seed{seed}_log10" for seed in SEEDS]].to_numpy()
    catpred_error = catpred - y[:, None]
    contributions = np.column_stack((
        np.ones(len(rows)),
        y,
        y ** 2,
        (catpred_error ** 2).mean(axis=1) - lance_error ** 2,
        np.abs(catpred_error).mean(axis=1) - np.abs(lance_error),
        (np.abs(lance_error) < 1).astype(float) - (np.abs(catpred_error) < 1).mean(axis=1),
    ))
    if not np.isfinite(contributions).all() or rows["cluster99"].isna().any():
        raise ValueError("Every row needs finite predictions, an observation and a cluster assignment.")
    return pd.DataFrame(contributions).groupby(rows["cluster99"].to_numpy(), sort=True).sum().to_numpy()


def gains(totals):
    n, sum_y, sum_y2, squared_error_gain, abs_error_gain, hit_gain = totals.T
    variance_sum = sum_y2 - sum_y ** 2 / n
    if np.any(variance_sum <= 0):
        raise ValueError("R-squared is undefined for a constant-target sample.")
    return np.column_stack((squared_error_gain / variance_sum, abs_error_gain / n, 100 * hit_gain / n))


def intervals(rows, rng):
    grouped = cluster_totals(rows)
    observed = gains(grouped.sum(axis=0, keepdims=True))[0]
    draws = []
    for start in range(0, RESAMPLES, 128):
        size = min(128, RESAMPLES - start)
        sample = rng.integers(len(grouped), size=(size, len(grouped)))
        draws.append(gains(grouped[sample].sum(axis=1)))
    limits = np.quantile(np.vstack(draws), [0.025, 0.975], axis=0)
    return len(grouped), observed, limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing benchmark_predictions.csv.gz")
    parser.add_argument("--output", type=Path, help="Destination CSV (defaults to the input directory)")
    args = parser.parse_args()
    data = pd.read_csv(args.data_dir / "benchmark_predictions.csv.gz")
    output = args.output or args.data_dir / "paired_benchmark_intervals.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261123)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "cohort", "n", "clusters99", "metric", "gain", "lower95", "upper95"])
        for cohort in ("Heldout", "OOD99", "UnionOOD99"):
            for task in ("kcat", "km", "ki"):
                rows = data.loc[data["task"] == task]
                if cohort == "OOD99":
                    rows = rows.loc[rows["target_ood99"] == 1]
                elif cohort == "UnionOOD99":
                    rows = rows.loc[rows["union_ood99"] == 1]
                groups, estimate, bounds = intervals(rows, rng)
                for j, metric in enumerate(("R2", "MAE", "p1mag")):
                    writer.writerow([task, cohort, len(rows), groups, metric, estimate[j], bounds[0, j], bounds[1, j]])


if __name__ == "__main__":
    main()
