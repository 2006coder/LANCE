"""Compute the change in ensemble R² after removing both JEPA objectives."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def loss_contrasts(metrics):
    benchmark = metrics.loc[(metrics.stage == "memory") & (metrics.role == "benchmark")]
    scores = benchmark.pivot(index=["parameter", "split"], columns="variant", values="R2")
    scores = scores.reindex(columns=["full", "full_no_jepa", "full_no_sigreg", "full_no_jepa_no_sigreg"])
    if scores.empty or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Each task and benchmark split needs a finite R² for all four configurations.")
    return pd.DataFrame({
        "sigreg_retained": scores.full_no_jepa - scores.full,
        "sigreg_removed": scores.full_no_jepa_no_sigreg - scores.full_no_sigreg,
    }).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="Ensemble metrics CSV from the full-objective ablation")
    parser.add_argument("output", type=Path, help="Destination CSV for the unrounded contrasts")
    args = parser.parse_args()
    contrasts = loss_contrasts(pd.read_csv(args.metrics))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
