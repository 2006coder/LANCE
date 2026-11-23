"""Reproduce the same-reactant mean control from the released CatPred-DB split files."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESAMPLES = 20_000


@lru_cache(maxsize=None)
def valid_molecule(smiles):
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    return molecule is not None and molecule.GetNumHeavyAtoms() > 0


def load_partition(split_dir, split):
    columns = ["reactant_smiles", "sequence_99cluster", "log10kcat_max"]
    rows = pd.read_csv(split_dir / f"kcat-random_{split}.csv", usecols=columns)
    rows.index.name = "source_row_index"
    smiles = rows.reactant_smiles.fillna("").map(lambda value: " ".join(value.split()))
    rows = rows.assign(reactant_smiles=smiles).loc[smiles.map(valid_molecule)].copy()
    if rows.sequence_99cluster.isna().any() or not np.isfinite(rows.log10kcat_max).all():
        raise ValueError("Each retained row needs a cluster ID and a finite turnover label.")
    return rows


def control_predictions(split_dir, reference):
    train = load_partition(split_dir, "trainval")
    test = load_partition(split_dir, "test")
    test, reference = test.sort_index(), reference.sort_index()
    if not test.index.equals(reference.index) or len(train) != 20_833:
        raise ValueError("The supplied partitions differ from the archived model inputs.")
    np.testing.assert_allclose(test.log10kcat_max, reference.observed_log10, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(test.sequence_99cluster, reference.cluster99)
    label_groups = {smiles: group for smiles, group in train.groupby("reactant_smiles")}
    counts = []
    control = []
    for row in test.itertuples():
        group = label_groups.get(row.reactant_smiles)
        labels = (np.empty(0) if group is None else
                  group.loc[group.sequence_99cluster != row.sequence_99cluster, "log10kcat_max"].to_numpy())
        counts.append(len(labels))
        pred = reference.loc[row.Index, "backbone_log10"]
        if len(labels):
            weight = reference.loc[row.Index, "blend_alpha"]
            pred += weight * (labels.mean() - pred)
        control.append(pred)
    joined = reference.assign(eligible_training_rows=counts, mean_control_log10=control)
    np.testing.assert_array_equal(joined.eligible_training_rows > 0, joined.exact_match_eligible)
    return joined


def score(summary):
    count, sum_y, sum_y2, backbone_sq, control_sq, full_sq, backbone_abs, control_abs, full_abs = summary.T
    denom = sum_y2 - sum_y**2 / count
    if np.any(denom <= 0):
        raise ValueError("R-squared is undefined with constant targets.")
    return np.column_stack((1 - backbone_sq / denom, 1 - control_sq / denom, 1 - full_sq / denom,
                            backbone_abs / count, control_abs / count, full_abs / count,
                            (control_sq - full_sq) / denom, (control_abs - full_abs) / count))


def summarize(rows, rng):
    y = rows.observed_log10.to_numpy()
    predictions = rows[["backbone_log10", "mean_control_log10", "final_log10"]].to_numpy()
    errors = predictions - y[:, None]
    totals = np.column_stack((np.ones(len(y)), y, y**2, errors**2, np.abs(errors)))
    groups = pd.DataFrame(totals).groupby(rows.cluster99.to_numpy(), sort=True).sum().to_numpy()
    estimates = score(groups.sum(axis=0, keepdims=True))[0]
    draws = []
    for start in range(0, RESAMPLES, 128):
        n = min(128, RESAMPLES - start)
        index = rng.integers(len(groups), size=(n, len(groups)))
        draws.append(score(groups[index].sum(axis=1))[:, -2:])
    limits = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
    names = ("backbone_R2", "mean_control_R2", "full_R2", "backbone_MAE", "mean_control_MAE", "full_MAE",
             "full_R2_gain_vs_control", "full_MAE_gain_vs_control")
    return {"records": len(rows), "eligible": int(rows.exact_match_eligible.sum()), "clusters": len(groups),
            **dict(zip(names, estimates)), "R2_gain_low95": limits[0, 0], "R2_gain_high95": limits[1, 0],
            "MAE_gain_low95": limits[0, 1], "MAE_gain_high95": limits[1, 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path, help="CatPred-DB datasets/splits directory")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing turnover_retrieval_predictions.csv.gz")
    parser.add_argument("--output", type=Path, help="Output directory (defaults to the input directory)")
    args = parser.parse_args()
    output = args.output or args.data_dir
    reference = pd.read_csv(args.data_dir / "turnover_retrieval_predictions.csv.gz", index_col="source_row_index")
    rows = control_predictions(args.split_dir, reference)
    rng = np.random.default_rng(20261123)
    cohorts = {"Heldout": rows, "TargetOOD99": rows.loc[rows.target_ood99 == 1],
               "UnionOOD99": rows.loc[rows.union_ood99 == 1]}
    results = [{"cohort": name, **summarize(group, rng)} for name, group in cohorts.items()]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output / "reactant_mean_control.csv", index=False)
    rows[["eligible_training_rows", "mean_control_log10"]].to_csv(
        output / "reactant_mean_predictions.csv.gz", compression={"method": "gzip", "mtime": 0})


if __name__ == "__main__":
    main()
