import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import reference_root, PACKAGE, study, verify_release, write
from run import completed

KEYS = ["parameter", "split", "source_row_index", "sequence_key", "smiles_key"]
METRICS = ["R2", "MAE", "rmse", "p1mag"]


def load_rows(path):
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate prediction rows: " + str(path))
    return frame.set_index(KEYS).sort_index()


def check_rows(reference, candidate):
    if not reference.index.equals(candidate.index):
        raise ValueError("Evaluation rows differ from the original model")
    if not np.allclose(reference.target, candidate.target, atol=1e-6, rtol=0):
        raise ValueError("Evaluation targets differ from the original model")
    if not np.isfinite(candidate[["target", "pred"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite evaluation output")


def score(frame):
    y = frame.target.to_numpy(dtype=float)
    pred = frame.pred.to_numpy(dtype=float)
    errors = pred - y
    return {
        "R2": float(1 - np.sum(errors**2) / np.sum((y - y.mean()) ** 2)),
        "MAE": float(np.mean(abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p1mag": float(100 * np.mean(abs(errors) < 1)),
        "n": len(frame),
    }


def comparison_tables(metrics):
    comparisons = [
        ("effect_of_jepa_in_full", "full", "full_no_jepa"),
        ("effect_of_sigreg_in_full", "full", "full_no_sigreg"),
        ("effect_of_jepa_without_sigreg", "full_no_sigreg", "full_no_jepa_no_sigreg"),
        ("effect_of_sigreg_without_jepa", "full_no_jepa", "full_no_jepa_no_sigreg"),
        ("effect_of_jepa_and_sigreg", "full", "full_no_jepa_no_sigreg"),
    ]
    required = {"full", "full_no_jepa", "full_no_sigreg", "full_no_jepa_no_sigreg"}
    deltas = []
    interactions = []
    for (stage, task, split), group in metrics.groupby(["stage", "parameter", "split"]):
        if group.variant.duplicated().any() or set(group.variant) != required:
            raise ValueError("Each comparison needs all four configurations exactly once")
        values = group.set_index("variant")
        context = {
            "stage": stage,
            "parameter": task,
            "split": split,
            "role": "training_overlap" if split == "val" else "benchmark",
        }
        for label, a, b in comparisons:
            deltas.append(
                {
                    "contrast": label,
                    "a": a,
                    "b": b,
                    **context,
                    **{
                        metric + "_a_minus_b": values.loc[a, metric] - values.loc[b, metric]
                        for metric in METRICS
                    },
                }
            )
        interactions.append(
            {
                **context,
                "formula": "full - full_no_jepa - full_no_sigreg + full_no_jepa_no_sigreg",
                **{
                    metric + "_interaction": values.loc["full", metric]
                    - values.loc["full_no_jepa", metric]
                    - values.loc["full_no_sigreg", metric]
                    + values.loc["full_no_jepa_no_sigreg", metric]
                    for metric in METRICS
                },
            }
        )
    return pd.DataFrame(deltas), pd.DataFrame(interactions)


def summarize(root):
    records = []
    for stage, reference_name in [("backbone", "backbone"), ("memory", "full")]:
        reference = load_rows(
            reference_root() / reference_name / "per_row_predictions.csv"
        )
        frames = {"full": reference}
        for name in study()["variants"]:
            folder = root / name / "ensemble10" / stage
            if not completed(folder):
                raise RuntimeError("Evaluation is incomplete: " + str(folder))
            frames[name] = load_rows(folder / "per_row_predictions.csv")
            check_rows(reference, frames[name])
        for name, frame in frames.items():
            for (task, split), group in frame.reset_index().groupby(
                ["parameter", "split"], sort=True
            ):
                records.append(
                    {
                        "variant": name,
                        "stage": stage,
                        "parameter": task,
                        "split": split,
                        "role": "training_overlap" if split == "val" else "benchmark",
                        "ensemble_size": 10,
                        **score(group),
                    }
                )
    metrics = pd.DataFrame(records)
    out = root / "summary"
    out.mkdir(exist_ok=True)
    metrics.to_csv(out / "ensemble_metrics.csv", index=False)
    deltas, interactions = comparison_tables(metrics)
    deltas.to_csv(out / "contrasts.csv", index=False)
    interactions.to_csv(out / "interaction.csv", index=False)
    write(
        out / "complete.json",
        {
            "variants": study()["variants"],
            "reference": study()["reference"],
            "aggregation": "mean predictions over seeds 0-9 before scoring",
            "uncertainty": "No seed standard error is inferred from one ensemble.",
            "interaction": "Descriptive metric difference of differences, not a significance test.",
            "interaction_direction": "Positive favors complementarity for R2/p1mag; negative for MAE/rmse.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    a = p.parse_args()
    verify_release()
    summarize(a.run_root)
