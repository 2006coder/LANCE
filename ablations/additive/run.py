import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

from common import (
    PACKAGE,
    backbone_args,
    digest,
    memory_args,
    read,
    study,
    verify_release,
    write,
)


def execute(module, args):
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(PACKAGE / "runtime") + os.pathsep + env.get("PYTHONPATH", "")
    )
    subprocess.run(
        [sys.executable, "-u", "-m", module, *map(str, args)], env=env, check=True
    )


def completed(folder):
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    for name, expected in read(marker)["files"].items():
        if not (folder / name).is_file() or digest(folder / name) != expected:
            raise RuntimeError("Completed output changed: " + str(folder / name))
    return True


def mark(folder, names):
    write(
        folder / "complete.json",
        {"files": {name: digest(folder / name) for name in names}},
    )


def train(a):
    folder = a.run_root / a.variant / f"seed{a.seed}"
    for stage, module, args in [
        (
            "backbone",
            "LANCE.train",
            backbone_args(a.variant, a.seed, a.run_root, a.data_root, a.esm_cache),
        ),
        (
            "memory",
            "LANCE.lance7r.train",
            memory_args(a.variant, a.seed, a.run_root, a.data_root, a.esm_cache),
        ),
    ]:
        output = folder / stage
        if completed(output):
            print("Already completed:", output, flush=True)
            continue
        execute(module, args)
        from LANCE.data import load_torch

        checkpoint = load_torch(output / "checkpoint_best.pt")
        if stage == "memory" and checkpoint["epoch"] != study()["memory_epochs"] - 1:
            raise RuntimeError("Incomplete memory training")
        mark(output, ["checkpoint_best.pt", "config.json"])
    mark(folder, ["backbone/complete.json", "memory/complete.json"])


def evaluate(a):
    for stage, module in [
        ("backbone", "LANCE.evaluate"),
        ("memory", "LANCE.lance7r.evaluate"),
    ]:
        output = a.run_root / a.variant / "ensemble10" / stage
        if completed(output):
            continue
        checkpoints = []
        for seed in study()["seeds"]:
            folder = a.run_root / a.variant / f"seed{seed}" / stage
            if not completed(folder):
                raise RuntimeError("Missing completed seed: " + str(folder))
            checkpoints.append(folder / "checkpoint_best.pt")
        output.mkdir(parents=True, exist_ok=True)
        execute(
            module,
            [
                "--checkpoint",
                *checkpoints,
                "--data-root",
                a.data_root,
                "--esm-cache",
                a.esm_cache,
                "--output",
                output / "evaluation_summary.json",
                "--predictions-output",
                output / "per_row_predictions.csv",
                "--batch-size",
                "256",
                "--num-workers",
                "0",
                "--device",
                "cuda",
            ],
        )
        mark(
            output,
            [
                "evaluation_summary.json",
                "evaluation_summary_catpred.csv",
                "per_row_predictions.csv",
            ],
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["train", "evaluate"])
    p.add_argument("--variant", choices=study()["variants"], required=True)
    p.add_argument("--seed", type=int, choices=study()["seeds"])
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--esm-cache", type=Path, required=True)
    a = p.parse_args()
    verify_release()
    preflight = read(a.run_root / "preflight.json")
    if preflight["manifest_sha256"] != digest(PACKAGE / "MANIFEST.json"):
        raise RuntimeError("Preflight belongs to a different package")
    if preflight["data_root"] != str(a.data_root) or preflight["esm_cache"] != str(
        a.esm_cache
    ):
        raise RuntimeError("Input paths differ from preflight")
    cache_stat = a.esm_cache.stat()
    if (cache_stat.st_size, cache_stat.st_mtime_ns) != (
        preflight["cache_size"],
        preflight["cache_mtime_ns"],
    ):
        raise RuntimeError("ESM++ cache changed after preflight")
    from preflight import check_data

    check_data(a.data_root)
    if a.stage == "train" and a.seed is None:
        p.error("--seed is required for training")
    folder = (
        a.run_root
        / a.variant
        / (f"seed{a.seed}" if a.stage == "train" else "ensemble10")
    )
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (train if a.stage == "train" else evaluate)(a)


if __name__ == "__main__":
    main()
