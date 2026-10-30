import argparse
import getpass
import json
import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from common import PACKAGE, digest, read, write


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--protocols", nargs="+", choices=["random", "cluster60"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--refit", action="store_true")
    p.add_argument("--schedule", choices=["sequential", "array"], default="sequential")
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--max-tasks", type=int, default=0)
    a = p.parse_args()
    root = Path(os.environ["RUN_ROOT"])
    s = read(PACKAGE / "configs/study.json")
    names = a.variants or s["variants"]
    protocols = a.protocols or s["protocols"]
    seeds = a.seeds or s["seeds"]
    if any(n not in s["variants"] for n in names) or len(set(names)) != len(names):
        raise ValueError("Invalid or duplicate variants")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Invalid seeds")
    if a.max_tasks < 0:
        raise ValueError("max-tasks must be nonnegative")
    if not 1 <= a.chunk_size <= 1000:
        raise ValueError("chunk-size must be between 1 and 1000")
    parallel = 1 if a.schedule == "sequential" else int(os.environ["MAX_PARALLEL"])
    if parallel < 1:
        raise ValueError("MAX_PARALLEL must be positive")
    s.update(variants=names, protocols=protocols, seeds=seeds)
    tasks = [
        dict(protocol=pr, variant=v, seed=seed)
        for pr in protocols
        for v in names
        for seed in seeds
    ]
    if root.exists():
        if not a.resume:
            raise FileExistsError(
                "RUN_ROOT exists; use --resume or choose a new RUN_ROOT"
            )
        if read(root / "study.json") != s or read(root / "tasks.json") != tasks:
            raise ValueError("Resume selections differ from the frozen study")
    source = root / "source"
    script = source / "job.sh"
    log = root / "logs"
    if a.submit:
        if a.resume:
            job_ids = [read(f)["job"] for f in (root / "submissions").glob("*.json")]
            if job_ids:
                queued = subprocess.check_output(
                    [
                        "squeue",
                        "--noheader",
                        "--user=" + getpass.getuser(),
                        "--format=%i",
                    ],
                    text=True,
                ).splitlines()
                active = set(job_ids) & {row.strip().split("_")[0] for row in queued}
                if active:
                    raise RuntimeError(
                        "Previous jobs are still queued or running. Finish or cancel those jobs before resubmitting."
                    )
        if not root.exists():
            root.mkdir(parents=True)
            shutil.copytree(
                PACKAGE,
                source,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.zip", "test_outputs", "validation_local"
                ),
            )
            shutil.copytree(source / "configs", root / "configs")
            write(root / "study.json", s)
            write(root / "tasks.json", tasks)
            write(
                root / "source_hashes.json",
                {
                    str(f.relative_to(source)): digest(f)
                    for f in source.rglob("*")
                    if f.is_file()
                },
            )
        for rel, sha in read(root / "source_hashes.json").items():
            if digest(source / rel) != sha:
                raise RuntimeError("Frozen source changed: " + rel)
        log.mkdir(exist_ok=True)
    else:
        print("Dry run: no files written and no jobs submitted.")
    defaults = [
        "sbatch",
        "--parsable",
        "--account=" + os.environ["SLURM_ACCOUNT"],
        "--partition=" + os.environ["SLURM_PARTITION"],
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=" + os.environ["SLURM_CPUS"],
        "--mem=" + os.environ["SLURM_MEM"],
        "--chdir=" + str(source),
        "--export=ALL",
        "--signal=B:USR1@600",
        "--output=" + str(log / "%x_%A_%a.out"),
        "--error=" + str(log / "%x_%A_%a.err"),
    ]
    submitted = []

    def submit(extra, stage, index_map=None):
        cmd = (
            defaults
            + extra
            + [str(script), str(source), stage]
            + ([str(index_map)] if index_map else [])
        )
        print(shlex.join(cmd), flush=True)
        if a.submit:
            raw = subprocess.check_output(cmd, text=True).strip()
            job = raw.split(";")[0]
            if not job.isdigit():
                raise RuntimeError("Invalid sbatch response: " + raw)
            submitted.append(dict(stage=stage, job=job, command=cmd))
            write(root / "submissions" / f"{job}.json", submitted[-1])
            return job
        submitted.append(dict(stage=stage))
        return str(900000 + len(submitted))

    pending = [
        i
        for i, t in enumerate(tasks)
        if not (
            root
            / ("refit_runs" if a.refit else "runs")
            / t["protocol"]
            / t["variant"]
            / f"seed{t['seed']}"
            / "complete.json"
        ).exists()
    ]
    if a.refit:
        for t in tasks:
            if not (
                root
                / "runs"
                / t["protocol"]
                / t["variant"]
                / f"seed{t['seed']}"
                / "complete.json"
            ).exists():
                raise RuntimeError("Refit requires every validation-stage run complete")
    scheduled = pending[: a.max_tasks] if a.max_tasks else pending
    preflight = submit(
        [
            "--job-name=l7r_preflight",
            "--gres=" + os.environ["SLURM_GRES"],
            "--time=02:00:00",
        ],
        "preflight",
    )
    ids = []
    chunk_size = 1 if a.schedule == "sequential" else a.chunk_size
    for start in range(0, len(scheduled), chunk_size):
        chunk = scheduled[start : start + chunk_size]
        index_map = root / "chunks" / f"{uuid.uuid4().hex}.json"
        if a.submit:
            write(index_map, chunk)
        array_options = (
            [f"--array=0-{len(chunk)-1}%{parallel}"] if a.schedule == "array" else []
        )
        ids.append(
            submit(
                [
                    "--job-name=l7r_ablation",
                    "--gres=" + os.environ["SLURM_GRES"],
                    "--time=" + os.environ["SLURM_TIME"],
                    "--dependency=afterok:" + (ids[-1] if ids else preflight),
                ]
                + array_options,
                "refit" if a.refit else "train",
                index_map,
            )
        )
    if not a.refit and len(scheduled) == len(pending):
        submit(
            [
                "--job-name=l7r_summary",
                "--partition=" + os.environ.get("SLURM_CPU_PARTITION", "standard"),
                "--mem=32G",
                "--cpus-per-task=4",
                "--time=12:00:00",
                "--dependency=afterok:" + (ids[-1] if ids else preflight),
            ],
            "summarize",
        )
    print(
        json.dumps(
            dict(
                backbone_runs=len(tasks),
                pending=len(pending),
                scheduled=len(scheduled),
                deferred=len(pending) - len(scheduled),
                protocols=protocols,
                seeds=seeds,
                schedule=a.schedule,
                training_jobs=len(scheduled),
                arrays=len(ids) if a.schedule == "array" else 0,
                maximum_simultaneous_gpus=parallel,
                requested_training_gpu_hours_upper_bound="number of pending runs × configured walltime; not an estimate of actual runtime",
            )
        )
    )


if __name__ == "__main__":
    main()
