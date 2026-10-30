import argparse
import fcntl
import getpass
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from common import reference_root, PACKAGE, digest, read, study, verify_release, write
from run import completed


def job_command(stage, root, code, dependencies=(), name=None, seeds=()):
    cpu = stage == "summarize"
    env = os.environ
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--export=ALL",
        "--kill-on-invalid-dep=yes",
        "--account=" + env["SLURM_ACCOUNT"],
        "--partition="
        + env.get(
            "SLURM_CPU_PARTITION" if cpu else "SLURM_PARTITION",
            "standard" if cpu else "gpu-h100",
        ),
        "--cpus-per-task=" + ("4" if cpu else env.get("SLURM_CPUS", "16")),
        "--mem=" + ("32G" if cpu else env.get("SLURM_MEM", "190G")),
        "--job-name=l7full_" + stage + ("_" + name if name else ""),
        "--output=" + str(root / "logs/%x_%A_%a.out"),
        "--error=" + str(root / "logs/%x_%A_%a.err"),
    ]
    if not cpu:
        command.append("--gres=" + env.get("SLURM_GRES", "gpu:h100:1"))
    hours = {
        "preflight": "02:00:00",
        "train": env.get("SLURM_TIME", "40:00:00"),
        "evaluate": env.get("SLURM_EVAL_TIME", "08:00:00"),
        "summarize": "02:00:00",
    }
    command.append("--time=" + hours[stage])
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(dependencies))
    if seeds:
        command.append("--array=" + ",".join(map(str, seeds)))
    command += [str(code / "job.sh"), str(code), stage]
    if name:
        command.append(name)
    return command


def plan(root, resume):
    tasks = []
    for name in study()["variants"]:
        seeds = [
            s
            for s in study()["seeds"]
            if not resume or not completed(root / name / f"seed{s}")
        ]
        eval_needed = seeds or not all(
            completed(root / name / "ensemble10" / stage)
            for stage in ("backbone", "memory")
        )
        tasks.append((name, seeds, bool(eval_needed)))
    return tasks


def submit(root, resume, dry_run):
    verify_release()
    code = root / "code"
    if dry_run:
        counter = 0

        def send(stage, deps=(), name=None, seeds=()):
            nonlocal counter
            counter += 1
            print(shlex.join(job_command(stage, root, code, deps, name, seeds)))
            return f"DRY{counter}"

    else:
        if not resume and (root / "submission.json").exists():
            raise RuntimeError("Study already submitted; use --resume")
        if not resume and any(root.iterdir()):
            existing = [p.name for p in root.iterdir() if p.name != ".submit.lock"]
            if existing:
                raise RuntimeError(
                    "Run directory is not empty; use --resume: " + str(existing)
                )
        if code.exists():
            if digest(code / "MANIFEST.json") != digest(PACKAGE / "MANIFEST.json"):
                raise RuntimeError(
                    "Existing run uses a different release; choose a new RUN_ROOT"
                )
            for name, expected in read(code / "MANIFEST.json").items():
                if digest(code / name) != expected:
                    raise RuntimeError("Frozen run source changed: " + name)
        else:
            shutil.copytree(
                PACKAGE, code, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store")
            )
        previous = (
            read(root / "submission.json")
            if (root / "submission.json").exists()
            else {"jobs": []}
        )
        ids = [job["id"] for job in previous["jobs"]]
        if ids:
            result = subprocess.run(
                ["squeue", "--noheader", "--user=" + getpass.getuser(), "--format=%i"],
                capture_output=True,
                text=True,
                check=True,
            )
            active = sorted(
                {line.strip().split("_")[0] for line in result.stdout.splitlines()}
                & set(ids)
            )
            if active:
                raise RuntimeError(
                    "Jobs are still active; resume would duplicate them: "
                    + ", ".join(active)
                )
        (root / "logs").mkdir(exist_ok=True)
        journal = {
            "manifest_sha256": digest(PACKAGE / "MANIFEST.json"),
            "jobs": previous["jobs"],
        }

        def send(stage, deps=(), name=None, seeds=()):
            command = job_command(stage, root, code, deps, name, seeds)
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            if not re.fullmatch(r"\d+", job_id):
                raise RuntimeError("Unexpected sbatch response: " + result.stdout)
            journal["jobs"].append(
                {
                    "id": job_id,
                    "stage": stage,
                    "variant": name,
                    "seeds": list(seeds),
                    "dependencies": list(deps),
                    "command": command,
                }
            )
            write(root / "submission.json", journal)
            print(f'{stage} {name or ""}: {job_id}', flush=True)
            return job_id

    tasks = plan(root, resume)
    if not any(needed for _, _, needed in tasks):
        send("summarize")
        return
    preflight = send("preflight")
    evaluations = []
    for name, seeds, needed in tasks:
        dependency = send("train", [preflight], name, seeds) if seeds else preflight
        if needed:
            evaluations.append(send("evaluate", [dependency], name))
    send("summarize", evaluations)


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    root = Path(os.environ["RUN_ROOT"]).resolve()
    os.environ["RUN_ROOT"] = str(root)
    if a.dry_run:
        submit(root, a.resume, True)
        return
    for key in ("ENV_PREFIX", "DATA_ROOT", "ESM_CACHE"):
        if not Path(os.environ[key]).exists():
            raise FileNotFoundError(f"{key}: {os.environ[key]}")
    for stage in ("full", "backbone"):
        predictions = reference_root() / stage / "per_row_predictions.csv"
        if not predictions.is_file():
            raise FileNotFoundError(f"Original evaluation artifact is required: {predictions}")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".submit.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(root, a.resume, False)


if __name__ == "__main__":
    main()
