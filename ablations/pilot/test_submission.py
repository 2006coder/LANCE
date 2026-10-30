import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import PACKAGE, read


def test_submission(schedule):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bins = root / "bin"
        bins.mkdir()
        scheduler = bins / "sbatch"
        scheduler.write_text(
            "#!/usr/bin/env python3\n"
            "import json,os,sys\n"
            "from pathlib import Path\n"
            "p=Path(os.environ['MOCK_LOG'])\n"
            "rows=p.read_text().splitlines() if p.exists() else []\n"
            "with p.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "print(str(7100+len(rows))+';mockcluster')\n"
        )
        scheduler.chmod(0o755)
        (bins / "squeue").write_text("#!/bin/sh\nexit 0\n")
        (bins / "squeue").chmod(0o755)
        env = dict(os.environ)
        env.update(
            PATH=str(bins) + os.pathsep + env["PATH"],
            MOCK_LOG=str(root / "jobs.jsonl"),
            RUN_ROOT=str(root / "runs"),
            SLURM_ACCOUNT="test-account",
            SLURM_PARTITION="test-gpu",
            SLURM_GRES="gpu:h100:1",
            SLURM_CPUS="16",
            SLURM_MEM="190G",
            SLURM_TIME="40:00:00",
            MAX_PARALLEL="2",
        )
        command = [sys.executable, str(PACKAGE / "submit.py")]
        selection = [
            "--variants",
            "full",
            "kinetic_only",
            "--protocols",
            "random",
            "--seeds",
            "0",
            "1",
            "--chunk-size",
            "3",
        ]
        if schedule == "array":
            selection += ["--schedule", "array"]
        subprocess.run(
            command + ["--dry-run"] + selection,
            env=env,
            check=True,
            capture_output=True,
        )
        assert not (root / "runs").exists()
        subprocess.run(
            command + ["--submit"] + selection, env=env, check=True, capture_output=True
        )
        jobs = [
            json.loads(line) for line in (root / "jobs.jsonl").read_text().splitlines()
        ]
        expected_maps = (
            [[0, 1, 2], [3]] if schedule == "array" else [[i] for i in range(4)]
        )
        assert len(jobs) == len(expected_maps) + 2
        for i, job in enumerate(jobs[1:], start=1):
            assert f"--dependency=afterok:{7100 + i - 1}" in job
        assert "--partition=standard" in jobs[-1]
        assert [read(job[-1]) for job in jobs[1:-1]] == expected_maps
        if schedule == "array":
            assert "--array=0-2%2" in jobs[1]
            assert "--array=0-0%2" in jobs[2]
        else:
            assert not any(arg.startswith("--array=") for job in jobs for arg in job)
        assert len(read(root / "runs/tasks.json")) == 4
        # A changed selection cannot reuse an existing experiment directory.
        bad = subprocess.run(
            command + ["--submit", "--resume", "--variants", "full"],
            env=env,
            capture_output=True,
        )
        assert bad.returncode != 0
        assert len((root / "jobs.jsonl").read_text().splitlines()) == len(jobs)
        (bins / "squeue").write_text("#!/bin/sh\nprintf '7101\\n'\n")
        active = subprocess.run(
            command + ["--submit", "--resume"] + selection,
            env=env,
            capture_output=True,
        )
        assert active.returncode != 0
        assert b"still queued or running" in active.stderr
        assert len((root / "jobs.jsonl").read_text().splitlines()) == len(jobs)
        (bins / "squeue").write_text("#!/bin/sh\nexit 0\n")
        complete = root / "runs/runs/random/full/seed0/complete.json"
        complete.parent.mkdir(parents=True)
        complete.write_text("{}")
        subprocess.run(
            command + ["--submit", "--resume", "--max-tasks", "2"] + selection,
            env=env,
            check=True,
            capture_output=True,
        )
        wave = [
            json.loads(line) for line in (root / "jobs.jsonl").read_text().splitlines()
        ][len(jobs) :]
        assert wave[0][-1] == "preflight"
        assert all(job[-2] == "train" for job in wave[1:])
        assert [index for job in wave[1:] for index in read(job[-1])] == [1, 2]
        for i, job in enumerate(wave[1:], start=1):
            assert f"--dependency=afterok:{7100 + len(jobs) + i - 1}" in job
        assert not any(job[-1] == "summarize" for job in wave)
        subprocess.run(["bash", "-n", str(PACKAGE / "submit.sh")], check=True)
        subprocess.run(["bash", "-n", str(PACKAGE / "job.sh")], check=True)


def test_job_dispatch():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bins = root / "env/bin"
        bins.mkdir(parents=True)
        python = bins / "python"
        python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
        python.chmod(0o755)
        env = dict(
            os.environ, PROJECT_ROOT=str(root), SLURM_ACCOUNT="test-account",
            ENV_PREFIX=str(root / "env"), RUN_ROOT=str(root / "runs")
        )
        env.pop("SLURM_ARRAY_TASK_ID", None)
        for stage, array_index in [("train", None), ("train", "2"), ("refit", None)]:
            current = dict(env)
            if array_index is not None:
                current["SLURM_ARRAY_TASK_ID"] = array_index
            result = subprocess.run(
                [
                    "bash",
                    str(PACKAGE / "job.sh"),
                    str(PACKAGE),
                    stage,
                    str(root / "map.json"),
                ],
                env=current,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            assert result[result.index("--index") + 1] == (array_index or "0")
            assert result[result.index("--index-map") + 1] == str(root / "map.json")
            assert ("--refit" in result) == (stage == "refit")


if __name__ == "__main__":
    test_submission("sequential")
    test_submission("array")
    test_job_dispatch()
    print("Submission checks passed")
