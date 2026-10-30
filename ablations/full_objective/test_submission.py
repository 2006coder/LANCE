import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import PACKAGE, read
from submit import submit


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, SLURM_ACCOUNT="test-account")
        environment.start()
        self.addCleanup(environment.stop)

    def test_shell_syntax(self):
        for script in [*PACKAGE.glob("*.sh"), PACKAGE / "configs/cluster.env"]:
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_concurrent_graph_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            root.mkdir()
            commands = []

            def scheduler(command, **kwargs):
                commands.append(command)
                return subprocess.CompletedProcess(
                    command, 0, str(1000 + len(commands)) + "\n", ""
                )

            with patch("submit.verify_release"), patch(
                "submit.subprocess.run", scheduler
            ):
                submit(root, False, False)
            jobs = read(root / "submission.json")["jobs"]
            self.assertEqual(len(jobs), 8)
            training = [job for job in jobs if job["stage"] == "train"]
            self.assertEqual(len(training), 3)
            self.assertEqual(sum(len(job["seeds"]) for job in training), 30)
            preflight = jobs[0]["id"]
            for job in training:
                self.assertEqual(job["dependencies"], [preflight])
                self.assertEqual(job["seeds"], list(range(10)))
                self.assertIn("--array=0,1,2,3,4,5,6,7,8,9", job["command"])
                self.assertFalse(
                    any(
                        "%" in arg
                        for arg in job["command"]
                        if arg.startswith("--array=")
                    )
                )
                self.assertIn(str(root / "code/job.sh"), job["command"])
                self.assertIn(str(root / "code"), job["command"])
                self.assertNotIn("bash", job["command"])
            evals = [job for job in jobs if job["stage"] == "evaluate"]
            for job in evals:
                train = next(x for x in training if x["variant"] == job["variant"])
                self.assertEqual(job["dependencies"], [train["id"]])
            self.assertEqual(jobs[-1]["dependencies"], [job["id"] for job in evals])
            self.assertNotIn("--gres=gpu:h100:1", jobs[-1]["command"])
            with patch("submit.verify_release"), patch(
                "submit.subprocess.run", scheduler
            ):
                with self.assertRaisesRegex(RuntimeError, "already submitted"):
                    submit(root, False, False)
                with patch(
                    "submit.subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        ["squeue"], 0, jobs[0]["id"], ""
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "still active"):
                        submit(root, True, False)

            def resumed_scheduler(command, **kwargs):
                if command[0] == "squeue":
                    return subprocess.CompletedProcess(command, 0, "", "")
                return scheduler(command, **kwargs)

            def done(path):
                return "seed0" in path.parts or "seed1" in path.parts

            with patch("submit.verify_release"), patch("submit.completed", done), patch(
                "submit.subprocess.run", resumed_scheduler
            ):
                submit(root, True, False)
            new_jobs = read(root / "submission.json")["jobs"][8:]
            for job in new_jobs:
                if job["stage"] == "train":
                    self.assertEqual(job["seeds"], list(range(2, 10)))

    def test_failed_submission_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            count = 0

            def scheduler(command, **kwargs):
                nonlocal count
                count += 1
                if count == 3:
                    raise subprocess.CalledProcessError(
                        1, command, stderr="Submission limit"
                    )
                return subprocess.CompletedProcess(command, 0, str(2000 + count), "")

            with patch("submit.verify_release"), patch(
                "submit.subprocess.run", scheduler
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    submit(root, False, False)
            self.assertEqual(len(read(root / "submission.json")["jobs"]), 2)

    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "absent"
            with patch("submit.verify_release"):
                submit(root, False, True)
            self.assertFalse(root.exists())

    def test_job_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "environment/bin"
            binary.mkdir(parents=True)
            fake_python = binary / "python"
            fake_python.write_text(
                '#!/usr/bin/env bash\nprintf "SCRIPT_CWD=%s\\n" "$PWD"\nprintf "%s\\n" "$@"\n'
            )
            fake_python.chmod(0o755)
            env = dict(
                os.environ,
                ENV_PREFIX=str(binary.parent),
                RUN_ROOT=str(root / "run space"),
                DATA_ROOT=str(root / "data space"),
                ESM_CACHE=str(root / "cache.pt"),
                SLURM_ARRAY_TASK_ID="9",
            )
            spool = root / "slurm spool"
            spool.mkdir()
            script = spool / "slurm_script"
            script.write_text((PACKAGE / "job.sh").read_text())
            for stage in ["preflight", "train", "evaluate", "summarize"]:
                result = subprocess.run(
                    ["bash", str(script), str(PACKAGE), stage, "full_no_jepa"],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn(str(root / "run space"), result.stdout.splitlines())
                self.assertEqual(
                    result.stdout.splitlines()[0], "SCRIPT_CWD=" + str(PACKAGE)
                )
                if stage == "train":
                    self.assertIn("9", result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
