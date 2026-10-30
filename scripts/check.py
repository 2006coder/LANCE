import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

subprocess.run([sys.executable, "tests/test_lance4r.py"], cwd=ROOT, env=ENV, check=True)
for study in ("exploratory", "pilot", "additive", "full_objective"):
    directory = ROOT / "ablations" / study
    for test in sorted(directory.glob("test_*.py")):
        subprocess.run([sys.executable, str(test)], cwd=directory, env=ENV, check=True)
