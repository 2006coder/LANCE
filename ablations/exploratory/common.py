import hashlib
import json
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def configure_imports():
    sys.path.insert(0, str(PACKAGE / "runtime"))


configure_imports()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )
    os.replace(tmp, path)


def study():
    return read(PACKAGE / "configs/study.json")


def variant(name):
    return read(PACKAGE / "configs" / f"{name}.json")


def verify_source():
    for name, sha in read(PACKAGE / "SOURCE_PROVENANCE.json").items():
        if digest(PACKAGE / "runtime/LANCE" / name) != sha:
            raise RuntimeError("Changed source snapshot: " + name)


def safe_json(x):
    import math

    if isinstance(x, dict):
        return {str(k): safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe_json(v) for v in x]
    if hasattr(x, "item"):
        return safe_json(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
