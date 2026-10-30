import hashlib
import json
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "runtime"))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def study():
    return read(PACKAGE / "configs/study.json")


def verify_release():
    for name, expected in read(PACKAGE / "MANIFEST.json").items():
        if digest(PACKAGE / name) != expected:
            raise RuntimeError("Package file changed: " + name)


def backbone_args(name, seed, root, data, cache, device="cuda"):
    if name not in study()["variants"] or seed not in study()["seeds"]:
        raise ValueError("Unknown configuration or seed")
    args = list(study()["backbone_args"])
    weights = read(PACKAGE / "configs" / f"{name}.json")["weights"]
    for key, value in weights.items():
        flag = "--" + key.replace("_", "-") + "-weight"
        if flag in args:
            args[args.index(flag) + 1] = str(value)
    args += [
        "--vrex-weight",
        "0",
        "--data-root",
        str(data),
        "--train-split",
        "trainval",
        "--esm-cache",
        str(cache),
        "--output-dir",
        str(root / name),
        "--run-name",
        f"seed{seed}/backbone",
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    return args


def memory_args(name, seed, root, data, cache, device="cuda"):
    return [
        "--protocol",
        "fixed-final",
        "--promotion-recipe",
        str(PACKAGE / "assets/promotion_recipe.json"),
        "--data-root",
        str(data),
        "--esm-cache",
        str(cache),
        "--incumbent-checkpoint",
        str(root / name / f"seed{seed}/backbone/checkpoint_best.pt"),
        "--incumbent-split",
        "trainval",
        "--memory-split",
        "trainval",
        "--output-dir",
        str(root / name),
        "--run-name",
        f"seed{seed}/memory",
        "--device",
        device,
        "--seed",
        str(seed),
        "--batch-size",
        "256",
        "--eval-batch-size",
        "128",
        "--num-workers",
        "0",
    ]


def reference_root():
    root = os.environ.get("LANCE_REFERENCE_ROOT")
    if not root:
        raise RuntimeError("Set LANCE_REFERENCE_ROOT to the original model evaluation directory containing full/ and backbone/.")
    return Path(root)
