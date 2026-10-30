import argparse
import platform
import subprocess
import sys
from pathlib import Path

import torch
from common import PACKAGE, digest, read, verify_source, write
from datasets import FrameDataset, frame, prepare, scalers_for, verify_partition
from LANCE.data import PARAM_SPECS, EmbeddingCache, collate_kinetics, move_batch
from models import count, make_model
from run import loss_for


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--esm-cache", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    verify_source()
    s = read(a.run_root / "study.json")
    expected_data = read(PACKAGE / "assets/reference_data_hashes.json")
    from LANCE.data import split_path

    for key, expected in expected_data.items():
        task, split = key.split("/")
        if digest(split_path(a.data_root, task, split)) != expected:
            raise ValueError(
                "Dataset differs from the inspected manuscript data: " + key
            )
    partitions = a.run_root / "partitions"
    if not all((partitions / pr / "manifest.json").exists() for pr in s["protocols"]):
        if partitions.exists() and any(partitions.iterdir()):
            raise RuntimeError(
                "Incomplete partition preparation; inspect and remove partitions before retrying"
            )
        prepare(a.data_root, partitions, s["protocols"], s["split_seed"])
    for pr in s["protocols"]:
        manifest = verify_partition(partitions / pr)
        if manifest["input_hashes"] != expected_data:
            raise ValueError("Prepared partitions came from a different dataset")
    device = torch.device(a.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in relocated environment")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This recipe requires bfloat16 support")
    cache = EmbeddingCache(a.esm_cache, kind="ESM++")
    if cache.dim != 1152:
        raise ValueError(f"Expected ESM++ width 1152, got {cache.dim}")
    for pr in s["protocols"]:
        for t in PARAM_SPECS:
            for role in ("fit", "select", "calibrate", "report"):
                f = frame(partitions / pr, t, role)
                for seq in f.sequence.unique():
                    if not cache.has_text(seq):
                        raise KeyError("Missing ESM++ sequence: " + str(seq)[:20])
    folder = partitions / s["protocols"][0]
    scalers = scalers_for(folder, PARAM_SPECS)
    checks = []
    for name in s["variants"]:
        v = read(a.run_root / "configs" / f"{name}.json")
        model = make_model(v, 0, cache.dim).to(device)
        examples = []
        for t in v["tasks"]:
            ds = FrameDataset(
                frame(folder, t, "fit").head(3),
                t,
                scalers[t],
                cache,
                64,
                v["intervention"].get("products"),
                0,
            )
            examples.extend(ds[i] for i in range(len(ds)))
        batch = move_batch(collate_kinetics(examples), device)
        model.train()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = model(batch)
            loss = loss_for(out, batch, v)["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Preflight loss: " + name)
        loss.backward()
        if not all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        ):
            raise FloatingPointError("Preflight gradients: " + name)
        checks.append(
            dict(
                variant=name,
                parameters=count(model),
                parameters_with_gradient=sum(
                    p.numel() for p in model.parameters() if p.grad is not None
                ),
                loss=float(loss.detach()),
                forward_backward=True,
            )
        )
        del model, loss, out, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Exercise retrieval masks and interventions in the cluster's actual environment.
    import test_study

    test_study.test_retrieval_interventions(device=device)
    write(
        a.run_root / "preflight.json",
        dict(
            status="passed",
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            cache=str(a.esm_cache),
            cache_bytes=a.esm_cache.stat().st_size,
            cache_sha256=digest(a.esm_cache),
            checks=checks,
        ),
    )
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True
    )
    (a.run_root / "environment.txt").write_text(
        packages.stdout if packages.returncode == 0 else packages.stderr
    )
    print("Preflight passed", flush=True)


if __name__ == "__main__":
    main()
