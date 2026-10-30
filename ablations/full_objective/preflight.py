import argparse
import gc
import platform
import sys
from pathlib import Path

from common import reference_root, PACKAGE, backbone_args, digest, read, study, verify_release, write


def check_data(data):
    expected = read(PACKAGE / "assets/data_hashes.json")
    for name, value in expected.items():
        if not (data / name).is_file() or digest(data / name) != value:
            raise RuntimeError("Original dataset does not match: " + str(data / name))
    actual = {
        str(p.relative_to(data))
        for p in (data / "datasets/splits").glob("*.csv")
        if p.name.startswith(("kcat-", "km-", "ki-")) and "trainvaltest" not in p.name
    }
    if actual != set(expected):
        raise RuntimeError(
            "Unexpected benchmark split files: " + str(actual ^ set(expected))
        )


def check_reference_rows(data):
    import numpy as np
    import pandas as pd
    from LANCE.data import (
        PARAM_SPECS,
        clean,
        filter_invalid_smiles_frame,
        load_split_frame,
        stable_key,
    )
    from summarize import KEYS, check_rows, load_rows

    reference = load_rows(reference_root() / "full/per_row_predictions.csv")
    records = []
    for task, split in reference.index.droplevel(KEYS[2:]).unique():
        spec = PARAM_SPECS[task]
        frame = filter_invalid_smiles_frame(
            load_split_frame(data, task, split), spec["smiles"]
        )
        for _, row in frame.iterrows():
            target = float(row[spec["target"]])
            if np.isfinite(target):
                records.append(
                    {
                        "parameter": task,
                        "split": split,
                        "source_row_index": int(row["_source_row_index"]),
                        "sequence_key": stable_key(clean(row["sequence"])),
                        "smiles_key": stable_key(clean(row[spec["smiles"]])),
                        "target": float(np.float32(target)),
                        "pred": 0.0,
                    }
                )
    check_rows(reference, pd.DataFrame(records).set_index(KEYS).sort_index())
    backbone = load_rows(reference_root() / "backbone/per_row_predictions.csv")
    check_rows(reference, backbone)


def check_retrieval(data, cache, scalers, device):
    import torch
    from LANCE.data import move_batch
    from LANCE.lance7r.retrieval import LabelAnchoredRetriever, RetrievalConfig
    from LANCE.lance7r.retrieval_data import (
        RetrievalKineticsDataset,
        build_retrieval_memory,
        collate_retrieval_training,
    )

    dataset = RetrievalKineticsDataset(
        data, "kcat", "trainval", scalers["kcat"], cache, max_residues=1024
    )
    memory = build_retrieval_memory(dataset)
    recipe = read(PACKAGE / "assets/promotion_recipe.json")
    model = LabelAnchoredRetriever(
        RetrievalConfig.from_dict(recipe["retrieval_config"]), memory
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    for start in range(0, len(dataset), 256):
        batch = move_batch(
            collate_retrieval_training(
                [dataset[i] for i in range(start, min(start + 256, len(dataset)))]
            ),
            device,
        )
        output = model(batch)
        valid = batch["target_mask"] & output["retrieval_available"]
        if not valid.any():
            continue
        prediction, target = output["retrieval_mu"][valid], batch["target_z"][valid]
        loss = torch.nn.functional.mse_loss(
            prediction, target
        ) + 0.15 * torch.nn.functional.smooth_l1_loss(prediction, target, beta=0.5)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite retrieval loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return {"memory_rows": memory.size, "valid_query_rows": int(valid.sum())}
    raise RuntimeError("No cross-cluster retrieval neighbors in trainval")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--esm-cache", type=Path, required=True)
    a = p.parse_args()
    verify_release()
    check_data(a.data_root)
    check_reference_rows(a.data_root)
    import torch
    from LANCE.data import (
        EmbeddingCache,
        build_scalers,
        move_batch,
        validate_cache_coverage,
    )
    from LANCE.losses import LANCELossWeights, lance_total_loss
    from LANCE.model import LANCEConfig, LANCEModel
    from LANCE.train import build_loaders, parse_args, seed_all
    from LANCE.lance7r.incumbent import (
        LANCEConfig as FrozenConfig,
        LANCEModel as FrozenModel,
    )

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with bfloat16 support is required")
    cache = EmbeddingCache(a.esm_cache, kind="ESM++")
    if cache.dim != 1152:
        raise RuntimeError(f"Expected ESM++ width 1152; got {cache.dim}")
    validate_cache_coverage(a.data_root, cache, splits=None)
    validate_cache_coverage(a.data_root, cache, splits=["trainval"])
    scalers = build_scalers(a.data_root, "trainval")
    checks = []
    for name in study()["variants"]:
        sys.argv = [
            "train",
            *backbone_args(name, 0, a.run_root, a.data_root, a.esm_cache),
        ]
        args = parse_args()
        seed_all(0)
        loader, _, _ = build_loaders(args, scalers, cache)
        config = read(PACKAGE / "configs" / f"{name}.json")
        model = LANCEModel(LANCEConfig(**config["model"])).cuda()
        frozen = FrozenModel(FrozenConfig.from_dict(config["model"]))
        frozen.load_state_dict(model.state_dict(), strict=True)
        del frozen
        batch = move_batch(next(iter(loader)), torch.device("cuda"))
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            parts = lance_total_loss(
                out,
                batch["target_z"],
                target_mask=batch["target_mask"],
                weights=LANCELossWeights(**config["weights"]),
                jepa_mask=batch["jepa_mask"],
                task_idx=batch["task_idx"],
                cluster_id=batch["cluster_id"],
                substrate_desc_target=batch["substrate_desc"],
                ec_target=batch["ec_level1"],
                ec_mask=batch["ec_mask"],
                reaction_group_id=batch["reaction_group_id"],
                target_scale=batch["target_scale"],
            )
        if not torch.isfinite(parts["loss"]):
            raise FloatingPointError(name + ": nonfinite loss")
        parts["loss"].backward()
        if not all(
            torch.isfinite(x.grad).all()
            for x in model.parameters()
            if x.grad is not None
        ):
            raise FloatingPointError(name + ": nonfinite gradients")
        checks.append(
            {
                "variant": name,
                "parameters": sum(x.numel() for x in model.parameters()),
                "train_rows": len(loader.dataset),
                "batches_per_epoch": len(loader),
            }
        )
        del parts, out, batch, model, loader
        gc.collect()
        torch.cuda.empty_cache()
    retrieval_check = check_retrieval(a.data_root, cache, scalers, torch.device("cuda"))
    cache_stat = a.esm_cache.stat()
    write(
        a.run_root / "preflight.json",
        {
            "manifest_sha256": digest(PACKAGE / "MANIFEST.json"),
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "platform": platform.platform(),
            "data_root": str(a.data_root),
            "esm_cache": str(a.esm_cache),
            "cache_size": cache_stat.st_size,
            "cache_mtime_ns": cache_stat.st_mtime_ns,
            "checks": checks,
            "retrieval": retrieval_check,
        },
    )


if __name__ == "__main__":
    main()
