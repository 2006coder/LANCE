import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import (
    PACKAGE,
    digest,
    read,
    safe_json,
    verify_source,
    write,
)
from datasets import (
    FrameDataset,
    RetrievalFrame,
    frame,
    verify_partition,
)
from LANCE.data import (
    PARAM_SPECS,
    EmbeddingCache,
    TargetScaler,
    collate_kinetics,
    load_torch,
    move_batch,
)
from LANCE.lance7r.retrieval_data import (
    build_retrieval_memory,
    collate_retrieval_training,
)
from LANCE.losses import LANCELossWeights, lance_total_loss, masked_gaussian_nll
from LANCE.train import (
    NoamLR,
    TaskReactionCohortBatchSampler,
    latent_diagnostics,
    regression_metrics,
    seed_all,
)
from models import count, make_model
from retrieval import AblatedRetriever, alpha_fit
from torch.utils.data import ConcatDataset, DataLoader

STOP = False


def stop_requested(signum, stack):
    global STOP
    STOP = True


class ExposureSampler(TaskReactionCohortBatchSampler):
    def __init__(self, *args, steps=None, reference_batch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = steps
        self.reference_batch_size = reference_batch_size

    def __len__(self):
        return self.steps if self.steps is not None else super().__len__()

    def __iter__(self):
        if self.reference_batch_size is None:
            yield from super().__iter__()
            return
        if len(self.task_pools) != 1:
            raise ValueError("Exposure matching expects one task")
        task = next(iter(self.task_pools))
        rng = np.random.default_rng(self.seed + 104729 * self.epoch)
        base, remainder = divmod(self.reference_batch_size, 3)
        for index in range(len(self)):
            count = base + sum(
                task == (index + extra) % 3 for extra in range(remainder)
            )
            batch = []
            if task == 0 and self.kcat_groups:
                pairs = int(count * self.cohort_fraction) // 2
                for _ in range(pairs):
                    batch.extend(self._draw_kcat_pair(rng))
                count -= 2 * pairs
            batch.extend(self._draw(self.task_pools[task], count, rng))
            rng.shuffle(batch)
            yield batch
        self.epoch += 1


def save_torch(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def append(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(safe_json(row), sort_keys=True) + "\n")


def loss_for(out, batch, variant_config):
    if variant_config["intervention"].get("kinetic") == "mse":
        out = {k: x for k, x in out.items() if k != "log_var"}
    if variant_config["intervention"].get("architecture"):
        value = masked_gaussian_nll(
            out["mu"],
            out["log_var"],
            batch["target_z"],
            batch["target_mask"],
            mode=variant_config["weights"]["gaussian_loss"],
            beta=variant_config["weights"]["beta_nll_beta"],
        )[0]
        return {"loss": value, "kinetic": value.detach()}
    return lance_total_loss(
        out,
        batch["target_z"],
        target_mask=batch["target_mask"],
        weights=LANCELossWeights(**variant_config["weights"]),
        jepa_mask=batch["jepa_mask"],
        cluster_id=batch["cluster_id"],
        task_idx=batch["task_idx"],
        substrate_desc_target=batch["substrate_desc"],
        ec_target=batch["ec_level1"],
        ec_mask=batch["ec_mask"],
        reaction_group_id=batch["reaction_group_id"],
        target_scale=batch["target_scale"],
    )


@torch.no_grad()
def predict(model, dataset, device, batch_size):
    model.eval()
    rows = []
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_kinetics
    )
    for batch in loader:
        batch["product_bmg"] = None
        batch = move_batch(batch, device)
        # Use float32 for final prediction and selection, as in the original evaluator.
        out = model(batch)
        mu = out["mu"].float()
        if not torch.isfinite(mu).all():
            raise FloatingPointError("Non-finite backbone prediction")
        scale = float(dataset.scaler.stds[0])
        mean = float(dataset.scaler.means[0])
        logvar = out.get("log_var", torch.zeros_like(mu)).float().clamp(-7, 7)
        if getattr(model, "intervention", {}).get("kinetic") == "mse":
            logvar.fill_(float("nan"))
        for y, p, z, var in zip(
            batch["target"].cpu().tolist(),
            (mu * scale + mean).cpu().tolist(),
            mu.cpu().tolist(),
            (logvar.exp() * scale**2).cpu().tolist(),
        ):
            rows.append(dict(target=y, prediction=p, prediction_z=z, variance=var))
    result = (
        dataset.frame[
            ["_row_id", "_sequence_key", "_union60", "_union99"]
            + [c for c in dataset.frame if c.startswith("_novel_union")]
        ]
        .reset_index(drop=True)
        .copy()
    )
    if len(rows) != len(result):
        raise AssertionError("Prediction row mismatch")
    return pd.concat([result, pd.DataFrame(rows)], axis=1)


def metrics(pred):
    m = regression_metrics(pred.target.tolist(), pred.prediction.tolist())
    m["n"] = len(pred)
    if "variance" in pred:
        var = np.maximum(pred.variance.to_numpy(), 1e-12)
        err = pred.target.to_numpy() - pred.prediction.to_numpy()
        m["gaussian_nll_diagnostic"] = float(
            np.mean(0.5 * (np.log(2 * np.pi * var) + err**2 / var))
        )
        m["coverage95_diagnostic"] = float(
            np.mean(np.abs(err) <= 1.959963984540054 * np.sqrt(var))
        )
    return safe_json(m)


def selection_score(rows):
    values = []
    for p in rows.values():
        m = metrics(p)
        if m["r2"] is None:
            raise ValueError("Selection cohort has undefined R2")
        values.append(m["mae"] - 0.10 * m["r2"] - 0.001 * m["p1mag"])
    return float(np.mean(values))


def train_backbone(
    variant_config, study_config, folder, out, cache, seed, device, refit_source=None
):
    study_config = dict(study_config)
    for key in ("init_lr", "max_lr", "final_lr"):
        study_config[key] *= variant_config.get("learning_rate_scale", 1.0)
    tasks = variant_config["tasks"]
    fit_frames = {t: frame(folder, t, "fit") for t in tasks}
    if refit_source:
        fit_frames = {
            t: pd.concat(
                [frame(folder, t, r) for r in ("fit", "select", "calibrate", "report")],
                ignore_index=True,
            )
            for t in tasks
        }
    scalers = {
        t: TargetScaler.from_values(fit_frames[t][PARAM_SPECS[t]["target"]])
        for t in tasks
    }
    model = make_model(variant_config, seed, cache.dim).to(device)
    datasets = [
        FrameDataset(
            fit_frames[t],
            t,
            scalers[t],
            cache,
            study_config["max_residues"],
            variant_config["intervention"].get("products"),
            seed,
        )
        for t in tasks
    ]
    all_n = sum(
        (
            len(
                pd.concat(
                    [
                        frame(folder, t, r)
                        for r in ("fit", "select", "calibrate", "report")
                    ],
                    ignore_index=True,
                )
            )
            if refit_source
            else len(frame(folder, t, "fit"))
        )
        for t in PARAM_SPECS
    )
    batch_size = (
        study_config["batch_size"] // 3
        if variant_config.get("exposure_matched")
        else study_config["batch_size"]
    )
    steps = (
        all_n // study_config["batch_size"]
        if variant_config.get("exposure_matched")
        else None
    )
    sampler = ExposureSampler(
        ConcatDataset(datasets),
        batch_size=batch_size,
        seed=seed + 17029,
        cohort_fraction=variant_config.get("cohort_fraction", 0.5),
        min_gap=0.25,
        steps=steps,
        reference_batch_size=(
            study_config["batch_size"]
            if variant_config.get("exposure_matched")
            else None
        ),
    )
    if len(sampler) < 1:
        raise ValueError("No training batches")
    loader = DataLoader(
        sampler.dataset,
        batch_sampler=sampler,
        collate_fn=collate_kinetics,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=study_config["init_lr"],
        weight_decay=study_config["weight_decay"],
    )
    scheduler = NoamLR(
        optimizer,
        [study_config["warmup_epochs"]],
        [study_config["scheduler_epochs"]],
        len(loader),
        [study_config["init_lr"]],
        [study_config["max_lr"]],
        [study_config["final_lr"]],
    )
    config = dict(
        variant=variant_config,
        study=study_config,
        seed=seed,
        model_config=model.config.__dict__,
        parameters=count(model),
        module_shapes={
            name: list(module.weight.shape)
            for name, module in model.named_modules()
            if isinstance(getattr(module, "weight", None), torch.Tensor)
        },
        steps_per_epoch=len(loader),
        batch_size=batch_size,
        scalers={t: x.state() for t, x in scalers.items()},
        fit_rows={t: len(f) for t, f in fit_frames.items()},
        partition_sha256=digest(Path(folder) / "manifest.json"),
        phase="refit" if refit_source else "validation",
        source_hashes=read(PACKAGE / "SOURCE_PROVENANCE.json"),
    )
    if (out / "config.json").exists() and read(out / "config.json") != config:
        raise ValueError("Run configuration changed; use a new RUN_ROOT")
    write(out / "config.json", config)
    best = float("inf")
    start = 0
    best_epoch = -1
    epochs = study_config["epochs"]
    if refit_source:
        epochs = int(read(refit_source / "backbone_complete.json")["best_epoch"]) + 1
    latest = out / "checkpoint_latest.pt"
    best_path = out / "checkpoint_best.pt"
    if latest.exists():
        ck = load_torch(latest)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        restore_rng(ck["rng"])
        start = ck["epoch"] + 1
        best = ck["best"]
        best_epoch = ck["best_epoch"]
        sampler.epoch = start
    for epoch in range(start, epochs):
        model.train()
        totals = {}
        seen = {t: 0 for t in tasks}
        started = time.monotonic()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(batch)
                parts = loss_for(output, batch, variant_config)
                loss = parts["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite backbone loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), study_config["grad_clip"]
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite backbone gradient")
            optimizer.step()
            scheduler.step()
            for key, x in parts.items():
                totals[key] = totals.get(key, 0.0) + float(x.detach())
            for t in tasks:
                seen[t] += int((batch["task_idx"] == list(PARAM_SPECS).index(t)).sum())
        diag = latent_diagnostics(output, batch.get("jepa_mask"))
        row = dict(
            epoch=epoch,
            train={k: x / len(loader) for k, x in totals.items()},
            task_draws=seen,
            lr=scheduler.get_lr()[0],
            seconds=time.monotonic() - started,
            diagnostics=diag,
        )
        if refit_source:
            score = 0.0
            improved = epoch == epochs - 1
        else:
            val = {
                t: predict(
                    model,
                    FrameDataset(
                        frame(folder, t, "select"),
                        t,
                        scalers[t],
                        cache,
                        study_config["max_residues"],
                        products="none",
                    ),
                    device,
                    study_config["eval_batch_size"],
                )
                for t in tasks
            }
            score = selection_score(val)
            row["selection"] = {t: metrics(x) for t, x in val.items()}
            improved = score < best
        if improved:
            best = score
            best_epoch = epoch
        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=rng_state(),
            epoch=epoch,
            best=best,
            best_epoch=best_epoch,
            config=config,
        )
        if improved:
            save_torch(
                best_path,
                {
                    key: value
                    for key, value in state.items()
                    if key not in {"optimizer", "scheduler", "rng"}
                },
            )
        save_torch(latest, state)
        append(out / "history.jsonl", row)
        print(json.dumps(safe_json(row)), flush=True)
        if STOP:
            raise SystemExit(75)
    if not best_path.exists():
        raise RuntimeError("No selected checkpoint")
    ck = load_torch(best_path)
    model.load_state_dict(ck["model"])
    write(
        out / "backbone_complete.json",
        dict(
            best_epoch=best_epoch,
            epochs=epochs,
            best_selection_score=best,
            checkpoint_sha256=digest(best_path),
        ),
    )
    return model, scalers, fit_frames


def retrieval_predictions(retriever, dataset, device, batch_size):
    retriever.eval()
    chunks = []
    with torch.no_grad():
        for batch in DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_retrieval_training
        ):
            r = retriever(move_batch(batch, device))
            chunks.append(
                pd.DataFrame({k: x.float().cpu().numpy() for k, x in r.items()})
            )
    return pd.concat(chunks, ignore_index=True)


def run_retrieval(
    retrieval_config,
    study_config,
    folder,
    out,
    base_preds,
    fit_frame,
    scaler,
    cache,
    seed,
    device,
    refit_source=None,
):
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed + 891)
    level = retrieval_config.get("cluster_level", 99)
    dataset = RetrievalFrame(
        fit_frame, "kcat", scaler, cache, study_config["max_residues"], level
    )
    memory = build_retrieval_memory(dataset)
    model = AblatedRetriever(retrieval_config, memory, seed).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = (
        torch.optim.AdamW(
            parameters,
            lr=study_config["retrieval_lr"],
            weight_decay=study_config["retrieval_weight_decay"],
        )
        if parameters
        else None
    )
    generator = torch.Generator().manual_seed(seed + 17029)
    loader = DataLoader(
        dataset,
        batch_size=study_config["batch_size"],
        shuffle=True,
        generator=generator,
        drop_last=False,
        collate_fn=collate_retrieval_training,
    )
    start = 0
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        ck = load_torch(checkpoint)
        model.load_state_dict(ck["model"])
        start = ck["epoch"] + 1
        if optimizer:
            optimizer.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
        generator.set_state(ck["loader_rng"])
    epochs = study_config["retrieval_epochs"] if parameters else 0
    for epoch in range(start, epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            batch = move_batch(batch, device)
            r = model(batch)
            mask = batch["target_mask"] & r["retrieval_available"]
            if not mask.any():
                continue
            y = r["retrieval_mu"][mask]
            target = batch["target_z"][mask]
            loss = F.mse_loss(y, target) + study_config[
                "retrieval_robust_weight"
            ] * F.smooth_l1_loss(y, target, beta=0.5)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite retrieval loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, study_config["grad_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite retrieval gradient")
            optimizer.step()
            total += float(loss.detach())
            n += 1
        if not n:
            raise RuntimeError("No eligible retrieval training rows")
        save_torch(
            checkpoint,
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(),
                loader_rng=generator.get_state(),
                epoch=epoch,
                variant=retrieval_config,
                scaler=scaler.state(),
            ),
        )
        append(out / "history.jsonl", dict(epoch=epoch, loss=total / n))
        if STOP:
            raise SystemExit(75)
    if not checkpoint.exists():
        save_torch(
            checkpoint,
            dict(
                model=model.state_dict(),
                optimizer=None,
                rng=rng_state(),
                loader_rng=generator.get_state(),
                epoch=-1,
                variant=retrieval_config,
                scaler=scaler.state(),
            ),
        )
    retrieved = {}
    for role, bp in base_preds.items():
        f = fit_frame if role == "fit" else frame(folder, "kcat", role)
        retrieved[role] = retrieval_predictions(
            model,
            RetrievalFrame(
                f, "kcat", scaler, cache, study_config["max_residues"], level
            ),
            device,
            study_config["eval_batch_size"],
        )
    if refit_source:
        alpha = read(
            refit_source / "retrieval" / retrieval_config["name"] / "summary.json"
        )["alpha"]
    elif retrieval_config.get("blend") in ("half", "one"):
        alpha = 0.5 if retrieval_config["blend"] == "half" else 1.0
    else:
        cal = retrieved["calibrate"]
        bp = base_preds["calibrate"]
        alpha = alpha_fit(
            bp.target.to_numpy(),
            bp.prediction.to_numpy(),
            cal.retrieval_mu.to_numpy() * scaler.stds[0] + scaler.means[0],
            cal.retrieval_available.to_numpy(),
            study_config["max_blend"],
        )
    summaries = {}
    for role, r in retrieved.items():
        bp = base_preds[role]
        result = bp.copy()
        available = r.retrieval_available.to_numpy().astype(bool)
        candidate = r.retrieval_mu.to_numpy() * scaler.stds[0] + scaler.means[0]
        result["base_prediction"] = bp.prediction
        result["prediction"] = bp.prediction.to_numpy() + alpha * available * (
            candidate - bp.prediction.to_numpy()
        )
        result["retrieval_prediction"] = candidate
        result["retrieval_available"] = available
        result["retrieval_cluster_seen"] = r.retrieval_cluster_seen.to_numpy().astype(
            bool
        )
        result["retrieval_exact_neighbors"] = r.retrieval_exact_neighbors
        result.to_csv(out / f"{role}_kcat.csv", index=False)
        summaries[role] = dict(
            metrics=metrics(result),
            coverage=float(available.mean()),
            covered_metrics=metrics(result.loc[available]) if available.any() else None,
        )
    write(
        out / "summary.json",
        dict(
            variant=retrieval_config,
            alpha=alpha,
            calibration_role=(
                "calibrate" if not refit_source else "locked_from_validation_stage"
            ),
            trainable_parameters=sum(p.numel() for p in parameters),
            memory_rows=model.memory_size,
            metrics=summaries,
            variance_policy="Unchanged backbone variance; NLL and coverage are diagnostics, not calibrated retrieval uncertainty.",
        ),
    )


def run_task(args):
    study_config = read(args.run_root / "study.json")
    tasks = read(args.run_root / "tasks.json")
    task = tasks[read(args.index_map)[args.index] if args.index_map else args.index]
    variant_config = read(args.run_root / "configs" / f"{task['variant']}.json")
    seed = task["seed"]
    protocol = task["protocol"]
    folder = args.run_root / "partitions" / protocol
    verify_source()
    verify_partition(folder)
    output = (
        args.run_root
        / ("refit_runs" if args.refit else "runs")
        / protocol
        / variant_config["name"]
        / f"seed{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        print("Complete:", output)
        return
    lock = output / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"Run is locked: {lock}. Check the Slurm job before removing a stale lock."
        )
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        cache = EmbeddingCache(args.esm_cache, kind="ESM++")
        refit_source = (
            args.run_root / "runs" / protocol / variant_config["name"] / f"seed{seed}"
            if args.refit
            else None
        )
        if refit_source and not (refit_source / "complete.json").exists():
            raise RuntimeError("Refit requires completed validation-stage run")
        model, scalers, fit_frames = train_backbone(
            variant_config,
            study_config,
            folder,
            output,
            cache,
            seed,
            device,
            refit_source,
        )
        roles = ["fit"] if args.refit else ["fit", "select", "calibrate", "report"]
        preds = {}
        summary = {}
        for t in variant_config["tasks"]:
            preds[t] = {}
            for role in roles:
                f = fit_frames[t] if role == "fit" else frame(folder, t, role)
                dataset = FrameDataset(
                    f,
                    t,
                    scalers[t],
                    cache,
                    study_config["max_residues"],
                    products="none",
                )
                p = predict(model, dataset, device, study_config["eval_batch_size"])
                p.to_csv(output / f"{role}_{t}.csv", index=False)
                preds[t][role] = p
                summary[f"{role}/{t}"] = metrics(p)
        write(output / "metrics.json", summary)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if "kcat" in variant_config["tasks"]:
            rvs = (
                study_config["retrieval_variants"]
                if variant_config["name"] == "full"
                else [
                    x for x in study_config["retrieval_variants"] if x["name"] == "full"
                ]
            )
            for retrieval_config in rvs:
                if retrieval_config["name"] == "none":
                    continue
                rout = output / "retrieval" / retrieval_config["name"]
                if (rout / "summary.json").exists():
                    continue
                run_retrieval(
                    retrieval_config,
                    study_config,
                    folder,
                    rout,
                    preds["kcat"],
                    fit_frames["kcat"],
                    scalers["kcat"],
                    cache,
                    seed,
                    device,
                    refit_source,
                )
        write(
            output / "complete.json",
            dict(
                task=task,
                phase="refit" if args.refit else "validation",
                code_sha256=digest(PACKAGE / "run.py"),
                seconds_finished=time.time(),
            ),
        )
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--esm-cache", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--refit", action="store_true")
    p.add_argument("--index-map", type=Path)
    a = p.parse_args()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGUSR1, stop_requested)
    run_task(a)


if __name__ == "__main__":
    main()
