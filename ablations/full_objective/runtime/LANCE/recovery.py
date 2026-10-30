import os
import random

import numpy as np
import torch


def restore(path, model, optimizer, scheduler, loader, args):
    if not path.exists():
        return 0, float("inf"), 0, 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["args"] != vars(args):
        raise RuntimeError("Resume arguments differ from the saved run")
    run = state["recovery"]
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(run["scheduler"])
    loader.batch_sampler.epoch = run["sampler_epoch"]
    random.setstate(run["python_rng"])
    np.random.set_state(run["numpy_rng"])
    torch.set_rng_state(run["torch_rng"])
    if run["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(run["cuda_rng"])
    metrics = path.parent / "metrics.jsonl"
    if metrics.exists():
        import json

        rows = []
        for line in metrics.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("epoch", -1) <= state["epoch"]:
                rows.append(line)
        metrics.write_text("\n".join(rows) + "\n")
    start = args.epochs if run["finished"] else state["epoch"] + 1
    return start, run["best"], run["bad_epochs"], run["global_step"]


def commit(
    folder,
    model,
    optimizer,
    scheduler,
    loader,
    epoch,
    metrics,
    scalers,
    args,
    best,
    bad_epochs,
    global_step,
    improved,
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "scalers": {k: v.state() for k, v in scalers.items()},
        "args": vars(args),
        "model_config": model.config_dict(),
        "experiment_family": "LANCE",
        "recovery": {
            "scheduler": scheduler.state_dict(),
            "sampler_epoch": loader.batch_sampler.epoch,
            "best": best,
            "bad_epochs": bad_epochs,
            "global_step": global_step,
            "finished": epoch + 1 >= args.epochs
            or (
                args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience
            ),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    names = (
        ["checkpoint_best.pt", "checkpoint_latest.pt"]
        if improved
        else ["checkpoint_latest.pt"]
    )
    for name in names:
        temporary = folder / (name + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, folder / name)
