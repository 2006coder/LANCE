import argparse
import tempfile
from pathlib import Path

import common
import numpy as np
import pandas as pd
import torch
from common import read, study, variant, write
from datasets import prepare
from LANCE.data import stable_key
from LANCE.lance7r.retrieval import LabelAnchoredRetriever, RetrievalConfig
from LANCE.lance7r.retrieval_data import RetrievalMemory
from LANCE.model import LANCEConfig, LANCEModel
from models import count, make_model
from retrieval import AblatedRetriever, alpha_fit
from run import loss_for, run_task

torch.set_num_threads(1)


def memory_and_batch():
    g = torch.Generator().manual_seed(8)
    n = 12
    memory = RetrievalMemory(
        torch.randn(n, 64, generator=g).half(),
        torch.randint(0, 2, (n, 128), generator=g).byte(),
        torch.linspace(-2, 2, n),
        torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
        torch.arange(n),
    )
    b = {
        "retrieval_protein_key": memory.protein_keys[:4].float(),
        "retrieval_chemical_key": memory.chemical_keys[:4].float(),
        "retrieval_reactant_id": memory.reactant_ids[:4],
        "retrieval_cluster_id": memory.cluster_ids[:4],
    }
    return memory, b


def test_retrieval_interventions(device="cpu"):
    memory, b = memory_and_batch()
    for rv in study()["retrieval_variants"]:
        if rv["name"] == "none":
            continue
        model = AblatedRetriever(rv, memory, 2).to(device)
        out = model({k: x.to(device) for k, x in b.items()})
        assert torch.isfinite(out["retrieval_mu"]).all(), rv["name"]
        params = [p for p in model.parameters() if p.requires_grad]
        if params:
            out["retrieval_mu"].square().mean().backward()
            assert any(p.grad is not None for p in params), rv["name"]
            assert all(
                torch.isfinite(p.grad).all() for p in params if p.grad is not None
            ), rv["name"]
    reference = LabelAnchoredRetriever(RetrievalConfig(mode="exact"), memory)
    model = AblatedRetriever({"name": "full", "learn": "all"}, memory, 0)
    model.load_state_dict(reference.state_dict())
    for key, value in reference(b).items():
        assert torch.equal(value, model(b)[key]), key
    # Training queries cannot read labels from their own cluster.
    changed = RetrievalMemory(
        memory.protein_keys,
        memory.chemical_keys,
        memory.target_z.clone(),
        memory.reactant_ids,
        memory.cluster_ids,
    )
    changed.target_z[0] = 1000
    a = AblatedRetriever({"learn": "none", "estimate": "mean"}, memory, 0)
    c = AblatedRetriever({"learn": "none", "estimate": "mean"}, changed, 0)
    assert torch.equal(a(b)["retrieval_mu"][:1], c(b)["retrieval_mu"][:1])
    strict = AblatedRetriever({"learn": "all", "strict_exact": True}, memory, 0)
    with torch.no_grad():
        strict.fusion_residual[-1].weight.fill_(0.2)
    x = torch.randn(3, 8)
    y = x.clone()
    y[:, [1, 3, 5, 7]] += 100
    assert torch.equal(strict.fusion_residual(x), strict.fusion_residual(y))
    assert (
        alpha_fit(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            np.array([2.0, 4.0]),
            np.ones(2),
            0.5,
        )
        == 0.5
    )
    assert (
        alpha_fit(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            np.array([2.0, 4.0]),
            np.zeros(2),
            0.5,
        )
        == 0
    )


def small_variant(name):
    v = variant(name)
    v["model"].update(
        protein_dim=12,
        model_dim=32,
        hidden_dim=64,
        action_dim=8,
        dmpnn_hidden=16,
        task_action_adapter_dim=32,
        dropout=0.0,
        num_interface_queries=2,
        interface_heads=4,
    )
    v.pop("capacity_fraction", None)
    return v


def toy_batch():
    from LANCE.dmpnn import mol2graph

    n = 6
    return dict(
        protein=torch.randn(n, 5, 12),
        protein_mask=torch.ones(n, 5, dtype=torch.bool),
        bmg=mol2graph(["CC", "CO", "CN"] * 2),
        product_bmg=mol2graph(["CO", "CN", "CC"] * 2),
        task_idx=torch.tensor([0, 1, 2] * 2),
        condition=torch.randn(n, 4),
        target_z=torch.linspace(-1, 1, n),
        target_mask=torch.ones(n, dtype=torch.bool),
        jepa_mask=torch.tensor([1, 0, 0] * 2, dtype=torch.bool),
        cluster_id=torch.arange(n),
        substrate_desc=torch.zeros(n, 8),
        ec_level1=torch.tensor([1, 2, 3] * 2),
        ec_mask=torch.ones(n, dtype=torch.bool),
        reaction_group_id=torch.tensor([1, 2, 3] * 2),
        target_scale=torch.ones(n),
    )


def test_models():
    b = toy_batch()
    for name in study()["variants"]:
        v = small_variant(name)
        model = make_model(v, 0, 12)
        if len(v["tasks"]) == 1:
            bb = dict(b)
            bb["task_idx"] = torch.full_like(
                b["task_idx"], ["kcat", "km", "ki"].index(v["tasks"][0])
            )
        else:
            bb = b
        out = model(bb)
        loss = loss_for(out, bb, v)["loss"]
        assert torch.isfinite(loss), name
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        ), name
        if name == "no_products":
            assert "product_state" not in out
        if name == "no_actions":
            assert (
                out["z_bind"].count_nonzero() == 0 and out["z_cat"].count_nonzero() == 0
            )
        if name == "one_action":
            assert torch.equal(out["z_bind"], out["z_cat"])
        if name == "no_cross_attention":
            expected = model.interface_encoder.norm(out["enzyme_context"])
            assert torch.allclose(expected, out["interface_context"])
    v = small_variant("full")
    model = make_model(v, 7, 12).eval()
    reference = LANCEModel(LANCEConfig.from_dict(v["model"]))
    reference.load_state_dict(model.state_dict())
    reference.eval()
    assert torch.equal(model(b)["mu"], reference(b)["mu"])
    for name in ("late_fusion_matched", "supervised_cross_attention_matched"):
        v = variant(name)
        m = make_model(v, 0)
        full = make_model(variant("full"), 0)
        assert abs(count(m) / count(full) - 1) < 0.01
        del m, full
    for task in ("kcat", "km", "ki"):
        v = variant("single_budget_" + task)
        m = make_model(v, 0)
        full = make_model(variant("full"), 0)
        assert abs(3 * count(m) / count(full) - 1) < 0.01
        del m, full


def make_fixture(root):
    data = root / "data"
    split = data / "datasets/splits"
    split.mkdir(parents=True)
    cache = {}
    clusters = []
    # Distinct pair IDs, cross-task sequence reuse, and enough clusters for every role.
    for t in ("kcat", "km", "ki"):
        spec = {
            "kcat": ("reactant_smiles", "log10kcat_max"),
            "km": ("substrate_smiles", "log10km_mean"),
            "ki": ("substrate_smiles", "log10ki_mean"),
        }[t]
        for role, start, n in [("train", 0, 90), ("val", 90, 90)]:
            rows = []
            for i in range(start, start + n):
                seq = "M" + str(i)
                key = stable_key(seq)
                cache[key] = torch.randn(3 + i % 4, 12)
                rows.append(
                    dict(
                        sequence=seq,
                        **{
                            spec[0]: "CC" if i % 2 else "CO",
                            spec[1]: float(i % 13) / 3,
                        },
                        product_smiles="CN",
                        reaction_smiles="CC>>CN",
                        ec="1.1.1.1",
                        sequence_99cluster=i,
                        sequence_80cluster=i,
                        sequence_60cluster=i,
                        temperature=25,
                        ph=7,
                    )
                )
                clusters.append(
                    dict(
                        sequence_key=key,
                        union_cluster_99=key,
                        union_cluster_80=key,
                        union_cluster_60=key,
                        union_cluster_40=key,
                    )
                )
            pd.DataFrame(rows).to_csv(split / f"{t}-random_{role}.csv", index=False)
    torch.save({"embeddings": cache}, root / "cache.pt")
    return data, pd.DataFrame(clusters).drop_duplicates()


def test_pipeline():
    import datasets

    original = datasets.PACKAGE
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        data, clusters = make_fixture(root)
        (root / "assets").mkdir()
        clusters.to_csv(root / "assets/union_sequence_clusters.csv", index=False)
        datasets.PACKAGE = root
        try:
            prepare(data, root / "partitions", ["random", "cluster60"], 20261016)
        finally:
            datasets.PACKAGE = original
        s = study()
        s.update(
            variants=["full", "full__lr_half", "full__lr_double", "no_actions"],
            protocols=["random", "cluster60"],
            seeds=[0, 1],
            epochs=2,
            scheduler_epochs=3,
            warmup_epochs=1,
            batch_size=18,
            eval_batch_size=24,
            retrieval_epochs=2,
            bootstrap_replicates=20,
        )
        s["retrieval_variants"] = [
            rv
            for rv in s["retrieval_variants"]
            if rv["name"] in ("full", "strict_exact", "fixed_heuristic")
        ]
        write(root / "study.json", s)
        for name in s["variants"]:
            write(root / "configs" / f"{name}.json", small_variant(name))
        tasks = [
            dict(protocol=protocol, variant=name, seed=seed)
            for protocol in s["protocols"]
            for name in s["variants"]
            for seed in s["seeds"]
        ]
        write(root / "tasks.json", tasks)
        for i in range(len(tasks)):
            run_task(
                argparse.Namespace(
                    run_root=root,
                    esm_cache=root / "cache.pt",
                    index=i,
                    index_map=None,
                    device="cpu",
                    refit=False,
                )
            )
        # An interrupted post-training evaluation resumes the same completed optimizer trajectory.
        out = root / "runs/random/full/seed0"
        before = common.digest(out / "checkpoint_latest.pt")
        (out / "complete.json").unlink()
        run_task(
            argparse.Namespace(
                run_root=root,
                esm_cache=root / "cache.pt",
                index=0,
                index_map=None,
                device="cpu",
                refit=False,
            )
        )
        assert before == common.digest(out / "checkpoint_latest.pt")
        # Resume after an actual optimizer epoch and compare every weight.
        import run as runner

        interrupted_root = root / "interrupted"
        import shutil

        shutil.copytree(root / "partitions", interrupted_root / "partitions")
        shutil.copytree(root / "configs", interrupted_root / "configs")
        write(interrupted_root / "study.json", s)
        write(interrupted_root / "tasks.json", tasks)
        args = argparse.Namespace(
            run_root=interrupted_root,
            esm_cache=root / "cache.pt",
            index=0,
            index_map=None,
            device="cpu",
            refit=False,
        )
        runner.STOP = True
        try:
            run_task(args)
            raise AssertionError("Expected epoch-boundary stop")
        except SystemExit as error:
            assert error.code == 75
        finally:
            runner.STOP = False
        run_task(args)
        original_state = runner.load_torch(out / "checkpoint_latest.pt")["model"]
        resumed_state = runner.load_torch(
            interrupted_root / "runs/random/full/seed0/checkpoint_latest.pt"
        )["model"]
        assert all(
            torch.equal(original_state[key], resumed_state[key])
            for key in original_state
        )
        from summarize import summarize

        summarize(root)
        assert read(root / "analysis/summary.json")["complete"]
        assert len(read(root / "analysis/tuning_choices.json")) == 2
        assert (
            root / "analysis/all_pairwise_member_contrasts.csv"
        ).stat().st_size > 100
        run_task(
            argparse.Namespace(
                run_root=root,
                esm_cache=root / "cache.pt",
                index=0,
                index_map=None,
                device="cpu",
                refit=True,
            )
        )
        assert (
            read(root / "refit_runs/random/full/seed0/complete.json")["phase"]
            == "refit"
        )


if __name__ == "__main__":
    import time

    started = time.monotonic()
    for f in [test_retrieval_interventions, test_models, test_pipeline]:
        print("RUN", f.__name__, flush=True)
        f()
        print("PASS", f.__name__, flush=True)
    print("All checks passed in", round(time.monotonic() - started, 2), "seconds")
