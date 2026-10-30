import os
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from common import reference_root, PACKAGE, backbone_args, memory_args, read, study
from LANCE import train as trainer
from LANCE.data import stable_key
from LANCE.lance7r import train as memory_trainer
from LANCE.lance7r.evaluate import evaluate_checkpoints as evaluate_memory
from LANCE.evaluate import evaluate_checkpoints as evaluate_backbone
from run import mark
from summarize import check_rows, load_rows, summarize

torch.set_num_threads(1)


def fixture(root):
    data = root / "data"
    splits = data / "datasets/splits"
    splits.mkdir(parents=True)
    (data / "datasets/processed").mkdir()
    cache = {}
    for task, smiles, target in [
        ("kcat", "reactant_smiles", "log10kcat_max"),
        ("km", "substrate_smiles", "log10km_mean"),
        ("ki", "substrate_smiles", "log10ki_mean"),
    ]:
        frames = {}
        for split, start, stop in [("train", 0, 12), ("val", 12, 18), ("test", 18, 24)]:
            rows = []
            for i in range(start, stop):
                seq = "M" + str(i)
                cache[stable_key(seq)] = torch.arange(48).reshape(4, 12).float() / (
                    i + 50
                )
                rows.append(
                    {
                        "sequence": seq,
                        smiles: "CC" if i % 2 else "CO",
                        target: (i % 7) / 3,
                        "product_smiles": "CN",
                        "reaction_smiles": "CC>>CN",
                        "ec": "1.1.1.1",
                        "sequence_99cluster": i,
                        "sequence_60cluster": i,
                        "temperature": 25,
                        "ph": 7,
                    }
                )
            frames[split] = pd.DataFrame(rows)
        frames["train"].to_csv(
            data
            / "datasets/processed"
            / {
                "kcat": "kcat_max_wt_singleSeqs_wpdbs.csv",
                "km": "km_mean_wt_singleSeqs_wpdbs.csv",
                "ki": "ki_mean_wt_singleSeqs_wpdbs.csv",
            }[task],
            index=False,
        )
        frames["trainval"] = pd.concat(
            [frames["train"], frames["val"]], ignore_index=True
        )
        frames["ood99"] = frames["test"]
        for split, frame in frames.items():
            name = (
                f"{task}-seq_test_sequence_99cluster.csv"
                if split == "ood99"
                else f"{task}-random_{split}.csv"
            )
            frame.to_csv(splits / name, index=False)
    torch.save({"embeddings": cache}, root / "cache.pt")
    return data, root / "cache.pt"


def arguments(name, root, data, cache):
    with patch(
        "sys.argv", ["train", *backbone_args(name, 0, root, data, cache, "cpu")]
    ):
        args = trainer.parse_args()
    for key, value in {
        "epochs": 2,
        "warmup_epochs": 1,
        "batch_size": 12,
        "eval_batch_size": 12,
        "dmpnn_hidden": 16,
        "model_dim": 16,
        "action_dim": 8,
        "hidden_dim": 24,
        "task_action_adapter_dim": 16,
        "max_residues": 8,
        "log_every": 0,
    }.items():
        setattr(args, key, value)
    return args


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class PipelineTests(unittest.TestCase):
    def test_recipe(self):
        self.assertEqual(study()["seeds"], list(range(10)))
        self.assertEqual(
            study()["variants"],
            ["full_no_jepa", "full_no_sigreg", "full_no_jepa_no_sigreg"],
        )
        for name in study()["variants"]:
            config = read(PACKAGE / "configs" / f"{name}.json")
            for seed in range(10):
                with patch(
                    "sys.argv",
                    [
                        "train",
                        *backbone_args(
                            name, seed, Path("/run"), Path("/data"), Path("/cache")
                        ),
                    ],
                ):
                    args = trainer.parse_args()
                self.assertEqual(
                    (
                        args.train_split,
                        args.seed,
                        args.epochs,
                        args.early_stop_patience,
                    ),
                    ("trainval", seed, 150, 20),
                )
                self.assertEqual((args.batch_size, args.eval_batch_size), (256, 128))
                for key, value in config["weights"].items():
                    if hasattr(args, key + "_weight"):
                        self.assertEqual(getattr(args, key + "_weight"), value, key)
                full_argv = list(study()["backbone_args"]) + [
                    "--data-root",
                    "/data",
                    "--train-split",
                    "trainval",
                    "--esm-cache",
                    "/cache",
                    "--output-dir",
                    str(Path("/run") / name),
                    "--run-name",
                    f"seed{seed}/backbone",
                    "--device",
                    "cuda",
                    "--seed",
                    str(seed),
                ]
                with patch("sys.argv", ["train", *full_argv]):
                    full_args = trainer.parse_args()
                removed = {
                    term + "_weight" for term in study()["removed_losses"][name]
                }
                for key, value in vars(full_args).items():
                    expected = 0.0 if key in removed else value
                    self.assertEqual(getattr(args, key), expected, f"{name}: {key}")
                self.assertTrue(args.reaction_cohort_sampler)
                self.assertEqual(args.score_r2_weight, 0.1)
                self.assertEqual(args.score_p1mag_weight, 0.001)
        for seed in range(10):
            reference = read(PACKAGE / "assets/reference/configs" / f"seed{seed}.json")
            self.assertEqual(reference["args"]["seed"], seed)
            self.assertEqual(
                reference["promotion_recipe"],
                read(PACKAGE / "assets/promotion_recipe.json"),
            )

    def test_training_retrieval_evaluation_and_resume(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(
            io.StringIO()
        ):
            root = Path(temp)
            data, cache = fixture(root)
            from preflight import check_retrieval
            from LANCE.data import EmbeddingCache, build_scalers

            check = check_retrieval(
                data,
                EmbeddingCache(cache, kind="test"),
                build_scalers(data, "trainval"),
                torch.device("cpu"),
            )
            self.assertEqual(check["memory_rows"], 18)
            self.assertGreater(check["valid_query_rows"], 0)
            for name in study()["variants"]:
                args = arguments(name, root, data, cache)
                trainer.train(args)
                checkpoint = root / name / "seed0/backbone/checkpoint_best.pt"
                state = load(checkpoint)
                self.assertTrue(
                    all(torch.isfinite(x).all() for x in state["model"].values())
                )
                self.assertEqual(state["args"]["train_split"], "trainval")
                self.assertEqual(state["recovery"]["sampler_epoch"], state["epoch"] + 1)
                with patch(
                    "sys.argv",
                    ["memory", *memory_args(name, 0, root, data, cache, "cpu")],
                ):
                    memory_options = memory_trainer.parse_args()
                memory_options.batch_size = 6
                memory_trainer.train(memory_options)
                memory_path = root / name / "seed0/memory/checkpoint_best.pt"
                memory_state = load(memory_path)
                self.assertEqual(memory_state["epoch"], 19)
                self.assertEqual(memory_state["model_config"]["memory_size"], 18)
                for stage, evaluator, path in [
                    ("backbone", evaluate_backbone, checkpoint),
                    ("memory", evaluate_memory, memory_path),
                ]:
                    out = root / name / "ensemble10" / stage
                    out.mkdir(parents=True)
                    evaluator(
                        [path] * 10,
                        data,
                        cache,
                        out / "evaluation_summary.json",
                        12,
                        0,
                        "cpu",
                        0,
                        0.5,
                        predictions_output=out / "per_row_predictions.csv",
                    )
                    report = read(out / "evaluation_summary.json")
                    self.assertEqual(report["ensemble_size"], 10)
                    self.assertEqual(report["catpred"]["kcat"]["Heldout"]["n"], 6)
                    mark(
                        out,
                        [
                            "evaluation_summary.json",
                            "evaluation_summary_catpred.csv",
                            "per_row_predictions.csv",
                        ],
                    )
                memory_rows = pd.read_csv(
                    root / name / "ensemble10/memory/per_row_predictions.csv"
                )
                protected = memory_rows[memory_rows.parameter.isin(["km", "ki"])]
                np.testing.assert_array_equal(protected.pred, protected.base_pred)

            name = "full_no_jepa"
            original_spec = importlib.util.spec_from_file_location(
                "original_trainer", PACKAGE / "assets/original_train.py"
            )
            original = importlib.util.module_from_spec(original_spec)
            original_spec.loader.exec_module(original)
            args = arguments(name, root / "original", data, cache)
            original.train(args)
            reference = load(root / name / "seed0/backbone/checkpoint_latest.pt")
            original_state = load(
                root / "original" / name / "seed0/backbone/checkpoint_latest.pt"
            )
            for key in reference["model"]:
                self.assertTrue(
                    torch.equal(reference["model"][key], original_state["model"][key]),
                    key,
                )

            args = arguments(name, root / "interrupted", data, cache)
            real_commit = trainer.recovery.commit

            def interrupt(*params, **kwargs):
                real_commit(*params, **kwargs)
                raise InterruptedError("Test interruption after an epoch")

            with patch.object(trainer.recovery, "commit", interrupt):
                with self.assertRaises(InterruptedError):
                    trainer.train(args)
            trainer.train(args)
            resumed = load(
                root / "interrupted" / name / "seed0/backbone/checkpoint_latest.pt"
            )
            for key in reference["model"]:
                self.assertTrue(
                    torch.equal(reference["model"][key], resumed["model"][key]), key
                )
            self.assertEqual(
                reference["recovery"]["global_step"], resumed["recovery"]["global_step"]
            )
            self.assertEqual(
                reference["recovery"]["scheduler"]["current_step"],
                resumed["recovery"]["scheduler"]["current_step"],
            )
            saved = resumed["model"]
            trainer.train(args)
            repeated = load(
                root / "interrupted" / name / "seed0/backbone/checkpoint_latest.pt"
            )
            self.assertTrue(
                all(torch.equal(v, repeated["model"][k]) for k, v in saved.items())
            )

    def test_original_data(self):
        reference_root = PACKAGE.parents[1] / "data/reference/CatPred-DB"
        if not reference_root.exists():
            self.skipTest("Local reference data is not included in the handoff")
        from preflight import check_data, check_reference_rows

        check_data(reference_root)
        check_reference_rows(reference_root)

    @unittest.skipUnless(os.environ.get("LANCE_REFERENCE_ROOT"), "Original evaluation artifacts are external")
    def test_summary_and_row_guard(self):
        import shutil

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in study()["variants"]:
                for stage, ref in [("backbone", "backbone"), ("memory", "full")]:
                    out = root / name / "ensemble10" / stage
                    shutil.copytree(reference_root() / ref, out)
                    mark(
                        out,
                        [
                            "evaluation_summary.json",
                            "evaluation_summary_catpred.csv",
                            "per_row_predictions.csv",
                        ],
                    )
            summarize(root)
            delta = pd.read_csv(root / "summary/contrasts.csv")
            self.assertTrue((delta.filter(like="_a_minus_b") == 0).all().all())
            frame = load_rows(reference_root() / "full/per_row_predictions.csv")
            with self.assertRaises(ValueError):
                check_rows(frame, frame.iloc[:-1])
            altered = frame.copy()
            altered.iloc[0, altered.columns.get_loc("target")] += 0.1
            with self.assertRaises(ValueError):
                check_rows(frame, altered)


if __name__ == "__main__":
    unittest.main()
