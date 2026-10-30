import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from common import PACKAGE, digest, read, write
from LANCE.data import (
    PARAM_SPECS,
    KineticsDataset,
    TargetScaler,
    clean,
    filter_invalid_smiles_frame,
    first_existing_product_column,
    is_valid_single_smiles,
    load_split_frame,
    stable_key,
)
from LANCE.lance7r.retrieval_data import RetrievalKineticsDataset


def bucket(key, seed):
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / 16**12


def prepare(data_root, out, protocols, seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(
        Path(os.environ.get("LANCE_SEQUENCE_CLUSTERS", str(PACKAGE / "assets/union_sequence_clusters.csv"))), dtype=str
    ).set_index("sequence_key")
    frames = {}
    hashes = {}
    for t, spec in PARAM_SPECS.items():
        parts = []
        for split in ("train", "val"):
            f = filter_invalid_smiles_frame(
                load_split_frame(data_root, t, split), spec["smiles"]
            )
            f = f[np.isfinite(pd.to_numeric(f[spec["target"]], errors="coerce"))].copy()
            f["_original_split"] = split
            f["_row_id"] = [f"{t}:{split}:{i}" for i in f["_source_row_index"]]
            f["_sequence_key"] = f.sequence.map(lambda x: stable_key(clean(x)))
            f["_pair_key"] = [
                stable_key(clean(s) + "|" + clean(m))
                for s, m in zip(f.sequence, f[spec["smiles"]])
            ]
            for level in (99, 80, 60, 40):
                f[f"_union{level}"] = f["_sequence_key"].map(
                    mapping[f"union_cluster_{level}"]
                )
                if f[f"_union{level}"].isna().any():
                    raise ValueError(f"Missing pooled clusters: {t}/{split}/{level}")
            for level in (99, 80, 60):
                col = f"sequence_{level}cluster"
                if col not in f or f[col].isna().any():
                    raise ValueError(f"Missing actual cluster identifiers: {col}")
            parts.append(f)
            from LANCE.data import split_path

            hashes[f"{t}/{split}"] = digest(split_path(data_root, t, split))
        # Exact same-task sequence/molecule pairs must not cross train/validation.
        if set(parts[0]._pair_key) & set(parts[1]._pair_key):
            raise ValueError(f"Duplicate train/val pair for {t}")
        frames[t] = pd.concat(parts, ignore_index=True)
    for protocol in protocols:
        counts = {}
        all_role_clusters = {r: set() for r in ("fit", "select", "calibrate", "report")}
        for t, f0 in frames.items():
            f = f0.copy()
            col = "_union60" if protocol == "cluster60" else "_union99"
            u = f[col].map(lambda x: bucket(x, seed))
            if protocol == "cluster60":
                roles = np.select(
                    [u < 0.7, u < 0.8, u < 0.9],
                    ["fit", "select", "calibrate"],
                    default="report",
                )
            elif protocol == "random":
                roles = np.where(
                    f["_original_split"] == "train",
                    "fit",
                    np.select(
                        [u < 0.4, u < 0.7], ["select", "calibrate"], default="report"
                    ),
                )
            else:
                raise ValueError(protocol)
            f["_role"] = roles
            train_union = (
                {
                    level: set(
                        pd.concat([x for x in frames.values()]).loc[
                            lambda x: x["_original_split"] == "train", f"_union{level}"
                        ]
                    )
                    for level in (99, 80, 60, 40)
                }
                if protocol == "random"
                else None
            )
            for role in ("fit", "select", "calibrate", "report"):
                sub = f[f["_role"] == role].copy()
                if len(sub) < 5:
                    raise ValueError(f"Too few rows: {protocol}/{t}/{role}={len(sub)}")
                if train_union:
                    for level in train_union:
                        sub[f"_novel_union{level}"] = ~sub[f"_union{level}"].isin(
                            train_union[level]
                        )
                path = out / protocol / f"{t}_{role}.csv"
                path.parent.mkdir(exist_ok=True)
                sub.to_csv(path, index=False)
                counts[f"{t}/{role}"] = len(sub)
                all_role_clusters[role].update(sub[col])
        pairs = [("select", "calibrate"), ("select", "report"), ("calibrate", "report")]
        if protocol == "cluster60":
            pairs += [("fit", r) for r in ("select", "calibrate", "report")]
        for a, b in pairs:
            if all_role_clusters[a] & all_role_clusters[b]:
                raise AssertionError("Cluster overlap " + a + "/" + b)
        # Novelty to every task in the fitting partition, for both protocols.
        for level in (99, 80, 60, 40):
            seen = set().union(
                *(
                    set(pd.read_csv(out / protocol / f"{t}_fit.csv")[f"_union{level}"])
                    for t in PARAM_SPECS
                )
            )
            for t in PARAM_SPECS:
                for role in ("fit", "select", "calibrate", "report"):
                    path = out / protocol / f"{t}_{role}.csv"
                    f = pd.read_csv(path)
                    f[f"_novel_union{level}"] = ~f[f"_union{level}"].isin(seen)
                    f.to_csv(path, index=False)
        write(
            out / protocol / "manifest.json",
            dict(
                protocol=protocol,
                seed=seed,
                counts=counts,
                input_hashes=hashes,
                cluster_map_sha256=digest(
                    Path(os.environ.get("LANCE_SEQUENCE_CLUSTERS", str(PACKAGE / "assets/union_sequence_clusters.csv")))
                ),
                files={p.name: digest(p) for p in (out / protocol).glob("*.csv")},
            ),
        )


def verify_partition(folder):
    m = read(Path(folder) / "manifest.json")
    for name, sha in m["files"].items():
        if digest(Path(folder) / name) != sha:
            raise RuntimeError("Changed partition " + name)
    return m


class FrameDataset(KineticsDataset):
    def __init__(
        self, frame, task, scaler, cache, max_residues=1024, products=None, seed=0
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.parameter = task
        self.task_idx = list(PARAM_SPECS).index(task)
        self.spec = PARAM_SPECS[task]
        self.split = "prepared"
        self.scaler = scaler
        self.esm_cache = cache
        self.max_residues = max_residues
        self.cluster_level = 60
        self.product_column = first_existing_product_column(self.frame.columns)
        if products == "none":
            self.product_column = None
        if products == "shuffle" and self.product_column:
            mask = self.frame[self.product_column].map(
                lambda x: is_valid_single_smiles(clean(x))
            )
            values = self.frame.loc[mask, self.product_column].to_numpy(copy=True)
            rng = np.random.default_rng(seed + 907)
            order = rng.permutation(len(values))
            # One cyclic permutation has no fixed row assignments when n > 1.
            if len(values) > 1:
                values[order] = np.roll(values[order], 1)
            self.frame.loc[mask, self.product_column] = values

    def __getitem__(self, i):
        x = super().__getitem__(i)
        r = self.frame.iloc[i]
        x["row_id"] = r["_row_id"]
        x["union_cluster"] = r["_union60"]
        return x


class RetrievalFrame(RetrievalKineticsDataset):
    def __init__(self, frame, task, scaler, cache, max_residues=1024, cluster_level=99):
        base = FrameDataset(frame, task, scaler, cache, max_residues, products="none")
        self.__dict__.update(base.__dict__)
        self.protein_key_dim = 64
        self.fingerprint_dim = 128
        self.retrieval_cluster_level = cluster_level
        self._retrieval_key_cache = [None] * len(self)


def frame(folder, t, role):
    return pd.read_csv(Path(folder) / f"{t}_{role}.csv")


def scalers_for(folder, tasks):
    return {
        t: TargetScaler.from_values(frame(folder, t, "fit")[PARAM_SPECS[t]["target"]])
        for t in tasks
    }
