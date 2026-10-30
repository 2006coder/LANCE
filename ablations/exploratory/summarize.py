import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from common import read, write
from run import metrics


def seed_interval(values, rng, repeats):
    values = np.asarray(values)
    draw = rng.integers(0, len(values), (repeats, len(values)))
    return np.quantile(values[draw].mean(1), [0.025, 0.975]).tolist()


def hierarchical_mae(a, b, clusters, rng, repeats):
    groups = [np.flatnonzero(clusters == g) for g in np.unique(clusters)]
    difference = a - b
    sums = np.stack([difference[:, ix].sum(1) for ix in groups], axis=1)
    sizes = np.asarray([len(ix) for ix in groups])
    seed_weights = (
        rng.multinomial(a.shape[0], np.ones(a.shape[0]) / a.shape[0], size=repeats)
        / a.shape[0]
    )
    cluster_weights = rng.multinomial(
        len(groups), np.ones(len(groups)) / len(groups), size=repeats
    )
    values = ((seed_weights @ sums) * cluster_weights).sum(1) / (
        cluster_weights @ sizes
    )
    return np.quantile(values, [0.025, 0.975]).tolist()


def summarize(root, partial=False):
    study_config = read(root / "study.json")
    tasks = read(root / "tasks.json")
    missing = []
    groups = {}
    member_rows = []
    for task in tasks:
        p = root / "runs" / task["protocol"] / task["variant"] / f"seed{task['seed']}"
        if not (p / "complete.json").exists():
            missing.append(task)
            continue
        files = [(task["variant"] + "/backbone", f) for f in p.glob("report_*.csv")]
        files += [
            (task["variant"] + "/memory_" + q.parent.name, q)
            for q in p.glob("retrieval/*/report_kcat.csv")
        ]
        for name, f in files:
            t = f.stem.split("_")[-1]
            df = pd.read_csv(f).sort_values("_row_id").reset_index(drop=True)
            key = (task["protocol"], name, t)
            groups.setdefault(key, {})[task["seed"]] = df
            for cohort, mask in [("all", np.ones(len(df), dtype=bool))] + [
                (f"union_ood{level}", df[f"_novel_union{level}"].to_numpy(dtype=bool))
                for level in (99, 80, 60, 40)
            ]:
                if not mask.any():
                    continue
                member_rows.append(
                    dict(
                        protocol=task["protocol"],
                        variant=name,
                        task=t,
                        seed=task["seed"],
                        cohort=cohort,
                        **metrics(df.loc[mask]),
                    )
                )
    # Retrieval preserves the other two outputs; reuse the exact stored predictions.
    for (protocol, name, task), members in list(groups.items()):
        if task != "kcat" or "/memory_" not in name:
            continue
        backbone = name.split("/")[0] + "/backbone"
        for protected in ("km", "ki"):
            key = (protocol, backbone, protected)
            if key in groups:
                groups[(protocol, name, protected)] = groups[key]
    for protocol in study_config["protocols"]:
        for family, prefix in (
            ("independent", "single_"),
            ("independent_budget", "single_budget_"),
        ):
            for task in ("kcat", "km", "ki"):
                key = (protocol, prefix + task + "/backbone", task)
                if key in groups:
                    groups[(protocol, family + "/backbone", task)] = groups[key]
    tuning_choices = []
    families = (
        "full",
        "kinetic_only",
        "late_fusion_matched",
        "supervised_cross_attention_matched",
    )
    for protocol in study_config["protocols"]:
        for family in families:
            candidates = [
                name
                for name in study_config["variants"]
                if read(root / "configs" / f"{name}.json").get("tuning_family")
                == family
            ]
            scores = {}
            for name in candidates:
                paths = [
                    root
                    / "runs"
                    / protocol
                    / name
                    / f"seed{seed}"
                    / "backbone_complete.json"
                    for seed in study_config["seeds"]
                ]
                if all(path.exists() for path in paths):
                    scores[name] = float(
                        np.mean([read(path)["best_selection_score"] for path in paths])
                    )
            if len(scores) != 3:
                continue
            selected = min(scores, key=lambda name: (scores[name], name))
            tuning_choices.append(
                dict(
                    protocol=protocol,
                    family=family,
                    selected=selected,
                    selection_scores=scores,
                )
            )
            for task in ("kcat", "km", "ki"):
                for stage in ("backbone", "memory_full"):
                    key = (protocol, selected + "/" + stage, task)
                    if key in groups:
                        groups[(protocol, "tuned_" + family + "/" + stage, task)] = (
                            groups[key]
                        )
    member_rows = []
    for (protocol, name, task), members in groups.items():
        for seed, df in members.items():
            for cohort, mask in [("all", np.ones(len(df), dtype=bool))] + [
                (f"union_ood{level}", df[f"_novel_union{level}"].to_numpy(dtype=bool))
                for level in (99, 80, 60, 40)
            ]:
                if mask.any():
                    member_rows.append(
                        dict(
                            protocol=protocol,
                            variant=name,
                            task=task,
                            seed=seed,
                            cohort=cohort,
                            **metrics(df.loc[mask]),
                        )
                    )
    write(root / "missing_runs.json", missing)
    if missing and not partial:
        raise RuntimeError(
            f"{len(missing)} runs missing; no complete-study summary written"
        )
    if not member_rows:
        raise RuntimeError("No completed predictions")
    report = root / "analysis"
    report.mkdir(exist_ok=True)
    pd.DataFrame(member_rows).to_csv(report / "member_metrics.csv", index=False)
    write(report / "tuning_choices.json", tuning_choices)
    member_frame = pd.DataFrame(member_rows)
    macro = member_frame[member_frame.cohort == "all"].groupby(
        ["protocol", "variant", "seed"]
    )
    macro_rows = []
    for (protocol, name, seed), group in macro:
        if set(group.task) != {"kcat", "km", "ki"}:
            continue
        macro_rows.append(
            dict(
                protocol=protocol,
                variant=name,
                seed=seed,
                mae=float(group.mae.mean()),
                rmse=float(group.rmse.mean()),
                r2=float(group.r2.mean()),
                p1mag=float(group.p1mag.mean()),
            )
        )
    pd.DataFrame(macro_rows).to_csv(report / "macro_member_metrics.csv", index=False)
    macro_contrasts = []
    macro_frame = pd.DataFrame(macro_rows)
    if len(macro_frame):
        for protocol, group in macro_frame.groupby("protocol"):
            pivot = group.pivot(index="seed", columns="variant", values="mae")
            for left, right in itertools.combinations(pivot.columns, 2):
                delta = (pivot[left] - pivot[right]).dropna().to_numpy()
                if len(delta):
                    low, high = seed_interval(
                        delta,
                        np.random.default_rng(study_config["split_seed"]),
                        study_config["bootstrap_replicates"],
                    )
                    macro_contrasts.append(
                        dict(
                            protocol=protocol,
                            a=left,
                            b=right,
                            mean_macro_mae_a_minus_b=float(delta.mean()),
                            ci_low=low,
                            ci_high=high,
                            paired_seeds=len(delta),
                        )
                    )
    pd.DataFrame(macro_contrasts).to_csv(
        report / "macro_mae_contrasts.csv", index=False
    )
    cached_metrics = {
        (protocol, name, task, seed): metrics(df)
        for (protocol, name, task), members in groups.items()
        for seed, df in members.items()
    }
    ensembles = []
    contrasts = []
    planned = []
    rng = np.random.default_rng(study_config["split_seed"])
    repeats = study_config["bootstrap_replicates"]
    bases = [
        "full/backbone",
        "full/memory_full",
        "kinetic_only/backbone",
        "supervised_cross_attention_matched/backbone",
        "late_fusion_matched/backbone",
        "tuned_full/backbone",
        "tuned_supervised_cross_attention_matched/backbone",
    ]
    for protocol in study_config["protocols"]:
        for t in ("kcat", "km", "ki"):
            available = {
                name: members
                for (pr, name, task), members in groups.items()
                if pr == protocol and task == t
            }
            for name, members in available.items():
                seedlist = sorted(members)
                ref = members[seedlist[0]]
                for df in members.values():
                    if not df["_row_id"].equals(ref["_row_id"]) or not np.array_equal(
                        df.target.to_numpy(), ref.target.to_numpy()
                    ):
                        raise ValueError("Ensemble row mismatch")
                ens = ref.copy()
                ens["prediction"] = np.stack(
                    [members[k].prediction.to_numpy() for k in seedlist]
                ).mean(0)
                if "variance" in ens:
                    means = np.stack(
                        [members[k].prediction.to_numpy() for k in seedlist]
                    )
                    aleatoric = np.stack(
                        [members[k].variance.to_numpy() for k in seedlist]
                    ).mean(0)
                    ens["variance"] = aleatoric + means.var(0)
                ep = report / "ensembles" / protocol
                ep.mkdir(parents=True, exist_ok=True)
                ens.to_csv(
                    ep / (name.replace("/", "__") + "_" + t + ".csv"), index=False
                )
                for cohort, mask in [("all", np.ones(len(ens), dtype=bool))] + [
                    (
                        f"union_ood{level}",
                        ens[f"_novel_union{level}"].to_numpy(dtype=bool),
                    )
                    for level in (99, 80, 60, 40)
                ]:
                    if mask.any():
                        ensembles.append(
                            dict(
                                protocol=protocol,
                                variant=name,
                                task=t,
                                cohort=cohort,
                                seeds=len(seedlist),
                                **metrics(ens.loc[mask]),
                            )
                        )
            for a, b in itertools.combinations(sorted(available), 2):
                seeds = sorted(set(available[a]) & set(available[b]))
                if len(seeds) != len(study_config["seeds"]) and not partial:
                    raise ValueError("Unpaired seeds")
                metric_differences = {m: [] for m in ("mae", "rmse", "r2", "p1mag")}
                errors_a = []
                errors_b = []
                for seed in seeds:
                    aa = available[a][seed]
                    bb = available[b][seed]
                    if not aa["_row_id"].equals(bb["_row_id"]) or not np.array_equal(
                        aa.target.to_numpy(), bb.target.to_numpy()
                    ):
                        raise ValueError(f"Contrast row mismatch: {a}/{b}")
                    ma = cached_metrics[(protocol, a, t, seed)]
                    mb = cached_metrics[(protocol, b, t, seed)]
                    for m in metric_differences:
                        if ma[m] is not None and mb[m] is not None:
                            metric_differences[m].append(ma[m] - mb[m])
                    errors_a.append(
                        abs(aa.target.to_numpy() - aa.prediction.to_numpy())
                    )
                    errors_b.append(
                        abs(bb.target.to_numpy() - bb.prediction.to_numpy())
                    )
                for m, values in metric_differences.items():
                    if not values:
                        continue
                    low, high = seed_interval(values, rng, repeats)
                    contrasts.append(
                        dict(
                            protocol=protocol,
                            task=t,
                            a=a,
                            b=b,
                            metric=m,
                            seeds=len(values),
                            mean_a_minus_b=np.mean(values),
                            paired_seed_ci_low=low,
                            paired_seed_ci_high=high,
                            positive_seeds=int(np.sum(np.asarray(values) > 0)),
                        )
                    )
                if a in bases or b in bases:
                    low, high = hierarchical_mae(
                        np.asarray(errors_a),
                        np.asarray(errors_b),
                        aa["_union60"].to_numpy(),
                        rng,
                        repeats,
                    )
                    planned.append(
                        dict(
                            protocol=protocol,
                            task=t,
                            a=a,
                            b=b,
                            metric="mean_member_mae",
                            mean_a_minus_b=float(
                                np.mean(np.asarray(errors_a) - np.asarray(errors_b))
                            ),
                            paired_seed_cluster_ci_low=low,
                            paired_seed_cluster_ci_high=high,
                        )
                    )
    # Explicit interactions use paired per-seed metrics, not a comparison of separate intervals.
    interactions = []
    factor_sets = {
        "jepa_sigreg": ("full", "no_jepa_losses", "no_sigreg", "no_jepa_no_sigreg"),
        "reaction_binding": (
            "full",
            "no_reaction_loss",
            "no_binding_loss",
            "no_jepa_losses",
        ),
        "ranking_sampler": (
            "full",
            "no_ranking",
            "no_cohort_pairs",
            "no_ranking_no_cohort_pairs",
        ),
        "supervised_buildup": (
            "kinetic_plus_jepa_sigreg",
            "kinetic_plus_sigreg",
            "kinetic_plus_jepa",
            "kinetic_only",
        ),
    }
    for protocol in study_config["protocols"]:
        for t in ("kcat", "km", "ki"):
            for name, cells in factor_sets.items():
                keys = [(protocol, x + "/backbone", t) for x in cells]
                if not all(k in groups for k in keys):
                    continue
                seeds = sorted(set.intersection(*(set(groups[k]) for k in keys)))
                for metric in ("mae", "rmse", "r2", "p1mag"):
                    vals = []
                    for seed in seeds:
                        ms = [metrics(groups[k][seed])[metric] for k in keys]
                        if all(x is not None for x in ms):
                            vals.append(ms[0] - ms[1] - ms[2] + ms[3])
                    if vals:
                        lo, hi = seed_interval(vals, rng, repeats)
                        interactions.append(
                            dict(
                                protocol=protocol,
                                task=t,
                                factorial=name,
                                metric=metric,
                                interaction=np.mean(vals),
                                ci_low=lo,
                                ci_high=hi,
                                seeds=len(vals),
                            )
                        )
    for name, rows in [
        ("ensemble_metrics", ensembles),
        ("all_pairwise_member_contrasts", contrasts),
        ("base_comparisons_seed_cluster", planned),
        ("factorial_interactions", interactions),
    ]:
        pd.DataFrame(rows).to_csv(report / f"{name}.csv", index=False)
    write(
        report / "summary.json",
        dict(
            complete=not missing,
            expected_runs=len(tasks),
            completed_runs=len(tasks) - len(missing),
            comparison_direction="a minus b: negative MAE/RMSE and positive R2/p1mag favor a",
            interval_scope="Unadjusted descriptive 95% bootstrap intervals; do not treat hundreds of contrasts as independent significance tests.",
            primary_base="supervised_cross_attention_matched/backbone",
            primary_endpoint="macro-average task MAE; also report every task separately",
            reports_are_new_internal_partitions=True,
            external_confirmation=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    summarize(a.run_root, a.partial)
