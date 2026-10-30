import json
from pathlib import Path

P = Path(__file__).resolve().parent
base = json.loads((P / "assets/reference_model.json").read_text())
loss = dict(
    kinetic=1.0,
    jepa=0.35,
    jepa_delta=0.25,
    jepa_contrastive=0.05,
    action_decorrelation=0.01,
    sigreg=0.02,
    binding_aux=0.12,
    binding_jepa=0.15,
    binding_jepa_delta=0.10,
    binding_jepa_contrastive=0.02,
    binding_jepa_exclude_kcat=True,
    kcat_reaction_ranking=0.03,
    kcat_ranking_temperature=0.35,
    kcat_ranking_min_gap=0.25,
    km_p1=0.03,
    km_p1_temperature=0.15,
    gaussian_loss="beta_nll",
    beta_nll_beta=0.5,
    ec_aux=0.10,
)
variants = []


def add(
    name, question, model=None, weights=None, intervention=None, tasks=None, **extra
):
    variants.append(
        dict(
            name=name,
            question=question,
            model={**base, **(model or {})},
            weights={**loss, **(weights or {})},
            intervention=intervention or {},
            tasks=tasks or ["kcat", "km", "ki"],
            **extra,
        )
    )


rx = dict(jepa=0.0, jepa_delta=0.0, jepa_contrastive=0.0)
bind = dict(binding_jepa=0.0, binding_jepa_delta=0.0, binding_jepa_contrastive=0.0)
aux = {
    **rx,
    **bind,
    **dict(
        sigreg=0.0,
        action_decorrelation=0.0,
        binding_aux=0.0,
        ec_aux=0.0,
        kcat_reaction_ranking=0.0,
        km_p1=0.0,
    ),
}
add("full", "Reference final LANCE-4R backbone; full LANCE-7R follows each checkpoint.")
add(
    "no_reaction_loss", "Reaction JEPA loss family, keeping product SIGReg.", weights=rx
)
add(
    "no_binding_loss",
    "Binding JEPA loss family, keeping its head features and SIGReg.",
    weights=bind,
)
add(
    "no_jepa_losses",
    "Both JEPA families, keeping representation regularization.",
    weights={**rx, **bind},
)
add("no_sigreg", "SIGReg independent of the JEPA losses.", weights={"sigreg": 0.0})
add(
    "no_jepa_no_sigreg",
    "Fourth cell of JEPA x SIGReg factorial.",
    weights={**rx, **bind, "sigreg": 0.0},
)
add(
    "no_products",
    "Remove every product graph and product-dependent target, including SIGReg.",
    weights=rx,
    intervention={"products": "none"},
)
add(
    "shuffled_products",
    "Negative control: products deterministically permuted among product-bearing training rows.",
    intervention={"products": "shuffle"},
)
add(
    "no_binding_branch",
    "Remove the internal binding-prediction branch, losses, head feature and projected states.",
    model={"use_binding_jepa": False, "binding_jepa_exclude_kcat": False},
    weights={**bind, "binding_jepa_exclude_kcat": False},
)
add(
    "no_state_losses",
    "Remove state matching in both JEPA branches.",
    weights={"jepa": 0.0, "binding_jepa": 0.0},
)
add(
    "no_delta_losses",
    "Remove displacement matching in both branches.",
    weights={"jepa_delta": 0.0, "binding_jepa_delta": 0.0},
)
add(
    "no_contrastive_losses",
    "Remove contrastive terms in both branches.",
    weights={"jepa_contrastive": 0.0, "binding_jepa_contrastive": 0.0},
)
add(
    "no_stop_product_target",
    "Product-target stop-gradient; reactant anchors and binding targets stay stopped.",
    model={"stop_jepa_target": False},
)
add(
    "no_action_decorrelation",
    "Action decorrelation penalty.",
    weights={"action_decorrelation": 0.0},
)
add(
    "no_descriptor_loss",
    "Molecular descriptor auxiliary supervision.",
    weights={"binding_aux": 0.0},
)
add("no_ec_loss", "EC class auxiliary supervision.", weights={"ec_aux": 0.0})
add(
    "no_ranking",
    "Ranking loss with unchanged cohort sampler.",
    weights={"kcat_reaction_ranking": 0.0},
)
add(
    "no_cohort_pairs",
    "Cohort sampling without changing balanced task allocation.",
    cohort_fraction=0.0,
)
add(
    "no_ranking_no_cohort_pairs",
    "Fourth cell of ranking x paired sampling factorial.",
    weights={"kcat_reaction_ranking": 0.0},
    cohort_fraction=0.0,
)
add("no_km_boundary", "Km factor-of-ten boundary loss.", weights={"km_p1": 0.0})
add(
    "kinetic_only",
    "Strong supervised base: full architecture, kinetic labels only; product tensors have zero loss weight.",
    weights=aux,
)
add(
    "no_cross_attention",
    "Replace the interface with LayerNorm of pooled enzyme context.",
    intervention={"interface": "pooled"},
)
add(
    "pooled_fusion_matched",
    "Replace residue cross-attention with a pooled pair MLP of matched trainable size.",
    intervention={"interface": "matched_mlp"},
)
add(
    "no_task_gate",
    "Remove task-dependent interface gates.",
    model={"use_task_interface_gate": False},
)
add(
    "no_task_adapter",
    "Keep shared actions but remove the task residual adapter.",
    model={"use_task_residual_actions": False},
)
add(
    "task_specific_actions",
    "Replace shared actions with separate task action networks; larger capacity control.",
    model={"use_task_residual_actions": False, "use_task_specific_actions": True},
)
add(
    "one_action",
    "Tie the two learned action vectors after the adapter; remove their decorrelation penalty.",
    weights={"action_decorrelation": 0.0},
    intervention={"action": "tied"},
)
add(
    "no_actions",
    "Zero both action vectors during retraining; remove degenerate decorrelation loss.",
    weights={"action_decorrelation": 0.0},
    intervention={"action": "zero"},
)
add(
    "no_actions_kinetic_only",
    "Action-route removal with every auxiliary off.",
    weights=aux,
    intervention={"action": "zero", "products": "none"},
)
add(
    "no_binding_action",
    "Zero only the binding action during retraining.",
    weights={"action_decorrelation": 0.0},
    intervention={"action": "no_bind"},
)
add(
    "no_catalytic_action",
    "Zero only the catalytic action during retraining.",
    weights={"action_decorrelation": 0.0},
    intervention={"action": "no_cat"},
)
add(
    "action_16",
    "Action bottleneck sensitivity: 16 coordinates.",
    model={"action_dim": 16},
)
add(
    "action_256",
    "Action bottleneck sensitivity: 256 coordinates.",
    model={"action_dim": 256},
)
add(
    "no_conditions",
    "Remove condition values and presence indicators.",
    intervention={"conditions": "zero"},
)
add(
    "condition_presence_only",
    "Keep condition missingness indicators but remove values.",
    intervention={"conditions": "presence"},
)
add(
    "no_global_bypass",
    "Remove kcat native-width protein bypass.",
    model={"use_kcat_global_rate_context": False},
)
add(
    "mean_protein_only",
    "Replace mean-plus-attentive pooling with mean pooling; keep residue attention.",
    intervention={"pool": "mean"},
)
add(
    "pooled_residue_input",
    "Broadcast each sequence mean to valid residue positions; remove residue-level variation.",
    intervention={"protein": "pooled"},
)
add(
    "nll",
    "Beta=0: standard heteroscedastic Gaussian NLL.",
    weights={"gaussian_loss": "nll"},
)
add(
    "mse",
    "MSE mean training with Gaussian-head architecture retained; variance not interpreted.",
    intervention={"kinetic": "mse"},
)
add(
    "half_width",
    "Capacity sensitivity with 256-dimensional states and 512 hidden units.",
    model={"model_dim": 256, "hidden_dim": 512, "task_action_adapter_dim": 128},
)
add(
    "late_fusion_matched",
    "Protein and molecule interact only in prediction heads; trainable parameter count matched within 1%.",
    weights=aux,
    intervention={"architecture": "late_fusion", "products": "none"},
)
add(
    "supervised_cross_attention_matched",
    "Strong primary base: cross-attention, same inputs and matched trainable parameter count, kinetic supervision only.",
    weights=aux,
    intervention={"architecture": "cross_attention_fusion", "products": "none"},
)
add(
    "kinetic_plus_jepa",
    "Build-up: kinetic-only backbone plus both JEPA losses, without SIGReg or other auxiliaries.",
    weights={**aux, **{k: loss[k] for k in [*rx, *bind]}},
)
add(
    "kinetic_plus_sigreg",
    "Build-up: kinetic-only backbone plus SIGReg, without JEPA losses or other auxiliaries.",
    weights={**aux, "sigreg": loss["sigreg"]},
)
add(
    "kinetic_plus_jepa_sigreg",
    "Build-up: kinetic-only backbone plus JEPA and SIGReg, other auxiliaries off.",
    weights={**aux, **{k: loss[k] for k in [*rx, *bind]}, "sigreg": loss["sigreg"]},
)
for t in ["kcat", "km", "ki"]:
    add(
        "single_" + t,
        "Independent "
        + t
        + " predictor with same backbone width; compare target exposure and parameter counts.",
        tasks=[t],
        exposure_matched=True,
    )
    add(
        "single_budget_" + t,
        "Independent "
        + t
        + " predictor: each has one third of shared parameter budget (within 1%).",
        tasks=[t],
        exposure_matched=True,
        capacity_fraction=1 / 3,
    )
add(
    "no_temperature",
    "Remove temperature and its availability indicator; retain pH.",
    intervention={"conditions": "no_temperature"},
)
add(
    "no_ph",
    "Remove pH and its availability indicator; retain temperature.",
    intervention={"conditions": "no_ph"},
)
add(
    "no_bond_updates",
    "D-MPNN depth one: initialization and atom readout, without iterative bond-message updates.",
    model={"dmpnn_depth": 1},
)
add(
    "mean_molecule_only",
    "Replace learned molecular attention pooling with a masked mean.",
    intervention={"molecule_pool": "mean"},
)

for family in (
    "full",
    "kinetic_only",
    "late_fusion_matched",
    "supervised_cross_attention_matched",
):
    parent = next(v for v in variants if v["name"] == family)
    parent["tuning_family"] = family
    for suffix, scale in (("lr_half", 0.5), ("lr_double", 2.0)):
        candidate = json.loads(json.dumps(parent))
        candidate.update(
            name=family + "__" + suffix,
            learning_rate_scale=scale,
            question=parent["question"] + " Equal-budget learning-rate sensitivity.",
        )
        variants.append(candidate)

retrieval = [
    dict(name="full", learn="all"),
    dict(name="none", learn="none"),
    dict(name="exact_label_mean", learn="none", estimate="mean"),
    dict(name="fixed_heuristic", learn="none"),
    dict(name="metric_only", learn="metric"),
    dict(name="residual_only", learn="residual"),
    dict(name="channel_only", learn="channel"),
    dict(name="metric_channels", learn="metric_channels"),
    dict(name="no_residual", learn="metric_channels"),
    dict(name="strict_exact", learn="all", strict_exact=True),
    dict(name="dual", learn="all", mode="dual"),
    dict(name="no_shrinkage", learn="all", no_shrinkage=True),
    dict(name="shuffled_labels", learn="all", shuffle_labels=True),
    dict(name="half_memory", learn="all", memory_fraction=0.5),
    dict(name="quarter_memory", learn="all", memory_fraction=0.25),
    dict(name="top1", learn="all", topk_exact=1),
    dict(name="top32", learn="all", topk_exact=32),
    dict(name="temperature_01", learn="all", temperature=0.1),
    dict(name="temperature_04", learn="all", temperature=0.4),
    dict(name="exclude80", learn="all", cluster_level=80),
    dict(name="exclude60", learn="all", cluster_level=60),
    dict(name="fixed_half_blend", learn="all", blend="half"),
    dict(name="no_blend", learn="all", blend="one"),
]
# metric_channels and no_residual describe the same intervention; run it once.
retrieval = [x for x in retrieval if x["name"] != "metric_channels"]
study = dict(
    schema=1,
    seeds=list(range(10)),
    protocols=["random", "cluster60"],
    split_seed=20261016,
    epochs=150,
    scheduler_epochs=150,
    early_stop_patience=0,
    batch_size=256,
    eval_batch_size=128,
    init_lr=0.00006,
    max_lr=0.00035,
    final_lr=0.000006,
    warmup_epochs=3,
    weight_decay=0.00008,
    grad_clip=1.0,
    max_residues=1024,
    selection="macro_mae_minus_0.10_r2_minus_0.001_p1mag",
    retrieval_epochs=20,
    retrieval_lr=0.001,
    retrieval_weight_decay=0.0001,
    retrieval_robust_weight=0.15,
    max_blend=0.5,
    bootstrap_replicates=2000,
    variants=[v["name"] for v in variants],
    retrieval_variants=retrieval,
)
for v in variants:
    (P / "configs" / f"{v['name']}.json").write_text(json.dumps(v, indent=2) + "\n")
(P / "configs/study.json").write_text(json.dumps(study, indent=2) + "\n")
descriptions = {
    "full": "Full LANCE-4R backbone; adding full retrieval gives LANCE-7R.",
    "no_reaction_loss": "Remove reaction JEPA losses; keep product SIGReg.",
    "no_binding_loss": "Remove binding JEPA losses; keep predictor features and SIGReg.",
    "no_jepa_losses": "Remove both JEPA loss families; keep SIGReg.",
    "no_sigreg": "Remove SIGReg; keep JEPA losses.",
    "no_jepa_no_sigreg": "Remove both JEPA and SIGReg to test their combined effect.",
    "no_stop_product_target": "Allow gradients through product targets; keep reactant anchors and binding targets stopped.",
    "no_action_decorrelation": "Remove the action decorrelation penalty.",
    "no_descriptor_loss": "Remove molecular descriptor supervision.",
    "no_ec_loss": "Remove enzyme-class supervision.",
    "no_ranking": "Remove ranking loss; keep the cohort sampler.",
    "no_cohort_pairs": "Remove reaction-pair sampling; keep balanced task allocation.",
    "no_ranking_no_cohort_pairs": "Remove ranking and reaction-pair sampling together.",
    "no_km_boundary": "Remove the Km factor-of-ten boundary loss.",
    "kinetic_only": "Same full architecture, trained only on kinetic labels.",
    "nll": "Use standard Gaussian negative log-likelihood (beta = 0).",
    "mse": "Train the mean with mean squared error; leave variance metrics undefined.",
}


def matrix_description(v):
    name = v["name"]
    if "__lr_" in name:
        family, scale = name.split("__lr_")
        factor = "0.5" if scale == "half" else "2"
        return f"Same as `{family}`, with all three learning-rate settings multiplied by {factor}."
    return descriptions.get(name, v["question"])


(P / "ABLATION_MATRIX.md").write_text(
    "# Ablation configs\n\n"
    "I included these 64 configs so we can compare against the full model, the supervised bases and each other. "
    "Each uses the same ten seeds and two split protocols. "
    "The reasons for the comparisons are in [STUDY_DESIGN.md](STUDY_DESIGN.md).\n\n"
    "| Config | What changes |\n|---|---|\n"
    + "".join(f"| `{v['name']}` | {matrix_description(v)} |\n" for v in variants)
    + "\nEvery backbone trains from scratch and is evaluated before retrieval. "
    "Each backbone with kcat also gets a newly fitted full retriever. "
    "The `full` backbone gets all 22 retrieval conditions.\n"
)
print(len(variants), "backbones;", len(retrieval), "retrieval variants")
