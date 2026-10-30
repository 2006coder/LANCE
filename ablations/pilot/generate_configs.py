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
    vrex=0.0,
    kcat_tail=0.0,
    kcat_mechanism_ranking=0.0,
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


loss_terms = [
    "kinetic",
    "jepa",
    "jepa_delta",
    "jepa_contrastive",
    "action_decorrelation",
    "sigreg",
    "vrex",
    "binding_aux",
    "binding_jepa",
    "binding_jepa_delta",
    "binding_jepa_contrastive",
    "kcat_tail",
    "kcat_mechanism_ranking",
    "kcat_reaction_ranking",
    "km_p1",
    "ec_aux",
]
kinetic_only = {key: (1.0 if key == "kinetic" else 0.0) for key in loss_terms}
binding = {
    key: loss[key]
    for key in ("binding_jepa", "binding_jepa_delta", "binding_jepa_contrastive")
}
reaction = {key: loss[key] for key in ("jepa", "jepa_delta", "jepa_contrastive")}
add(
    "kinetic_only",
    "Same full architecture; only kinetic loss is active.",
    weights=kinetic_only,
)
add(
    "kinetic_plus_binding_jepa",
    "Kinetic plus binding JEPA; every other auxiliary loss is zero.",
    weights={**kinetic_only, **binding},
)
add(
    "kinetic_plus_reaction_jepa",
    "Kinetic plus reaction JEPA; every other auxiliary loss is zero.",
    weights={**kinetic_only, **reaction},
)
add(
    "kinetic_plus_jepa",
    "Kinetic plus both JEPA families; every other auxiliary loss is zero.",
    weights={**kinetic_only, **binding, **reaction},
)
add(
    "kinetic_plus_jepa_sigreg",
    "Kinetic plus both JEPA families and SIGReg; other auxiliaries are zero.",
    weights={**kinetic_only, **binding, **reaction, "sigreg": loss["sigreg"]},
)
add(
    "full",
    "Original full loss recipe; reference for the remaining auxiliary losses as a group.",
)

study = dict(
    schema=1,
    design="kinetic_binding_reaction_sigreg",
    loss_terms=loss_terms,
    primary_base="kinetic_only/backbone",
    seeds=[0, 1],
    protocols=["random"],
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
    primary_contrasts=[
        dict(name="binding_added", a="kinetic_plus_binding_jepa", b="kinetic_only"),
        dict(name="reaction_added", a="kinetic_plus_reaction_jepa", b="kinetic_only"),
        dict(
            name="reaction_after_binding",
            a="kinetic_plus_jepa",
            b="kinetic_plus_binding_jepa",
        ),
        dict(
            name="binding_after_reaction",
            a="kinetic_plus_jepa",
            b="kinetic_plus_reaction_jepa",
        ),
        dict(
            name="sigreg_after_jepa",
            a="kinetic_plus_jepa_sigreg",
            b="kinetic_plus_jepa",
        ),
        dict(name="remaining_auxiliaries", a="full", b="kinetic_plus_jepa_sigreg"),
        dict(name="full_vs_kinetic", a="full", b="kinetic_only"),
    ],
    retrieval_backbones=["full"],
    retrieval_variants=[
        dict(name="full", learn="all"),
        dict(name="none", learn="none"),
        dict(name="exact_label_mean", learn="none", estimate="mean"),
    ],
)
for v in variants:
    (P / "configs" / f"{v['name']}.json").write_text(json.dumps(v, indent=2) + "\n")
(P / "configs/study.json").write_text(json.dumps(study, indent=2) + "\n")
(P / "ABLATION_MATRIX.md").write_text(
    "# Core loss ablations\n\n"
    "I kept the same architecture and changed only the loss coefficients. "
    "Two paired seeds on the random protocol give 12 training jobs.\n\n"
    "| Config | What changes |\n|---|---|\n"
    + "".join(f"| `{v['name']}` | {v['question']} |\n" for v in variants)
    + "\nAll six predict kcat, Km and Ki. Backbone comparisons use the same report rows and seeds. "
    "Only `full` gets retrieval: learned memory, an exact-label average and no memory. "
    "Those reuse its checkpoint and do not add backbone training jobs.\n"
)
print(len(variants), "configs;", len(variants) * len(study["seeds"]), "backbone runs")
