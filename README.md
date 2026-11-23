# LANCE-7R

LANCE-7R predicts enzyme turnover numbers, Michaelis constants and inhibition constants. The LANCE-4R backbone uses the molecular graph to attend to protein residues. Two learned vectors update the molecular representation along binding and catalytic paths, and each task head reads the features it needs. LANCE-7R adds a learned turnover correction from training measurements with matching reactants outside the query enzyme’s 99% sequence-identity cluster. When no match is available, it returns the backbone prediction.

## Code

- [Backbone](LANCE/model.py) and [training objectives](LANCE/losses.py) implement the shared representation, task heads and auxiliary losses.
- [Label memory](LANCE/lance7r/) implements retrieval, fitting and evaluation of the turnover correction.
- [Training jobs](LANCE/jobs/) and [ablation studies](ablations/) cover checkpoint recovery, ten-seed ensembles and evaluation across sequence-identity thresholds.
- [Analysis](analysis/) computes loss contrasts, paired benchmark intervals and a same-reactant averaging control.

This repository is an interim research release. The manuscript contains the evaluation results and study limitations. CatPred-derived utilities carry their [original attribution](third_party/NOTICE). [Manuscript on Overleaf](https://www.overleaf.com/project/6a87c05999040fba5e8548b8/share#62e271e0199234af5fdf054f98e7814f830612f1f87647a5) · [Figures](assets/figures/)
