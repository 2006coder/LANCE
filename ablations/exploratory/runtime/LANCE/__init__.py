"""LANCE: Latent Action Networks for Catalytic Enzyme Kinetics.

This package is intentionally small and separate from the active CatPred
training code.  It implements the architectural idea from the Claude/Codex
discussion:

    enzyme + substrate -> latent catalytic action -> reaction/product embedding
    enzyme + molecule + action latents -> kcat/Km/Ki prediction

The code is a research prototype.  It is built to be easy to inspect, test, and
later wire into the existing training scripts if the design looks promising.
"""

from .model import LANCEConfig, LANCEModel
from .losses import LANCELossWeights, lance_total_loss

__all__ = [
    "LANCEConfig",
    "LANCEModel",
    "LANCELossWeights",
    "lance_total_loss",
]
