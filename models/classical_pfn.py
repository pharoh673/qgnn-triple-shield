"""Classical baseline — Particle Flow Network / Deep Sets (Komiske-Metodiev-Thaler).

The honest permutation-invariant reference curve:

    jet embedding = ρ( pool_i φ(x_i) ),   logit = head(embedding)

with a shared per-particle MLP φ, a masked symmetric (mean) pool, and a post-pool
MLP ρ. Permutation invariance comes from the symmetric pool, exactly as for the
equivariant quantum model — so this isolates "quantum vs classical" at equal
inductive bias.

Width is config-driven (``phi_hidden``, ``rho_hidden``); use ``count_trainable``
to report/compare parameter budgets against the quantum models.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.quantum_equiv import masked_mean


def _mlp(sizes: Sequence[int], act=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:  # no activation on the final layer
            layers.append(act())
    return nn.Sequential(*layers)


class ParticleFlowNetwork(nn.Module):
    """Classical Deep Sets / PFN baseline."""

    def __init__(
        self,
        n_feat: int = 4,
        phi_hidden: Sequence[int] = (16, 16),
        rho_hidden: Sequence[int] = (16,),
        seed: int = 0,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.phi = _mlp([n_feat, *phi_hidden])
        self.latent = phi_hidden[-1]
        self.rho = _mlp([self.latent, *rho_hidden, 1])

    def embed(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, p, f = X.shape
        z = self.phi(X.reshape(b * p, f)).reshape(b, p, self.latent)
        return masked_mean(z, mask)  # padded constituents excluded

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.rho(self.embed(X, mask)).squeeze(-1)  # [B] logits
