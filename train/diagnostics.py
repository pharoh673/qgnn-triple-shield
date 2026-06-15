"""Trainability diagnostic — the barren-plateau gradient-variance probe.

Barren plateaus (McClean et al.) manifest as the variance of a cost gradient
component vanishing (exponentially in qubit count) over random initializations,
so the landscape is flat and training stalls. Schatzki et al. argue
permutation-equivariant QNNs avoid this.

``gradient_variance`` estimates, for a FIXED circuit parameter, the variance of
∂L/∂θ over many random initializations, at a given qubit count. Running it across
a qubit grid for the equivariant vs non-equivariant model gives the headline
trainability figure: does the equivariant model's gradient variance stay flat
while the non-equivariant one decays?

Note (honest caveat): models 3a/3b share the same per-constituent ansatz, so at
the small scales here (2-8 qubits, depth 3) the difference may be modest; the
infrastructure also accepts any factory exposing ``model.encoder.qlayer.weights``,
so a deliberately non-equivariant ansatz can be plugged in for a stronger contrast.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

# A model factory maps (n_qubits, seed) -> nn.Module exposing encoder.qlayer.weights.
ModelFactory = Callable[[int, int], nn.Module]


def make_probe_batch(
    n_jets: int = 12,
    n_const: int = 12,
    n_feat: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A fixed random batch. Inputs are held fixed; only initial params vary."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_jets, n_const, n_feat, generator=g)
    mask = torch.ones(n_jets, n_const, dtype=torch.bool)
    y = (torch.rand(n_jets, generator=g) > 0.5).float()
    return X, mask, y


def gradient_variance(
    factory: ModelFactory,
    n_qubits: int,
    n_inits: int,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target_index: int = 0,
    seed0: int = 0,
) -> dict:
    """Variance of ∂L/∂θ_target over ``n_inits`` random initializations.

    target_index indexes into the flattened circuit weight tensor; index 0 exists
    for every qubit count, so it is comparable across the qubit grid.
    """
    X, mask, y = batch
    lossf = nn.BCEWithLogitsLoss()
    grads = np.empty(n_inits, dtype=float)
    for i in range(n_inits):
        model = factory(n_qubits, seed0 + i)
        model.train()
        model.zero_grad(set_to_none=True)
        loss = lossf(model(X, mask), y)
        loss.backward()
        g = model.encoder.qlayer.weights.grad
        if g is None:
            raise RuntimeError("no gradient on circuit weights — autodiff not wired")
        grads[i] = g.flatten()[target_index].item()  # the GRADIENT, not the weight
    return {
        "n_qubits": n_qubits,
        "n_inits": n_inits,
        "grad_var": float(grads.var()),
        "grad_mean": float(grads.mean()),
        "grad_absmean": float(np.abs(grads).mean()),
    }


def gradient_variance_curve(
    factory: ModelFactory,
    n_qubits_grid: list[int],
    n_inits: int,
    batch=None,
    target_index: int = 0,
    seed0: int = 0,
) -> list[dict]:
    """Run the probe across a qubit grid; returns one row per qubit count."""
    if batch is None:
        batch = make_probe_batch()
    return [
        gradient_variance(factory, nq, n_inits, batch, target_index, seed0)
        for nq in n_qubits_grid
    ]
