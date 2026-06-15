"""Non-equivariant control (Fork A1) — budget-matched to QuantumDeepSets.

Symmetry is broken by adding a per-slot **positional encoding** to each
constituent's features before the shared circuit. Because the positional vector
depends on the constituent's slot index, permuting constituents changes the
per-slot inputs and hence the (still mean-pooled) jet embedding — so the model is
NOT permutation invariant.

Budget match (exact): the circuit reuses the *same* ``QuantumEncoder`` weight
shape and the *same* head architecture as the equivariant model. The positional
encoding is a fixed (non-trainable) sinusoidal buffer, so it adds zero trainable
parameters. Hence trainable-parameter counts are identical, isolating the effect
of the symmetry constraint itself.

(An alternative "per-slot independent circuit weights" break — A2 — can be added
later behind a flag; A1 is the cleaner controlled ablation.)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.quantum_equiv import QuantumEncoder, make_head, masked_mean


def sinusoidal_positional_encoding(n_positions: int, dim: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal PE, shape [n_positions, dim], O(1) values."""
    pe = torch.zeros(n_positions, dim)
    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 2))
    )
    pe[:, 0::2] = torch.sin(position * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe


class NonEquivariantQuantumTagger(nn.Module):
    """Budget-matched, order-SENSITIVE quantum tagger."""

    def __init__(
        self,
        n_qubits: int,
        n_feat: int = 4,
        reupload: int = 3,
        trainable_per_qubit: int = 3,
        head_bottleneck: bool = True,
        max_positions: int = 64,
        pos_scale: float = 1.0,
        noise=None,
        seed: int = 0,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = QuantumEncoder(
            n_qubits, n_feat, reupload, trainable_per_qubit, noise, seed
        )
        self.head = make_head(n_qubits, head_bottleneck)
        # Non-trainable positional encoding in feature space (adds 0 parameters).
        pe = pos_scale * sinusoidal_positional_encoding(max_positions, n_feat)
        self.register_buffer("pos", pe)

    def embed(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, p, f = X.shape
        X = X + self.pos[:p].unsqueeze(0)  # break permutation symmetry
        z = self.encoder(X.reshape(b * p, f)).reshape(b, p, self.n_qubits)
        return masked_mean(z, mask)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(X, mask)).squeeze(-1)
