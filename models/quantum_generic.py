"""Generic deep ansatz with a GLOBAL cost — diagnostic-only barren-plateau comparator.

This model exists solely for the trainability figure, NOT the performance sweep. It
is the canonical Cerezo-et-al. barren-plateau setup: an expressive, deep
hardware-efficient circuit read out through a GLOBAL observable ⟨Z_0 Z_1 ... Z_{n-1}⟩.
Under random initialization the gradient variance of such a global-cost, expressive
circuit decays exponentially in the qubit count — in contrast to the equivariant
model's shallow, LOCAL cost (Σ_i ⟨Z_i⟩), which stays trainable.

It exposes the same ``.encoder.qlayer.weights`` interface as the other taggers so
it plugs straight into ``train/diagnostics.py``.
"""

from __future__ import annotations

import functools
import operator

import pennylane as qml
import torch
import torch.nn as nn

from models.quantum_equiv import make_device, masked_mean


def generic_circuit(inputs, weights, n_qubits, n_feat, depth):
    """Deep hardware-efficient ansatz; returns a single GLOBAL parity expectation."""
    # Encode features once at the input.
    for q in range(n_qubits):
        qml.RY(inputs[..., q % n_feat], wires=q)
        qml.RZ(inputs[..., (q + 1) % n_feat], wires=q)
    # Deep trainable block: RX-RY-RZ per qubit + ring CZ each layer.
    for layer in range(depth):
        for q in range(n_qubits):
            qml.RX(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)
            qml.RZ(weights[layer, q, 2], wires=q)
        if n_qubits == 2:
            qml.CZ(wires=[0, 1])
        elif n_qubits > 2:
            for q in range(n_qubits):
                qml.CZ(wires=[q, (q + 1) % n_qubits])
    # GLOBAL observable: product Z_0 ⊗ Z_1 ⊗ ... ⊗ Z_{n-1}.
    obs = functools.reduce(operator.matmul, [qml.PauliZ(q) for q in range(n_qubits)])
    return qml.expval(obs)


class GenericEncoder(nn.Module):
    """Deep global-cost per-constituent circuit: [M, n_feat] -> [M, 1]."""

    def __init__(self, n_qubits: int, n_feat: int = 4, depth: int = 12, seed: int = 0):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        dev = make_device(n_qubits, mixed=False)

        def _qnode(inputs, weights):
            return generic_circuit(inputs, weights, n_qubits, n_feat, depth)

        qnode = qml.QNode(_qnode, dev, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (depth, n_qubits, 3)})
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.qlayer.weights.copy_(
                0.1 * torch.randn(self.qlayer.weights.shape, generator=g)
            )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # [M, n_feat] -> [M, 1]
        out = self.qlayer(feats)
        return out.unsqueeze(-1) if out.dim() == 1 else out


class GenericQuantumTagger(nn.Module):
    """Diagnostic-only deep/global-cost tagger (barren-plateau comparator)."""

    def __init__(self, n_qubits: int, n_feat: int = 4, depth: int = 12, seed: int = 0):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = GenericEncoder(n_qubits, n_feat, depth, seed)
        self.head = nn.Linear(1, 1)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, p, f = X.shape
        z = self.encoder(X.reshape(b * p, f)).reshape(b, p, 1)
        return self.head(masked_mean(z, mask)).squeeze(-1)
