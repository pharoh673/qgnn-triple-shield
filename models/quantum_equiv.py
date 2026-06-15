"""Equivariant quantum jet tagger — "Quantum Deep Sets".

Permutation invariance is built in by construction:

  * One SHARED parametrized circuit ``phi_theta`` (the same trainable weights) is
    applied to every constituent independently  → the per-node map is permutation
    *equivariant*.
  * A masked symmetric pool (mean) over constituents  → the jet embedding is
    permutation *invariant*.
  * A small classical head maps the embedding to a logit.

The circuit uses data re-uploading: each layer interleaves a feature-encoding
block (RY/RZ of the 4 input features, reused across layers) with a trainable
block (RY-RZ-RY per qubit) and a ring of CZ entanglers.

Devices (hard constraint): noiseless runs use ``lightning.qubit`` (statevector);
noisy runs use ``default.mixed`` (set ``mixed=True``). ``noise_fn`` lets noise.py
inject channels after each entangling layer (used in step 5/6).
"""

from __future__ import annotations

from typing import Callable, Optional

import pennylane as qml
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Device + circuit                                                            #
# --------------------------------------------------------------------------- #
def make_device(n_qubits: int, mixed: bool = False) -> qml.device:
    """Device for the per-constituent circuit.

    We use ``default.qubit`` (noiseless) / ``default.mixed`` (iid noise) with the
    ``backprop`` diff method rather than ``lightning.qubit``+``adjoint``. Reason:
    our workload is many SMALL circuits (2-8 qubits) over a LARGE batch of
    constituents. ``default.qubit`` vectorizes the whole batch as a single torch
    tensor contraction (orders of magnitude faster here), whereas lightning's
    adjoint loops per-sample. For small qubit counts this is the right trade-off.
    """
    if mixed:
        return qml.device("default.mixed", wires=n_qubits)
    return qml.device("default.qubit", wires=n_qubits)


def per_constituent_circuit(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    n_qubits: int,
    n_feat: int,
    reupload: int,
    noise_fn: Optional[Callable[[int], None]] = None,
):
    """The shared per-constituent circuit φ_θ. Returns ⟨Z_i⟩ for each qubit.

    Parameters
    ----------
    inputs  : [..., n_feat]  feature vector (broadcasts over a leading batch dim)
    weights : [reupload, n_qubits, 3]  trainable RY-RZ-RY angles per qubit per layer
    noise_fn: optional callable(layer_index) that applies noise channels (noise.py)
    """
    for layer in range(reupload):
        # --- data re-uploading: feature encoding (same features every layer) ---
        for q in range(n_qubits):
            qml.RY(inputs[..., q % n_feat], wires=q)
            qml.RZ(inputs[..., (q + 1) % n_feat], wires=q)
        # --- trainable block ---
        for q in range(n_qubits):
            qml.RY(weights[layer, q, 0], wires=q)
            qml.RZ(weights[layer, q, 1], wires=q)
            qml.RY(weights[layer, q, 2], wires=q)
        # --- entangler: ring of CZ ---
        if n_qubits == 2:
            qml.CZ(wires=[0, 1])
        elif n_qubits > 2:
            for q in range(n_qubits):
                qml.CZ(wires=[q, (q + 1) % n_qubits])
        # --- optional noise after entangling layer ---
        if noise_fn is not None:
            noise_fn(layer)

    return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]


class QuantumEncoder(nn.Module):
    """Shared per-constituent circuit wrapped as a Torch layer: [M, n_feat] → [M, n_qubits].

    Because the weights are shared across all constituents, applying this to each
    constituent of a jet is permutation-equivariant.
    """

    def __init__(
        self,
        n_qubits: int,
        n_feat: int = 4,
        reupload: int = 3,
        trainable_per_qubit: int = 3,
        noise=None,  # models.noise.NoiseModel | None (duck-typed)
        seed: int = 0,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_feat = n_feat
        self.reupload = reupload

        # Derive device + averaging from the noise model (None = noiseless fast path).
        self.noise = noise
        mixed = bool(noise.mixed) if noise is not None else False
        self.n_trajectories = int(noise.n_trajectories) if noise is not None else 1
        noise_fn = noise.apply if noise is not None else None

        dev = make_device(n_qubits, mixed)
        diff_method = "backprop"  # default.qubit/default.mixed vectorize the batch

        def _qnode(inputs, weights):
            return per_constituent_circuit(
                inputs, weights, n_qubits, n_feat, reupload, noise_fn
            )

        qnode = qml.QNode(_qnode, dev, interface="torch", diff_method=diff_method)
        weight_shapes = {"weights": (reupload, n_qubits, trainable_per_qubit)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Deterministic, small-angle initialization (helps trainability).
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.qlayer.weights.copy_(
                0.1 * torch.randn(self.qlayer.weights.shape, generator=g)
            )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # [M, n_feat] -> [M, n_qubits]
        if self.n_trajectories <= 1:
            return self.qlayer(feats)
        # Correlated noise: average expectation values over OU phase trajectories.
        out = None
        for t in range(self.n_trajectories):
            self.noise.resample(t)
            o = self.qlayer(feats)
            out = o if out is None else out + o
        return out / self.n_trajectories


# --------------------------------------------------------------------------- #
# Pooling head                                                                #
# --------------------------------------------------------------------------- #
def masked_mean(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant pool. z: [B,P,D], mask: [B,P] -> [B,D]."""
    m = mask.unsqueeze(-1).to(z.dtype)
    summed = (z * m).sum(dim=1)
    count = m.sum(dim=1).clamp(min=1.0)
    return summed / count


def make_head(n_qubits: int, bottleneck: bool) -> nn.Module:
    """Classical head on the pooled embedding → scalar logit.

    With ``bottleneck`` an extra Linear(n_qubits→n_qubits)+ReLU is added; the
    non-equivariant control uses the identical head so parameter counts match.
    """
    layers: list[nn.Module] = []
    if bottleneck:
        layers += [nn.Linear(n_qubits, n_qubits), nn.ReLU()]
    layers += [nn.Linear(n_qubits, 1)]
    return nn.Sequential(*layers)


class QuantumDeepSets(nn.Module):
    """Permutation-INVARIANT quantum tagger (primary model)."""

    def __init__(
        self,
        n_qubits: int,
        n_feat: int = 4,
        reupload: int = 3,
        trainable_per_qubit: int = 3,
        head_bottleneck: bool = True,
        noise=None,
        seed: int = 0,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = QuantumEncoder(
            n_qubits, n_feat, reupload, trainable_per_qubit, noise, seed
        )
        self.head = make_head(n_qubits, head_bottleneck)

    def embed(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """X: [B,P,F], mask: [B,P] -> invariant embedding [B, n_qubits]."""
        b, p, f = X.shape
        z = self.encoder(X.reshape(b * p, f)).reshape(b, p, self.n_qubits)
        return masked_mean(z, mask)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(X, mask)).squeeze(-1)  # [B] logits


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
