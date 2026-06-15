"""Training-plumbing sanity checks (build order step 4).

  * test_pfn_overfits_tiny_batch: the classical PFN can drive a tiny, learnable
    batch to ~zero loss / 100% accuracy — confirms the full forward/backward/optim
    loop works.
  * test_quantum_grads_flow: a single backward pass through the quantum models
    produces finite, non-zero gradients on the circuit weights — confirms the
    PennyLane↔Torch autodiff wiring before we rely on it in the sweep.

Run: python tests/test_overfit.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classical_pfn import ParticleFlowNetwork  # noqa: E402
from models.quantum_equiv import QuantumDeepSets, count_trainable  # noqa: E402
from models.quantum_noneq import NonEquivariantQuantumTagger  # noqa: E402


def _toy_batch(b=16, p=8, f=4, seed=0):
    """A small batch with a learnable rule: label = (mean Δη > 0)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(b, p, f, generator=g)
    mask = torch.ones(b, p, dtype=torch.bool)
    y = (X[:, :, 0].mean(dim=1) > 0).float()
    return X, mask, y


def test_pfn_overfits_tiny_batch():
    X, mask, y = _toy_batch()
    model = ParticleFlowNetwork(phi_hidden=(16, 16), rho_hidden=(16,), seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(400):
        opt.zero_grad()
        out = model(X, mask)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
    acc = ((out > 0).float() == y).float().mean().item()
    assert acc == 1.0 or loss.item() < 0.05, f"PFN failed to overfit: acc={acc}, loss={loss.item():.3f}"


def test_quantum_grads_flow():
    X, mask, y = _toy_batch(b=4, p=4, seed=1)
    lossf = nn.BCEWithLogitsLoss()
    for name, model in [
        ("equivariant", QuantumDeepSets(n_qubits=2, seed=1)),
        ("non_equivariant", NonEquivariantQuantumTagger(n_qubits=2, seed=1)),
    ]:
        model.train()
        out = model(X, mask)
        loss = lossf(out, y)
        loss.backward()
        w = model.encoder.qlayer.weights
        assert w.grad is not None, f"{name}: no grad on circuit weights"
        assert torch.isfinite(w.grad).all(), f"{name}: non-finite grad"
        assert w.grad.abs().sum() > 0, f"{name}: zero grad (autodiff not wired)"


def _report():
    eq = QuantumDeepSets(n_qubits=4, seed=1)
    neq = NonEquivariantQuantumTagger(n_qubits=4, seed=1)
    pfn = ParticleFlowNetwork(phi_hidden=(16, 16), rho_hidden=(16,))
    print("trainable parameter budgets (n_qubits=4):")
    print(f"  equivariant quantum     : {count_trainable(eq)}")
    print(f"  non-equivariant quantum : {count_trainable(neq)}")
    print(f"  classical PFN (16,16|16): {count_trainable(pfn)}")
    print()

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("training-plumbing sanity (step 4)\n" + "-" * 34)
    ok = _report()
    sys.exit(0 if ok else 1)
