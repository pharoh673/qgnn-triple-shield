"""Scientific-correctness test (build order step 3).

Numerically verifies the central architectural claim:
  * QuantumDeepSets is permutation-INVARIANT: a jet and a randomly permuted copy
    give the same output.
  * NonEquivariantQuantumTagger is NOT invariant.
  * Both have identical trainable-parameter counts (budget-matched control).

Run: python tests/test_invariance.py   (prints a report)
or:  python -m pytest tests/test_invariance.py -q
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.quantum_equiv import QuantumDeepSets, count_trainable  # noqa: E402
from models.quantum_noneq import NonEquivariantQuantumTagger  # noqa: E402

N_QUBITS = 4
N_CONST = 16
N_FEAT = 4


def _jet_and_permutation():
    torch.manual_seed(7)
    X = torch.randn(1, N_CONST, N_FEAT)
    mask = torch.ones(1, N_CONST, dtype=torch.bool)
    perm = torch.randperm(N_CONST)
    Xp = X[:, perm, :].contiguous()
    maskp = mask[:, perm].contiguous()
    return X, mask, Xp, maskp


def test_equivariant_is_invariant():
    model = QuantumDeepSets(n_qubits=N_QUBITS, seed=1).eval()
    X, mask, Xp, maskp = _jet_and_permutation()
    with torch.no_grad():
        o1 = model(X, mask)
        o2 = model(Xp, maskp)
    diff = (o1 - o2).abs().max().item()
    assert diff < 1e-5, f"equivariant output changed under permutation by {diff:.2e}"


def test_nonequivariant_is_not_invariant():
    model = NonEquivariantQuantumTagger(n_qubits=N_QUBITS, seed=1).eval()
    X, mask, Xp, maskp = _jet_and_permutation()
    with torch.no_grad():
        o1 = model(X, mask)
        o2 = model(Xp, maskp)
    diff = (o1 - o2).abs().max().item()
    assert diff > 1e-4, f"non-equivariant model was (wrongly) invariant: diff {diff:.2e}"


def test_param_budgets_match():
    eq = QuantumDeepSets(n_qubits=N_QUBITS, seed=1)
    neq = NonEquivariantQuantumTagger(n_qubits=N_QUBITS, seed=1)
    pe, pn = count_trainable(eq), count_trainable(neq)
    assert pe == pn, f"param mismatch: equivariant={pe}, non-equivariant={pn}"


def _report():
    X, mask, Xp, maskp = _jet_and_permutation()
    eq = QuantumDeepSets(n_qubits=N_QUBITS, seed=1).eval()
    neq = NonEquivariantQuantumTagger(n_qubits=N_QUBITS, seed=1).eval()
    with torch.no_grad():
        eq_diff = (eq(X, mask) - eq(Xp, maskp)).abs().max().item()
        neq_diff = (neq(X, mask) - neq(Xp, maskp)).abs().max().item()
    print(f"n_qubits={N_QUBITS}, n_const={N_CONST}")
    print(f"  equivariant   Δ(perm) = {eq_diff:.3e}   (want ≈0, invariant)")
    print(f"  non-equivar.  Δ(perm) = {neq_diff:.3e}   (want >0, order-sensitive)")
    print(f"  trainable params: equivariant={count_trainable(eq)}  "
          f"non-equivariant={count_trainable(neq)}")

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
    print("permutation-invariance test (step 3)\n" + "-" * 37)
    ok = _report()
    sys.exit(0 if ok else 1)
