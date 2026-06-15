"""Step-6 demo: (1) the trainer runs end-to-end on real data and reports metrics;
(2) the barren-plateau gradient-variance probe across a qubit grid for the
equivariant vs non-equivariant model.

Usage: python scripts/check_train_diag.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(6)

from data.loader import derive_seed, load_jets, make_splits, stratified_subsample  # noqa: E402
from data.paths import resolve  # noqa: E402
from models.quantum_equiv import QuantumDeepSets  # noqa: E402
from models.quantum_generic import GenericQuantumTagger  # noqa: E402
from models.quantum_noneq import NonEquivariantQuantumTagger  # noqa: E402
from train.diagnostics import gradient_variance_curve, make_probe_batch  # noqa: E402
from train.trainer import TrainConfig, train_and_evaluate  # noqa: E402


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(here, "config.yaml")))
    paths = resolve()
    d = cfg["data"]
    X, mask, y = load_jets(d["n_const"], d["classes"], data_dir=paths.data)
    pool, test = make_splits(y, n_test=d["n_test"], seed=cfg["seed"])

    # ---- (1) trainer end-to-end on a real subsample -------------------------
    print("=== trainer end-to-end (equivariant, n_qubits=4) ===")
    tcfg = TrainConfig.from_dict(cfg["train"])
    te = test[:600]
    for N in (100, 300):
        seed = derive_seed(cfg["seed"], N, 0)
        tr = pool[stratified_subsample(y[pool], N, seed)]
        t0 = time.time()
        m = train_and_evaluate(
            lambda s: QuantumDeepSets(n_qubits=4, reupload=3, seed=s),
            X[tr], mask[tr], y[tr], X[te], mask[te], y[te], tcfg, seed,
        )
        print(f"  N={N:4d}: test AUC={m['test_auc']:.3f}  train AUC={m['train_auc']:.3f}  "
              f"gap={m['gen_gap']:+.3f}  restarts={m['restarts_used']} escaped={m['escaped']}  "
              f"[{time.time()-t0:.0f}s]")

    # ---- (2) gradient-variance probe (barren-plateau) -----------------------
    # Probe the LANDSCAPE under uniform-random init (not the small-angle training
    # init). Contrast: equivariant (shallow, local cost) vs generic (deep, GLOBAL
    # cost) — the canonical Cerezo barren-plateau setup. A1 shown for reference.
    diag = cfg["diagnostics"]["grad_variance"]
    grid = diag["n_qubits_grid"]
    n_inits = min(int(diag["n_inits"]), 20)  # cap for the demo
    batch = make_probe_batch(n_jets=12, n_const=12)

    def uniform_init(model, seed):
        g = torch.Generator().manual_seed(seed)
        w = model.encoder.qlayer.weights
        with torch.no_grad():
            w.copy_((torch.rand(w.shape, generator=g) * 2 - 1) * math.pi)
        return model

    def eq_factory(nq, seed):
        return uniform_init(QuantumDeepSets(n_qubits=nq, reupload=3, seed=seed), seed)

    def neq_factory(nq, seed):
        return uniform_init(NonEquivariantQuantumTagger(n_qubits=nq, reupload=3, seed=seed), seed)

    def gen_factory(nq, seed):
        return uniform_init(GenericQuantumTagger(n_qubits=nq, depth=12, seed=seed), seed)

    print(f"\n=== gradient-variance probe (uniform init, n_inits={n_inits}, θ[0]) ===")
    t0 = time.time()
    eq = gradient_variance_curve(eq_factory, grid, n_inits, batch)
    neq = gradient_variance_curve(neq_factory, grid, n_inits, batch)
    gen = gradient_variance_curve(gen_factory, grid, n_inits, batch)
    print(f"  {'n_qubits':>8} | {'equiv (local)':>14} | {'A1 (local)':>14} | {'generic (global)':>16}")
    for e, n, g in zip(eq, neq, gen):
        print(f"  {e['n_qubits']:>8} | {e['grad_var']:>14.3e} | {n['grad_var']:>14.3e} | {g['grad_var']:>16.3e}")
    decay = lambda c: c[0]['grad_var'] / max(c[-1]['grad_var'], 1e-30)
    print(f"\n  Var(2q)/Var(8q) decay:  equiv={decay(eq):.1f}x   "
          f"A1={decay(neq):.1f}x   generic={decay(gen):.1f}x  (bigger = more plateau)")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
