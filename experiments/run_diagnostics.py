"""Compute the barren-plateau gradient-variance curves and save them for plotting.

Probes ∂L/∂θ₀ variance over random (uniform) initializations vs n_qubits, for:
  * equivariant  (shallow, LOCAL cost Σ⟨Z_i⟩)
  * non_equivariant (A1, shares the equivariant ansatz — reference)
  * generic      (deep, GLOBAL cost ⟨Z_0…Z_{n-1}⟩ — the barren-plateau comparator)

Saves results/diagnostics_gradvar.{parquet,csv}. Cheap (~1 min on one core).

Usage: python experiments/run_diagnostics.py
"""

from __future__ import annotations

import math
import os
import sys

import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)

from data.paths import resolve  # noqa: E402
from models.quantum_equiv import QuantumDeepSets  # noqa: E402
from models.quantum_generic import GenericQuantumTagger  # noqa: E402
from models.quantum_noneq import NonEquivariantQuantumTagger  # noqa: E402
from train.diagnostics import gradient_variance_curve, make_probe_batch  # noqa: E402


def _uniform_init(model, seed):
    g = torch.Generator().manual_seed(seed)
    w = model.encoder.qlayer.weights
    with torch.no_grad():
        w.copy_((torch.rand(w.shape, generator=g) * 2 - 1) * math.pi)
    return model


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(here, "config.yaml")))
    diag = cfg["diagnostics"]["grad_variance"]
    grid = diag["n_qubits_grid"]
    n_inits = int(diag["n_inits"])
    reup = cfg["model"]["equivariant"]["reuploading_depth"]
    batch = make_probe_batch(n_jets=12, n_const=12)

    factories = {
        "equivariant": lambda nq, s: _uniform_init(QuantumDeepSets(n_qubits=nq, reupload=reup, seed=s), s),
        "non_equivariant": lambda nq, s: _uniform_init(NonEquivariantQuantumTagger(n_qubits=nq, reupload=reup, seed=s), s),
        "generic": lambda nq, s: _uniform_init(GenericQuantumTagger(n_qubits=nq, depth=12, seed=s), s),
    }

    rows = []
    for name, fac in factories.items():
        for r in gradient_variance_curve(fac, grid, n_inits, batch):
            rows.append({"model": name, **r})
            print(f"{name:>16} nq={r['n_qubits']}  Var={r['grad_var']:.3e}")

    paths = resolve()
    out = paths.results() / "diagnostics_gradvar.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    df.to_csv(str(out).replace(".parquet", ".csv"), index=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
