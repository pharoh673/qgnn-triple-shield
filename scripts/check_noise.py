"""Step-5 demo: test ROC-AUC degrades as noise strength p rises.

Trains one small equivariant tagger noiselessly, then re-evaluates the SAME
trained weights under increasing iid and correlated noise. A quick, inline train
loop is used on purpose — the real trainer (with early stopping) is step 6.

Usage:  python scripts/check_noise.py
"""

from __future__ import annotations

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

from sklearn.metrics import roc_auc_score  # noqa: E402

from data.loader import (  # noqa: E402
    Standardizer,
    derive_seed,
    load_jets,
    make_splits,
    stratified_subsample,
)
from data.paths import resolve  # noqa: E402
from models.noise import noise_from_config  # noqa: E402
from models.quantum_equiv import QuantumDeepSets  # noqa: E402

N_TRAIN = 400
N_EVAL = 400
N_QUBITS = 4
REUPLOAD = 3
EPOCHS = 200
LR = 0.05


def _tensor(x, dtype=torch.float32):
    return torch.tensor(np.asarray(x), dtype=dtype)


def train_inplace(model, X, m, y, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(X, m), y)
        loss.backward()
        opt.step()
    return model


def auc(model, X, m, y) -> float:
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(X, m)).numpy()
    return roc_auc_score(y, s)


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    paths = resolve()
    d, ncfg = cfg["data"], cfg["noise"]

    X, mask, y = load_jets(d["n_const"], d["classes"], data_dir=paths.data)
    pool, test = make_splits(y, n_test=d["n_test"], seed=cfg["seed"])

    sub = stratified_subsample(y[pool], N_TRAIN, derive_seed(cfg["seed"], N_TRAIN, 0))
    tr = pool[sub]
    te = test[:N_EVAL]

    sc = Standardizer.fit(X[tr], mask[tr])
    Xtr, mtr = _tensor(sc.transform(X[tr], mask[tr])), _tensor(mask[tr], torch.bool)
    ytr = _tensor(y[tr])
    Xte, mte = _tensor(sc.transform(X[te], mask[te])), _tensor(mask[te], torch.bool)
    yte = y[te]

    print(f"train N={N_TRAIN}, eval N={N_EVAL}, n_qubits={N_QUBITS}, epochs={EPOCHS}")
    t0 = time.time()
    base = QuantumDeepSets(n_qubits=N_QUBITS, reupload=REUPLOAD, seed=1)
    train_inplace(base, Xtr, mtr, ytr, EPOCHS, LR)
    state = base.state_dict()
    train_auc = auc(base, Xtr, mtr, ytr.numpy())
    base_auc = auc(base, Xte, mte, yte)
    print(f"  noiseless: train AUC={train_auc:.3f}  test AUC={base_auc:.3f}  "
          f"({time.time()-t0:.0f}s)\n")

    def eval_noise(regime: str, level: str) -> tuple[float, float]:
        nm = noise_from_config(ncfg, regime, level, N_QUBITS, REUPLOAD, seed=0)
        model = QuantumDeepSets(n_qubits=N_QUBITS, reupload=REUPLOAD, noise=nm, seed=1)
        model.load_state_dict(state)  # same trained weights, now evaluated under noise
        return nm.spec.p, auc(model, Xte, mte, yte)

    print("=== iid amplitude_damping ===")
    print(f"  p=0.000  AUC={base_auc:.3f}  (noiseless ref)")
    for name in ["low", "mid", "high"]:
        p, a = eval_noise("iid", name)
        print(f"  p={p:.3f}  AUC={a:.3f}   [{name}]")

    cc = ncfg["correlated"]
    print("\n=== correlated OU coherent %s-rotation (tau=%.1f, n_traj=%d) ==="
          % (cc["axis"], cc["tau"], ncfg["n_trajectories"]))
    print(f"  p=0.000  AUC={base_auc:.3f}  (noiseless ref)")
    for name in ["low", "mid", "high"]:
        p, a = eval_noise("correlated", name)
        print(f"  p={p:.3f}  AUC={a:.3f}   [{name}]")


if __name__ == "__main__":
    main()
