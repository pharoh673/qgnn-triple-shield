"""Step-2 smoke check (runs on Colab/Kaggle, needs the `jetnet` package).

Prints shapes, class balance, NaN/finite checks, average multiplicity, and a
couple of example jets, then demonstrates leakage-free standardization on a
stratified subsample. This is the cell to eyeball before building models.

Usage:
    python scripts/check_data.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml

# Windows consoles default to cp1252, which cannot encode the Unicode block
# characters JetNet's download progress bar prints. Force UTF-8 so the download
# does not crash with UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 — best-effort; harmless if unsupported
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import (  # noqa: E402
    Standardizer,
    derive_seed,
    load_jets,
    make_splits,
    stratified_subsample,
)
from data.paths import resolve  # noqa: E402

FEAT_NAMES = ["Δη", "Δφ", "log pT", "log E"]


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    paths = resolve()
    print(f"platform = {paths.platform} | data_dir = {paths.data}\n")

    d = cfg["data"]
    classes = d["classes"]
    n_const = d["n_const"]

    X, mask, y = load_jets(n_const=n_const, classes=classes, data_dir=paths.data)

    print("=== shapes ===")
    print(f"X    {X.shape}  ({X.dtype})")
    print(f"mask {mask.shape} ({mask.dtype})")
    print(f"y    {y.shape}  ({y.dtype})\n")

    print("=== class balance ===")
    for c, name in enumerate(classes):
        print(f"  {name}: {int((y == c).sum())} jets")
    print()

    print("=== sanity ===")
    print(f"all finite: {np.isfinite(X).all()}")
    print(f"mean constituents/jet: {mask.sum(1).mean():.2f} (cap {n_const})")
    print(f"pads are zero: {np.allclose(X[~mask], 0.0)}\n")

    print("=== example jets (first valid 3 constituents) ===")
    for cls in range(len(classes)):
        j = int(np.flatnonzero(y == cls)[0])
        print(f"  class {classes[cls]} (jet {j}), feats = {FEAT_NAMES}")
        valid = np.flatnonzero(mask[j])[:3]
        for p in valid:
            print(f"    {np.round(X[j, p], 3)}")
    print()

    # Leakage-free standardization demo on a stratified subsample.
    pool_idx, test_idx = make_splits(y, n_test=d["n_test"], seed=cfg["seed"])
    sub_seed = derive_seed(cfg["seed"], 100, 0)
    sub = stratified_subsample(y[pool_idx], n=100, seed=sub_seed)
    tr_idx = pool_idx[sub]

    sc = Standardizer.fit(X[tr_idx], mask[tr_idx])
    z_tr = sc.transform(X[tr_idx], mask[tr_idx])
    print("=== standardizer (fit on N=100 train subsample only) ===")
    print(f"  train pool size: {len(pool_idx)} | fixed test size: {len(test_idx)}")
    print(f"  fit mean: {np.round(sc.mean, 3)}")
    print(f"  fit std : {np.round(sc.std, 3)}")
    print(f"  standardized train valid mean≈0: {np.round(z_tr[mask[tr_idx]].mean(0), 4)}")
    print(f"  standardized train valid std ≈1: {np.round(z_tr[mask[tr_idx]].std(0), 4)}")


if __name__ == "__main__":
    main()
