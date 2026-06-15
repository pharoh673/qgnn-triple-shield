"""Local, JetNet-free unit tests for the pure preprocessing in data/loader.py.

Run: python -m pytest tests/test_loader_logic.py -q
or:  python tests/test_loader_logic.py   (prints a human-readable report)

These verify the math (relative features, top-pT truncation, no-leakage
standardization, balanced subsampling, fixed test set, seed determinism) on
hand-built synthetic arrays, so we get real signal without downloading JetNet.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import (  # noqa: E402
    Standardizer,
    derive_seed,
    make_splits,
    relative_features,
    stratified_subsample,
    truncate_pad,
)


def _toy(n_jets=6, n_part=5):
    """Build a toy JetNet-shaped pair with known values."""
    rng = np.random.default_rng(0)
    # particle columns: etarel, phirel, ptrel, mask
    particle = np.zeros((n_jets, n_part, 4), np.float32)
    particle[..., 0] = rng.normal(0, 0.2, (n_jets, n_part))   # etarel
    particle[..., 1] = rng.normal(0, 0.2, (n_jets, n_part))   # phirel
    particle[..., 2] = rng.uniform(0.01, 1.0, (n_jets, n_part))  # ptrel
    # First 3 particles valid, rest padded, for every jet.
    particle[:, :3, 3] = 1.0
    particle[:, 3:, :3] = 0.0  # padded slots carry zeros
    # jet columns: type, pt, eta, mass, num_particles
    jet = np.zeros((n_jets, 5), np.float32)
    jet[:, 0] = np.array([0, 1] * (n_jets // 2))   # alternating labels
    jet[:, 1] = 500.0    # jet pt
    jet[:, 2] = 0.3      # jet eta
    jet[:, 4] = 3
    return particle, jet


def test_relative_features_shapes_and_mask():
    particle, jet = _toy()
    feats, mask = relative_features(particle, jet)
    assert feats.shape == (6, 5, 4)
    assert mask.shape == (6, 5)
    assert mask[:, :3].all() and not mask[:, 3:].any()
    # Padded slots must be exactly zero in every feature.
    assert np.allclose(feats[~mask], 0.0)
    # Δη, Δφ are passed through unchanged for valid slots.
    assert np.allclose(feats[..., 0][mask], particle[..., 0][mask])
    # log pT, log E finite and ordered (E >= pT since cosh(eta) >= 1).
    assert np.isfinite(feats[mask]).all()
    assert (feats[..., 3][mask] >= feats[..., 2][mask] - 1e-4).all()


def test_truncate_keeps_hardest():
    # One jet, 4 valid particles with distinct ptrel; keep top-2.
    feats = np.zeros((1, 4, 4), np.float32)
    feats[0, :, 2] = [0.1, 0.9, 0.4, 0.7]   # put ptrel in feature col 2 too
    mask = np.ones((1, 4), bool)
    ptrel = np.array([[0.1, 0.9, 0.4, 0.7]], np.float32)
    out, om = truncate_pad(feats, mask, ptrel, n_const=2)
    assert out.shape == (1, 2, 4) and om.shape == (1, 2)
    # Hardest two are 0.9 then 0.7.
    assert np.allclose(out[0, :, 2], [0.9, 0.7])


def test_truncate_pads_when_short():
    feats = np.ones((1, 2, 4), np.float32)
    mask = np.ones((1, 2), bool)
    ptrel = np.array([[0.5, 0.3]], np.float32)
    out, om = truncate_pad(feats, mask, ptrel, n_const=5)
    assert out.shape == (1, 5, 4)
    assert om[0, :2].all() and not om[0, 2:].any()
    assert np.allclose(out[0, 2:], 0.0)


def test_standardizer_no_leakage_and_zero_pads():
    rng = np.random.default_rng(1)
    feats = rng.normal(5.0, 3.0, (20, 4, 4)).astype(np.float32)
    mask = np.zeros((20, 4), bool)
    mask[:, :2] = True
    feats = feats * mask[..., None]
    sc = Standardizer.fit(feats, mask)
    z = sc.transform(feats, mask)
    valid = z[mask]
    # Standardized valid features: ~zero mean, ~unit std.
    assert np.allclose(valid.mean(0), 0.0, atol=1e-5)
    assert np.allclose(valid.std(0), 1.0, atol=1e-5)
    # Pads stay exactly zero.
    assert np.allclose(z[~mask], 0.0)


def test_stratified_subsample_balanced_and_deterministic():
    y = np.array([0] * 50 + [1] * 50)
    a = stratified_subsample(y, n=20, seed=123)
    b = stratified_subsample(y, n=20, seed=123)
    assert np.array_equal(a, b)               # deterministic
    assert len(a) == 20
    assert (y[a] == 0).sum() == 10 and (y[a] == 1).sum() == 10  # balanced


def test_make_splits_fixed_and_disjoint():
    y = np.array([0] * 100 + [1] * 100)
    pool, test = make_splits(y, n_test=40, seed=42)
    pool2, test2 = make_splits(y, n_test=40, seed=42)
    assert np.array_equal(test, test2)        # test set is fixed
    assert len(test) == 40
    assert (y[test] == 0).sum() == 20 and (y[test] == 1).sum() == 20
    assert len(np.intersect1d(pool, test)) == 0   # disjoint


def test_derive_seed_distinct():
    s1 = derive_seed(42, 20, 0)
    s2 = derive_seed(42, 20, 1)
    s3 = derive_seed(42, 50, 0)
    assert len({s1, s2, s3}) == 3             # distinct streams


def _report():
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
    print("data/loader.py pure-logic tests\n" + "-" * 34)
    ok = _report()
    sys.exit(0 if ok else 1)
