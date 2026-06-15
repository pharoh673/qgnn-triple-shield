"""JetNet → model-ready particle clouds.

Design split (so the science is testable without a JetNet download):

  * Pure functions  (`relative_features`, `truncate_pad`, `Standardizer`,
    `stratified_subsample`, `make_splits`)  operate on plain numpy arrays and
    carry no I/O.  They are unit-tested in tests/test_loader_logic.py and run
    anywhere numpy is installed.
  * `load_jets(...)` is the only function that touches the `jetnet` package and
    the disk cache.  It calls the pure functions above.

Feature convention (n_features = 4), per constituent:
    [ Δη , Δφ , log pT , log E ]
with Δη, Δφ already relative to the jet axis (JetNet's etarel/phirel), and
pT, E reconstructed to absolute scale from the jet kinematics so the log is
physically meaningful.  A separate boolean validity mask is returned alongside.

All standardization is fit on the *training subsample only* — see Standardizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Particle/jet feature layouts we request from JetNet.  Keep these explicit so
# index assumptions below never silently drift.
_PARTICLE_FEATURES = ["etarel", "phirel", "ptrel", "mask"]   # -> columns 0,1,2,3
_JET_FEATURES = ["type", "pt", "eta", "mass", "num_particles"]  # -> columns 0..4

# JetNet's 'type' jet feature is a GLOBAL class index into this canonical order,
# NOT an index into the user's requested `classes`. We remap to local labels.
_JETNET_TYPE_ORDER = ["g", "q", "t", "w", "z"]

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Pure preprocessing (no I/O — unit tested locally)                           #
# --------------------------------------------------------------------------- #
def relative_features(
    particle_data: np.ndarray,
    jet_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw JetNet arrays to (features, mask).

    Parameters
    ----------
    particle_data : [n_jets, n_part, 4]  columns = (etarel, phirel, ptrel, mask)
    jet_data      : [n_jets, 5]          columns = (type, pt, eta, mass, num_part)

    Returns
    -------
    feats : [n_jets, n_part, 4]  columns = (Δη, Δφ, log pT, log E)
    mask  : [n_jets, n_part]     bool, True = valid constituent
    """
    if particle_data.ndim != 3 or particle_data.shape[-1] != 4:
        raise ValueError(f"particle_data must be [J,P,4], got {particle_data.shape}")
    if jet_data.ndim != 2 or jet_data.shape[-1] != 5:
        raise ValueError(f"jet_data must be [J,5], got {jet_data.shape}")

    etarel = particle_data[..., 0]
    phirel = particle_data[..., 1]
    ptrel = particle_data[..., 2]
    mask = particle_data[..., 3] > 0.5

    jet_pt = jet_data[:, 1][:, None]    # [J,1] broadcast over particles
    jet_eta = jet_data[:, 2][:, None]

    # Reconstruct absolute kinematics. Massless-constituent approximation: E = pT cosh(η).
    pt = np.clip(ptrel * jet_pt, _EPS, None)
    eta_abs = etarel + jet_eta
    energy = np.clip(pt * np.cosh(eta_abs), _EPS, None)

    log_pt = np.log(pt)
    log_e = np.log(energy)

    feats = np.stack([etarel, phirel, log_pt, log_e], axis=-1).astype(np.float32)
    # Zero out padded slots so they never carry stray values into the pool.
    feats = feats * mask[..., None]
    return feats, mask


def truncate_pad(
    feats: np.ndarray,
    mask: np.ndarray,
    ptrel: np.ndarray,
    n_const: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the top-`n_const` constituents by ptrel, pad/truncate to n_const.

    Sorting by ptrel (descending) before truncation means that when n_const is
    smaller than the stored multiplicity we keep the hardest particles, which is
    the physically sensible choice for a qubit-frugal tagger.

    Parameters
    ----------
    feats : [J, P, F]
    mask  : [J, P]  bool
    ptrel : [J, P]  used only as the sort key
    """
    j, p, f = feats.shape
    # Invalid slots get -inf key so they sort last.
    key = np.where(mask, ptrel, -np.inf)
    order = np.argsort(-key, axis=1)                      # [J,P] descending
    take = order[:, :n_const]                             # [J,n_const]

    rows = np.arange(j)[:, None]
    out_feats = feats[rows, take]                         # [J,n_const,F]
    out_mask = mask[rows, take]                           # [J,n_const]

    if n_const > p:  # pad if we asked for more than available
        pad = n_const - p
        out_feats = np.concatenate(
            [out_feats, np.zeros((j, pad, f), out_feats.dtype)], axis=1
        )
        out_mask = np.concatenate([out_mask, np.zeros((j, pad), bool)], axis=1)

    out_feats = out_feats * out_mask[..., None]           # re-zero padded slots
    return out_feats.astype(np.float32), out_mask


@dataclass
class Standardizer:
    """Per-feature mean/std, fit on valid constituents of the TRAIN subsample only.

    Padded constituents are excluded from the fit and remain zero after transform.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, feats: np.ndarray, mask: np.ndarray) -> "Standardizer":
        valid = feats[mask]                               # [n_valid, F]
        if valid.size == 0:
            raise ValueError("Standardizer.fit got no valid constituents")
        mean = valid.mean(axis=0)
        std = valid.std(axis=0)
        std = np.where(std < _EPS, 1.0, std)              # guard constant features
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = (feats - self.mean) / self.std
        return (out * mask[..., None]).astype(np.float32)  # keep pads at zero


def stratified_subsample(
    y: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """Return indices for a class-balanced subsample of total size `n`.

    Balanced = n//2 per class for the binary task. Deterministic given `seed`.
    Returns shuffled indices (so batching does not see class-sorted data).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"expected binary labels, found classes {classes}")
    per = n // 2
    picks = []
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        if len(idx_c) < per:
            raise ValueError(
                f"class {c} has {len(idx_c)} jets, need {per} for N={n}"
            )
        picks.append(rng.choice(idx_c, size=per, replace=False))
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


def make_splits(
    y: np.ndarray,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into a fixed, class-balanced test set and a train pool.

    The test set is identical across every sweep cell (it depends only on the
    global seed and n_test), which is required for cross-cell comparability.

    Returns (train_pool_idx, test_idx).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = n_test // 2
    test_picks = []
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        if len(idx_c) < per:
            raise ValueError(f"class {c} too small for n_test={n_test}")
        test_picks.append(rng.choice(idx_c, size=per, replace=False))
    test_idx = np.concatenate(test_picks)
    rng.shuffle(test_idx)

    mask_test = np.zeros(len(y), bool)
    mask_test[test_idx] = True
    train_pool_idx = np.flatnonzero(~mask_test)
    return train_pool_idx, test_idx


def derive_seed(global_seed: int, n: int, replicate: int) -> int:
    """Deterministic per-cell seed from (global_seed, N, replicate_index).

    Uses SeedSequence so nearby (n, replicate) values do not give correlated streams.
    """
    ss = np.random.SeedSequence([global_seed, int(n), int(replicate)])
    return int(ss.generate_state(1)[0])


# --------------------------------------------------------------------------- #
# JetNet I/O (the only part that needs the `jetnet` package + network)        #
# --------------------------------------------------------------------------- #
def load_jets(
    n_const: int,
    classes: Sequence[str] = ("g", "t"),
    data_dir: str | Path = "data_cache",
    n_part_stock: int = 30,
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load JetNet, preprocess to particle clouds, and cache to .npz.

    Returns
    -------
    X    : [n_jets, n_const, 4]  raw (unstandardized) features
    mask : [n_jets, n_const]     bool validity mask
    y    : [n_jets]              int label, index into `classes` (0 = first class)

    Standardization is intentionally NOT applied here — it must be fit per
    training subsample downstream to avoid leakage (see Standardizer).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{'-'.join(classes)}_nc{n_const}_stock{n_part_stock}"
    cache_file = data_dir / f"jets_{tag}.npz"

    if cache and cache_file.exists():
        d = np.load(cache_file)
        return d["X"], d["mask"], d["y"]

    from jetnet.datasets import JetNet  # imported lazily so tests don't need it

    particle_data, jet_data = JetNet.getData(
        jet_type=list(classes),
        data_dir=str(data_dir),
        particle_features=_PARTICLE_FEATURES,
        jet_features=_JET_FEATURES,
        num_particles=n_part_stock,
        split="all",
        download=True,
    )
    particle_data = np.asarray(particle_data, dtype=np.float32)
    jet_data = np.asarray(jet_data, dtype=np.float32)

    feats, full_mask = relative_features(particle_data, jet_data)
    ptrel = particle_data[..., 2]
    X, mask = truncate_pad(feats, full_mask, ptrel, n_const)

    # Remap JetNet's GLOBAL type codes to local labels 0..k-1 in `classes` order.
    raw_type = jet_data[:, 0].astype(np.int64)
    y = np.full(raw_type.shape, -1, dtype=np.int64)
    for local, name in enumerate(classes):
        if name not in _JETNET_TYPE_ORDER:
            raise ValueError(f"unknown jet class {name!r}; known: {_JETNET_TYPE_ORDER}")
        y[raw_type == _JETNET_TYPE_ORDER.index(name)] = local
    if (y < 0).any():
        raise ValueError(
            f"unexpected type codes {np.unique(raw_type).tolist()} for classes={list(classes)}"
        )

    if cache:
        np.savez_compressed(cache_file, X=X, mask=mask, y=y)
    return X, mask, y
