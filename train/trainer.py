"""Training loop + evaluation for one sweep cell.

``train_and_evaluate`` is the single entry point the sweep calls per config. It is
deliberately model-agnostic: it accepts any ``torch.nn.Module`` whose ``forward(X,
mask)`` returns logits, so the equivariant/non-equivariant quantum models and the
classical PFN all go through the same code path.

No-leakage contract: the feature Standardizer is fit ONLY on the train portion of
the training subsample (not val, not test). Early stopping is on validation ROC-AUC
(falling back to negative BCE when a tiny validation split is single-class). The
best-by-val weights are restored before the final train/test metrics are computed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data.loader import Standardizer


@dataclass
class TrainConfig:
    lr: float = 0.01
    weight_decay: float = 0.0
    batch_size: int = 32
    max_epochs: int = 80
    patience: int = 12
    val_frac: float = 0.2
    max_restarts: int = 4
    warmup_epochs: int = 15
    stuck_auc: float = 0.55

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(
            lr=float(d.get("lr", 0.01)),
            weight_decay=float(d.get("weight_decay", 0.0)),
            batch_size=int(d.get("batch_size", 32)),
            max_epochs=int(d.get("max_epochs", 80)),
            patience=int(d.get("early_stop_patience", d.get("patience", 12))),
            val_frac=float(d.get("val_frac", 0.2)),
            max_restarts=int(d.get("max_restarts", 4)),
            warmup_epochs=int(d.get("warmup_epochs", 15)),
            stuck_auc=float(d.get("stuck_auc", 0.55)),
        )


def _tensor(x, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(x), dtype=dtype)


@torch.no_grad()
def _auc(model: nn.Module, X: torch.Tensor, m: torch.Tensor, y_np: np.ndarray) -> float:
    model.eval()
    s = torch.sigmoid(model(X, m)).cpu().numpy()
    return float(roc_auc_score(y_np, s))


@torch.no_grad()
def _val_score(model: nn.Module, X: torch.Tensor, m: torch.Tensor, y_np: np.ndarray) -> float:
    """Early-stopping score: ROC-AUC if both classes present, else negative BCE."""
    model.eval()
    logits = model(X, m)
    if len(np.unique(y_np)) >= 2:
        return float(roc_auc_score(y_np, torch.sigmoid(logits).cpu().numpy()))
    yt = _tensor(y_np)
    return float(-F.binary_cross_entropy_with_logits(logits, yt).item())


def _split_train_val(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]      # (train, val)


def train_and_evaluate(
    model_factory,
    X_train: np.ndarray,
    mask_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    mask_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig,
    seed: int,
) -> dict:
    """Train (with random restarts), early-stop on val AUC, return metrics.

    ``model_factory(init_seed) -> nn.Module`` is called per restart so each attempt
    gets a fresh initialization. If a restart is still at chance-level train loss
    after ``warmup_epochs``, it is abandoned and re-seeded (escapes vanishing-grad
    inits). The best-by-validation weights across restarts are restored before the
    final train/test metrics.
    """
    n = len(y_train)
    tr, val = _split_train_val(n, cfg.val_frac, seed)
    sc = Standardizer.fit(X_train[tr], mask_train[tr])  # no leakage: train portion only

    def prep(X, m):
        return _tensor(sc.transform(X, m)), _tensor(m, torch.bool)

    Xtr, mtr = prep(X_train[tr], mask_train[tr])
    ytr = _tensor(y_train[tr])
    Xval, mval = prep(X_train[val], mask_train[val])
    yval_np = y_train[val]
    Xte, mte = prep(X_test, mask_test)
    lossf = nn.BCEWithLogitsLoss()
    n_tr = len(ytr)

    best_score, best_state, best_epoch = -np.inf, None, -1
    restarts_used, escaped = 0, False

    for r in range(cfg.max_restarts):
        restarts_used = r + 1
        model = model_factory(seed + r * 100_003)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(seed)
        local_best, local_state, bad, stuck = -np.inf, None, 0, False

        epoch = 0
        for epoch in range(cfg.max_epochs):
            model.train()
            perm = torch.randperm(n_tr, generator=gen)
            for i in range(0, n_tr, cfg.batch_size):
                b = perm[i : i + cfg.batch_size]
                opt.zero_grad()
                lossf(model(Xtr[b], mtr[b]), ytr[b]).backward()
                opt.step()

            # Stuck detector: abandon a dead init at the warmup checkpoint, judged by
            # TRAIN AUC ~ chance (signal, not BCE loss). Must run before any early-stop
            # break, so patience is only counted AFTER warmup.
            if epoch + 1 == cfg.warmup_epochs and _auc(model, Xtr, mtr, y_train[tr]) < cfg.stuck_auc:
                stuck = True
                break

            score = _val_score(model, Xval, mval, yval_np)
            if score > local_best + 1e-4:
                local_best = score
                local_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            elif epoch + 1 > cfg.warmup_epochs:  # don't early-stop during warmup
                bad += 1
                if bad >= cfg.patience:
                    break

        if local_state is not None and local_best > best_score:
            best_score, best_state, best_epoch = local_best, local_state, epoch
        if not stuck:
            escaped = True
            break  # a healthy init finished; no need for more restarts

    final = model_factory(seed)
    if best_state is not None:
        final.load_state_dict(best_state)

    train_auc = _auc(final, Xtr, mtr, y_train[tr])
    test_auc = _auc(final, Xte, mte, y_test)
    return {
        "train_auc": train_auc,
        "val_score": float(best_score),
        "test_auc": test_auc,
        "gen_gap": train_auc - test_auc,
        "best_epoch": int(best_epoch),
        "restarts_used": int(restarts_used),
        "escaped": bool(escaped),
        "n_train": int(n),
    }
