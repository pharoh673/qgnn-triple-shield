"""Config-driven sweep runner.

One result row per config, keyed by a stable hash of the config dict; a config
whose result JSON already exists is skipped (so a killed run resumes for free).
Quantum models (equivariant, non_equivariant) sweep N x n_qubits x noise x
replicate; classical models depend only on N x replicate (n_qubits/noise are
not applicable), so those axes are collapsed to avoid redundant work.

Parallelism: a process pool of single-threaded workers (benchmarked optimal on
this CPU). Each worker loads the cached dataset once via an initializer.

CLI:
    python experiments/sweep.py --grid smoke              # the tiny sign-off grid
    python experiments/sweep.py --grid full --workers 9   # the real run
    python experiments/sweep.py --grid smoke --collect-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import (  # noqa: E402
    derive_seed,
    load_jets,
    make_splits,
    stratified_subsample,
)
from data.paths import resolve  # noqa: E402

QUANTUM_MODELS = ("equivariant", "non_equivariant")
CLASSICAL_MODELS = ("classical_matched", "classical_reference")

# Per-process globals, set by _init_worker.
_CFG: dict | None = None
_DATA: dict | None = None
_CACHE_DIR: str | None = None


# --------------------------------------------------------------------------- #
# Grid expansion + hashing                                                    #
# --------------------------------------------------------------------------- #
def expand_grid(sweep_cfg: dict) -> list[dict]:
    """Expand an OAT block grid into a de-duplicated flat list of per-config dicts.

    Each block is a small cross-product of N x n_qubits x noise x models x replicates.
    Blocks are unioned; identical cells (e.g. the shared anchor) appear once. Classical
    models collapse the n_qubits/noise axes (not applicable), so they are emitted once
    per (N, replicate) across all blocks.
    """
    models = sweep_cfg["models"]
    default_reps = sweep_cfg["n_replicates"]
    seen: set[str] = set()
    configs: list[dict] = []

    def add(c: dict) -> None:
        h = config_hash(c)
        if h not in seen:
            seen.add(h)
            configs.append(c)

    for block in sweep_cfg["blocks"]:
        reps = int(block.get("n_replicates", default_reps))  # per-block override
        for model in models:
            for N in block["N"]:
                for rep in range(reps):
                    if model in QUANTUM_MODELS:
                        for nq in block["n_qubits"]:
                            for noise in block["noise"]:
                                add({
                                    "model": model, "N": int(N), "n_qubits": int(nq),
                                    "noise_regime": noise["regime"],
                                    "noise_level": noise["level"], "replicate": int(rep),
                                })
                    else:  # classical: no qubit / noise axes
                        add({
                            "model": model, "N": int(N), "n_qubits": 0,
                            "noise_regime": "none", "noise_level": "none",
                            "replicate": int(rep),
                        })
    return configs


def config_hash(c: dict) -> str:
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #
def _init_worker(config_path: str) -> None:
    """Pin threads to 1 and load the dataset once per process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    import torch
    torch.set_num_threads(1)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    global _CFG, _DATA, _CACHE_DIR
    with open(config_path) as f:
        _CFG = yaml.safe_load(f)
    paths = resolve()
    _CACHE_DIR = str(paths.cache())
    d = _CFG["data"]
    X, mask, y = load_jets(d["n_const"], d["classes"], data_dir=paths.data)
    pool, test = make_splits(y, n_test=d["n_test"], seed=_CFG["seed"])
    _DATA = {"X": X, "mask": mask, "y": y, "pool": pool, "test": test}


def _build_model(model: str, n_qubits: int, noise_model, seed: int):
    mc = _CFG["model"]
    if model == "equivariant":
        from models.quantum_equiv import QuantumDeepSets
        e = mc["equivariant"]
        return QuantumDeepSets(
            n_qubits, reupload=e["reuploading_depth"],
            trainable_per_qubit=e["trainable_per_qubit"],
            head_bottleneck=e["head_bottleneck"], noise=noise_model, seed=seed,
        )
    if model == "non_equivariant":
        from models.quantum_noneq import NonEquivariantQuantumTagger
        n = mc["non_equivariant"]
        return NonEquivariantQuantumTagger(
            n_qubits, reupload=n["reuploading_depth"],
            trainable_per_qubit=n["trainable_per_qubit"],
            head_bottleneck=n["head_bottleneck"], pos_scale=n["pos_scale"],
            noise=noise_model, seed=seed,
        )
    if model in CLASSICAL_MODELS:
        from models.classical_pfn import ParticleFlowNetwork
        p = mc[model]
        return ParticleFlowNetwork(
            phi_hidden=p["phi_hidden"], rho_hidden=p["rho_hidden"], seed=seed,
        )
    raise ValueError(f"unknown model {model!r}")


def run_one(c: dict) -> dict:
    """Train+evaluate one config (or load its cached result). Always returns a row."""
    cache_file = os.path.join(_CACHE_DIR, f"{config_hash(c)}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            r = json.load(f)
        r["_cached"] = True
        return r

    import torch
    from models.noise import noise_from_config
    from models.quantum_equiv import count_trainable
    from train.trainer import TrainConfig, train_and_evaluate

    X, mask, y = _DATA["X"], _DATA["mask"], _DATA["y"]
    pool, test = _DATA["pool"], _DATA["test"]

    seed = derive_seed(_CFG["seed"], c["N"], c["replicate"])
    tr = pool[stratified_subsample(y[pool], c["N"], seed)]
    reupload = _CFG["model"]["equivariant"]["reuploading_depth"]

    def factory(init_seed: int):
        if c["model"] in QUANTUM_MODELS:
            nm = noise_from_config(
                _CFG["noise"], c["noise_regime"], c["noise_level"],
                c["n_qubits"], reupload, seed=init_seed,
            )
        else:
            nm = None
        torch.manual_seed(init_seed)
        return _build_model(c["model"], c["n_qubits"], nm, init_seed)

    tcfg = TrainConfig.from_dict(_CFG["train"])
    t0 = time.time()
    metrics = train_and_evaluate(
        factory, X[tr], mask[tr], y[tr], X[test], mask[test], y[test], tcfg, seed
    )
    elapsed = time.time() - t0

    row = {**c, **metrics,
           "n_params": int(count_trainable(factory(seed))),
           "seed": int(seed), "train_secs": round(elapsed, 2)}
    tmp = cache_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, cache_file)  # atomic write so a kill mid-write can't corrupt cache
    row["_cached"] = False
    return row


# --------------------------------------------------------------------------- #
# Driver + collection                                                         #
# --------------------------------------------------------------------------- #
def collect_results(cache_dir: str, out_path: str) -> int:
    """Gather all cached result JSONs into one parquet table."""
    import pandas as pd
    rows = []
    for fn in os.listdir(cache_dir):
        if fn.endswith(".json"):
            with open(os.path.join(cache_dir, fn)) as f:
                rows.append(json.load(f))
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    df.to_csv(out_path.replace(".parquet", ".csv"), index=False)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default=None)
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config or os.path.join(here, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    paths = resolve()
    cache_dir = str(paths.cache())
    out_path = str(paths.results() / f"results_{args.grid}.parquet")

    configs = expand_grid(cfg["sweep"][args.grid])
    todo = [c for c in configs if not os.path.exists(os.path.join(cache_dir, f"{config_hash(c)}.json"))]
    print(f"[{args.grid}] {len(configs)} configs total, {len(todo)} to run, "
          f"{len(configs)-len(todo)} cached. workers={args.workers}")

    if not args.collect_only and todo:
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(config_path,)
        ) as ex:
            futures = {ex.submit(run_one, c): c for c in todo}
            for fut in as_completed(futures):
                r = fut.result()
                done += 1
                print(f"  [{done}/{len(todo)}] {r['model']:>20} N={r['N']:<4} "
                      f"nq={r['n_qubits']} {r['noise_regime']}/{r['noise_level']:<5} "
                      f"rep={r['replicate']} -> AUC={r['test_auc']:.3f} "
                      f"({r.get('train_secs','?')}s)")
        print(f"ran {len(todo)} configs in {time.time()-t0:.0f}s")

    n = collect_results(cache_dir, out_path)
    print(f"collected {n} results -> {out_path}")


if __name__ == "__main__":
    main()
