"""Generate the payoff figures from cached sweep + diagnostic results.

Five figures (each: mean ± std band over replicates, saved as PNG + the plotted
data as CSV under results/figures/):

  1. learning_curve       test AUC vs N            (fixed nq, noise=none)
  2. qubit_frugality      test AUC vs n_qubits     (fixed N, noise=none)
  3. noise_robustness     test AUC vs noise p      (iid & correlated, fixed N/nq)
  4. generalization_gap   (train-test AUC) vs N
  5. gradient_variance    Var ∂L/∂θ vs n_qubits    (barren-plateau probe)

Usage:
    python analysis/plots.py                      # uses results_full.parquet
    python analysis/plots.py --results results/results_smoke.parquet
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.paths import resolve  # noqa: E402

QUANTUM = ("equivariant", "non_equivariant")
CLASSICAL = ("classical_matched", "classical_reference")

STYLE = {
    "equivariant":         dict(label="Equivariant QNN",        color="#1f77b4", marker="o", ls="-"),
    "non_equivariant":     dict(label="Non-equivariant QNN",    color="#d62728", marker="s", ls="--"),
    "classical_matched":   dict(label="Classical PFN (matched)", color="#2ca02c", marker="^", ls=":"),
    "classical_reference": dict(label="Classical PFN (reference)", color="#9467bd", marker="v", ls="-."),
    "generic":             dict(label="Generic QNN (global cost)", color="#ff7f0e", marker="D", ls="--"),
}


def _agg(df: pd.DataFrame, x: str, value: str = "test_auc") -> pd.DataFrame:
    """Mean/std/count of `value` grouped by `x`, sorted by `x`."""
    g = df.groupby(x)[value].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g.sort_values(x)


def _band(ax, x, mean, std, model):
    s = STYLE[model]
    ax.plot(x, mean, marker=s["marker"], color=s["color"], ls=s["ls"], label=s["label"], lw=1.8, ms=5)
    ax.fill_between(x, mean - std, mean + std, color=s["color"], alpha=0.15)


def _finish(ax, title, xlabel, ylabel, outdir, name, rows):
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, framealpha=0.9)
    fig = ax.figure
    fig.tight_layout()
    png = os.path.join(outdir, f"{name}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
    print(f"  saved {name}.png (+csv)")


# --------------------------------------------------------------------------- #
def fig_learning_curve(df, cfg, outdir, nq=4):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    rows = []
    for model in (*QUANTUM, *CLASSICAL):
        sub = df[(df.model == model) & (df.noise_regime == "none")]
        sub = sub[sub.n_qubits == (nq if model in QUANTUM else 0)]
        if sub.empty:
            continue
        g = _agg(sub, "N")
        _band(ax, g["N"].values, g["mean"].values, g["std"].values, model)
        for _, r in g.iterrows():
            rows.append(dict(model=model, N=r["N"], auc_mean=r["mean"], auc_std=r["std"], n=r["count"]))
    ax.set_xscale("log")
    _finish(ax, f"Learning curves (n_qubits={nq}, noiseless)", "training-set size N",
            "test ROC-AUC", outdir, "learning_curve", rows)


def fig_qubit_frugality(df, cfg, outdir, N=200):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    rows = []
    for model in QUANTUM:
        sub = df[(df.model == model) & (df.noise_regime == "none") & (df.N == N)]
        if sub.empty:
            continue
        g = _agg(sub, "n_qubits")
        _band(ax, g["n_qubits"].values, g["mean"].values, g["std"].values, model)
        for _, r in g.iterrows():
            rows.append(dict(model=model, n_qubits=r["n_qubits"], auc_mean=r["mean"], auc_std=r["std"]))
    # classical baselines: flat reference lines (no qubit axis)
    for model in CLASSICAL:
        sub = df[(df.model == model) & (df.N == N)]
        if sub.empty:
            continue
        m = sub["test_auc"].mean()
        ax.axhline(m, color=STYLE[model]["color"], ls=STYLE[model]["ls"], lw=1.4,
                   label=STYLE[model]["label"] + " (N-only)")
        rows.append(dict(model=model, n_qubits="all", auc_mean=m, auc_std=sub["test_auc"].std()))
    _finish(ax, f"Qubit frugality (N={N}, noiseless)", "working qubits n_qubits",
            "test ROC-AUC", outdir, "qubit_frugality", rows)


def fig_noise_robustness(df, cfg, outdir, N=200, nq=4):
    levels_by_regime = {
        "iid": cfg["noise"]["iid"]["levels"],
        "correlated": cfg["noise"]["correlated"]["levels"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    rows = []
    for ax, regime in zip(axes, ("iid", "correlated")):
        pmap = {"none": 0.0, **levels_by_regime[regime]}
        for model in QUANTUM:
            sub = df[(df.model == model) & (df.N == N) & (df.n_qubits == nq) &
                     (df.noise_regime.isin(["none", regime]))].copy()
            if sub.empty:
                continue
            sub["p"] = sub["noise_level"].map(pmap)
            g = _agg(sub.dropna(subset=["p"]), "p")
            _band(ax, g["p"].values, g["mean"].values, g["std"].values, model)
            for _, r in g.iterrows():
                rows.append(dict(regime=regime, model=model, p=r["p"], auc_mean=r["mean"], auc_std=r["std"]))
        ax.set_title(f"{regime} noise"); ax.set_xlabel("noise strength p"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("test ROC-AUC")
    fig.suptitle(f"Noise robustness (N={N}, n_qubits={nq})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "noise_robustness.png"), dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "noise_robustness.csv"), index=False)
    print("  saved noise_robustness.png (+csv)")


def fig_generalization_gap(df, cfg, outdir, nq=4):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    rows = []
    for model in (*QUANTUM, *CLASSICAL):
        sub = df[(df.model == model) & (df.noise_regime == "none")]
        sub = sub[sub.n_qubits == (nq if model in QUANTUM else 0)]
        if sub.empty:
            continue
        g = _agg(sub, "N", value="gen_gap")
        _band(ax, g["N"].values, g["mean"].values, g["std"].values, model)
        for _, r in g.iterrows():
            rows.append(dict(model=model, N=r["N"], gap_mean=r["mean"], gap_std=r["std"]))
    ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    _finish(ax, f"Generalization gap (n_qubits={nq}, noiseless)", "training-set size N",
            "train AUC − test AUC", outdir, "generalization_gap", rows)


def fig_gradient_variance(diag_df, outdir):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    rows = []
    for model in ("equivariant", "non_equivariant", "generic"):
        sub = diag_df[diag_df.model == model].sort_values("n_qubits")
        if sub.empty:
            continue
        s = STYLE[model]
        ax.plot(sub["n_qubits"], sub["grad_var"], marker=s["marker"], color=s["color"],
                ls=s["ls"], label=s["label"], lw=1.8, ms=5)
        for _, r in sub.iterrows():
            rows.append(dict(model=model, n_qubits=r["n_qubits"], grad_var=r["grad_var"]))
    ax.set_yscale("log")
    _finish(ax, "Trainability: gradient variance (barren-plateau probe)",
            "working qubits n_qubits", "Var(∂L/∂θ₀)  [log]", outdir, "gradient_variance", rows)


def fig_shield_surface(df, cfg, outdir, N=200):
    """Bonus 2-D heatmap: equivariant − non_equivariant test-AUC advantage over (nq, p)."""
    pmap = {"none": 0.0, **cfg["noise"]["correlated"]["levels"]}
    sub = df[(df.N == N) & (df.noise_regime.isin(["none", "correlated"])) &
             (df.model.isin(QUANTUM))].copy()
    if sub.empty:
        return
    sub["p"] = sub["noise_level"].map(pmap)
    piv = sub.groupby(["model", "n_qubits", "p"]).test_auc.mean().reset_index()
    eq = piv[piv.model == "equivariant"].pivot(index="n_qubits", columns="p", values="test_auc")
    ne = piv[piv.model == "non_equivariant"].pivot(index="n_qubits", columns="p", values="test_auc")
    common = eq.index.intersection(ne.index)
    diff = (eq.loc[common] - ne.loc[common])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    im = ax.imshow(diff.values, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-0.2, vmax=0.2,
                   extent=[min(diff.columns), max(diff.columns), min(diff.index), max(diff.index)])
    ax.set_xticks(sorted(diff.columns)); ax.set_yticks(sorted(diff.index))
    fig.colorbar(im, ax=ax, label="equiv − non-equiv  test AUC")
    _finish(ax, f"Shield surface: equivariant advantage (N={N}, correlated noise)",
            "noise strength p", "working qubits n_qubits", outdir, "shield_surface",
            diff.reset_index().to_dict("records"))


def main() -> None:
    ap = argparse.ArgumentParser()
    paths = resolve()
    ap.add_argument("--results", default=str(paths.results() / "results_full.parquet"))
    ap.add_argument("--diag", default=str(paths.results() / "diagnostics_gradvar.parquet"))
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(here, "config.yaml")))
    outdir = str(paths.figures())
    print(f"reading {args.results}")
    df = pd.read_parquet(args.results)

    fig_learning_curve(df, cfg, outdir)
    fig_qubit_frugality(df, cfg, outdir)
    fig_noise_robustness(df, cfg, outdir)
    fig_generalization_gap(df, cfg, outdir)
    fig_shield_surface(df, cfg, outdir)
    if os.path.exists(args.diag):
        fig_gradient_variance(pd.read_parquet(args.diag), outdir)
    else:
        print(f"  (skipping gradient_variance: {args.diag} not found — run experiments/run_diagnostics.py)")
    print(f"figures -> {outdir}")


if __name__ == "__main__":
    main()
