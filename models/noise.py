"""Noise models injected after each entangling layer of the per-constituent circuit.

Two regimes, exposed via config ``{regime, p, ...}``:

* **iid "easy mode"** — per-qubit, per-layer single-qubit channels (Depolarizing
  and/or AmplitudeDamping) with probability ``p``. These are non-unitary, so the
  circuit must run on ``default.mixed`` (``NoiseModel.mixed == True``), single shot.

* **correlated "hard mode"** — Ornstein-Uhlenbeck (OU) correlated DEPHASING. A single
  phase trajectory ``φ_l`` is drawn from a stationary AR(1) process (the OU
  discretization) with correlation time ``τ`` measured in layers, and applied as a
  coherent ``RZ(φ_l)`` on every qubit (so it is BOTH temporally correlated across
  layers and spatially correlated across qubits). Because each trajectory is unitary
  we run it on the fast statevector device and average expectation values over
  ``n_trajectories`` samples — trajectory-averaged dephasing. Phase amplitude is
  ``σ·√p·π`` so larger ``p`` and larger ``σ`` mean stronger dephasing; larger ``τ``
  means longer memory. ``τ`` is intended as a secondary sweep axis.

The ``NoiseModel`` is duck-typed for ``models.quantum_equiv.QuantumEncoder``:
it exposes ``.mixed``, ``.n_trajectories``, ``.apply(layer)`` and ``.resample(t)``.
``regime="none"`` (or ``p==0``) is a no-op that keeps the fast noiseless path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class NoiseSpec:
    """Declarative noise configuration for one sweep cell."""

    regime: str = "none"                       # "none" | "iid" | "correlated"
    p: float = 0.0
    channels: Sequence[str] = field(           # iid: depolarizing | amplitude_damping
        default_factory=lambda: ("amplitude_damping", "depolarizing")
    )
    axis: str = "x"                            # correlated rotation axis; "z" commutes with Z-readout
    tau: float = 2.0                           # correlated: correlation time (layers)
    sigma: float = 1.0                         # correlated: amplitude scale
    n_trajectories: int = 8                    # correlated: trajectories to average

    @property
    def active(self) -> bool:
        return self.regime in ("iid", "correlated") and self.p > 0.0


class NoiseModel:
    """Stateful, injectable noise for the per-constituent circuit."""

    def __init__(self, spec: NoiseSpec, n_qubits: int, reupload: int, seed: int = 0):
        self.spec = spec
        self.n_qubits = n_qubits
        self.reupload = reupload
        # iid channels are non-unitary -> need the mixed-state device.
        self.mixed = spec.regime == "iid" and spec.active
        # correlated coherent dephasing runs on the (fast) statevector device, averaged.
        self.n_trajectories = (
            spec.n_trajectories if (spec.regime == "correlated" and spec.active) else 1
        )
        self._rng = np.random.default_rng(seed)
        self._phases = np.zeros(reupload, dtype=float)
        if spec.regime == "correlated" and spec.active:
            self.resample(0)

    # --- called inside the circuit, after each entangling layer ---------------
    def apply(self, layer: int) -> None:
        if not self.spec.active:
            return
        if self.spec.regime == "iid":
            for q in range(self.n_qubits):
                if "depolarizing" in self.spec.channels:
                    qml.DepolarizingChannel(self.spec.p, wires=q)
                if "amplitude_damping" in self.spec.channels:
                    qml.AmplitudeDamping(self.spec.p, wires=q)
        elif self.spec.regime == "correlated":
            phi = float(self._phases[layer])
            # Same phase on all qubits = spatial correlation. Default axis is X:
            # an RZ phase would commute with the Z readout (no-op), so an OU-correlated
            # coherent rotation about X actually corrupts ⟨Z⟩ (correlated control/drift error).
            gate = qml.RX if self.spec.axis == "x" else qml.RZ
            for q in range(self.n_qubits):
                gate(phi, wires=q)

    # --- called once per trajectory before re-running the circuit -------------
    def resample(self, traj_index: int) -> None:
        """Draw a fresh OU (AR(1)) phase trajectory of length ``reupload``."""
        if not (self.spec.regime == "correlated" and self.spec.active):
            return
        rho = float(np.exp(-1.0 / max(self.spec.tau, 1e-6)))   # AR(1) coefficient
        amp = self.spec.sigma * np.sqrt(self.spec.p) * np.pi   # phase std
        x = np.empty(self.reupload, dtype=float)
        x[0] = self._rng.standard_normal()                     # stationary unit variance
        step_std = np.sqrt(max(1.0 - rho * rho, 0.0))
        for l in range(1, self.reupload):
            x[l] = rho * x[l - 1] + step_std * self._rng.standard_normal()
        self._phases = amp * x


def noise_from_config(
    noise_cfg: dict,
    regime: str,
    level: str,
    n_qubits: int,
    reupload: int,
    seed: int = 0,
) -> NoiseModel:
    """Build a NoiseModel from the config ``noise`` block + a (regime, level) pair.

    Each regime carries its own ``levels`` map (iid and correlated have different
    sensitivity). ``regime="none"`` (or level "none") gives the noiseless fast path.
    """
    if regime == "none" or level == "none":
        return NoiseModel(NoiseSpec(regime="none"), n_qubits, reupload, seed)
    if regime not in ("iid", "correlated"):
        raise ValueError(f"unknown noise regime {regime!r}")
    sub = noise_cfg[regime]
    p = float(sub["levels"][level])
    if p == 0.0:
        return NoiseModel(NoiseSpec(regime="none"), n_qubits, reupload, seed)
    if regime == "iid":
        spec = NoiseSpec(regime="iid", p=p, channels=tuple(sub["channels"]))
    else:  # correlated
        spec = NoiseSpec(
            regime="correlated",
            p=p,
            axis=str(sub.get("axis", "x")),
            tau=float(sub["tau"]),
            sigma=float(sub["sigma"]),
            n_trajectories=int(noise_cfg["n_trajectories"]),
        )
    return NoiseModel(spec, n_qubits, reupload, seed)
