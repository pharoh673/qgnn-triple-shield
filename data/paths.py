"""Platform detection + path resolution.

We detect Kaggle / Colab / local and place everything (results, cache, data) under a
single platform-appropriate root. Callers do not hard-code paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def detect_platform() -> str:
    """Return one of 'kaggle', 'colab', 'local'."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_URL_BASE" in os.environ:
        return "kaggle"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except ImportError:
        return "local"


@dataclass(frozen=True)
class Paths:
    platform: str
    root: Path           # repo root (where code lives)
    work: Path           # writable persistent dir for results
    data: Path           # where JetNet .h5 files get cached

    def results(self, sub: str = "") -> Path:
        p = self.work / "results" / sub if sub else self.work / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def cache(self, sub: str = "") -> Path:
        p = self.work / "results" / "cache" / sub if sub else self.work / "results" / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def figures(self) -> Path:
        p = self.work / "results" / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p


def resolve(repo_name: str = "qgnn-triple-shield") -> Paths:
    """Resolve paths for the current platform.

    - Kaggle: code lives at /kaggle/input/<repo_name>/ if uploaded as dataset,
      else /kaggle/working/<repo_name>/ if cloned. Writes go to /kaggle/working.
    - Colab: code wherever, writes to /content/drive/MyDrive/<repo_name>/ if mounted,
      else /content/<repo_name>/.
    - Local: cwd is root and work.
    """
    plat = detect_platform()

    if plat == "kaggle":
        kaggle_input = Path("/kaggle/input") / repo_name
        kaggle_work = Path("/kaggle/working")
        root = kaggle_input if kaggle_input.exists() else kaggle_work / repo_name
        work = kaggle_work / repo_name
        data = kaggle_work / "data_cache"
    elif plat == "colab":
        drive = Path("/content/drive/MyDrive") / repo_name
        if drive.parent.exists():  # drive mounted
            work = drive
        else:
            work = Path("/content") / repo_name
        root = Path("/content") / repo_name
        data = work / "data_cache"
    else:
        root = Path.cwd()
        work = Path.cwd()
        data = Path.cwd() / "data_cache"

    for p in (work, data):
        p.mkdir(parents=True, exist_ok=True)

    return Paths(platform=plat, root=root, work=work, data=data)


if __name__ == "__main__":
    p = resolve()
    print(f"platform={p.platform}\nroot={p.root}\nwork={p.work}\ndata={p.data}")
