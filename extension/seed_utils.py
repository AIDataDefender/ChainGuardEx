"""Seeding utilities for reproducible analysis runs.

The VS Code extension can pass a seed into the Python backend so that any
random initialization (e.g., missing checkpoint keys) is stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Optional


@dataclass(frozen=True)
class SeedConfig:
    seed: int
    deterministic: bool = True


def parse_seed(value: Optional[str | int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = str(value).strip()
    if value == "":
        return None
    return int(value)


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set seeds for Python, NumPy and PyTorch (if installed).

    This is best-effort: if torch/numpy aren't available, it still seeds Python.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # cuDNN knobs (safe on CPU too)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

            # Prefer determinism over performance if supported
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    # Some ops may not have deterministic kernels; ignore.
                    pass

    except Exception:
        pass
