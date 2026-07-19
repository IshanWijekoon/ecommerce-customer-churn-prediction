"""Deterministic seeding helpers for Python, NumPy, and optional backends."""

from __future__ import annotations

import os
import random
from typing import Optional


def seed_everything(seed: int = 42, *, deterministic_torch: bool = False) -> None:
    """Seed built-in ``random``, NumPy, and optionally PyTorch.

    Args:
        seed: Integer seed applied to all available RNGs.
        deterministic_torch: If True and torch is installed, enable deterministic
            CuDNN settings (may reduce performance).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_numpy_generator(seed: Optional[int] = 42):
    """Return a NumPy ``Generator`` bound to ``seed``, or ``None`` if NumPy is absent."""
    try:
        import numpy as np

        return np.random.default_rng(seed)
    except ImportError:
        return None
