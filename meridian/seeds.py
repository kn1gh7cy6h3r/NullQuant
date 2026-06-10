"""
seeds.py — deterministic seeding across every RNG the project touches.

Reproducibility is table stakes for credible research: the same config + seed
must yield the same numbers. Call set_global_seed() once at the start of any
pipeline or notebook run.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, scikit-learn (via NumPy), and torch if present."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy always present here
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no CUDA on mac
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch is optional for the non-LSTM parts of the pipeline.
        pass
