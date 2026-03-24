from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fix Python and NumPy RNG for reproducible sampling and auxiliary randomness."""
    random.seed(seed)
    np.random.seed(seed)
