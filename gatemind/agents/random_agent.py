"""Random agent: picks any valid gate. Lower bound for comparison."""

import numpy as np


class RandomAgent:

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask):
        valid = np.flatnonzero(mask)
        return int(self.rng.choice(valid))