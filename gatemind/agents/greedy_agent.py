"""Greedy agent: picks the valid gate with the best immediate reward.

It only looks at the current flight and never thinks about future flights.
This is the reference the DQN agent has to beat.
"""

import numpy as np


class GreedyAgent:

    def __init__(self, env):
        self.env = env  # needs the env to score each gate for the current flight

    def act(self, obs, mask):
        flight = self.env.current_flight()
        valid = np.flatnonzero(mask)
        rewards = [self.env.reward(flight, a) for a in valid]
        return int(valid[np.argmax(rewards)])