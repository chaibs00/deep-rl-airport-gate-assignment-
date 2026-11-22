"""Gymnasium environment for airport gate assignment.

One step = one arriving flight. One episode = one simulated day.
"""

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from gatemind.data.schedule_generator import ScheduleGenerator
from gatemind.environement.airport import Airport


class GateAssignmentEnv(gym.Env):

    def __init__(
        self,
        airport_config="configs/airport.yaml",
        schedule_config="configs/schedule.yaml",
        default_config="configs/default.yaml",
    ):
        self.airport = Airport(airport_config)
        self.generator = ScheduleGenerator(self.airport.sizes, schedule_config)

        with open(default_config) as f:
            config = yaml.safe_load(f)
        self.reward_cfg = config["reward"]
        self.max_wait = config["state"]["max_wait"]
        self.max_turnaround = config["state"]["max_turnaround"]

        self.categories = list(self.airport.sizes.keys())  # regional, narrow, wide
        self.day_start = self.generator.config["day_start"]
        self.day_end = self.generator.config["day_end"]

        # Contact gates only (the remote stand is the last gate).
        self.contact = self.airport.gates[: self.airport.remote_index]
        self.max_walk = max(g.walk_minutes for g in self.contact)
        self.max_size = max(self.airport.sizes.values())

        # Actions: one per contact gate + 1 remote stand.
        self.action_space = spaces.Discrete(self.airport.num_gates)

        # State: flight (category one-hot, arrival, turnaround, flights left)
        #        + for each contact gate (time until free, fits the aircraft).
        n_flight = len(self.categories) + 3
        n_gates = 2 * len(self.contact)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n_flight + n_gates,), dtype=np.float32)

        self.flights = []
        self.index = 0
        self.assignments = []  # (flight, gate_name), used later for Gantt charts

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.airport.reset()
        self.flights = self.generator.generate(self.np_random)
        self.index = 0
        self.assignments = []
        return self._observation(), self._info()

    def step(self, action):
        flight = self.flights[self.index]

        if not self.airport.is_valid(action, flight.size, flight.arrival):
            raise ValueError(
                f"Invalid action {action} for flight {flight.flight_id}: use the action mask."
            )

        reward = self._reward(flight, action)
        self.airport.assign(action, flight.departure)
        self.assignments.append((flight, self.airport.gates[action].name))

        self.index += 1
        terminated = self.index == len(self.flights)
        return self._observation(), reward, terminated, False, self._info()

    # ------------------------------------------------------------------ #
    # MDP components
    # ------------------------------------------------------------------ #

    def action_mask(self):
        """True for each gate the current flight can use."""
        if self.index >= len(self.flights):
            return np.ones(self.airport.num_gates, dtype=bool)
        flight = self.flights[self.index]
        return self.airport.valid_mask(flight.size, flight.arrival)

    def _reward(self, flight, action):
        gate = self.airport.gates[action]
        r = self.reward_cfg

        if gate.is_remote:
            return r["remote"]

        distance_penalty = r["max_distance_penalty"] * gate.walk_minutes / self.max_walk
        mismatch_penalty = r["size_mismatch"] * (gate.max_size - flight.size)
        return r["contact"] - distance_penalty - mismatch_penalty

    def _observation(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.index >= len(self.flights):
            return obs  # end of the day

        flight = self.flights[self.index]

        # Flight features
        one_hot = [1.0 if c == flight.category else 0.0 for c in self.categories]
        arrival = (flight.arrival - self.day_start) / (self.day_end - self.day_start)
        turnaround = min(flight.turnaround / self.max_turnaround, 1.0)
        flights_left = (len(self.flights) - self.index) / len(self.flights)
        flight_features = one_hot + [arrival, turnaround, flights_left]

        # Gate features
        wait = self.airport.time_until_free(flight.arrival)[: len(self.contact)]
        wait = np.minimum(wait / self.max_wait, 1.0)
        fits = [1.0 if flight.size <= g.max_size else 0.0 for g in self.contact]

        obs[:] = np.concatenate([flight_features, wait, fits])
        return obs

    def _info(self):
        return {"action_mask": self.action_mask()}