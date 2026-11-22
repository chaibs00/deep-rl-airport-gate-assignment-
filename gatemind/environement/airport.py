"""Airport model: the gates and when each one becomes free."""

from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class Gate:
    """One gate of the airport."""

    name: str           # e.g. "D3", or "REMOTE"
    group: str          # e.g. "D"
    max_size: int       # largest aircraft size the gate accepts
    walk_minutes: float # walking time from the terminal center
    is_remote: bool = False
    free_at: float = 0.0  # time (minutes) when the gate becomes free


class Airport:
    """Loads the gates from a YAML file and tracks their occupancy.

    The last gate in the list is always the remote stand.
    """

    def __init__(self, config_path="configs/airport.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.sizes = config["aircraft_sizes"]
        self.buffer = config["buffer_minutes"]
        self.gates = []

        for group in config["gate_groups"]:
            for i in range(1, group["count"] + 1):
                self.gates.append(
                    Gate(
                        name=f"{group['name']}{i}",
                        group=group["name"],
                        max_size=self.sizes[group["max_size"]],
                        walk_minutes=group["walk_minutes"],
                    )
                )

        remote = config["remote"]
        self.gates.append(
            Gate(
                name="REMOTE",
                group="REMOTE",
                max_size=self.sizes[remote["max_size"]],
                walk_minutes=remote["walk_minutes"],
                is_remote=True,
            )
        )

        self.remote_index = len(self.gates) - 1

    @property
    def num_gates(self):
        """Number of actions: contact gates + 1 remote stand."""
        return len(self.gates)

    def reset(self):
        """Free all gates at the start of a new day."""
        for gate in self.gates:
            gate.free_at = 0.0

    def is_valid(self, gate_index, aircraft_size, arrival_time):
        """A gate is valid if the aircraft fits and the gate is free."""
        gate = self.gates[gate_index]
        if gate.is_remote:
            return True
        fits = aircraft_size <= gate.max_size
        free = gate.free_at <= arrival_time
        return fits and free

    def valid_mask(self, aircraft_size, arrival_time):
        """Boolean array: True for each gate the flight can use."""
        return np.array(
            [self.is_valid(i, aircraft_size, arrival_time) for i in range(self.num_gates)]
        )

    def assign(self, gate_index, departure_time):
        """Occupy a gate until the flight departs (+ buffer)."""
        gate = self.gates[gate_index]
        if not gate.is_remote:
            gate.free_at = departure_time + self.buffer

    def time_until_free(self, current_time):
        """For each gate, minutes left before it is free (0 if already free)."""
        return np.array(
            [max(0.0, gate.free_at - current_time) for gate in self.gates]
        )