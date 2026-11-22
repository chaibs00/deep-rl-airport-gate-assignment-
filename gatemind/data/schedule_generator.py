"""Generates one simulated day of arriving flights."""

from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class Flight:
    """One flight that needs a gate."""

    flight_id: int
    category: str     # "regional", "narrow" or "wide"
    size: int         # numeric size, from airport.yaml
    arrival: float    # minutes after midnight
    departure: float  # minutes after midnight

    @property
    def turnaround(self):
        return self.departure - self.arrival


class ScheduleGenerator:
    """Creates a random but realistic list of flights for one day."""

    def __init__(self, sizes, config_path="configs/schedule.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.sizes = sizes  # e.g. {"regional": 0, "narrow": 1, "wide": 2}

    def _arrival_times(self, n, rng):
        """Mix of peak hours (normal distributions) and uniform background traffic."""
        cfg = self.config
        times = []

        n_peak_total = 0
        for peak in cfg["peaks"]:
            n_peak = int(round(n * peak["share"]))
            times.append(rng.normal(peak["mean"], peak["std"], size=n_peak))
            n_peak_total += n_peak

        n_uniform = n - n_peak_total
        times.append(rng.uniform(cfg["day_start"], cfg["day_end"], size=n_uniform))

        times = np.concatenate(times)
        return np.clip(times, cfg["day_start"], cfg["day_end"])

    def _turnaround(self, category, rng):
        """Ground time for one aircraft, never below the category minimum."""
        t = self.config["turnaround"][category]
        return max(t["min"], rng.normal(t["mean"], t["std"]))

    def generate(self, rng):
        """Return the day's flights, sorted by arrival time."""
        n = self.config["flights_per_day"]
        mix = self.config["fleet_mix"]
        categories = list(mix.keys())
        probabilities = list(mix.values())

        arrivals = np.sort(self._arrival_times(n, rng))
        chosen = rng.choice(categories, size=n, p=probabilities)

        flights = []
        for i, (arrival, category) in enumerate(zip(arrivals, chosen)):
            category = str(category)
            departure = arrival + self._turnaround(category, rng)
            flights.append(
                Flight(
                    flight_id=i,
                    category=category,
                    size=self.sizes[category],
                    arrival=round(float(arrival), 1),
                    departure=round(float(departure), 1),
                )
            )
        return flights