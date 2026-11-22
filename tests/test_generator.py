"""
Tests for the ScheduleGenerator class.
    python -m pytest tests/test_generator.py
"""


import numpy as np

from gatemind.data.schedule_generator import ScheduleGenerator
from gatemind.environement.airport import Airport


def make_flights(seed=0):
    airport = Airport("configs/airport.yaml")
    generator = ScheduleGenerator(airport.sizes, "configs/schedule.yaml")
    return generator.generate(np.random.default_rng(seed))


def test_number_of_flights():
    assert len(make_flights()) == 100


def test_flights_sorted_by_arrival():
    arrivals = [f.arrival for f in make_flights()]
    assert arrivals == sorted(arrivals)


def test_departure_after_arrival():
    for f in make_flights():
        assert f.departure > f.arrival


def test_same_seed_gives_same_day():
    assert make_flights(seed=42) == make_flights(seed=42)


def test_different_seeds_give_different_days():
    assert make_flights(seed=1) != make_flights(seed=2)