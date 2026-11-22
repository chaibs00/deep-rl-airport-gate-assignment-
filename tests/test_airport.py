"""
Tests for the Airport class.
    python -m pytest tests/test_airport.py
"""


from gatemind.environement.airport import Airport


def make_airport():
    airport = Airport("configs/airport.yaml")
    airport.reset()
    return airport


def test_number_of_gates():
    airport = make_airport()
    assert airport.num_gates == 24  # 23 contact gates + 1 remote
    assert airport.gates[airport.remote_index].is_remote


def test_wide_body_cannot_use_narrow_gate():
    airport = make_airport()
    wide = airport.sizes["wide"]
    b1 = 0  # first gate is B1 (narrow-body)
    assert not airport.is_valid(b1, wide, arrival_time=0)


def test_occupied_gate_is_invalid_until_free():
    airport = make_airport()
    narrow = airport.sizes["narrow"]
    airport.assign(0, departure_time=100)           # B1 busy until 100 + 10 buffer
    assert not airport.is_valid(0, narrow, arrival_time=105)
    assert airport.is_valid(0, narrow, arrival_time=110)


def test_remote_is_always_valid():
    airport = make_airport()
    wide = airport.sizes["wide"]
    airport.assign(airport.remote_index, departure_time=500)
    assert airport.is_valid(airport.remote_index, wide, arrival_time=0)