"""Metrics describing the quality of one day's gate schedule."""

import numpy as np


def day_metrics(env):
    """Computes the metrics of the day that was just played in `env`."""
    gates = {gate.name: gate for gate in env.airport.gates}
    n = len(env.assignments)

    contact, walk, mismatch = 0, [], 0
    for flight, gate_name in env.assignments:
        gate = gates[gate_name]
        walk.append(gate.walk_minutes)
        if not gate.is_remote:
            contact += 1
            if gate.max_size > flight.size:
                mismatch += 1

    return {
        "contact_rate": 100 * contact / n,       # % of flights at a contact gate
        "remote": n - contact,                   # flights sent to a remote stand
        "walk": float(np.mean(walk)),            # average walking time (minutes)
        "size_mismatch": 100 * mismatch / n,     # % of flights at a gate larger than needed
    }