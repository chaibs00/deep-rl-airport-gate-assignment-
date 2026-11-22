"""
Plays one simulated day with random valid actions.

    python -m scripts.random_day
"""

import numpy as np

from gatemind.environement.gate_env import GateAssignmentEnv

env = GateAssignmentEnv()
rng = np.random.default_rng(0)

obs, info = env.reset(seed=0)
print("Observation size:", obs.shape[0], "| Actions:", env.action_space.n)

total, done = 0.0, False
while not done:
    valid = np.flatnonzero(info["action_mask"])
    action = int(rng.choice(valid))
    obs, reward, done, _, info = env.step(action)
    total += reward

remote = sum(1 for _, gate in env.assignments if gate == "REMOTE")
print(f"Total reward: {total:.1f} | Flights: {len(env.assignments)} | Remote stands: {remote}")
for flight, gate in env.assignments[:5]:
    print(f"  flight {flight.flight_id:2d} ({flight.category:8s}) -> {gate}")