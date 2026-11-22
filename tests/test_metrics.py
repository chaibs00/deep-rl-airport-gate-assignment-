from gatemind.agents.greedy_agent import GreedyAgent
from gatemind.environement.gate_env import GateAssignmentEnv
from gatemind.utils.runner import run_episode


def test_metrics_are_consistent():
    env = GateAssignmentEnv()
    result = run_episode(env, GreedyAgent(env), seed=0)
    assert 0 <= result["contact_rate"] <= 100
    assert 0 <= result["size_mismatch"] <= 100
    n = len(env.flights)
    assert round(result["contact_rate"] * n / 100) + result["remote"] == n
    assert 3 <= result["walk"] <= 15   # between closest gate and remote stand


def test_all_flights_at_remote_gives_zero_contact():
    env = GateAssignmentEnv()

    class RemoteAgent:
        def act(self, obs, mask):
            return env.airport.remote_index

    result = run_episode(env, RemoteAgent(), seed=0)
    assert result["contact_rate"] == 0
    assert result["remote"] == len(env.flights)
    assert result["walk"] == 15