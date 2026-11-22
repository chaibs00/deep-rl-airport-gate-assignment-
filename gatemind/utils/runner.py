"""Runs one simulated day with any agent and returns a summary."""

from gatemind.utils.metrics import day_metrics


def run_episode(env, agent, seed):
    obs, info = env.reset(seed=seed)
    total_reward, done = 0.0, False

    while not done:
        action = agent.act(obs, info["action_mask"])
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

    return {"reward": total_reward, **day_metrics(env)}