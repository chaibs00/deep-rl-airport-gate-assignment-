"""Deep Q-Network agent with action masking."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from gatemind.agents.networks import DuelingQNetwork, QNetwork, masked_q_values
from gatemind.agents.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(self, obs_dim, n_actions, config, seed=0):
        self.cfg = config
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Options (step 8): Double DQN target and dueling architecture.
        self.double = config.get("double", False)
        network = DuelingQNetwork if config.get("dueling", False) else QNetwork

        # Online network (learns) and target network (slow copy, used for targets).
        self.q_net = network(obs_dim, n_actions, config["hidden"])
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config["lr"])

        self.buffer = ReplayBuffer(config["buffer_capacity"], obs_dim, n_actions)
        self.steps = 0          # transitions seen so far
        self.training = True    # False = no exploration (evaluation)

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #

    @property
    def epsilon(self):
        """Linear decay from epsilon_start to epsilon_end."""
        c = self.cfg
        progress = min(self.steps / c["epsilon_decay_steps"], 1.0)
        return c["epsilon_start"] + progress * (c["epsilon_end"] - c["epsilon_start"])

    def act(self, obs, mask):
        """Epsilon-greedy over valid actions only."""
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))

        with torch.no_grad():
            q = self.q_net(torch.as_tensor(obs).unsqueeze(0))
            q = masked_q_values(q, torch.as_tensor(mask).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def remember(self, obs, action, reward, next_obs, next_mask, done):
        self.buffer.push(obs, action, reward, next_obs, next_mask, done)
        self.steps += 1

    def learn(self):
        """One gradient step on a random mini-batch. Returns the loss, or None."""
        c = self.cfg
        if len(self.buffer) < max(c["learning_starts"], c["batch_size"]):
            return None

        batch = self.buffer.sample(c["batch_size"], self.rng)
        obs = torch.as_tensor(batch["obs"])
        actions = torch.as_tensor(batch["actions"])
        rewards = torch.as_tensor(batch["rewards"])
        next_obs = torch.as_tensor(batch["next_obs"])
        next_masks = torch.as_tensor(batch["next_masks"])
        dones = torch.as_tensor(batch["dones"])

        # Q(s, a) for the actions that were actually taken.
        q_taken = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * value of the best valid next gate.
        with torch.no_grad():
            if self.double:
                # Double DQN: the online network chooses a', the target network evaluates it.
                online_next = masked_q_values(self.q_net(next_obs), next_masks)
                next_actions = online_next.argmax(dim=1, keepdim=True)
                next_value = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                # DQN: the target network both chooses and evaluates a'.
                next_q = masked_q_values(self.target_net(next_obs), next_masks)
                next_value = next_q.max(dim=1).values

            next_value = torch.where(dones.bool(), torch.zeros_like(next_value), next_value)
            target = rewards + c["gamma"] * next_value

        loss = F.smooth_l1_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        if self.steps % c["target_update"] == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())