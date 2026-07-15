"""
replay_buffer.py — minimal replay buffer for the Phase 1 D1-only agent.

Deliberately minimal: no causal history window, no per-step SCM context,
no priority weighting yet. Those fields feed ICN/D3, which isn't wired
up in this phase. Adding them back later (when D1+D3 integration starts)
is a straightforward extension, not a rewrite -- but building them now
would be dead weight in a v1 that's supposed to be minimal.
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, gru_dim: int, seed: int = 0):
        self.capacity = capacity
        self.state_dim = state_dim
        self.gru_dim = gru_dim
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, state_dim), np.float32)
        self.next_states = np.zeros((capacity, state_dim), np.float32)
        self.hiddens = np.zeros((capacity, gru_dim), np.float32)
        self.next_hiddens = np.zeros((capacity, gru_dim), np.float32)
        self.actions = np.zeros(capacity, np.int32)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)

        self._size = 0
        self._ptr = 0

    def add(self, state, hidden, action, reward, next_state, next_hidden, done):
        i = self._ptr
        self.states[i] = state
        self.hiddens[i] = hidden
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.next_hiddens[i] = next_hidden
        self.dones[i] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self):
        return self._size

    def sample(self, batch_size: int):
        if self._size < batch_size:
            return None
        idx = self.rng.integers(0, self._size, size=batch_size)
        return {
            "states": self.states[idx],
            "hiddens": self.hiddens[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "next_hiddens": self.next_hiddens[idx],
            "dones": self.dones[idx],
            "weights": np.ones(batch_size, np.float32),
        }
