"""
replay_buffer.py — replay buffer family for Genesis.

Three levels, each a drop-in replacement for the one before:
    ReplayBuffer        — uniform sampling (Phase 1 original)
    PrioritizedReplay   — TD-error prioritized sampling
    SequenceReplay      — temporally correlated chunks for RNN training
"""
import numpy as np


class ReplayBuffer:
    """Uniform replay. Base for all others."""
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
        return self._index(idx)

    def _index(self, idx):
        return {
            "states": self.states[idx],
            "hiddens": self.hiddens[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "next_hiddens": self.next_hiddens[idx],
            "dones": self.dones[idx],
            "weights": np.ones(len(idx), np.float32),
        }


class PrioritizedReplay(ReplayBuffer):
    """TD-error prioritized replay (Schaul et al. 2016).

    Samples transitions with probability proportional to |TD error|^alpha.
    Importance-sampling weights correct the bias toward high-error samples.
    """
    def __init__(self, capacity, state_dim, gru_dim, alpha=0.6, beta=0.4,
                 beta_anneal=0.001, seed=0):
        super().__init__(capacity, state_dim, gru_dim, seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal
        self.priorities = np.ones(capacity, np.float32)
        self._max_priority = 1.0

    def add(self, state, hidden, action, reward, next_state, next_hidden, done):
        i = self._ptr
        super().add(state, hidden, action, reward, next_state, next_hidden, done)
        self.priorities[i] = self._max_priority

    def sample(self, batch_size: int):
        if self._size < batch_size:
            return None
        probs = self.priorities[:self._size] ** self.alpha
        probs /= probs.sum() + 1e-8
        idx = self.rng.choice(self._size, size=batch_size, replace=True, p=probs)
        batch = self._index(idx)
        weights = (self._size * probs[idx]) ** (-self.beta)
        batch["weights"] = (weights / weights.max()).astype(np.float32)
        batch["indices"] = idx
        self.beta = min(1.0, self.beta + self.beta_anneal)
        return batch

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = abs(err) + 1e-6
            if self.priorities[i] > self._max_priority:
                self._max_priority = self.priorities[i]


class SequenceReplay(ReplayBuffer):
    """Samples contiguous trajectory chunks for RNN training.

    Returns (batch_size, sequence_length) tensors preserving temporal order
    within each chunk. Crucial for training recurrent policies on actual
    trajectory structure rather than shuffled transitions.
    """
    def __init__(self, capacity, state_dim, gru_dim, seq_len=8, seed=0):
        super().__init__(capacity, state_dim, gru_dim, seed)
        self.seq_len = seq_len

    def sample(self, batch_size: int):
        if self._size < batch_size * self.seq_len:
            return None
        starts = self.rng.integers(0, self._size - self.seq_len, size=batch_size)
        idx = np.array([np.arange(s, s + self.seq_len) for s in starts])
        batch = self._index(idx.ravel())
        for k in ("states", "hiddens", "actions", "rewards", "next_states",
                  "next_hiddens", "dones"):
            batch[k] = batch[k].reshape(batch_size, self.seq_len, -1)
        batch["actions"] = batch["actions"].squeeze(-1)
        batch["rewards"] = batch["rewards"].squeeze(-1)
        batch["dones"] = batch["dones"].squeeze(-1)
        return batch
