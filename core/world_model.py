"""
world_model.py — Step 1 of concept formation: a forward model on h.

Predicts h_{t+1} from (h_t, action_t). This is the foundation the other two
concept-formation steps (contrastive learning, clustering) will build on:
it's the first thing that gives h a reason to encode "what happens next"
rather than only "what's my current value estimate."

Deliberately kept separate from D1Agent's own GRU forward pass -- this
module CONSUMES h_t and h_{t+1} that the agent already produces during
training, it does not run its own copy of the GRU. That keeps this an
additive, bolt-on loss rather than a second competing recurrent pathway.

Architecture: MLP([gru_dim + action_dim, hidden, hidden, gru_dim])
Loss: MSE(predicted h_{t+1}, actual h_{t+1})

What "working" means for this module in isolation (checked before moving
to contrastive learning): prediction loss should go down over training on
real trajectories, and should be higher for random (h, action, h_next)
mismatches than for real transitions -- i.e. it should actually be
learning transition structure, not just regressing to the mean of h.
"""
import numpy as np
from core.networks_min import MLP, Adam


class ForwardWorldModel:
    def __init__(self, gru_dim, action_dim, hidden_dim=64, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.gru_dim = gru_dim
        self.action_dim = action_dim
        self.net = MLP([gru_dim + action_dim, hidden_dim, hidden_dim, gru_dim], rng)
        self.optim = Adam(self.net.all_params(), lr=lr)

    def _action_onehot(self, actions):
        actions = np.asarray(actions, np.int32)
        B = len(actions)
        onehot = np.zeros((B, self.action_dim), np.float32)
        onehot[np.arange(B), actions] = 1.0
        return onehot

    def predict(self, h_batch, actions):
        """h_batch: (B, gru_dim), actions: (B,) int -> predicted h_next (B, gru_dim)"""
        h_batch = np.asarray(h_batch, np.float32)
        onehot = self._action_onehot(actions)
        x = np.concatenate([h_batch, onehot], axis=-1)
        return self.net.forward(x)

    def update_step(self, h_batch, actions, h_next_actual):
        """One gradient step. Returns scalar MSE loss."""
        h_batch = np.asarray(h_batch, np.float32)
        h_next_actual = np.asarray(h_next_actual, np.float32)
        onehot = self._action_onehot(actions)
        x = np.concatenate([h_batch, onehot], axis=-1)

        pred = self.net.forward(x)
        diff = pred - h_next_actual
        loss = float(np.mean(diff ** 2))

        d_out = (2.0 / diff.shape[-1]) * diff / len(h_batch)
        _, grads = self.net.backward(d_out)
        self.optim.step(grads)
        return loss

    def prediction_error(self, h_batch, actions, h_next_actual):
        """Same as update_step but no gradient step -- for eval/diagnostics."""
        pred = self.predict(h_batch, actions)
        h_next_actual = np.asarray(h_next_actual, np.float32)
        return np.mean((pred - h_next_actual) ** 2, axis=-1)
