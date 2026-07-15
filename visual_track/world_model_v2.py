"""
world_model_v2.py -- Latent-space forward world model.

Predicts latent_{t+1} from (latent_t, action), operating in the vision
encoder's representational space rather than the GRU hidden space.

This replaces world_model.py in the visual pipeline. The architecture is
identical (MLP), only dimensions change: latent_dim (128) instead of
gru_dim (32). The key addition is update_step_with_grad() which returns
the gradient w.r.t. the input latent_t so the vision encoder can be
trained by the prediction objective.

Why latent space and not pixel space?
    Predicting pixels (video prediction) is an order of magnitude harder
    and introduces wasted model capacity on rendering details (textures,
    exact wall colors) that don't matter for behaviour. Latent prediction
    forces the encoder to extract only what is predictable, which is
    precisely the spatial and object structure the later phases need.
"""
import numpy as np
from core.networks_min import MLP, Adam


class LatentWorldModel:
    """Forward model on latent space: (latent_t, action) -> pred_latent_{t+1}.

    Args:
        latent_dim: dimension of vision encoder output (default 128).
        action_dim: number of discrete actions.
        hidden_dim: MLP hidden layer size (default 128, matching latent_dim).
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=128, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = MLP([latent_dim + action_dim, hidden_dim, hidden_dim, latent_dim], rng)
        self.optim = Adam(self.net.all_params(), lr=lr)

    def _action_onehot(self, actions):
        actions = np.asarray(actions, np.int32)
        B = len(actions)
        oh = np.zeros((B, self.action_dim), np.float32)
        oh[np.arange(B), actions] = 1.0
        return oh

    def predict(self, latents, actions):
        """Forward pass only. latents: (B, latent_dim), actions: (B,) int."""
        latents = np.asarray(latents, np.float32)
        x = np.concatenate([latents, self._action_onehot(actions)], axis=-1)
        return self.net.forward(x)

    def update_step(self, latents, actions, next_latents):
        """Standard update without input gradient. Returns scalar MSE loss.

        Provided for backwards compatibility / simplicity when encoder
        backprop is not needed.
        """
        latents = np.asarray(latents, np.float32)
        next_latents = np.asarray(next_latents, np.float32)
        x = np.concatenate([latents, self._action_onehot(actions)], axis=-1)

        pred = self.net.forward(x)
        diff = pred - next_latents
        loss = float(np.mean(diff ** 2))

        d_out = (2.0 / diff.shape[-1]) * diff / len(latents)
        _, grads = self.net.backward(d_out)
        self.optim.step(grads)
        return loss

    def update_step_with_grad(self, latents, actions, next_latents):
        """Update + return (loss, d_latent) for encoder backprop.

        next_latents should have stop_gradient applied (i.e. treated as
        fixed targets, not outputs of the current encoder during this
        update). If you pass encoder-produced next_latents, detach them
        first by calling .copy() on the array before passing it here.

        Returns:
            loss: float MSE.
            d_latent: (B, latent_dim) gradient of loss w.r.t. latent_t.
        """
        latents = np.asarray(latents, np.float32)
        next_latents = np.asarray(next_latents, np.float32)
        onehot = self._action_onehot(actions)
        x = np.concatenate([latents, onehot], axis=-1)

        pred = self.net.forward(x)
        diff = pred - next_latents
        loss = float(np.mean(diff ** 2))

        d_out = (2.0 / diff.shape[-1]) * diff / len(latents)
        d_input, grads = self.net.backward(d_out)
        self.optim.step(grads)

        d_latent = d_input[:, :self.latent_dim]
        return loss, d_latent

    def prediction_error(self, latents, actions, next_latents):
        """Eval only. Returns per-sample MSE (B,)."""
        pred = self.predict(latents, actions)
        return np.mean((pred - np.asarray(next_latents, np.float32)) ** 2, axis=-1)
