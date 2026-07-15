"""
rnd.py — Random Network Distillation intrinsic reward (Phase 1's "curiosity" drive).

Mechanism (Burda et al. 2018):
    - target: a fixed, randomly-initialized MLP. Never trained.
    - predictor: a trainable MLP trying to match target's output.
    - intrinsic reward = ||predictor(obs) - target(obs)||^2

States the agent hasn't seen much of are states the predictor hasn't learned
to match yet -> higher prediction error -> higher reward -> agent is drawn
toward novelty. As the predictor learns a state, reward there decays, so the
agent moves on. This is a real mechanism, not a vibe: the loss function IS
the curiosity signal.

Known failure mode (the "noisy TV" problem): pure novelty-seeking can get
stuck on states that are inherently unpredictable (e.g. injected noise)
rather than states that are novel-but-learnable. Nothing here corrects for
that yet -- that's precisely the gap D3 (causal attribution) is meant to
close in a later phase. Flagging it now so it isn't a surprise later.
"""
import numpy as np
from core.networks_min import MLP, Adam


class RNDModule:
    def __init__(self, state_dim, hidden_dim=32, out_dim=16, lr=1e-3, seed=0):
        rng_target = np.random.default_rng(seed)
        rng_pred = np.random.default_rng(seed + 1)

        self.target = MLP([state_dim, hidden_dim, out_dim], rng_target)
        self.predictor = MLP([state_dim, hidden_dim, out_dim], rng_pred)
        self.optim = Adam(self.predictor.all_params(), lr=lr)

        # Running normalization of intrinsic reward (RND is sensitive to scale
        # drift over training; without this, rewards can blow up or vanish).
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 1e-4

    def intrinsic_reward(self, obs_batch: np.ndarray, update_norm=True) -> np.ndarray:
        """
        obs_batch: (B, state_dim) or (state_dim,)
        Returns: (B,) raw prediction-error rewards (not yet normalized).
        """
        obs_batch = np.asarray(obs_batch, np.float32)
        scalar = obs_batch.ndim == 1
        if scalar:
            obs_batch = obs_batch[None]

        target_out = self.target.forward(obs_batch)
        pred_out = self.predictor.forward(obs_batch)
        err = np.mean((pred_out - target_out) ** 2, axis=-1)

        if update_norm:
            batch_mean = err.mean()
            batch_var = err.var()
            n = len(err)
            delta = batch_mean - self._reward_mean
            tot = self._reward_count + n
            self._reward_mean += delta * n / tot
            self._reward_var = (
                self._reward_var * self._reward_count + batch_var * n +
                delta ** 2 * self._reward_count * n / tot
            ) / tot
            self._reward_count = tot

        return float(err[0]) if scalar else err

    def normalize(self, raw_reward: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._reward_var) + 1e-8
        return raw_reward / std

    def update_step(self, obs_batch: np.ndarray) -> float:
        """Train the predictor to match the fixed target. Returns scalar loss."""
        obs_batch = np.asarray(obs_batch, np.float32)
        target_out = self.target.forward(obs_batch)
        pred_out = self.predictor.forward(obs_batch)

        diff = pred_out - target_out
        loss = float(np.mean(diff ** 2))

        d_out = (2.0 / diff.shape[-1]) * diff / len(obs_batch)
        _, grads = self.predictor.backward(d_out)
        self.optim.step(grads)
        return loss
