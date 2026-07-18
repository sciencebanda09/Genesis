"""
world_model_v3.py — Multi-step world model with imagination.

Extends Phase 1's single-step ForwardWorldModel with:
  1. Multi-step prediction (rollout k steps into the future)
  2. Uncertainty estimation (ensemble disagreement)
  3. Imagination — generate hypothetical (h, action) trajectories
     for planning without environment interaction

Architecture:
  - Ensemble of ForwardWorldModel copies for uncertainty quantification
  - Rollout using predicted h as input for subsequent steps
  - Evaluation metric: prediction error divergence over horizon length
"""
import numpy as np
from core.networks_min import MLP, Adam


class MultiStepWorldModel:
    """K-step forward model with ensemble uncertainty."""
    def __init__(self, gru_dim, action_dim, hidden_dim=64, 
                 n_ensemble=3, lr=1e-3, seed=0):
        self.gru_dim = gru_dim
        self.action_dim = action_dim
        self.n_ensemble = n_ensemble
        self.models = []
        self.optimizers = []
        rng = np.random.default_rng(seed)
        for i in range(n_ensemble):
            m = MLP([gru_dim + action_dim, hidden_dim, hidden_dim, gru_dim], 
                    np.random.default_rng(seed + i * 100))
            self.models.append(m)
            self.optimizers.append(Adam(m.all_params(), lr=lr))

    def _action_onehot(self, actions):
        actions = np.asarray(actions, np.int32)
        B = len(actions)
        onehot = np.zeros((B, self.action_dim), np.float32)
        onehot[np.arange(B), actions] = 1.0
        return onehot

    def predict_step(self, h_batch, actions, ensemble_idx=None):
        h_batch = np.asarray(h_batch, np.float32)
        if np.any(np.isnan(h_batch)):
            return np.zeros_like(h_batch) if ensemble_idx is not None else np.zeros((self.n_ensemble, *h_batch.shape))
        onehot = self._action_onehot(actions)
        x = np.concatenate([h_batch, onehot], axis=-1)
        if ensemble_idx is not None:
            return self.models[ensemble_idx].forward(x)
        predictions = np.stack([m.forward(x) for m in self.models], axis=0)
        return predictions

    def predict_with_uncertainty(self, h_batch, actions):
        """Returns (mean_prediction, uncertainty) where uncertainty is ensemble std."""
        preds = self.predict_step(h_batch, actions, ensemble_idx=None)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    def rollout(self, h_start, action_sequence):
        """Roll out a sequence of actions from h_start, returning predicted h trajectory.

        Args:
            h_start: (gru_dim,) initial hidden state
            action_sequence: (seq_len,) int actions

        Returns:
            h_trajectory: (seq_len+1, gru_dim) — includes h_start as first element
            uncertainties: (seq_len, gru_dim) — per-step ensemble std
        """
        seq_len = len(action_sequence)
        h_traj = [np.asarray(h_start, np.float32).ravel()]
        uncertainties = []
        h = h_traj[0]
        for a in action_sequence:
            mean_h, std_h = self.predict_with_uncertainty(
                h.reshape(1, -1), np.array([a], np.int32))
            h = mean_h[0]
            h_traj.append(h)
            uncertainties.append(std_h[0])
        return np.stack(h_traj), np.stack(uncertainties) if uncertainties else None

    def imagine(self, h_start, horizon=10, n_candidates=8, 
                action_set=None):
        """Generate candidate future trajectories by random action sequences.

        Args:
            h_start: starting hidden state
            horizon: how many steps to imagine
            n_candidates: number of trajectories to generate
            action_set: list of possible actions (default: 0..action_dim-1)

        Returns:
            trajectories: (n_candidates, horizon+1, gru_dim)
            action_seqs: (n_candidates, horizon) int
            uncertainties: (n_candidates, horizon, gru_dim)
        """
        if action_set is None:
            action_set = list(range(self.action_dim))
        rng = np.random.default_rng()
        trajectories = []
        action_seqs = []
        uncertainties_list = []
        for _ in range(n_candidates):
            seq = rng.choice(action_set, size=horizon).astype(np.int32)
            traj, unc = self.rollout(h_start, seq)
            trajectories.append(traj)
            action_seqs.append(seq)
            uncertainties_list.append(unc if unc is not None else np.zeros((horizon, self.gru_dim)))
        return (np.stack(trajectories), np.stack(action_seqs), 
                np.stack(uncertainties_list))

    def update_step(self, h_batch, actions, h_next_actual):
        """Update all ensemble members on the same batch.

        ponytail: each ensemble member trains on the full batch.
        Proper bagging would use bootstrap sampling per member.
        """
        h_batch = np.asarray(h_batch, np.float32)
        h_next_actual = np.asarray(h_next_actual, np.float32)
        if np.any(np.isnan(h_batch)) or np.any(np.isnan(h_next_actual)):
            return float('inf')
        onehot = self._action_onehot(actions)
        x = np.concatenate([h_batch, onehot], axis=-1)

        losses = []
        for i, (model, optim) in enumerate(zip(self.models, self.optimizers)):
            pred = model.forward(x)
            diff = pred - h_next_actual
            loss = float(np.mean(diff ** 2))
            d_out = (2.0 / diff.shape[-1]) * diff / len(h_batch)
            _, grads = model.backward(d_out)
            optim.step(grads)
            losses.append(loss)
        return np.mean(losses)

    def multi_step_loss(self, h_trajectory, actions_taken, lookahead=3):
        """Compute prediction error over k-step rollouts.

        Evaluates how prediction error accumulates over the lookahead horizon.
        Used as a diagnostic and training signal for temporal consistency.
        """
        h_traj = np.asarray(h_trajectory, np.float32)
        actions = np.asarray(actions_taken, np.int32)
        T = len(actions)
        k = min(lookahead, T)
        errors = []
        for t in range(T - k):
            h_current = h_traj[t]
            for j in range(k):
                pred, _ = self.predict_with_uncertainty(
                    h_current.reshape(1, -1), np.array([actions[t + j]], np.int32))
                h_current = pred[0]
            actual = h_traj[t + k]
            errors.append(float(np.mean((h_current - actual) ** 2)))
        return np.mean(errors) if errors else 0.0
