"""
rnd.py — Intrinsic curiosity modules for Genesis.

Two mechanisms provided:
    RNDModule  — Random Network Distillation (Burda et al. 2018)
                 novelty = ||predictor(obs) - target(obs)||^2
    ICMModule  — Intrinsic Curiosity Module (Pathak et al. 2017)
                 novelty = ||forward_model_feature(obs, action) - encoded_next_obs||^2

RND is the Phase 1 default. ICM is provided for comparison and understanding
tradeoffs between the two approaches (RND: pure novelty; ICM: prediction error
in learned feature space, less prone to noisy-TV in theory).
"""
import numpy as np
from core.networks_min import MLP, Adam, Linear


class _RunningNorm:
    """Shared running normalization used by all curiosity modules."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, batch):
        batch_mean = batch.mean()
        batch_var = batch.var()
        n = len(batch)
        delta = batch_mean - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        self.var = (self.var * self.count + batch_var * n +
                    delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def normalize(self, raw):
        return raw / (np.sqrt(self.var) + 1e-8)


class RNDModule:
    """Random Network Distillation: novelty via prediction error.

    Mechanism (Burda et al. 2018):
        - target: a fixed, randomly-initialized MLP. Never trained.
        - predictor: a trainable MLP trying to match target's output.
        - intrinsic reward = ||predictor(obs) - target(obs)||^2

    States the agent hasn't seen much of are states the predictor hasn't learned
    to match yet -> higher prediction error -> higher reward -> agent is drawn
    toward novelty. As the predictor learns a state, reward there decays, so the
    agent moves on.

    Known failure mode (the "noisy TV" problem): pure novelty-seeking can get
    stuck on states that are inherently unpredictable (e.g. injected noise)
    rather than states that are novel-but-learnable.
    """
    def __init__(self, state_dim, hidden_dim=32, out_dim=16, lr=1e-3, seed=0):
        rng_target = np.random.default_rng(seed)
        rng_pred = np.random.default_rng(seed + 1)

        self.target = MLP([state_dim, hidden_dim, out_dim], rng_target)
        self.predictor = MLP([state_dim, hidden_dim, out_dim], rng_pred)
        self.optim = Adam(self.predictor.all_params(), lr=lr)
        self._norm = _RunningNorm()

    def intrinsic_reward(self, obs_batch: np.ndarray, update_norm=True) -> np.ndarray:
        obs_batch = np.asarray(obs_batch, np.float32)
        scalar = obs_batch.ndim == 1
        if scalar:
            obs_batch = obs_batch[None]
        target_out = self.target.forward(obs_batch)
        pred_out = self.predictor.forward(obs_batch)
        err = np.mean((pred_out - target_out) ** 2, axis=-1)
        if update_norm:
            self._norm.update(err)
        return float(err[0]) if scalar else err

    def normalize(self, raw_reward: np.ndarray) -> np.ndarray:
        return self._norm.normalize(raw_reward)

    def update_step(self, obs_batch: np.ndarray) -> float:
        obs_batch = np.asarray(obs_batch, np.float32)
        target_out = self.target.forward(obs_batch)
        pred_out = self.predictor.forward(obs_batch)
        diff = pred_out - target_out
        loss = float(np.mean(diff ** 2))
        d_out = (2.0 / diff.shape[-1]) * diff / len(obs_batch)
        _, grads = self.predictor.backward(d_out)
        self.optim.step(grads)
        return loss


class ICMModule:
    """Intrinsic Curiosity Module (Pathak et al. 2017).

    Mechanism: a forward model predicts encoded(next_obs) from
    encoded(obs) + action. Intrinsic reward = prediction error in
    a learned feature space. The feature encoder is shared between
    the forward and inverse models.

    Unlike RND (pure novelty), ICM's prediction error is action-conditional:
    the reward is high when the agent cannot predict the outcome of its own
    actions. This is less prone to the noisy-TV problem because unpredictable
    but action-independent noise doesn't produce prediction error.

    Tradeoff: ICM's feature encoder adds capacity and ICM can learn to
    ignore action-relevant features (feature hypothesis: the encoder learns
    to discard predictable information to minimize its own loss).
    """
    def __init__(self, state_dim, action_dim, feat_dim=32, hidden_dim=32,
                 lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        # Feature encoder: state -> feat_dim
        self.feature_net = MLP([state_dim, hidden_dim, feat_dim], rng)
        # Forward model: concat(feature, action_onehot) -> predicted feature_next
        fwd_in = feat_dim + action_dim
        self.forward_net = MLP([fwd_in, hidden_dim, feat_dim], rng)
        all_params = self.feature_net.all_params() + self.forward_net.all_params()
        self.optim = Adam(all_params, lr=lr)
        self._norm = _RunningNorm()

    def _feat(self, obs_batch):
        obs_batch = np.asarray(obs_batch, np.float32)
        return self.feature_net.forward(obs_batch)

    def intrinsic_reward(self, obs_batch, actions, next_obs_batch,
                         update_norm=True):
        obs_batch = np.asarray(obs_batch, np.float32)
        next_obs_batch = np.asarray(next_obs_batch, np.float32)
        actions = np.asarray(actions, np.int32)
        B = len(obs_batch)
        feat = self._feat(obs_batch)
        feat_next = self._feat(next_obs_batch)
        onehot = np.zeros((B, actions.max() + 1), np.float32)
        onehot[np.arange(B), actions] = 1.0
        x = np.concatenate([feat, onehot], axis=-1)
        feat_pred = self.forward_net.forward(x)
        err = 0.5 * np.mean((feat_pred - feat_next) ** 2, axis=-1)
        if update_norm:
            self._norm.update(err)
        return err

    def normalize(self, raw_reward):
        return self._norm.normalize(raw_reward)

    def update_step(self, obs_batch, actions, next_obs_batch):
        obs_batch = np.asarray(obs_batch, np.float32)
        next_obs_batch = np.asarray(next_obs_batch, np.float32)
        actions = np.asarray(actions, np.int32)
        B = len(obs_batch)
        feat = self._feat(obs_batch)
        feat_next = self._feat(next_obs_batch)
        onehot = np.zeros((B, actions.max() + 1), np.float32)
        onehot[np.arange(B), actions] = 1.0
        x = np.concatenate([feat, onehot], axis=-1)
        feat_pred = self.forward_net.forward(x)
        fwd_diff = feat_pred - feat_next
        fwd_loss = float(np.mean(fwd_diff ** 2))

        d_fwd = fwd_diff / B
        d_fwd_in, fwd_grads = self.forward_net.backward(d_fwd)
        d_feat_from_fwd = d_fwd_in[:, :feat.shape[1]]
        _, feat_grads = self.feature_net.backward(d_feat_from_fwd)
        self.optim.step(feat_grads + fwd_grads)
        return fwd_loss
