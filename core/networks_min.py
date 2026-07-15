"""
networks_min.py — Primitives for the Genesis D1 agent
======================================================
Standalone copy, trimmed to only what DelayCorrectedBellman (D1) and a
GRU policy net need: activations, Linear, LayerNorm, MLP, GRUCell, Adam,
GRUPolicyNet, gru_batch_forward.

Source: vendored 2026-07-13 from an external project, unmodified logic.
"""
import numpy as np

# ── Activations ───────────────────────────────────────────────────────────────

def relu(x):     return np.maximum(0.0, x)
def sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def tanh(x):     return np.tanh(np.clip(x, -20, 20))
def softplus(x):
    x = np.clip(x, -20, 20)
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
def softmax(x):
    e = np.exp(np.clip(x, -30, 30) - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

def d_relu(x):     return (x > 0).astype(np.float32)
def d_sigmoid(s):  return s * (1.0 - s)
def d_tanh(t):     return 1.0 - t**2
def d_softplus(x): return sigmoid(x)


# ── Primitives ────────────────────────────────────────────────────────────────

class Linear:
    def __init__(self, in_dim, out_dim, rng, scale=None):
        scale = scale or np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, np.float32)
        self._last_x = None

    def forward(self, x):
        x = np.asarray(x, np.float32)
        self._last_x = x
        return x @ self.W + self.b

    def backward(self, d_out):
        dW = self._last_x.T @ d_out
        db = d_out.sum(axis=0)
        dx = d_out @ self.W.T
        return dx, dW, db

    def params(self): return [self.W, self.b]


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.g = np.ones(dim, np.float32)
        self.b = np.zeros(dim, np.float32)
        self.eps = eps
        self._cache = None

    def forward(self, x):
        mu    = x.mean(-1, keepdims=True)
        std   = x.std(-1, keepdims=True) + self.eps
        x_hat = (x - mu) / std
        self._cache = (x_hat, std)
        return self.g * x_hat + self.b

    def backward(self, d_out):
        x_hat, std = self._cache
        B, D = d_out.shape[0], d_out.shape[-1]
        dg     = (d_out * x_hat).sum(0)
        db_g   = d_out.sum(0)
        dx_hat = d_out * self.g
        # BUG FIX (2026-07-14): was dividing by (B * std) -- B is the batch
        # size, an unrelated axis. LayerNorm normalizes across D (the
        # feature dimension, x.mean(-1)/x.std(-1) above), so the backward
        # formula's coefficient must be 1/(D * std), not 1/(B * std).
        # Confirmed via finite-difference: old version gave dx off by a
        # large, wrong-signed factor (ratio ~ -1321 in a B=4, D=8 test).
        # dg/db_g were unaffected (they sum over the batch axis correctly,
        # by design), which is why every module using MLP looked fine up
        # to now -- nothing had ever needed to backprop THROUGH an MLP
        # into an earlier stage before GRUReconstructionTrainer.
        dx     = (1.0 / (D * std)) * (
            D * dx_hat
            - dx_hat.sum(-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).sum(-1, keepdims=True)
        )
        return dx, dg, db_g

    def params(self): return [self.g, self.b]


class MLP:
    """Linear -> LayerNorm -> ReLU trunk with full backprop."""
    def __init__(self, dims, rng, scale=None):
        self.layers, self.norms = [], []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i+1], rng, scale))
            if i < len(dims) - 2:
                self.norms.append(LayerNorm(dims[i+1]))
        self._pre_acts = []

    def forward(self, x):
        x = np.asarray(x, np.float32)
        self._pre_acts = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.norms):
                x = self.norms[i].forward(x)
                self._pre_acts.append(x.copy())
                x = relu(x)
        return x

    def backward(self, d_out):
        grads = []
        d = d_out
        n_hidden = len(self.norms)
        for i in reversed(range(len(self.layers))):
            if i < n_hidden:
                d = d * d_relu(self._pre_acts[i])
                d, dg, db_n = self.norms[i].backward(d)
                grads = [dg, db_n] + grads
            dx, dW, db = self.layers[i].backward(d)
            grads = [dW, db] + grads
            d = dx
        return d, grads

    def all_params(self):
        p = []
        for i, layer in enumerate(self.layers):
            p.extend(layer.params())
            if i < len(self.norms):
                p.extend(self.norms[i].params())
        return p

    def hidden_features(self, x):
        x = np.asarray(x, np.float32)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward(x)
            if i < len(self.norms):
                x = self.norms[i].forward(x)
                x = relu(x)
        return x


# ── GRU Cell with full backprop ────────────────────────────────────────────────

class GRUCell:
    def __init__(self, input_dim, hidden_dim, rng):
        s = np.sqrt(2.0 / (input_dim + hidden_dim))
        def _W(): return rng.normal(0, s, (input_dim,  hidden_dim)).astype(np.float32)
        def _U(): return rng.normal(0, s, (hidden_dim, hidden_dim)).astype(np.float32)
        def _b(): return np.zeros(hidden_dim, np.float32)

        self.Wr, self.Ur, self.br = _W(), _U(), _b()
        self.Wz, self.Uz, self.bz = _W(), _U(), _b()
        self.Wn, self.Un, self.bn = _W(), _U(), _b()
        self._cache = None

    def forward(self, x, h):
        x = np.asarray(x, np.float32)
        h = np.asarray(h, np.float32)
        r = sigmoid(x @ self.Wr + h @ self.Ur + self.br)
        z = sigmoid(x @ self.Wz + h @ self.Uz + self.bz)
        n = tanh(   x @ self.Wn + (r * h) @ self.Un + self.bn)
        h_new = (1.0 - z) * n + z * h
        self._cache = (x, h, r, z, n)
        return h_new

    def backward(self, d_h_new):
        x, h, r, z, n = self._cache
        d_n = d_h_new * (1.0 - z) * d_tanh(n)
        d_z = d_h_new * (h - n)   * d_sigmoid(z)

        dWn = x.T @ d_n;        dUn = (r * h).T @ d_n;  dbn = d_n.sum(0)
        dWz = x.T @ d_z;        dUz = h.T @ d_z;         dbz = d_z.sum(0)

        d_r_raw = d_n @ self.Un.T * h
        d_r     = d_r_raw * d_sigmoid(r)
        dWr = x.T @ d_r;        dUr = h.T @ d_r;         dbr = d_r.sum(0)

        dx = d_n @ self.Wn.T + d_z @ self.Wz.T + d_r @ self.Wr.T
        dh = (d_h_new * z + d_n @ self.Un.T * r +
              d_z @ self.Uz.T + d_r @ self.Ur.T)
        return dx, dh, [dWr, dUr, dbr, dWz, dUz, dbz, dWn, dUn, dbn]

    def params(self):
        return [self.Wr, self.Ur, self.br,
                self.Wz, self.Uz, self.bz,
                self.Wn, self.Un, self.bn]

    def zero_state(self, batch=1):
        return np.zeros((batch, self.Wr.shape[1]), np.float32)


# ── Adam ──────────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip=3.0):
        self.params = list(params)
        self.lr, self.beta1, self.beta2, self.eps, self.clip = lr, beta1, beta2, eps, clip
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = np.clip(g, -self.clip, self.clip)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)

    def add_params(self, new_params):
        for p in new_params:
            self.params.append(p)
            self.m.append(np.zeros_like(p))
            self.v.append(np.zeros_like(p))


# ── GRU Policy Network (dueling Q-head) ─────────────────────────────────────────

class GRUPolicyNet:
    def __init__(self, state_dim, action_dim, gru_dim=64, hidden_dim=128,
                 n_layers=2, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.gru_dim    = gru_dim
        self.action_dim = action_dim

        self.gru      = GRUCell(state_dim, gru_dim, rng)
        trunk_dims    = [gru_dim] + [hidden_dim] * n_layers
        self.trunk    = MLP(trunk_dims, rng)
        self.val_head = Linear(hidden_dim, 1,          rng, scale=0.01)
        self.adv_head = Linear(hidden_dim, action_dim, rng, scale=0.01)

        all_p = (self.gru.params() + self.trunk.all_params() +
                 self.val_head.params() + self.adv_head.params())
        self.optim = Adam(all_p, lr=lr)

    def forward(self, state, h):
        state = np.asarray(state, np.float32)
        if state.ndim == 1: state = state[None]
        h_new = self.gru.forward(state, h)
        feat  = self.trunk.forward(h_new)
        v     = self.val_head.forward(feat)
        a     = self.adv_head.forward(feat)
        q     = v + a - a.mean(axis=-1, keepdims=True)
        return q, h_new

    def backward_update(self, state, h, actions, td_errors, weights):
        # BUG FOUND during Phase 1 divergence debugging (2026-07-14):
        # Q-values diverged to 6-figure magnitudes within ~2000 training
        # steps in a supposedly converging TD setup. Isolated to this
        # function via a minimal single-sample gradient check: training
        # toward a fixed target of 0 made q_sa grow monotonically instead
        # of shrinking. Root cause: d_q[i, ac] was set to -delta[i] while
        # val_head.backward() below receives +delta directly for the same
        # td_error. dL/dq_sa = td_error for both v and a (since
        # q = v + a - mean(a)), so both heads must receive the SAME sign.
        # The negation here made the advantage head's gradient fight the
        # value head's gradient on every update -- slow-building internal
        # contradiction, not a clean explosion, which is why it wasn't
        # obvious from loss curves alone. Fixed: d_q now uses +delta[i],
        # matching val_head's sign convention. Verified via isolated
        # single-sample and 64-sample synthetic tests: q_sa now damps
        # toward the target instead of diverging.
        B     = len(actions)
        state = np.asarray(state, np.float32)
        h     = np.asarray(h, np.float32)

        h_new = self.gru.forward(state, h)
        feat  = self.trunk.forward(h_new)
        v     = self.val_head.forward(feat)
        a     = self.adv_head.forward(feat)

        delta = (td_errors * weights / B).astype(np.float32)

        d_q = np.zeros((B, self.action_dim), np.float32)
        for i, ac in enumerate(actions):
            d_q[i, ac] = delta[i]
        d_q -= d_q.mean(axis=-1, keepdims=True)

        d_feat_v, dWv, dbv = self.val_head.backward(delta[:, None])
        d_feat_a, dWa, dba = self.adv_head.backward(d_q)
        d_feat = d_feat_v + d_feat_a

        d_h_new, trunk_grads = self.trunk.backward(d_feat)
        _, _, gru_grads      = self.gru.backward(d_h_new)

        all_grads = gru_grads + trunk_grads + [dWv, dbv, dWa, dba]
        self.optim.step(all_grads)

    def zero_state(self, batch=1): return self.gru.zero_state(batch)

    def copy_weights_from(self, other):
        for s, t in zip(other._flat(), self._flat()): t[:] = s

    def soft_update_from(self, other, tau=0.005):
        for s, t in zip(other._flat(), self._flat()): t[:] = tau * s + (1 - tau) * t

    def _flat(self):
        return (self.gru.params() + self.trunk.all_params() +
                self.val_head.params() + self.adv_head.params())


def gru_batch_forward(cell: GRUCell, X: np.ndarray, H: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float32); H = np.asarray(H, np.float32)
    r = sigmoid(X @ cell.Wr + H @ cell.Ur + cell.br)
    z = sigmoid(X @ cell.Wz + H @ cell.Uz + cell.bz)
    n = tanh(   X @ cell.Wn + (r * H) @ cell.Un + cell.bn)
    return (1.0 - z) * n + z * H
