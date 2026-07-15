import numpy as np
from core.networks_min import MLP, Adam


class SelfEditPolicy:
    def __init__(self, metric_dim, edit_dim, hidden_dim=32, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.metric_dim = metric_dim
        self.edit_dim = edit_dim
        self.net = MLP([metric_dim, hidden_dim, hidden_dim, edit_dim], rng)
        self.optim = Adam(self.net.all_params(), lr=lr)

    def generate(self, metric_state, rng=None):
        metric_state = np.asarray(metric_state, np.float32)
        if metric_state.ndim == 1:
            metric_state = metric_state[None]
        raw = self.net.forward(metric_state)
        return sigmoid(raw)[0]

    def train_on_edits(self, metric_states, target_edits):
        metric_states = np.asarray(metric_states, np.float32)
        target_edits = np.asarray(target_edits, np.float32)
        pred = self.net.forward(metric_states)
        diff = pred - target_edits
        loss = float(np.mean(diff ** 2))
        d_out = (2.0 / diff.shape[-1]) * diff / len(metric_states)
        _, grads = self.net.backward(d_out)
        self.optim.step(grads)
        return loss

    def get_metric_state(self, cortex, agent, env, wm, rnd):
        state = []
        state.append(env.coverage())
        state.append(cortex.epsilon if hasattr(cortex, 'epsilon') else agent.epsilon())
        state.append(cortex.curiosity_weights.get('rnd', 1.0))
        td = cortex.buf.get('td_error_mean')
        state.append(td.mean(50) if td and len(td) > 1 else 0.0)
        wml = cortex.buf.get('wm_loss')
        state.append(wml.mean(50) if wml and len(wml) > 1 else 0.0)
        rr = cortex.buf.get('rnd_reward')
        state.append(rr.mean(50) if rr and len(rr) > 1 else 0.0)
        cov_buf = cortex.buf.get('coverage')
        cov_trend = cov_buf.trend(50) if cov_buf and len(cov_buf) > 50 else 0.0
        state.append(cov_trend)
        state.append(agent.policy_net.optim.lr)
        return np.array(state, np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
