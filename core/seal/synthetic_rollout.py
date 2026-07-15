import numpy as np


SYNTHETIC_EDIT_SPEC = {
    "param_names": ["num_steps", "noise_scale", "mix_ratio"],
    "dims": 3,
    "ranges": [(1.0, 8.0), (0.0, 0.05), (0.0, 0.5)],
}


def scale_edit(raw_vector, spec=None):
    if spec is None:
        spec = SYNTHETIC_EDIT_SPEC
    scaled = np.zeros_like(raw_vector)
    for i, (lo, hi) in enumerate(spec["ranges"]):
        scaled[i] = lo + (hi - lo) * float(raw_vector[i])
    return scaled


class SyntheticRollout:
    def __init__(self, agent, wm, action_dim):
        self.agent = agent
        self.wm = wm
        self.action_dim = action_dim

    def generate(self, edit, n_transitions=32, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        edit = scale_edit(edit)
        num_steps = int(round(edit[0]))
        noise_scale = float(edit[1])
        mix_ratio = float(edit[2])

        buffer = self.agent.buffer
        if len(buffer) < 4:
            return [], 0.0

        idx = rng.integers(0, len(buffer), size=min(n_transitions, len(buffer)))
        batch = buffer._index(idx)
        h_starts = batch["hiddens"]
        actions_real = batch["actions"]

        synthetic_h = []
        synthetic_actions = []
        synthetic_h_next = []

        for i in range(len(h_starts)):
            h = h_starts[i:i+1].copy()
            for _ in range(num_steps):
                a = int(rng.integers(self.action_dim))
                h_pred = self.wm.predict(h, np.array([a], np.int32))
                if noise_scale > 0:
                    h_pred = h_pred + rng.normal(0, noise_scale, h_pred.shape).astype(np.float32)
                synthetic_h.append(h[0])
                synthetic_actions.append(a)
                synthetic_h_next.append(h_pred[0])
                h = h_pred

        n_synthetic = len(synthetic_h)
        if n_synthetic == 0:
            return [], mix_ratio

        n_mix = int(n_synthetic * mix_ratio)
        if n_mix == 0:
            return [], mix_ratio

        indices = rng.choice(n_synthetic, size=min(n_mix, n_synthetic), replace=False)
        result = {
            "states": np.zeros((len(indices), buffer.state_dim), np.float32),
            "hiddens": np.array([synthetic_h[i] for i in indices]),
            "actions": np.array([synthetic_actions[i] for i in indices], np.int32),
            "rewards": np.zeros(len(indices), np.float32),
            "next_states": np.zeros((len(indices), buffer.state_dim), np.float32),
            "next_hiddens": np.array([synthetic_h_next[i] for i in indices]),
            "dones": np.zeros(len(indices), np.float32),
            "weights": np.ones(len(indices), np.float32),
        }
        return result, mix_ratio
