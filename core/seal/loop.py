import numpy as np
from core.seal.self_edit_policy import SelfEditPolicy
from core.seal.restem import ReSTEM


class SEALLoop:
    def __init__(self, metric_dim, edit_dim, inner_loop_fn,
                 hidden_dim=32, policy_lr=1e-3, keep_top_k=0.5,
                 reward_threshold=None, seed=0):
        self.policy = SelfEditPolicy(metric_dim, edit_dim, hidden_dim, policy_lr, seed)
        self.restem = ReSTEM(self.policy, keep_top_k, reward_threshold)
        self.inner_loop_fn = inner_loop_fn
        self.edit_dim = edit_dim
        self.metric_dim = metric_dim
        self.history = []

    def outer_step(self, metric_state, n_edits=5, seed_offset=0):
        metric_state = np.asarray(metric_state, np.float32)
        edits = []
        for i in range(n_edits):
            edit = self.policy.generate(metric_state)
            edits.append(edit)

        rewards = []
        infos = []
        for i, edit in enumerate(edits):
            reward, info = self.inner_loop_fn.run(edit, seed_offset=seed_offset + i * 1000)
            rewards.append(reward)
            infos.append(info)

        restem_result = self.restem.step(metric_state, np.array(edits), np.array(rewards))
        best_idx = int(np.argmax(rewards))

        result = {
            "restem": restem_result,
            "best_reward": float(rewards[best_idx]),
            "mean_reward": float(np.mean(rewards)),
            "best_edit": edits[best_idx].tolist(),
            "infos": infos,
        }
        self.history.append(result)
        return result
