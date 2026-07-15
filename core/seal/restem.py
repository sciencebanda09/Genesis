import numpy as np


class ReSTEM:
    def __init__(self, policy, keep_top_k=0.5, reward_threshold=None, max_buffer=200):
        self.policy = policy
        self.keep_top_k = keep_top_k
        self.reward_threshold = reward_threshold
        self.buffer_metric_states = []
        self.buffer_target_edits = []
        self.buffer_rewards = []
        self.max_buffer = max_buffer
        self.iterations = 0

    def step(self, metric_state, edits, rewards):
        self.iterations += 1
        metric_state = np.asarray(metric_state, np.float32)
        edits = np.asarray(edits, np.float32)
        rewards = np.asarray(rewards, np.float32)

        threshold = self.reward_threshold
        if threshold is None:
            threshold = float(np.median(rewards)) if len(rewards) > 0 else 0.0

        good = rewards >= threshold
        if not good.any():
            return {"kept": 0, "mean_reward": float(rewards.mean()), "best_reward": float(rewards.max())}

        kept_edits = edits[good]
        kept_rewards = rewards[good]

        for i in range(len(kept_edits)):
            self.buffer_metric_states.append(metric_state.copy())
            self.buffer_target_edits.append(kept_edits[i].copy())
            self.buffer_rewards.append(float(kept_rewards[i]))

        if len(self.buffer_metric_states) > self.max_buffer:
            idx = np.argsort(self.buffer_rewards)[-self.max_buffer:]
            self.buffer_metric_states = [self.buffer_metric_states[i] for i in idx]
            self.buffer_target_edits = [self.buffer_target_edits[i] for i in idx]
            self.buffer_rewards = [self.buffer_rewards[i] for i in idx]

        if len(self.buffer_metric_states) >= 4:
            ms = np.array(self.buffer_metric_states)
            te = np.array(self.buffer_target_edits)
            loss = self.policy.train_on_edits(ms, te)

        return {
            "kept": int(good.sum()),
            "total": len(rewards),
            "mean_reward": float(rewards.mean()),
            "best_reward": float(rewards.max()),
            "threshold": float(threshold),
            "buffer_size": len(self.buffer_metric_states),
        }
