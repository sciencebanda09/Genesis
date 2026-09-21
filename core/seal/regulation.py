import numpy as np
from core.replay_buffer import PrioritizedReplay


REGULATION_EDIT_SPEC = {
    "param_names": ["curiosity_beta", "lr_mult", "replay_priority_exp", "exploration_eps", "memory_mix_ratio"],
    "dims": 5,
    "ranges": [(0.1, 2.0), (0.5, 2.0), (0.0, 1.0), (0.01, 1.0), (0.0, 1.0)],
}


def scale_edit(raw_vector, spec=None):
    if spec is None:
        spec = REGULATION_EDIT_SPEC
    scaled = np.zeros_like(raw_vector)
    for i, (lo, hi) in enumerate(spec["ranges"]):
        scaled[i] = lo + (hi - lo) * float(raw_vector[i])
    return scaled


class RegulationInnerLoop:
    def __init__(self, env_fn, agent_fn, rnd_fn, wm_fn, cortex_fn,
                 inner_steps=100, seed=0):
        self.env_fn = env_fn
        self.agent_fn = agent_fn
        self.rnd_fn = rnd_fn
        self.wm_fn = wm_fn
        self.cortex_fn = cortex_fn
        self.inner_steps = inner_steps
        self.seed = seed

    def run(self, edit_raw, seed_offset=0):
        edit = scale_edit(edit_raw)
        curiosity_beta, lr_mult, replay_priority_exp, exploration_eps, memory_mix_ratio = edit

        base_seed = self.seed + seed_offset
        env = self.env_fn(base_seed + 1000)
        rnd = self.rnd_fn(base_seed + 2000)
        agent = self.agent_fn(base_seed + 3000)
        wm = self.wm_fn(base_seed + 4000)
        cortex = self.cortex_fn()

        _hijack_exploration(agent, exploration_eps)
        _hijack_lr(agent, lr_mult)

        pre_coverage = 0.0
        post_coverage = 0.0
        wm_losses = []
        total_intrinsic = 0.0
        prioritized = PrioritizedReplay(
            agent.buffer.capacity, agent.state_dim, agent.gru_dim,
            alpha=float(replay_priority_exp), seed=base_seed + 5000)

        for t in range(self.inner_steps):
            if t == 0:
                obs = env.reset()
                agent.reset_hidden()
                pre_coverage = env.coverage()
            else:
                obs = next_obs

            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, ext_r, done, info = env.step(action)
            intr_r_raw = rnd.intrinsic_reward(next_obs)
            intr_r = rnd.normalize(np.array([intr_r_raw]))[0]
            h_after = agent._h.copy()

            rnd_reward = curiosity_beta * intr_r
            agent.store(obs, action, rnd_reward, next_obs, done, h_before, h_after)
            prioritized.add(obs, h_before, action, rnd_reward,
                            next_obs, h_after, done)
            total_intrinsic += rnd_reward

            cortex.observe(global_step=t, coverage=env.coverage(),
                           rnd_reward=intr_r, td_error_mean=0.0, wm_loss=0.0)

            if t > 10:
                rnd.update_step(np.array([next_obs]))
                # Candidate evaluation uses the same adaptive-memory control
                # surface as the live experiment.  The prioritized replay
                # buffer is local to this candidate, so evaluating an edit
                # cannot leave priority or gradient residue in another edit.
                batch_size = agent.batch_size
                n_p = int(round(batch_size * memory_mix_ratio))
                n_u = batch_size - n_p
                uniform_batch = agent.buffer.sample(n_u) if n_u > 0 else None
                prio_batch = (prioritized.sample(n_p)
                              if n_p > 0 and len(prioritized) >= n_p else None)
                combined = _merge_batches(uniform_batch, prio_batch)
                stats = agent.update(batch=combined) if combined is not None else None
                if (stats is not None and prio_batch is not None
                        and "indices" in prio_batch
                        and stats.get("td_error") is not None):
                    n_uniform = len(uniform_batch["states"]) if uniform_batch is not None else 0
                    prioritized.update_priorities(
                        prio_batch["indices"],
                        stats["td_error"][n_uniform:])
                if wm is not None and stats is not None:
                    wml = wm.update_step(h_before, [action], h_after)
                    wm_losses.append(wml)

            if done:
                break

        post_coverage = env.coverage()
        coverage_gain = post_coverage - pre_coverage
        wm_loss_mean = float(np.mean(wm_losses)) if wm_losses else 0.0
        reward = coverage_gain - 0.1 * wm_loss_mean

        info = {
            "coverage_gain": float(coverage_gain),
            "pre_coverage": float(pre_coverage),
            "post_coverage": float(post_coverage),
            "wm_loss_mean": wm_loss_mean,
            "total_intrinsic": float(total_intrinsic),
            "replay_mix_ratio": float(memory_mix_ratio),
            "replay_priority_exp": float(replay_priority_exp),
            "edit": {REGULATION_EDIT_SPEC["param_names"][i]: float(edit[i])
                     for i in range(len(edit))},
        }
        return float(reward), info


def _merge_batches(uniform_batch, prioritized_batch):
    """Merge replay batches without sharing mutable candidate state."""
    if uniform_batch is None:
        return prioritized_batch
    if prioritized_batch is None:
        return uniform_batch
    merged = {}
    for key in ("states", "hiddens", "actions", "rewards", "next_states",
                "next_hiddens", "dones", "weights"):
        merged[key] = np.concatenate([uniform_batch[key], prioritized_batch[key]], axis=0)
    merged["indices"] = prioritized_batch["indices"]
    return merged


def _hijack_exploration(agent, fixed_eps):
    agent._saved_epsilon_fn = agent.epsilon
    agent.epsilon = lambda: fixed_eps

def _hijack_lr(agent, mult):
    agent.policy_net.optim.lr *= mult
