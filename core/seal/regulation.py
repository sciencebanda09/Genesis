import numpy as np


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
            total_intrinsic += rnd_reward

            cortex.observe(global_step=t, coverage=env.coverage(),
                           rnd_reward=intr_r, td_error_mean=0.0, wm_loss=0.0)

            if t > 10:
                rnd.update_step(np.array([next_obs]))
                stats = agent.update()
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
            "edit": {REGULATION_EDIT_SPEC["param_names"][i]: float(edit[i])
                     for i in range(len(edit))},
        }
        return float(reward), info


def _hijack_exploration(agent, fixed_eps):
    agent._saved_epsilon_fn = agent.epsilon
    agent.epsilon = lambda: fixed_eps

def _hijack_lr(agent, mult):
    agent.policy_net.optim.lr *= mult
