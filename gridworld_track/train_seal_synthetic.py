import argparse
import numpy as np
from tqdm import tqdm

from core.rnd import RNDModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from core.logger import JsonlLogger
from core.seal.loop import SEALLoop
from core.seal.self_edit_policy import SelfEditPolicy
from core.seal.synthetic_rollout import SyntheticRollout, scale_edit, SYNTHETIC_EDIT_SPEC
from .gridworld import GridWorld, ACTIONS


def run(episodes=300, max_steps=200, seed=0, log_path="logs/seal_synthetic.jsonl",
        warmup_steps=300, outer_every=1000, n_edits_per_step=5,
        inner_steps=100, metric_dim=8, edit_dim=3):
    logger = JsonlLogger(log_path)

    env = GridWorld(max_steps=max_steps, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)
    wm = ForwardWorldModel(gru_dim=agent.gru_dim, action_dim=env.action_dim, seed=seed)
    synth = SyntheticRollout(agent, wm, env.action_dim)

    class _InnerLoop:
        def __init__(self, _agent, _wm, _synth, _steps):
            self._agent = _agent
            self._wm = _wm
            self._synth = _synth
            self.inner_steps = _steps
            self.rng = np.random.default_rng(0)

        def _save_wm(self):
            return [p.copy() for p in self._wm.net.all_params()]

        def _restore_wm(self, saved):
            for t, s in zip(self._wm.net.all_params(), saved):
                t[:] = s

        def run(self, edit_raw, seed_offset=0):
            edit = scale_edit(edit_raw, SYNTHETIC_EDIT_SPEC)
            saved = self._save_wm()
            pre_wm_err = _eval_wm_error(self._wm, self._agent)
            synth_data, mix_ratio = self._synth.generate(
                edit_raw, rng=np.random.default_rng(int(seed_offset)))

            for t in range(min(self.inner_steps, 50)):
                batch = self._agent.buffer.sample(64)
                if batch is None:
                    continue
                if synth_data and len(synth_data["hiddens"]) > 0 and t < 30:
                    n_synth = min(len(synth_data["hiddens"]), 16)
                    idx = self.rng.integers(0, len(synth_data["hiddens"]), size=n_synth)
                    for k in ["states", "hiddens", "actions", "rewards",
                              "next_states", "next_hiddens", "dones", "weights"]:
                        batch[k] = np.concatenate([batch[k], synth_data[k][idx]])
                self._wm.update_step(batch["hiddens"], batch["actions"],
                                     batch["next_hiddens"])

            post_wm_err = _eval_wm_error(self._wm, self._agent)
            self._restore_wm(saved)
            wm_improvement = pre_wm_err - post_wm_err
            reward = float(np.clip(wm_improvement, -1.0, 1.0))

            return reward, {
                "wm_improvement": float(wm_improvement),
                "pre_wm_err": float(pre_wm_err),
                "post_wm_err": float(post_wm_err),
                "n_synth": len(synth_data["hiddens"]) if synth_data else 0,
                "mix_ratio": float(mix_ratio),
                "edit": {SYNTHETIC_EDIT_SPEC["param_names"][i]: float(edit[i])
                         for i in range(len(edit))},
            }

    inner = _InnerLoop(agent, wm, synth, inner_steps)
    seal_loop = SEALLoop(metric_dim=metric_dim, edit_dim=edit_dim,
                          inner_loop_fn=inner, seed=seed + 2000)

    global_step = 0
    outer_count = 0
    coverage_history = []
    all_rewards = []

    for ep in tqdm(range(episodes), desc="SealSyn"):
        obs = env.reset()
        agent.reset_hidden()
        ep_intrinsic = 0.0

        for t in range(max_steps):
            if global_step > warmup_steps and global_step % outer_every == 0:
                metric_state = _get_metric_state(agent, env, wm, rnd)
                result = seal_loop.outer_step(
                    metric_state, n_edits=n_edits_per_step,
                    seed_offset=outer_count * 1000)
                outer_count += 1
                all_rewards.append(result["best_reward"])
                logger.log_update(global_step, {
                    "type": "seal_synthetic_outer",
                    "outer_iteration": outer_count,
                    "mean_reward": result["mean_reward"],
                    "best_reward": result["best_reward"],
                })

            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, ext_r, done, info = env.step(action)
            intr_r_raw = rnd.intrinsic_reward(next_obs)
            intr_r = rnd.normalize(np.array([intr_r_raw]))[0]
            h_after = agent._h.copy()
            agent.store(obs, action, intr_r, next_obs, done, h_before, h_after)
            ep_intrinsic += intr_r
            obs = next_obs
            global_step += 1

            if global_step > warmup_steps:
                rnd.update_step(np.array([next_obs]))
                stats = agent.update()
                if wm is not None and stats is not None:
                    wml = wm.update_step(h_before, [action], h_after)
                    stats["world_model_loss"] = wml
                if stats is not None:
                    logger.log_update(global_step, stats)

            if done:
                break

        coverage_history.append(env.coverage())
        if (ep + 1) % 20 == 0:
            print(f"ep {ep+1:4d} | coverage {np.mean(coverage_history[-20:]):.3f} "
                  f"| outer_iters {outer_count}")

    logger.close()
    return coverage_history, all_rewards


def _get_metric_state(agent, env, wm, rnd):
    state = [env.coverage(), agent.epsilon(), 0.0, 0.0,
             agent.policy_net.optim.lr, 0.0, 0.0, 0.0]
    return np.array(state, np.float32)


def _eval_wm_error(wm, agent):
    if len(agent.buffer) < 16:
        return 0.0
    batch = agent.buffer.sample(16)
    if batch is None:
        return 0.0
    err = wm.prediction_error(batch["hiddens"], batch["actions"],
                              batch["next_hiddens"])
    return float(np.mean(err))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", type=str, default="logs/seal_synthetic.jsonl")
    parser.add_argument("--outer-every", type=int, default=1000)
    parser.add_argument("--n-edits", type=int, default=5)
    parser.add_argument("--inner-steps", type=int, default=50)
    args = parser.parse_args()

    run(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
        log_path=args.log_path, outer_every=args.outer_every,
        n_edits_per_step=args.n_edits, inner_steps=args.inner_steps)
