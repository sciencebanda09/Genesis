import argparse
import numpy as np
from tqdm import tqdm

from core.rnd import RNDModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from core.executive_cortex.cortex import ExecutiveCortex
from core.logger import JsonlLogger
from core.seal.loop import SEALLoop
from core.seal.regulation import RegulationInnerLoop
from core.seal.self_edit_policy import SelfEditPolicy
from .gridworld import GridWorld, ACTIONS


def run(episodes=300, max_steps=200, seed=0, log_path="logs/seal_regulation.jsonl",
        warmup_steps=300, outer_every=1000, n_edits_per_step=5,
        inner_steps=100, metric_dim=8, edit_dim=5):
    logger = JsonlLogger(log_path)

    def make_env(s=None): return GridWorld(max_steps=max_steps, seed=s if s is not None else seed)
    def make_agent(s=None): return D1Agent(state_dim=8, action_dim=5, seed=s if s is not None else seed)
    def make_rnd(s=None): return RNDModule(state_dim=8, seed=s if s is not None else seed)
    def make_wm(s=None): return ForwardWorldModel(gru_dim=32, action_dim=5, seed=s if s is not None else seed)
    def make_cortex(s=None): return ExecutiveCortex()

    inner = RegulationInnerLoop(make_env, make_agent, make_rnd, make_wm, make_cortex,
                                 inner_steps=inner_steps, seed=seed)

    seal_loop = SEALLoop(metric_dim=metric_dim, edit_dim=edit_dim,
                          inner_loop_fn=inner, seed=seed + 1000)

    env = make_env()
    rnd = make_rnd()
    agent = make_agent()
    wm = make_wm()
    cortex = make_cortex()

    global_step = 0
    outer_count = 0
    coverage_history = []
    all_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        agent.reset_hidden()
        ep_intrinsic = 0.0

        for t in range(max_steps):
            if global_step > warmup_steps and global_step % outer_every == 0:
                metric_state = SelfEditPolicy.get_metric_state(
                    None, cortex, agent, env, wm, rnd)
                result = seal_loop.outer_step(
                    metric_state, n_edits=n_edits_per_step,
                    seed_offset=outer_count * 1000)
                outer_count += 1
                all_rewards.append(result["best_reward"])
                logger.log_update(global_step, {
                    "type": "seal_outer",
                    "outer_iteration": outer_count,
                    "mean_reward": result["mean_reward"],
                    "best_reward": result["best_reward"],
                    "kept_edits": result["restem"]["kept"],
                    "buffer_size": result["restem"]["buffer_size"],
                })

                best_edit = result["best_edit"]
                curiosity_beta, lr_mult, replay_priority_exp, exploration_eps, memory_mix_ratio = \
                    _unpack_edit(best_edit)
                _apply_edit(agent, curiosity_beta, lr_mult,
                            replay_priority_exp, exploration_eps)

            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, ext_r, done, info = env.step(action)
            intr_r_raw = rnd.intrinsic_reward(next_obs)
            intr_r = rnd.normalize(np.array([intr_r_raw]))[0]
            h_after = agent._h.copy()

            curiosity_w = cortex.curiosity_weights.get("rnd", 1.0)
            rnd_reward = curiosity_w * intr_r
            agent.store(obs, action, rnd_reward, next_obs, done, h_before, h_after)
            ep_intrinsic += rnd_reward

            obs = next_obs
            global_step += 1

            if global_step > warmup_steps:
                rnd.update_step(np.array([next_obs]))
                stats = agent.update()
                if wm is not None and stats is not None:
                    wml = wm.update_step(h_before, [action], h_after)
                    stats["world_model_loss"] = wml

                cortex.observe(
                    global_step=global_step,
                    coverage=env.coverage(),
                    rnd_reward=intr_r,
                    td_error_mean=stats["td_error_mean"] if stats else 0.0,
                    wm_loss=stats["world_model_loss"] if stats else 0.0,
                )

                if stats is not None:
                    logger.log_update(global_step, stats)

            if done:
                break

        coverage_history.append(env.coverage())
        if (ep + 1) % 20 == 0:
            recent_cov = np.mean(coverage_history[-20:])
            print(f"ep {ep+1:4d} | coverage {recent_cov:.3f} "
                  f"| outer_iters {outer_count} | buffer {len(agent.buffer):4d} "
                  f"| best_reward {np.mean(all_rewards[-10:]):.3f}" if all_rewards else "")

    logger.close()
    return coverage_history, all_rewards


def _unpack_edit(edit):
    return (float(edit[0]), float(edit[1]), float(edit[2]),
            float(edit[3]), float(edit[4]))


def _apply_edit(agent, curiosity_beta, lr_mult, replay_priority_exp, exploration_eps):
    agent.policy_net.optim.lr = agent.policy_net.optim.lr * lr_mult


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", type=str, default="logs/seal_regulation.jsonl")
    parser.add_argument("--outer-every", type=int, default=1000)
    parser.add_argument("--n-edits", type=int, default=5)
    parser.add_argument("--inner-steps", type=int, default=100)
    args = parser.parse_args()

    run(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
        log_path=args.log_path, outer_every=args.outer_every,
        n_edits_per_step=args.n_edits, inner_steps=args.inner_steps)
