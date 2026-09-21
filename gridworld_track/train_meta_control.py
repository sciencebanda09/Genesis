"""Train and evaluate the paper-1 learning-process controllers.

Candidate edits are evaluated from cloned snapshots of the current live
learner.  Only the selected edit is applied to the live learner; rejected
candidate training is discarded.  Static and random modes provide matched
controls for the learned meta-controller.
"""

import argparse
import copy
import numpy as np
from tqdm import tqdm

from core.agent import D1Agent
from core.executive_cortex.cortex import ExecutiveCortex
from core.logger import JsonlLogger
from core.rnd import RNDModule
from core.seal.meta_controller import (
    LearnedMetaController,
    scale_meta_edit,
)
from core.seal.regulation import _merge_batches
from core.seal.regulation import REGULATION_EDIT_SPEC
from core.world_model import ForwardWorldModel
from core.replay_buffer import PrioritizedReplay
from .gridworld import GridWorld


def _make_factories(max_steps, seed, local_walls=False):
    state_dim = 12 if local_walls else 8

    def make_env(s):
        return GridWorld(max_steps=max_steps, seed=int(s),
                         local_walls=local_walls)

    def make_agent(s):
        return D1Agent(state_dim=state_dim, action_dim=5, seed=int(s))

    def make_rnd(s):
        return RNDModule(state_dim=state_dim, seed=int(s))

    def make_wm(s):
        return ForwardWorldModel(gru_dim=32, action_dim=5, seed=int(s))

    def make_cortex(_s=None):
        return ExecutiveCortex()

    return make_env, make_agent, make_rnd, make_wm, make_cortex


def _apply_live_edit(agent, physical_edit, base_lr):
    """Apply all live controls without accumulating LR/epsilon wrappers."""
    beta, lr_mult, priority_exp, exploration_eps, memory_mix = physical_edit
    agent.policy_net.optim.lr = float(base_lr) * float(lr_mult)
    agent._meta_curiosity_beta = float(beta)
    agent._meta_replay_priority_exp = float(priority_exp)
    agent._meta_memory_mix_ratio = float(memory_mix)
    agent.exploration_multiplier = float(exploration_eps)


class LiveSnapshotEvaluator:
    """Evaluate an edit from a cloned snapshot of the current live learner.

    Every candidate starts from the same state: environment position, agent
    weights and replay buffers, RND statistics, world model, cortex metrics,
    and RNG state.  The clone is discarded after the rollout, so candidate
    training cannot modify the live learner or another candidate.
    """

    def __init__(self, snapshot, obs, inner_steps, base_lr,
                 max_steps, probe_tasks=1, probe_seed=0,
                 local_walls=False,
                 risk_penalty=0.5, worst_case_weight=0.5):
        self.snapshot = snapshot
        self.obs = np.asarray(obs, np.float32).copy()
        self.inner_steps = int(inner_steps)
        self.base_lr = float(base_lr)
        self.max_steps = int(max_steps)
        self.probe_tasks = max(1, int(probe_tasks))
        self.probe_seed = int(probe_seed)
        self.local_walls = bool(local_walls)
        self.risk_penalty = float(risk_penalty)
        self.worst_case_weight = float(worst_case_weight)
        if self.risk_penalty < 0.0 or self.worst_case_weight < 0.0:
            raise ValueError("risk coefficients must be non-negative")

    def __call__(self, edit, seed_offset=0):
        rewards = []
        infos = []
        physical = scale_meta_edit(np.asarray(edit, np.float32))
        for task_index in range(self.probe_tasks):
            state = copy.deepcopy(self.snapshot)
            if task_index > 0:
                state["env"] = GridWorld(
                    max_steps=self.max_steps,
                    seed=self.probe_seed + int(seed_offset) + task_index,
                    local_walls=self.local_walls)
                task_obs = state["env"].reset()
                state["agent"].reset_hidden()
            else:
                task_obs = self.obs.copy()
            reward, info = self._run_one(state, task_obs, physical)
            rewards.append(reward)
            infos.append(info)
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        worst_reward = float(np.min(rewards))
        robust_score = (mean_reward
                        - self.risk_penalty * std_reward
                        + self.worst_case_weight * worst_reward)
        return float(robust_score), {
            "task_rewards": [float(value) for value in rewards],
            "task_infos": infos,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "worst_reward": worst_reward,
            "robust_score": float(robust_score),
            "edit": {REGULATION_EDIT_SPEC["param_names"][i]: float(physical[i])
                     for i in range(len(physical))},
        }

    def _run_one(self, state, obs, physical):
        env = state["env"]
        rnd = state["rnd"]
        agent = state["agent"]
        wm = state["wm"]
        cortex = state["cortex"]
        prioritized = state["prioritized"]
        obs = self.obs.copy()
        _apply_live_edit(agent, physical, self.base_lr)
        start_coverage = float(env.coverage())
        wm_losses = []
        moved = 0

        for step in range(self.inner_steps):
            if env.done:
                obs = env.reset()
                agent.reset_hidden()
            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, _, done, info = env.step(action)
            intrinsic = rnd.normalize(np.array([
                rnd.intrinsic_reward(next_obs)]))[0]
            h_after = agent._h.copy()
            reward = float(physical[0]) * float(intrinsic)
            agent.store(obs, action, reward, next_obs, done, h_before, h_after)
            prioritized.add(obs, h_before, action, reward,
                            next_obs, h_after, done)
            moved += int(info.get("moved", False))

            rnd.update_step(np.array([next_obs]))
            mix = float(physical[4])
            prioritized.alpha = float(physical[2])
            n_p = int(round(agent.batch_size * mix))
            n_u = agent.batch_size - n_p
            uniform_batch = agent.buffer.sample(n_u) if n_u > 0 else None
            priority_batch = (
                prioritized.sample(n_p)
                if n_p > 0 and len(prioritized) >= n_p else None)
            batch = _merge_batches(uniform_batch, priority_batch)
            if batch is not None:
                stats = agent.update(batch=batch)
                if priority_batch is not None and stats is not None:
                    n_uniform = len(uniform_batch["states"]) \
                        if uniform_batch is not None else 0
                    prioritized.update_priorities(
                        priority_batch["indices"],
                        stats["td_error"][n_uniform:])
            wm_loss = wm.update_step(h_before, [action], h_after)
            wm_losses.append(float(wm_loss))
            cortex.observe(
                global_step=step,
                coverage=env.coverage(),
                rnd_reward=float(intrinsic),
                td_error_mean=float(stats["td_error_mean"])
                if batch is not None else 0.0,
                wm_loss=float(wm_loss),
            )
            obs = next_obs

        coverage_gain = float(env.coverage()) - start_coverage
        reward = coverage_gain + 0.001 * moved - 0.1 * float(np.mean(wm_losses))
        return float(reward), {
            "coverage_gain": coverage_gain,
            "moved": int(moved),
            "wm_loss_mean": float(np.mean(wm_losses)) if wm_losses else 0.0,
            "edit": {REGULATION_EDIT_SPEC["param_names"][i]: float(physical[i])
                     for i in range(len(physical))},
        }


def _snapshot(env, agent, rnd, wm, cortex, prioritized):
    """Create one immutable base snapshot for all candidates."""
    return copy.deepcopy({
        "env": env,
        "agent": agent,
        "rnd": rnd,
        "wm": wm,
        "cortex": cortex,
        "prioritized": prioritized,
    })


def _fixed_edit():
    """Neutral, fixed schedule used by the primary baseline."""
    return np.array([1.0, 1.0, 0.6, 1.0, 0.5], np.float32)


def evaluate_policy(agent, max_steps, seeds, deterministic=True,
                    local_walls=False):
    """Evaluate a frozen policy on held-out layouts without training."""
    values = []
    for eval_seed in seeds:
        env = GridWorld(max_steps=max_steps, seed=int(eval_seed),
                        local_walls=local_walls)
        probe = copy.deepcopy(agent)
        if deterministic:
            probe.exploration_multiplier = 0.0
        obs = env.reset()
        probe.reset_hidden()
        for _ in range(max_steps):
            action = probe.select_action(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break
        values.append(float(env.coverage()))
    return np.asarray(values, np.float32)


def run(episodes=100, max_steps=100, seed=0,
        log_path="logs/meta_control.jsonl", warmup_steps=300,
        outer_every=100, n_candidates=5, inner_steps=50,
        controller_mode="learned", eval_layouts=10, eval_probe_tasks=1,
        risk_penalty=0.5, worst_case_weight=0.5,
        transfer_features=True):
    """Run one controller trial and return train/held-out summaries.

    ``controller_mode`` is one of ``learned``, ``fixed``, or ``random``.
    The latter two are matched controls for the paper rather than debugging
    conveniences.
    """
    if controller_mode not in ("learned", "fixed", "random"):
        raise ValueError("controller_mode must be learned, fixed, or random")
    make_env, make_agent, make_rnd, make_wm, make_cortex = _make_factories(
        max_steps, seed, local_walls=transfer_features)
    env = make_env(seed)
    rnd = make_rnd(seed)
    agent = make_agent(seed)
    wm = make_wm(seed)
    cortex = make_cortex(seed)
    prioritized = PrioritizedReplay(
        agent.buffer.capacity, agent.state_dim, agent.gru_dim, seed=seed + 5000)
    base_lr = agent.policy_net.optim.lr

    controller = (LearnedMetaController(metric_dim=8, edit_dim=5,
                                         seed=seed + 20000)
                  if controller_mode == "learned" else None)
    random_rng = np.random.default_rng(seed + 30000)
    logger = JsonlLogger(log_path)

    episode_coverages = []
    outer_rewards = []
    global_step = 0
    outer_count = 0
    if controller_mode == "learned":
        # Start from the neutral baseline and let the controller earn every
        # deviation through a measured improvement over the incumbent.
        current_raw_edit = np.array([0.4736842, 0.3333333, 0.6,
                                     1.0, 0.5], np.float32)
        current_edit = scale_meta_edit(current_raw_edit)
    elif controller_mode == "fixed":
        current_edit = _fixed_edit()
        current_raw_edit = np.array([0.4736842, 0.3333333, 0.6,
                                     1.0, 0.5], np.float32)
    else:
        current_raw_edit = random_rng.random(5).astype(np.float32)
        current_edit = scale_meta_edit(current_raw_edit)
    _apply_live_edit(agent, current_edit, base_lr)

    for episode in tqdm(range(episodes), desc=f"MetaControl[{controller_mode}]" ):
        obs = env.reset()
        agent.reset_hidden()
        done = False
        while not done:
            if (global_step >= warmup_steps and global_step > 0
                    and global_step % outer_every == 0):
                if controller_mode == "learned":
                    metric_state = controller.policy.get_metric_state(
                        cortex, agent, env, wm, rnd)
                    evaluator = LiveSnapshotEvaluator(
                        _snapshot(env, agent, rnd, wm, cortex, prioritized),
                        obs, inner_steps, base_lr, max_steps,
                        probe_tasks=eval_probe_tasks,
                        probe_seed=seed + 70000,
                        local_walls=transfer_features,
                        risk_penalty=risk_penalty,
                        worst_case_weight=worst_case_weight)
                    result = controller.outer_step(
                        metric_state, evaluator, n_candidates=n_candidates,
                        seed_offset=outer_count * 10000,
                        incumbent=current_raw_edit,
                        min_improvement=0.005)
                    current_raw_edit = result["best_edit"].copy()
                    current_edit = scale_meta_edit(result["best_edit"])
                    outer_record = {
                        "best_reward": result["best_reward"],
                        "mean_reward": result["mean_reward"],
                        "policy_loss": result["update"]["policy_loss"],
                        "best_edit": current_edit.tolist(),
                        "candidate_rewards": result["rewards"].tolist(),
                        "candidate_task_rewards": [
                            info.get("task_rewards", []) for info in result["infos"]
                        ],
                    }
                elif controller_mode == "random":
                    current_raw_edit = random_rng.random(5).astype(np.float32)
                    current_edit = scale_meta_edit(current_raw_edit)
                    outer_record = {
                        "best_reward": 0.0,
                        "mean_reward": 0.0,
                        "policy_loss": 0.0,
                        "best_edit": current_edit.tolist(),
                        "candidate_rewards": [],
                    }
                else:
                    outer_record = {
                        "best_reward": 0.0,
                        "mean_reward": 0.0,
                        "policy_loss": 0.0,
                        "best_edit": current_edit.tolist(),
                        "candidate_rewards": [],
                    }
                _apply_live_edit(agent, current_edit, base_lr)
                outer_count += 1
                outer_rewards.append(outer_record["best_reward"])
                logger.log_update(global_step, {
                    "type": "meta_control_outer",
                    "controller_mode": controller_mode,
                    "outer_iteration": outer_count,
                    **outer_record,
                })

            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, _, done, _ = env.step(action)
            intrinsic = rnd.normalize(np.array([
                rnd.intrinsic_reward(next_obs)]))[0]
            h_after = agent._h.copy()
            beta = getattr(agent, "_meta_curiosity_beta", 1.0)
            reward = beta * intrinsic
            agent.store(obs, action, reward, next_obs, done, h_before, h_after)
            prioritized.add(obs, h_before, action, reward,
                            next_obs, h_after, done)

            stats = None
            if global_step >= warmup_steps:
                rnd.update_step(np.array([next_obs]))
                mix = getattr(agent, "_meta_memory_mix_ratio", 0.5)
                prioritized.alpha = getattr(agent, "_meta_replay_priority_exp", 0.6)
                n_p = int(round(agent.batch_size * mix))
                n_u = agent.batch_size - n_p
                ub = agent.buffer.sample(n_u) if n_u > 0 else None
                pb = prioritized.sample(n_p) if n_p > 0 else None
                batch = _merge_batches(ub, pb)
                if batch is not None:
                    stats = agent.update(batch=batch)
                    if pb is not None and stats is not None:
                        n_uniform = len(ub["states"]) if ub is not None else 0
                        prioritized.update_priorities(
                            pb["indices"], stats["td_error"][n_uniform:])
                wm_loss = wm.update_step(h_before, [action], h_after)
                cortex.observe(
                    global_step=global_step,
                    coverage=env.coverage(),
                    rnd_reward=float(intrinsic),
                    td_error_mean=stats["td_error_mean"] if stats else 0.0,
                    wm_loss=float(wm_loss),
                )
                if stats is not None:
                    logger.log_update(global_step, _scalar_stats(stats))

            obs = next_obs
            global_step += 1

        episode_coverages.append(float(env.coverage()))
        if (episode + 1) % 20 == 0:
            print(f"ep {episode + 1:4d} | coverage {np.mean(episode_coverages[-20:]):.3f} "
                  f"| outer_iters {outer_count}")

    logger.close()
    heldout_seeds = [seed + 100000 + i for i in range(int(eval_layouts))]
    heldout_coverage = evaluate_policy(
        agent, max_steps, heldout_seeds, True, local_walls=transfer_features)
    heldout_exploratory = evaluate_policy(
        agent, max_steps, heldout_seeds, False, local_walls=transfer_features)
    return {
        "coverage": np.asarray(episode_coverages, np.float32),
        "outer_rewards": np.asarray(outer_rewards, np.float32),
        "heldout_coverage": heldout_coverage,
        "heldout_exploratory_coverage": heldout_exploratory,
        "controller_mode": controller_mode,
        "controller": controller,
    }


def _scalar_stats(stats):
    """Keep JSONL update records numeric and compact.

    ``D1Agent.update`` returns per-sample TD errors for priority updates.  The
    vector is useful in memory, but is not a paper metric and should not be
    written into every JSONL record.
    """
    return {
        key: float(value) for key, value in stats.items()
        if np.isscalar(value)
    }


def summarize_trial(result, tail=20):
    """Return paper-facing scalar metrics for one seed."""
    coverage = np.asarray(result["coverage"], np.float32)
    tail_values = coverage[-min(tail, len(coverage)):]
    heldout = np.asarray(result.get("heldout_coverage", []), np.float32)
    heldout_exploratory = np.asarray(
        result.get("heldout_exploratory_coverage", []), np.float32)
    return {
        "episodes": int(len(coverage)),
        "controller_mode": result.get("controller_mode", "learned"),
        "final_coverage_mean": float(np.mean(tail_values)) if len(tail_values) else 0.0,
        "coverage_auc": float(np.mean(coverage)) if len(coverage) else 0.0,
        "heldout_coverage_mean": float(np.mean(heldout)) if len(heldout) else 0.0,
        "heldout_coverage_std": float(np.std(heldout)) if len(heldout) else 0.0,
        "heldout_exploratory_mean": (
            float(np.mean(heldout_exploratory))
            if len(heldout_exploratory) else 0.0
        ),
        "outer_steps": int(len(result["outer_rewards"])),
        "outer_reward_mean": (
            float(np.mean(result["outer_rewards"]))
            if len(result["outer_rewards"]) else 0.0
        ),
    }


def run_sweep(seeds, **kwargs):
    """Run independent seeds and return raw results plus aggregate metrics."""
    seeds = [int(seed) for seed in seeds]
    results = []
    summaries = []
    base_log_path = kwargs.pop("log_path", "logs/meta_control.jsonl")
    for seed in seeds:
        if len(seeds) == 1:
            log_path = base_log_path
        else:
            if base_log_path.endswith(".jsonl"):
                log_path = base_log_path[:-6] + f"_seed{seed}.jsonl"
            else:
                log_path = f"{base_log_path}_seed{seed}.jsonl"
        result = run(seed=seed, log_path=log_path, **kwargs)
        results.append(result)
        summaries.append({"seed": seed, **summarize_trial(result)})

    final_values = [item["final_coverage_mean"] for item in summaries]
    aggregate = {
        "n_seeds": len(seeds),
        "controller_mode": summaries[0]["controller_mode"] if summaries else None,
        "final_coverage_mean": float(np.mean(final_values)) if final_values else 0.0,
        "final_coverage_std": float(np.std(final_values)) if final_values else 0.0,
        "heldout_coverage_mean": float(np.mean(
            [item["heldout_coverage_mean"] for item in summaries]))
        if summaries else 0.0,
        "heldout_exploratory_mean": float(np.mean(
            [item["heldout_exploratory_mean"] for item in summaries]))
        if summaries else 0.0,
        "trials": summaries,
    }
    return {"results": results, "aggregate": aggregate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", type=str, default="logs/meta_control.jsonl")
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--outer-every", type=int, default=100)
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--controller", choices=("learned", "fixed", "random"),
                        default="learned")
    parser.add_argument("--eval-layouts", type=int, default=10)
    parser.add_argument("--eval-probe-tasks", type=int, default=1,
                        help="task layouts per candidate evaluation")
    parser.add_argument("--risk-penalty", type=float, default=0.5,
                        help="penalty on cross-layout candidate variance")
    parser.add_argument("--worst-case-weight", type=float, default=0.5,
                        help="weight on the worst probe-layout reward")
    parser.add_argument("--transfer-features", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="include local wall features in the observation")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seeds; overrides --seed")
    args = parser.parse_args()
    common = dict(
        episodes=args.episodes, max_steps=args.max_steps,
        warmup_steps=args.warmup_steps, outer_every=args.outer_every,
        n_candidates=args.n_candidates, inner_steps=args.inner_steps,
        controller_mode=args.controller, eval_layouts=args.eval_layouts,
        eval_probe_tasks=args.eval_probe_tasks,
        risk_penalty=args.risk_penalty,
        worst_case_weight=args.worst_case_weight,
        transfer_features=args.transfer_features,
    )
    if args.seeds:
        sweep = run_sweep([int(value) for value in args.seeds.split(",")],
                          log_path=args.log_path, **common)
        print("sweep summary:", sweep["aggregate"])
    else:
        result = run(seed=args.seed, log_path=args.log_path, **common)
        print("trial summary:", summarize_trial(result))
