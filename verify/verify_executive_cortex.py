"""
verify_executive_cortex.py — Executive Cortex verification experiments.

Tests whether the Executive Cortex's adaptive regulation outperforms
static baselines in at least one of: curiosity, memory, exploration.

Experiments
───────────
  1. Curiosity — adaptive RND+ICM mixing vs static RND
  2. Memory — adaptive uniform+prioritized mixing vs static uniform
  3. Exploration — adaptive epsilon vs fixed schedule

Each experiment reports: coverage, TD error, world model loss, intrinsic return.
The Executive Cortex must beat at least one static baseline to pass.

Usage:
    python -m verify.verify_executive_cortex [--seeds 3] [--episodes 50]
"""
import argparse
import numpy as np
import time

from core.rnd import RNDModule, ICMModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from core.replay_buffer import ReplayBuffer, PrioritizedReplay
from core.executive_cortex import ExecutiveCortex
from gridworld_track.gridworld import GridWorld, ACTIONS

SEP = "-" * 72


# ── Shared Training Loop ──────────────────────────────────────────────────

def create_agent_and_modules(env, use_icm=False, use_cortex=False, seed=0):
    """Create D1 agent, curiosity modules, world model, and optional cortex."""
    rng = np.random.default_rng(seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim,
                    eps_start=1.0, eps_end=0.05, eps_decay=3000, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    icm = ICMModule(state_dim=env.state_dim, action_dim=env.action_dim,
                    seed=seed + 100) if use_icm else None
    wm = ForwardWorldModel(gru_dim=agent.gru_dim, action_dim=env.action_dim,
                            seed=seed + 200)
    cortex = ExecutiveCortex() if use_cortex else None
    return agent, rnd, icm, wm, cortex


def run_experiment(env, agent, rnd, icm, wm, cortex,
                   episodes=50, max_steps=200, warmup=100,
                   use_adaptive_curiosity=False,
                   use_adaptive_memory=False,
                   use_adaptive_exploration=False,
                   label="experiment"):
    """Run an experiment with given configuration.

    Parameters
    ----------
    use_adaptive_curiosity : bool
        If True, intrinsic reward = w_rnd * RND + w_icm * ICM, weighted by cortex.
        If False, use static RND only.
    use_adaptive_memory : bool
        If True, sample from both uniform and prioritized replay with cortex weights.
        If False, use static uniform replay.
    use_adaptive_exploration : bool
        If True, override agent epsilon with cortex-regulated value.
        If False, use agent's built-in epsilon schedule.
    """
    # Setup adaptive memory buffers
    prio_buffer = None
    if use_adaptive_memory:
        prio_buffer = PrioritizedReplay(
            capacity=agent.buffer.capacity,
            state_dim=agent.state_dim,
            gru_dim=agent.gru_dim,
            seed=abs(hash(label)) % (2**31),
        )

    metrics = {
        'coverage': [], 'td_error': [], 'wm_loss': [], 'intrinsic_return': [],
        'episode_steps': [], 'curiosity_w_rnd': [], 'curiosity_w_icm': [],
        'memory_w_uniform': [], 'memory_w_prioritized': [],
        'epsilon': [], 'rnd_reward_mean': [], 'icm_reward_mean': [],
    }

    global_step = 0
    last_cortex_params = {}

    for ep in range(episodes):
        obs = env.reset()
        agent.reset_hidden()
        ep_return = 0.0
        ep_rnd_rewards = []
        ep_icm_rewards = []

        for t in range(max_steps):
            # ── Observe metrics before action ──
            if cortex is not None:
                cortex.observe(
                    global_step=global_step,
                    episode=ep,
                    coverage=env.coverage(),
                )

            # ── Exploration: adaptive or static ──
            if use_adaptive_exploration and cortex is not None:
                eps = cortex.epsilon
            else:
                eps = agent.epsilon()
            agent._eps_override = eps  # we'll patch select_action below

            h_before = agent._h.copy()

            # ── Select action (with adaptive epsilon) ──
            if use_adaptive_exploration and cortex is not None:
                if agent.rng.random() < cortex.epsilon:
                    action = int(agent.rng.integers(agent.action_dim))
                    _, agent._h = agent.policy_net.forward(obs, agent._h)
                else:
                    q, h_new = agent.policy_net.forward(obs, agent._h)
                    agent._h = h_new
                    action = int(np.argmax(q[0]))
            else:
                action = agent.select_action(obs)

            next_obs, extrinsic_r, done, info = env.step(action)

            # ── Curiosity: adaptive or static ──
            rnd_reward_raw = rnd.intrinsic_reward(next_obs)
            rnd_reward = rnd.normalize(np.array([rnd_reward_raw]))[0]
            ep_rnd_rewards.append(rnd_reward)

            if use_adaptive_curiosity and cortex is not None and icm is not None:
                # Pad actions to ensure ICM onehot has action_dim columns
                # (ICM's onehot size = actions.max()+1, which fails with single action 0)
                act_batch = np.array([action, env.action_dim - 1], np.int32)
                obs_batch = np.array([obs, obs])
                next_obs_batch = np.array([next_obs, next_obs])
                icm_raw_batch = icm.intrinsic_reward(obs_batch, act_batch, next_obs_batch)
                icm_reward = icm.normalize(np.array([icm_raw_batch[0]]))[0]
                ep_icm_rewards.append(icm_reward)

                cw = cortex.curiosity_weights
                intrinsic_r = cw.get('rnd', 1.0) * rnd_reward + cw.get('icm', 0.0) * icm_reward
            else:
                intrinsic_r = rnd_reward

            h_after = agent._h.copy()
            agent.store(obs, action, intrinsic_r, next_obs, done, h_before, h_after)

            if use_adaptive_memory and prio_buffer is not None:
                prio_buffer.add(obs, h_before, action, intrinsic_r,
                                next_obs, h_after, done)

            ep_return += intrinsic_r
            obs = next_obs
            global_step += 1

            # ── Updates ──
            if global_step > warmup:
                rnd.update_step(np.array([next_obs]))
                if icm is not None and use_adaptive_curiosity:
                    icm.update_step(
                        np.array([obs, obs]),
                        np.array([action, env.action_dim - 1]),
                        np.array([next_obs, next_obs]),
                    )

                # Sample batch for agent update
                if use_adaptive_memory and cortex is not None and prio_buffer is not None:
                    mw = cortex.memory_weights
                    batch_size = agent.batch_size
                    n_uniform = max(1, int(batch_size * mw.get('uniform', 1.0)))
                    n_prio = max(1, int(batch_size * mw.get('prioritized', 0.0)))
                    n_total = n_uniform + n_prio

                    # Override batch_size temporarily
                    old_bs = agent.batch_size

                    # Uniform sample
                    agent.batch_size = n_uniform
                    uniform_batch = agent.buffer.sample(n_uniform)

                    # Prioritized sample
                    prio_batch = prio_buffer.sample(n_prio) if len(prio_buffer) >= n_prio else None

                    agent.batch_size = old_bs

                    if uniform_batch is not None or prio_batch is not None:
                        # Merge batches
                        combined = {}
                        if uniform_batch is not None and prio_batch is not None:
                            for k in uniform_batch:
                                if isinstance(uniform_batch[k], np.ndarray):
                                    combined[k] = np.concatenate(
                                        [uniform_batch[k], prio_batch[k]], axis=0)
                            combined['indices'] = (
                                list(uniform_batch.get('indices', []))
                                + list(prio_batch.get('indices', []))
                            )
                            combined['weights'] = np.concatenate(
                                [uniform_batch['weights'], prio_batch['weights']])
                        elif uniform_batch is not None:
                            combined = uniform_batch
                        else:
                            combined = prio_batch

                        # Use combined batch for agent update
                        stats = agent.update(batch=combined)
                    else:
                        stats = None
                else:
                    stats = agent.update()

                # World model update
                wm_loss = wm.update_step(h_before, [action], h_after)

                # Collect metrics
                if stats is not None:
                    td_err = stats.get('td_error_mean', 0.0)
                    metrics['td_error'].append(td_err)

                    if use_adaptive_memory and prio_buffer is not None:
                        per_sample = stats.get('td_error', None)
                        if per_sample is not None:
                            n_u = len(uniform_batch['states']) if uniform_batch is not None else 0
                            n_p = len(prio_batch['states']) if prio_batch is not None else 0
                            prio_errs = per_sample[n_u:n_u + n_p]
                            if 'indices' in combined and len(prio_errs) > 0:
                                prio_buffer.update_priorities(combined['indices'], prio_errs)
                            # Track per-buffer TD error for cortex regulation
                            if n_u > 0:
                                cortex.observe(uniform_td_error=float(np.mean(np.abs(per_sample[:n_u]))))
                            if n_p > 0:
                                cortex.observe(prioritized_td_error=float(np.mean(np.abs(prio_errs))))

                metrics['wm_loss'].append(wm_loss)

                # Cortex observations
                if cortex is not None:
                    cortex.observe(
                        td_error_mean=stats.get('td_error_mean', 0.0) if stats else 0.0,
                        wm_loss=wm_loss,
                        rnd_reward=rnd_reward,
                        rnd_loss=stats.get('delay_loss', 0.0) if stats else 0.0,
                        coverage=env.coverage(),
                        intrinsic_return=ep_return / (t + 1),
                    )
                    if icm is not None:
                        cortex.observe(icm_reward=icm_reward if use_adaptive_curiosity else 0.0)

                    params = cortex.regulate()
                    if params:
                        last_cortex_params.update(params)

            if done:
                break

        # ── Episode-end metrics ──
        metrics['coverage'].append(env.coverage())
        metrics['intrinsic_return'].append(ep_return)
        metrics['episode_steps'].append(t + 1)
        metrics['epsilon'].append(eps)
        metrics['rnd_reward_mean'].append(np.mean(ep_rnd_rewards) if ep_rnd_rewards else 0.0)
        metrics['icm_reward_mean'].append(np.mean(ep_icm_rewards) if ep_icm_rewards else 0.0)

        if cortex is not None:
            cw = cortex.curiosity_weights
            metrics['curiosity_w_rnd'].append(cw.get('rnd', 1.0))
            metrics['curiosity_w_icm'].append(cw.get('icm', 0.0))
            mw = cortex.memory_weights
            metrics['memory_w_uniform'].append(mw.get('uniform', 1.0))
            metrics['memory_w_prioritized'].append(mw.get('prioritized', 0.0))

    return {k: np.array(v) for k, v in metrics.items()}


# ── Experiment 1: Adaptive Curiosity ──────────────────────────────────────

def experiment_adaptive_curiosity(env, episodes=50, seed=0):
    """Compare adaptive RND+ICM vs static RND."""
    print(f"\n{SEP}")
    print("EXPERIMENT 1: Adaptive Curiosity")
    print(f"  Comparing: static RND vs adaptive (RND + ICM weighted)")
    print(f"{SEP}")

    # Static RND baseline
    agent_s, rnd_s, _, wm_s, _ = create_agent_and_modules(env, use_icm=False, use_cortex=False, seed=seed)
    t0 = time.time()
    base = run_experiment(env, agent_s, rnd_s, None, wm_s, None,
                          episodes=episodes, label="static_rnd")
    t_base = time.time() - t0
    print(f"  static RND done in {t_base:.1f}s | final coverage: {base['coverage'][-5:].mean():.3f}")

    # Adaptive curiosity
    agent_a, rnd_a, icm_a, wm_a, cortex_a = create_agent_and_modules(
        env, use_icm=True, use_cortex=True, seed=seed + 100)
    t0 = time.time()
    adaptive = run_experiment(env, agent_a, rnd_a, icm_a, wm_a, cortex_a,
                              episodes=episodes,
                              use_adaptive_curiosity=True,
                              label="adaptive_curiosity")
    t_adaptive = time.time() - t0
    print(f"  adaptive  done in {t_adaptive:.1f}s | final coverage: {adaptive['coverage'][-5:].mean():.3f}")

    return {
        'baseline': ('Static RND', base),
        'adaptive': ('Adaptive (RND+ICM)', adaptive),
        'baseline_label': 'Static RND',
        'adaptive_label': 'Adaptive (RND+ICM)',
    }


# ── Experiment 2: Adaptive Memory ─────────────────────────────────────────

def experiment_adaptive_memory(env, episodes=50, seed=0):
    """Compare adaptive uniform+prioritized vs static uniform."""
    print(f"\n{SEP}")
    print("EXPERIMENT 2: Adaptive Memory")
    print(f"  Comparing: static uniform replay vs adaptive (uniform + prioritized)")
    print(f"{SEP}")

    # Static uniform baseline
    agent_s, rnd_s, _, wm_s, _ = create_agent_and_modules(env, use_icm=False, use_cortex=False, seed=seed)
    t0 = time.time()
    base = run_experiment(env, agent_s, rnd_s, None, wm_s, None,
                          episodes=episodes, label="static_uniform")
    t_base = time.time() - t0
    print(f"  static uniform done in {t_base:.1f}s | final td_err: {base['td_error'][-50:].mean():.4f}")

    # Adaptive memory
    agent_a, rnd_a, _, wm_a, cortex_a = create_agent_and_modules(
        env, use_icm=False, use_cortex=True, seed=seed + 100)
    t0 = time.time()
    adaptive = run_experiment(env, agent_a, rnd_a, None, wm_a, cortex_a,
                              episodes=episodes,
                              use_adaptive_memory=True,
                              label="adaptive_memory")
    t_adaptive = time.time() - t0
    print(f"  adaptive  done in {t_adaptive:.1f}s | final td_err: {adaptive['td_error'][-50:].mean():.4f}")

    return {
        'baseline': ('Static Uniform', base),
        'adaptive': ('Adaptive (Uniform+Prio)', adaptive),
        'baseline_label': 'Static Uniform',
        'adaptive_label': 'Adaptive (Uniform+Prio)',
    }


# ── Experiment 3: Adaptive Exploration ────────────────────────────────────

def experiment_adaptive_exploration(env, episodes=50, seed=0):
    """Compare adaptive epsilon vs fixed schedule."""
    print(f"\n{SEP}")
    print("EXPERIMENT 3: Adaptive Exploration")
    print(f"  Comparing: fixed epsilon schedule vs cortex-regulated epsilon")
    print(f"{SEP}")

    # Fixed epsilon schedule baseline
    agent_s, rnd_s, _, wm_s, _ = create_agent_and_modules(env, use_icm=False, use_cortex=False, seed=seed)
    t0 = time.time()
    base = run_experiment(env, agent_s, rnd_s, None, wm_s, None,
                          episodes=episodes, label="static_eps")
    t_base = time.time() - t0
    print(f"  static eps done in {t_base:.1f}s | final coverage: {base['coverage'][-5:].mean():.3f}")

    # Adaptive exploration
    agent_a, rnd_a, _, wm_a, cortex_a = create_agent_and_modules(
        env, use_icm=False, use_cortex=True, seed=seed + 100)
    t0 = time.time()
    adaptive = run_experiment(env, agent_a, rnd_a, None, wm_a, cortex_a,
                              episodes=episodes,
                              use_adaptive_exploration=True,
                              label="adaptive_eps")
    t_adaptive = time.time() - t0
    print(f"  adaptive  done in {t_adaptive:.1f}s | final coverage: {adaptive['coverage'][-5:].mean():.3f}")

    return {
        'baseline': ('Fixed Epsilon', base),
        'adaptive': ('Adaptive Epsilon', adaptive),
        'baseline_label': 'Fixed Epsilon',
        'adaptive_label': 'Adaptive Epsilon',
    }


# ── Results Analysis ──────────────────────────────────────────────────────

def compare_metrics(baseline, adaptive, key, higher_better=True):
    """Compare two experiments on a metric. Returns improvement statistics."""
    b = baseline[key]
    a = adaptive[key]
    if len(b) == 0 or len(a) == 0:
        return {'baseline_mean': 0, 'adaptive_mean': 0, 'improvement': 0, 'better': False}

    b_mean = float(np.mean(b[-50:])) if len(b) > 50 else float(np.mean(b))
    a_mean = float(np.mean(a[-50:])) if len(a) > 50 else float(np.mean(a))

    if abs(b_mean) < 1e-8:
        improvement = float('inf') if a_mean > 1e-8 else 0.0
    else:
        improvement = (a_mean - b_mean) / abs(b_mean) * 100

    better = (a_mean > b_mean) if higher_better else (a_mean < b_mean)
    return {
        'baseline_mean': b_mean,
        'adaptive_mean': a_mean,
        'improvement_pct': improvement,
        'better': better,
    }


def print_comparison(exp_result, metrics_config):
    """Print comparison table for an experiment."""
    baseline_label = exp_result['baseline_label']
    adaptive_label = exp_result['adaptive_label']
    baseline = exp_result['baseline'][1]
    adaptive = exp_result['adaptive'][1]

    print(f"\n  {baseline_label} vs {adaptive_label}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Adaptive':>12} {'Change':>10} {'Winner':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    wins = 0
    total = 0
    for key, higher_better, label in metrics_config:
        comp = compare_metrics(baseline, adaptive, key, higher_better)
        total += 1
        if comp['better']:
            wins += 1
        change_str = f"{comp['improvement_pct']:+.1f}%" if comp['improvement_pct'] != float('inf') else "inf"
        winner = adaptive_label if comp['better'] else baseline_label
        print(f"  {label:<25} {comp['baseline_mean']:>12.4f} {comp['adaptive_mean']:>12.4f} {change_str:>10} {winner:<10}")

    print(f"\n  Result: {wins}/{total} metrics favor {adaptive_label}")
    return wins, total


def print_dynamics(exp_result, label="Dynamics"):
    """Print adaptive dynamics summary."""
    adaptive = exp_result['adaptive'][1]

    print(f"\n  {label}:")
    for key, name in [('curiosity_w_rnd', 'RND weight'),
                       ('curiosity_w_icm', 'ICM weight'),
                       ('memory_w_uniform', 'Uniform weight'),
                       ('memory_w_prioritized', 'Priority weight'),
                       ('epsilon', 'Epsilon')]:
        vals = adaptive.get(key, [])
        if len(vals) > 0:
            print(f"    {name:<20}: start={vals[0]:.3f} end={vals[-1]:.3f} "
                  f"min={vals.min():.3f} max={vals.max():.3f}")


# ── Main ──────────────────────────────────────────────────────────────────

def main(seeds=1, episodes=50, max_steps=200, quiet=False):
    all_results = {}

    for s in range(seeds):
        seed = s * 1000
        env = GridWorld(max_steps=max_steps, seed=seed)
        if not quiet:
            print(f"\n{'='*72}")
            print(f"Seed {seed} | {episodes} episodes, {max_steps} max steps")
            print(f"{'='*72}")

        # Experiment 1: Curiosity
        result_c = experiment_adaptive_curiosity(env, episodes, seed)
        all_results[f'curiosity_seed{s}'] = result_c

        # Experiment 2: Memory
        result_m = experiment_adaptive_memory(env, episodes, seed)
        all_results[f'memory_seed{s}'] = result_m

        # Experiment 3: Exploration
        result_e = experiment_adaptive_exploration(env, episodes, seed)
        all_results[f'exploration_seed{s}'] = result_e

    # ── Summary for first seed ──
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)

    metrics_configs = {
        'curiosity': [
            ('coverage', True, 'Coverage'),
            ('intrinsic_return', True, 'Intrinsic Return'),
            ('wm_loss', False, 'World Model Loss'),
            ('td_error', False, 'TD Error'),
        ],
        'memory': [
            ('td_error', False, 'TD Error'),
            ('coverage', True, 'Coverage'),
            ('wm_loss', False, 'World Model Loss'),
            ('intrinsic_return', True, 'Intrinsic Return'),
        ],
        'exploration': [
            ('coverage', True, 'Coverage'),
            ('intrinsic_return', True, 'Intrinsic Return'),
            ('wm_loss', False, 'World Model Loss'),
            ('td_error', False, 'TD Error'),
        ],
    }

    total_wins = 0
    total_comparisons = 0
    passed_experiments = []

    for exp_key, metrics_cfg in metrics_configs.items():
        result_key = f'{exp_key}_seed0'
        if result_key not in all_results:
            continue
        result = all_results[result_key]
        print(f"\n[{exp_key.upper()}]")
        wins, tot = print_comparison(result, metrics_cfg)
        print_dynamics(result, f"{exp_key.capitalize()} dynamics")
        total_wins += wins
        total_comparisons += tot
        if wins > tot / 2:
            passed_experiments.append(exp_key)

    print(f"\n{SEP}")
    print(f"PASSED experiments: {passed_experiments}")
    print(f"Overall: {total_wins}/{total_comparisons} metrics favor Executive Cortex")

    if len(passed_experiments) >= 1:
        print("\n[PASS] VERDICT: Executive Cortex outperforms at least one static baseline.")
        print("  Recommendation: proceed to integration as default architecture.")
    else:
        print("\n[FAIL] VERDICT: Executive Cortex did not outperform any static baseline.")
        print("  Recommendation: investigate regulation hyperparameters and retry.")

    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Executive Cortex")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per experiment")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    main(seeds=args.seeds, episodes=args.episodes,
         max_steps=args.max_steps, quiet=args.quiet)
