"""
compare_coverage.py — Phase 1's concrete quantitative claim.

Question: does RND-driven curiosity + D1 value learning actually explore
MORE of the environment than a uniform-random policy, given the same
number of steps? If not, the curiosity mechanism isn't doing anything
useful yet, regardless of how clean the training curves look.

Metric: fraction of non-wall cells visited by end of episode, averaged
over the last N episodes (after learning has had time to matter) and
compared against a random-action baseline on the identical grid/seed.

This is a real head-to-head, not just "RND agent explores lots of cells"
in isolation -- coverage numbers alone are meaningless without the
random-policy reference point.
"""
import numpy as np

from core.rnd import RNDModule
from core.agent import D1Agent
from .gridworld import GridWorld, ACTIONS


def run_random_baseline(episodes, max_steps, seed):
    env = GridWorld(max_steps=max_steps, seed=seed)
    rng = np.random.default_rng(seed + 1000)  # separate stream from env's own rng
    coverages = []
    for ep in range(episodes):
        env.reset()
        done = False
        while not done:
            action = int(rng.integers(env.action_dim))
            _, _, done, _ = env.step(action)
        coverages.append(env.coverage())
    return coverages


def run_rnd_d1_agent(episodes, max_steps, seed, warmup_steps=300):
    env = GridWorld(max_steps=max_steps, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)

    coverages = []
    global_step = 0
    for ep in range(episodes):
        obs = env.reset()
        agent.reset_hidden()
        done = False
        while not done:
            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, ext_r, done, info = env.step(action)

            intr_r_raw = rnd.intrinsic_reward(next_obs)
            intr_r = rnd.normalize(np.array([intr_r_raw]))[0]
            h_after = agent._h.copy()
            agent.store(obs, action, intr_r, next_obs, done, h_before, h_after)

            obs = next_obs
            global_step += 1
            if global_step > warmup_steps:
                rnd.update_step(np.array([next_obs]))
                agent.update()

        coverages.append(env.coverage())
    return coverages


def main(episodes=150, max_steps=100, seed=42, last_n=30):
    print(f"Running random baseline ({episodes} episodes)...")
    random_cov = run_random_baseline(episodes, max_steps, seed)

    print(f"Running RND+D1 agent ({episodes} episodes)...")
    agent_cov = run_rnd_d1_agent(episodes, max_steps, seed)

    random_cov = np.array(random_cov)
    agent_cov = np.array(agent_cov)

    print()
    print("=" * 60)
    print(f"Coverage (fraction of free cells visited per episode)")
    print("=" * 60)
    print(f"{'':20s} {'all episodes':>15s} {'last '+str(last_n):>15s}")
    print(f"{'Random baseline':20s} {random_cov.mean():15.4f} {random_cov[-last_n:].mean():15.4f}")
    print(f"{'RND+D1 agent':20s} {agent_cov.mean():15.4f} {agent_cov[-last_n:].mean():15.4f}")
    print()

    delta_all = agent_cov.mean() - random_cov.mean()
    delta_last = agent_cov[-last_n:].mean() - random_cov[-last_n:].mean()
    print(f"Delta (all episodes):      {delta_all:+.4f}")
    print(f"Delta (last {last_n} episodes):  {delta_last:+.4f}")
    print()
    if delta_last > 0.01:
        print("RESULT: agent explores MORE than random by end of training.")
    elif delta_last < -0.01:
        print("RESULT: agent explores LESS than random -- curiosity mechanism")
        print("        is not adding value over random exploration yet.")
    else:
        print("RESULT: no meaningful difference from random -- inconclusive.")

    return random_cov, agent_cov


if __name__ == "__main__":
    main()
