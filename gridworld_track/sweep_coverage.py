"""
sweep_coverage.py — multi-seed statistical version of compare_coverage.py.

Single seeds are anecdotes, not evidence (we already saw this: 2/3 seeds
positive, 1/3 a wash). This runs N seeds and reports mean +/- std of the
coverage delta (agent minus random), which is the actual claim Phase 1
can defend: not "the agent beats random" but "the agent beats random by
X +/- Y percentage points across N seeds."

Run: python3 sweep_coverage.py --seeds 10 --episodes 100
"""
import argparse
import numpy as np

from .compare_coverage import run_random_baseline, run_rnd_d1_agent


def sweep(n_seeds=10, episodes=100, max_steps=100, last_n=25, base_seed=1000):
    deltas = []
    agent_finals = []
    random_finals = []

    for i in range(n_seeds):
        seed = base_seed + i
        print(f"[{i+1}/{n_seeds}] seed={seed} ...", flush=True)
        random_cov = np.array(run_random_baseline(episodes, max_steps, seed))
        agent_cov = np.array(run_rnd_d1_agent(episodes, max_steps, seed))

        r_final = random_cov[-last_n:].mean()
        a_final = agent_cov[-last_n:].mean()
        delta = a_final - r_final

        deltas.append(delta)
        agent_finals.append(a_final)
        random_finals.append(r_final)
        print(f"    random={r_final:.4f}  agent={a_final:.4f}  delta={delta:+.4f}")

    deltas = np.array(deltas)
    agent_finals = np.array(agent_finals)
    random_finals = np.array(random_finals)

    print()
    print("=" * 60)
    print(f"SWEEP RESULT over {n_seeds} seeds (last {last_n} episodes each)")
    print("=" * 60)
    print(f"Random baseline coverage: {random_finals.mean():.4f} +/- {random_finals.std():.4f}")
    print(f"RND+D1 agent coverage:    {agent_finals.mean():.4f} +/- {agent_finals.std():.4f}")
    print(f"Delta (agent - random):   {deltas.mean():+.4f} +/- {deltas.std():.4f}")
    print(f"Seeds with positive delta: {(deltas > 0).sum()}/{n_seeds}")
    print()

    # one-sample t-test style check: is the mean delta distinguishable from 0
    # given the spread? (rough, no scipy dependency -- just mean/std/n)
    se = deltas.std(ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else float("nan")
    t_stat = deltas.mean() / se if se > 0 else float("nan")
    print(f"Standard error of delta: {se:.4f}")
    print(f"delta_mean / SE (rough t-stat): {t_stat:.2f}")
    print("(as a rough guide: |t| > ~2 suggests the effect is unlikely to be pure noise")
    print(" at this sample size, but this is not a substitute for a proper test)")

    return deltas, agent_finals, random_finals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--last-n", type=int, default=25)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    sweep(n_seeds=args.seeds, episodes=args.episodes, max_steps=args.max_steps,
          last_n=args.last_n, base_seed=args.base_seed)
