"""
verify_recon_context.py — does a reconstruction-shaped h retain more
context info than D1-only h did?

Trains a FRESH agent (own GRU, separate from any earlier-collected data)
under TWO conditions:
  1. D1 only (baseline, matches what we already measured: NMI~0.002-0.014)
  2. D1 + GRUReconstructionTrainer running alongside, additively

Then clusters h from each condition and checks NMI against the same
independent local_context_label() signal used throughout. This isolates
whether the reconstruction pressure -- which does NOT use the context
label anywhere -- recovers context information as a side effect of being
asked to not throw away observation content in general.
"""
import numpy as np

from core.rnd import RNDModule
from core.agent import D1Agent
from core.recon_auxiliary import GRUReconstructionTrainer
from core.clustering import OnlineKMeans
from gridworld_track.gridworld import GridWorld, WALL, OBJ_A, OBJ_B


def local_context_label(env):
    y, x = env.pos
    neighbors = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < env.height and 0 <= nx < env.width:
            neighbors.append(env.grid[ny, nx])
    if WALL in neighbors:
        return 0
    if OBJ_A in neighbors:
        return 1
    if OBJ_B in neighbors:
        return 2
    return 3


def normalized_mutual_info(labels_a, labels_b):
    labels_a = np.asarray(labels_a); labels_b = np.asarray(labels_b)
    n = len(labels_a)
    a_vals = np.unique(labels_a); b_vals = np.unique(labels_b)
    contingency = np.zeros((len(a_vals), len(b_vals)))
    for i, av in enumerate(a_vals):
        for j, bv in enumerate(b_vals):
            contingency[i, j] = np.sum((labels_a == av) & (labels_b == bv))
    p_ab = contingency / n
    p_a = p_ab.sum(axis=1, keepdims=True)
    p_b = p_ab.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi_terms = p_ab * np.log((p_ab + 1e-12) / (p_a @ p_b + 1e-12) + 1e-12)
    mi = np.nansum(np.where(p_ab > 0, mi_terms, 0.0))
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p + 1e-12))
    h_a = entropy(p_a.flatten()); h_b = entropy(p_b.flatten())
    denom = np.sqrt(h_a * h_b)
    return float(mi / denom) if denom > 0 else 0.0


def run_condition(use_recon, n_steps=4000, seed=0):
    env = GridWorld(max_steps=100, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)
    recon = GRUReconstructionTrainer(agent.policy_net.gru, obs_dim=env.state_dim,
                                      hidden_dim=32, lr=1e-3, seed=seed) if use_recon else None

    H, CTX = [], []
    step = 0
    obs = env.reset()
    agent.reset_hidden()
    recon_losses = []
    while step < n_steps:
        h_before = agent._h.copy()
        ctx = local_context_label(env)
        action = agent.select_action(obs)
        next_obs, ext_r, done, info = env.step(action)
        h_after = agent._h.copy()

        H.append(h_before[0]); CTX.append(ctx)

        intr_r_raw = rnd.intrinsic_reward(next_obs)
        intr_r = rnd.normalize(np.array([intr_r_raw]))[0]
        agent.store(obs, action, intr_r, next_obs, done, h_before, h_after)

        if recon is not None:
            rl = recon.update_step(obs[None], h_before)
            recon_losses.append(rl)

        obs = next_obs
        step += 1
        if step > 300:
            rnd.update_step(np.array([next_obs]))
            agent.update()
        if done:
            obs = env.reset()
            agent.reset_hidden()

    return np.array(H, np.float32), np.array(CTX, np.int32), recon_losses


def cluster_and_nmi(H, CTX, seed=0, n_clusters=5):
    Hn = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(seed)
    km = OnlineKMeans(n_clusters=n_clusters, embed_dim=H.shape[1], seed=seed)
    for i in range(500):
        idx = rng.integers(0, len(H), 64)
        km.update_step(Hn[idx])
    assignments = km.assign(Hn)
    return normalized_mutual_info(assignments, CTX)


def main(n_steps=4000, seed=0):
    print("Condition 1: D1 only (baseline)...")
    H_base, CTX_base, _ = run_condition(use_recon=False, n_steps=n_steps, seed=seed)
    nmi_base = cluster_and_nmi(H_base, CTX_base, seed=seed)

    print("Condition 2: D1 + GRUReconstructionTrainer (additive)...")
    H_recon, CTX_recon, recon_losses = run_condition(use_recon=True, n_steps=n_steps, seed=seed)
    nmi_recon = cluster_and_nmi(H_recon, CTX_recon, seed=seed)

    recon_losses = np.array(recon_losses)
    print(f"\nReconstruction loss: first 20 avg = {recon_losses[:20].mean():.4f} | "
          f"last 20 avg = {recon_losses[-20:].mean():.4f}")

    print()
    print("=" * 60)
    print("VERIFICATION: does reconstruction pressure recover context info?")
    print("=" * 60)
    print(f"NMI(clusters-on-h, context) -- D1 only:              {nmi_base:.4f}")
    print(f"NMI(clusters-on-h, context) -- D1 + reconstruction:  {nmi_recon:.4f}")
    print(f"Raw-observation ceiling (measured earlier):          0.0925")
    print()

    if nmi_recon > nmi_base + 0.02 and nmi_recon > 0.05:
        print("RESULT: reconstruction pressure recovers meaningful context info in h,")
        print("        without ever being told what 'context' means. This is real")
        print("        evidence that h's information loss was avoidable, not intrinsic")
        print("        to what D1's task requires.")
    elif nmi_recon > nmi_base + 0.01:
        print("RESULT: small improvement, but not strong. Reconstruction helps somewhat")
        print("        but doesn't fully recover the raw-observation ceiling.")
    else:
        print("RESULT: no meaningful improvement. Reconstruction pressure alone does")
        print("        not recover context info -- something else about the GRU's")
        print("        capacity or the joint training dynamics is the limiting factor.")

    return nmi_base, nmi_recon


if __name__ == "__main__":
    main()
