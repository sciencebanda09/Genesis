"""
verify_contrastive.py — isolated test of contrastive.py BEFORE wiring it
into the main training loop.

The bar: after training, embeddings for states sharing an action-context
should have measurably higher average cosine similarity than embeddings
from different action-contexts, AND this gap should be larger than at
random initialization. If the gap doesn't grow, the projector isn't
learning anything -- it would be indistinguishable from a random
projection that happens to produce a plausible-looking loss curve.
"""
import numpy as np

from core.rnd import RNDModule
from core.agent import D1Agent
from core.contrastive import ContrastiveProjector
from gridworld_track.gridworld import GridWorld


def collect_transitions(n_steps=3000, seed=0):
    env = GridWorld(max_steps=100, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)

    H, A = [], []
    step = 0
    obs = env.reset()
    agent.reset_hidden()
    while step < n_steps:
        h_before = agent._h.copy()
        action = agent.select_action(obs)
        next_obs, ext_r, done, info = env.step(action)
        h_after = agent._h.copy()

        H.append(h_before[0])
        A.append(action)

        intr_r_raw = rnd.intrinsic_reward(next_obs)
        intr_r = rnd.normalize(np.array([intr_r_raw]))[0]
        agent.store(obs, action, intr_r, next_obs, done, h_before, h_after)

        obs = next_obs
        step += 1
        if step > 300:
            rnd.update_step(np.array([next_obs]))
            agent.update()
        if done:
            obs = env.reset()
            agent.reset_hidden()

    return np.array(H, np.float32), np.array(A, np.int32)


def same_vs_diff_similarity(projector, H, A, rng, n_eval=500):
    """Average cosine similarity for same-action pairs vs different-action pairs."""
    idx = rng.integers(0, len(H), size=n_eval)
    z = projector.embed(H[idx])
    a = A[idx]

    sim = z @ z.T
    same = (a[:, None] == a[None, :])
    np.fill_diagonal(same, False)
    diff = ~same
    np.fill_diagonal(diff, False)

    same_sim = sim[same].mean() if same.any() else float("nan")
    diff_sim = sim[diff].mean() if diff.any() else float("nan")
    return same_sim, diff_sim


def main(n_steps=3000, train_updates=800, batch_size=64, seed=0):
    print(f"Collecting {n_steps} transitions from an RND+D1 agent...")
    H, A = collect_transitions(n_steps=n_steps, seed=seed)
    print(f"Collected. H shape: {H.shape}, actions used: {np.unique(A)}")

    gru_dim = H.shape[1]
    action_dim = int(A.max()) + 1
    proj = ContrastiveProjector(gru_dim=gru_dim, seed=seed)
    rng = np.random.default_rng(seed)

    same_before, diff_before = same_vs_diff_similarity(proj, H, A, rng)
    print(f"\nBEFORE training: same-action sim = {same_before:.4f}, "
          f"diff-action sim = {diff_before:.4f}, gap = {same_before - diff_before:+.4f}")

    print(f"\nTraining contrastive projector for {train_updates} steps...")
    n = len(H)
    losses = []
    skipped = 0
    for i in range(train_updates):
        idx = rng.integers(0, n, size=batch_size)
        a_batch = A[idx]
        positive_mask = a_batch[:, None] == a_batch[None, :]  # same-action pairing
        loss = proj.update_step(H[idx], positive_mask)
        if loss is None:
            skipped += 1
            continue
        losses.append(loss)
        if i % 100 == 0:
            print(f"  step {i:4d}  loss {loss:.5f}")

    if skipped:
        print(f"  ({skipped} batches skipped -- no positive pairs, e.g. all-unique actions)")

    same_after, diff_after = same_vs_diff_similarity(proj, H, A, rng)
    print(f"\nAFTER training:  same-action sim = {same_after:.4f}, "
          f"diff-action sim = {diff_after:.4f}, gap = {same_after - diff_after:+.4f}")

    gap_before = same_before - diff_before
    gap_after = same_after - diff_after

    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Gap before training: {gap_before:+.4f}")
    print(f"Gap after training:  {gap_after:+.4f}")
    print(f"Loss trend: first 10 avg = {np.mean(losses[:10]):.4f} | "
          f"last 10 avg = {np.mean(losses[-10:]):.4f}")
    print()

    if gap_after > gap_before + 0.02:
        print("RESULT: contrastive learning IS separating states by action-context")
        print("        (same-action states end up measurably closer in embedding")
        print("        space than different-action states, more so than at init).")
    else:
        print("RESULT: gap did not grow meaningfully -- projector is not clearly")
        print("        learning useful structure yet. Do not build clustering on")
        print("        top of this until it does.")

    return gap_before, gap_after, losses


if __name__ == "__main__":
    main()
