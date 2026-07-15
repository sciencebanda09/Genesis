"""
verify_contrastive_on_wm_features.py — retry of step 2, using the TRAINED
world model's internal MLP features as contrastive input instead of raw h.

Hypothesis: raw h collapsed to a narrow cone (verified: same/diff cosine
sim ~0.84 for both, no separable structure) because D1 alone only
pressures h to be useful for a scalar value prediction -- nothing pushes
it to spread out. The world model's hidden layer, by contrast, was
trained specifically to predict (h_t, action) -> h_t+1, a much richer
target. Its internal features may retain more of the directional
structure needed for contrastive separation to have something to work
with.

This is a real test, not an assumption -- if world-model features ALSO
collapse, that's further evidence the bottleneck is D1's value-only
training pressure on h itself, not which downstream features we pick.
"""
import numpy as np

from core.rnd import RNDModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from core.contrastive import ContrastiveProjector
from gridworld_track.gridworld import GridWorld


def collect_transitions_with_next(n_steps=3000, seed=0):
    env = GridWorld(max_steps=100, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)

    H, A, H_next = [], [], []
    step = 0
    obs = env.reset()
    agent.reset_hidden()
    while step < n_steps:
        h_before = agent._h.copy()
        action = agent.select_action(obs)
        next_obs, ext_r, done, info = env.step(action)
        h_after = agent._h.copy()

        H.append(h_before[0]); A.append(action); H_next.append(h_after[0])

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

    return np.array(H, np.float32), np.array(A, np.int32), np.array(H_next, np.float32)


def action_onehot(actions, action_dim):
    actions = np.asarray(actions, np.int32)
    onehot = np.zeros((len(actions), action_dim), np.float32)
    onehot[np.arange(len(actions)), actions] = 1.0
    return onehot


def same_vs_diff_similarity(embed_fn, feats, A, rng, n_eval=500):
    idx = rng.integers(0, len(feats), size=n_eval)
    z = embed_fn(feats[idx])
    a = A[idx]
    sim = z @ z.T
    same = (a[:, None] == a[None, :]); np.fill_diagonal(same, False)
    diff = ~same; np.fill_diagonal(diff, False)
    return sim[same].mean() if same.any() else float("nan"), \
           sim[diff].mean() if diff.any() else float("nan")


def main(n_steps=3000, wm_train_steps=800, contrastive_train_steps=800,
         batch_size=64, seed=0):
    print(f"Collecting {n_steps} transitions...")
    H, A, H_next = collect_transitions_with_next(n_steps=n_steps, seed=seed)
    gru_dim = H.shape[1]
    action_dim = int(A.max()) + 1
    rng = np.random.default_rng(seed)
    n = len(H)

    print(f"Training world model for {wm_train_steps} steps...")
    wm = ForwardWorldModel(gru_dim=gru_dim, action_dim=action_dim, seed=seed)
    for i in range(wm_train_steps):
        idx = rng.integers(0, n, size=batch_size)
        wm.update_step(H[idx], A[idx], H_next[idx])

    # extract world model's internal hidden-layer features (pre-final-layer
    # activations) for the full dataset -- this is the richer representation
    # we're testing as contrastive input instead of raw h.
    onehot = action_onehot(A, action_dim)
    x = np.concatenate([H, onehot], axis=-1)
    wm_features = wm.net.hidden_features(x)
    print(f"World-model feature shape: {wm_features.shape} (vs raw h: {H.shape})")

    # sanity check: are these features ALSO collapsed, before any contrastive
    # projection touches them? (raw cosine sim, same test as we ran on h)
    feat_n = wm_features / (np.linalg.norm(wm_features, axis=-1, keepdims=True) + 1e-8)
    sim = feat_n @ feat_n.T
    same = (A[:, None] == A[None, :]); np.fill_diagonal(same, False)
    diff = ~same; np.fill_diagonal(diff, False)
    print(f"\nRaw world-model feature cosine sim -- same_action: {sim[same].mean():.4f}, "
          f"diff_action: {sim[diff].mean():.4f}")
    print(f"(compare to raw h, which was ~0.84 / 0.84 -- fully collapsed, no gap)")

    feat_dim = wm_features.shape[1]
    proj = ContrastiveProjector(gru_dim=feat_dim, seed=seed, lr=1e-2)

    same_before, diff_before = same_vs_diff_similarity(proj.embed, wm_features, A, rng)
    print(f"\nBEFORE contrastive training: same={same_before:.4f} diff={diff_before:.4f} "
          f"gap={same_before - diff_before:+.4f}")

    print(f"\nTraining contrastive projector for {contrastive_train_steps} steps...")
    losses = []
    for i in range(contrastive_train_steps):
        idx = rng.integers(0, n, size=batch_size)
        a_batch = A[idx]
        positive_mask = a_batch[:, None] == a_batch[None, :]  # same-action pairing
        loss = proj.update_step(wm_features[idx], positive_mask)
        if loss is not None:
            losses.append(loss)
        if i % 100 == 0 and loss is not None:
            print(f"  step {i:4d}  loss {loss:.5f}")

    same_after, diff_after = same_vs_diff_similarity(proj.embed, wm_features, A, rng)
    print(f"\nAFTER contrastive training:  same={same_after:.4f} diff={diff_after:.4f} "
          f"gap={same_after - diff_after:+.4f}")

    gap_before = same_before - diff_before
    gap_after = same_after - diff_after

    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Gap before: {gap_before:+.4f}  |  Gap after: {gap_after:+.4f}")
    if losses:
        print(f"Loss: first 10 avg = {np.mean(losses[:10]):.4f} | last 10 avg = {np.mean(losses[-10:]):.4f}")

    if gap_after > gap_before + 0.02:
        print("\nRESULT: world-model features DO support contrastive separation")
        print("        (hypothesis confirmed -- richer training signal in the")
        print("        world model gave contrastive learning real structure to use).")
    else:
        print("\nRESULT: world-model features ALSO fail to separate meaningfully.")
        print("        This rules out 'wrong feature source' as the explanation --")
        print("        the real bottleneck is that nothing in this pipeline yet")
        print("        pressures ANY representation to spread out directionally.")
        print("        Next real fix would be an explicit variance/decorrelation")
        print("        regularizer, not a different feature source.")

    return gap_before, gap_after


if __name__ == "__main__":
    main()
