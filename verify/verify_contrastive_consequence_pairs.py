"""
verify_contrastive_consequence_pairs.py — the redesigned step 2.

Previous design: positive pairs = same action. Verified working
mechanically (gap +1.21 on world-model features) but then FAILED the
deeper test in verify_clustering.py: clusters built on those embeddings
had NMI=0.9988 with action (expected, that's the training signal) and
NMI=0.0016 with independent spatial/object context (i.e. no evidence of
anything beyond an action-label echo).

New design: positive pairs = states whose PREDICTED CONSEQUENCES are
similar, regardless of which action produced them. Two states h_i, h_j
(each paired with its own action a_i, a_j) are a positive pair if the
world model's predicted h_next for each lands close together in
prediction space. This is a genuinely different notion of "alike" --
it's about outcome similarity, not action-label equality, and is closer
to what "concept" should mean (things that lead to similar futures are
the same kind of thing), consistent with the reasoning that originally
motivated building the world model in step 1.

Test: does clustering on THESE embeddings correlate with independent
spatial/object context any better than the old action-based version did
(NMI 0.0016)? That's the real bar -- not just "does contrastive loss go
down," which we already know it will (InfoNCE will separate almost any
non-degenerate positive-pair signal to some extent).
"""
import numpy as np

from core.rnd import RNDModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from core.contrastive import ContrastiveProjector
from core.clustering import OnlineKMeans
from gridworld_track.gridworld import GridWorld


def local_context_label(env):
    """Same independent ground-truth signal used in verify_clustering.py --
    computed from env.grid directly, never seen by any trained component."""
    from gridworld import WALL, OBJ_A, OBJ_B
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


def collect_transitions_with_context(n_steps=4000, seed=0):
    env = GridWorld(max_steps=100, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)

    H, A, H_next, CTX = [], [], [], []
    step = 0
    obs = env.reset()
    agent.reset_hidden()
    while step < n_steps:
        h_before = agent._h.copy()
        action = agent.select_action(obs)
        ctx = local_context_label(env)
        next_obs, ext_r, done, info = env.step(action)
        h_after = agent._h.copy()

        H.append(h_before[0]); A.append(action); H_next.append(h_after[0]); CTX.append(ctx)

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

    return (np.array(H, np.float32), np.array(A, np.int32),
            np.array(H_next, np.float32), np.array(CTX, np.int32))


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


def build_consequence_positive_mask(wm, h_batch, a_batch, k_nearest=5):
    """
    For each anchor i in the batch, predict its consequence h_next_pred[i]
    using the world model. Positive pairs = the k_nearest other samples in
    the batch whose predicted consequence is closest (L2) to anchor i's
    predicted consequence -- regardless of whether they share an action.
    """
    h_next_pred = wm.predict(h_batch, a_batch)  # (B, gru_dim)
    B = len(h_batch)
    dists = np.linalg.norm(h_next_pred[:, None, :] - h_next_pred[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    nearest_idx = np.argsort(dists, axis=-1)[:, :k_nearest]
    mask = np.zeros((B, B), dtype=bool)
    rows = np.repeat(np.arange(B), k_nearest)
    cols = nearest_idx.flatten()
    mask[rows, cols] = True
    return mask


def same_vs_diff_similarity_for_context(embed_fn, feats, CTX, rng, n_eval=500):
    idx = rng.integers(0, len(feats), size=n_eval)
    z = embed_fn(feats[idx])
    c = CTX[idx]
    sim = z @ z.T
    same = (c[:, None] == c[None, :]); np.fill_diagonal(same, False)
    diff = ~same; np.fill_diagonal(diff, False)
    return sim[same].mean() if same.any() else float("nan"), \
           sim[diff].mean() if diff.any() else float("nan")


def main(n_steps=4000, wm_train_steps=1000, contrastive_train_steps=1000,
         batch_size=64, k_nearest=5, seed=0):
    print(f"Collecting {n_steps} transitions with independent context labels...")
    H, A, H_next, CTX = collect_transitions_with_context(n_steps=n_steps, seed=seed)
    print(f"Context label distribution (0=wall,1=objA,2=objB,3=none-adjacent): "
          f"{np.bincount(CTX, minlength=4)}")

    gru_dim = H.shape[1]
    action_dim = int(A.max()) + 1
    rng = np.random.default_rng(seed)
    n = len(H)

    print(f"\nTraining world model for {wm_train_steps} steps...")
    wm = ForwardWorldModel(gru_dim=gru_dim, action_dim=action_dim, seed=seed)
    for i in range(wm_train_steps):
        idx = rng.integers(0, n, size=batch_size)
        wm.update_step(H[idx], A[idx], H_next[idx])

    onehot_all = np.zeros((n, action_dim), np.float32)
    onehot_all[np.arange(n), A] = 1.0
    x_all = np.concatenate([H, onehot_all], axis=-1)
    wm_features_all = wm.net.hidden_features(x_all)
    feat_dim = wm_features_all.shape[1]

    print(f"\nTraining contrastive projector for {contrastive_train_steps} steps "
          f"(positive pairs = {k_nearest}-nearest by PREDICTED CONSEQUENCE, not action)...")
    proj = ContrastiveProjector(gru_dim=feat_dim, seed=seed, lr=1e-2)

    same_before, diff_before = same_vs_diff_similarity_for_context(
        proj.embed, wm_features_all, CTX, rng)
    print(f"BEFORE: context-same sim={same_before:.4f} context-diff sim={diff_before:.4f} "
          f"gap={same_before - diff_before:+.4f}")

    losses = []
    for i in range(contrastive_train_steps):
        idx = rng.integers(0, n, size=batch_size)
        h_batch = H[idx]
        a_batch = A[idx]
        wm_feats_batch = wm_features_all[idx]
        positive_mask = build_consequence_positive_mask(wm, h_batch, a_batch, k_nearest)
        loss = proj.update_step(wm_feats_batch, positive_mask)
        if loss is not None:
            losses.append(loss)
        if i % 100 == 0 and loss is not None:
            print(f"  step {i:4d}  loss {loss:.5f}")

    same_after, diff_after = same_vs_diff_similarity_for_context(
        proj.embed, wm_features_all, CTX, rng)
    print(f"AFTER:  context-same sim={same_after:.4f} context-diff sim={diff_after:.4f} "
          f"gap={same_after - diff_after:+.4f}")

    z_all = proj.embed(wm_features_all)
    km = OnlineKMeans(n_clusters=5, embed_dim=z_all.shape[1], seed=seed)
    for i in range(500):
        idx = rng.integers(0, n, size=batch_size)
        km.update_step(z_all[idx])
    cluster_assignments = km.assign(z_all)

    nmi_action = normalized_mutual_info(cluster_assignments, A)
    nmi_context = normalized_mutual_info(cluster_assignments, CTX)

    print()
    print("=" * 60)
    print("VERIFICATION: consequence-based pairing vs. old action-based pairing")
    print("=" * 60)
    print(f"NMI(clusters, action)  = {nmi_action:.4f}   (old design: 0.9988)")
    print(f"NMI(clusters, context) = {nmi_context:.4f}   (old design: 0.0016 -- FAILED)")
    print()
    print(f"Direct embedding-space context gap: before={same_before-diff_before:+.4f} "
          f"after={same_after-diff_after:+.4f}")
    print()

    if nmi_context > 0.15:
        print("RESULT: consequence-based pairing DOES produce clusters that correlate")
        print("        with independent spatial/object context. This is real evidence")
        print("        of emergent structure -- a genuine improvement over action-based")
        print("        pairing, not just a restatement of a different label.")
    elif nmi_context > 0.05:
        print("RESULT: weak improvement over the old design -- some context signal,")
        print("        not strong enough to confidently call this concept formation.")
    else:
        print("RESULT: still no meaningful correlation with context. Consequence-based")
        print("        pairing did not fix the underlying problem. Honest conclusion:")
        print("        the world model's predicted consequences in THIS gridworld may")
        print("        not vary enough with local context to support this distinction --")
        print("        or GRU h itself doesn't carry enough spatial information for any")
        print("        downstream pairing signal to exploit it.")

    return nmi_action, nmi_context


if __name__ == "__main__":
    main()
