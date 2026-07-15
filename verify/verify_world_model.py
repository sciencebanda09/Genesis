"""
verify_world_model.py — isolated test of world_model.py BEFORE wiring it
into the main training loop.

The bar: prediction loss on REAL (h_t, a_t, h_t+1) transitions should drop
with training AND should end up lower than loss on SHUFFLED transitions
(same h_t, a_t but a randomly mismatched h_t+1 from elsewhere in the
buffer). If the model can't beat the shuffled baseline, it hasn't learned
transition structure -- it's just regressing toward the mean of h, which
would be a silent failure mode indistinguishable from "working" if you
only look at the raw loss curve going down.
"""
import numpy as np

from core.rnd import RNDModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from gridworld_track.gridworld import GridWorld


def collect_transitions(n_steps=3000, seed=0):
    """Run a real RND+D1 agent and collect (h_t, action, h_t+1) transitions."""
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

        H.append(h_before[0])
        A.append(action)
        H_next.append(h_after[0])

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


def main(n_steps=3000, train_updates=500, batch_size=64, seed=0):
    print(f"Collecting {n_steps} real transitions from an RND+D1 agent...")
    H, A, H_next = collect_transitions(n_steps=n_steps, seed=seed)
    print(f"Collected. H shape: {H.shape}")

    gru_dim = H.shape[1]
    action_dim = int(A.max()) + 1
    wm = ForwardWorldModel(gru_dim=gru_dim, action_dim=action_dim, seed=seed)

    rng = np.random.default_rng(seed)
    n = len(H)

    print(f"\nTraining forward world model for {train_updates} steps...")
    losses = []
    for i in range(train_updates):
        idx = rng.integers(0, n, size=batch_size)
        loss = wm.update_step(H[idx], A[idx], H_next[idx])
        losses.append(loss)
        if i % 50 == 0:
            print(f"  step {i:4d}  loss {loss:.5f}")

    print()
    print("=" * 60)
    print("VERIFICATION: real transitions vs. shuffled (mismatched) targets")
    print("=" * 60)

    eval_idx = rng.integers(0, n, size=500)
    real_err = wm.prediction_error(H[eval_idx], A[eval_idx], H_next[eval_idx])

    shuffled_idx = rng.permutation(n)[:500]  # random h_next, unrelated to h_t/a_t
    shuffled_err = wm.prediction_error(H[eval_idx], A[eval_idx], H_next[shuffled_idx])

    print(f"Real-transition prediction error:     {real_err.mean():.5f} +/- {real_err.std():.5f}")
    print(f"Shuffled-transition prediction error: {shuffled_err.mean():.5f} +/- {shuffled_err.std():.5f}")
    print(f"Ratio (shuffled / real):               {shuffled_err.mean() / (real_err.mean() + 1e-8):.2f}x")
    print()

    print(f"Loss trend: first 20 avg = {np.mean(losses[:20]):.5f} | last 20 avg = {np.mean(losses[-20:]):.5f}")
    print()

    if real_err.mean() < shuffled_err.mean() * 0.8:
        print("RESULT: world model IS learning real transition structure")
        print("        (predicts real next-states meaningfully better than random ones).")
    else:
        print("RESULT: world model is NOT clearly learning transition structure --")
        print("        real vs shuffled error too close. Do not build contrastive")
        print("        learning on top of this yet; something needs fixing first.")

    return real_err, shuffled_err, losses


if __name__ == "__main__":
    main()
