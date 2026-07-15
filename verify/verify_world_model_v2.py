"""
verify_world_model_v2.py -- Isolated verification of LatentWorldModel.

The bar (same as world_model.py's verify_world_model.py):
    Prediction loss on REAL (latent_t, a_t, latent_{t+1}) transitions
    should drop with training AND should end up lower than loss on
    SHUFFLED transitions (same latent_t, a_t but a randomly mismatched
    latent_{t+1}). If the model can't beat shuffled, it's just regressing
    to the mean of the latent distribution.

BUG FIX (audit): the original version created a VisionEncoder and called
enc.encode() to build the transition dataset, but NEVER trained the
encoder (enc.backward() was never called anywhere in this script). An
untrained, randomly-initialized encoder produces near-arbitrary latents,
so the world model was being asked to predict noise from noise -- the
resulting 1.02x ratio (real error 0.00035 vs shuffled 0.00036, at
n_steps=800) reflected that setup, not a real limit on LatentWorldModel's
capacity. The prior version's comment ("random CNN latents have very low
total variance... prediction floor is near the mean-MSE") correctly
described the SYMPTOM but not the fix: the floor being low doesn't mean
the ratio can't be meaningfully above 1x once the encoder actually learns
something for the world model to exploit. Fixed by jointly training the
encoder (via the WM's own gradient, same mechanism as verify_encoder.py)
before evaluating, and raising the pass threshold's required margin.
"""
import numpy as np
from tqdm import tqdm
from visual_track.visual_gridworld import VisualGridWorld
from visual_track.vision_encoder import VisionEncoder
from visual_track.world_model_v2 import LatentWorldModel


def collect_and_train(n_steps=800, train_updates=400, batch_size=32, seed=0):
    """Collect image transitions, then JOINTLY train encoder + world model,
    returning post-training latents (not from a frozen/untrained encoder)."""
    env = VisualGridWorld(max_steps=100, render_size=64, seed=seed)
    enc = VisionEncoder(latent_dim=128, rng=np.random.default_rng(seed))
    rng = np.random.default_rng(seed)

    images_t, actions, images_next = [], [], []
    obs = env.reset()
    for _ in range(n_steps):
        action = int(rng.integers(env.action_dim))
        next_obs, _, done, _ = env.step(action)
        images_t.append(obs)
        actions.append(action)
        images_next.append(next_obs)
        obs = next_obs
        if done:
            obs = env.reset()

    images_t = np.stack(images_t).astype(np.float32) / 255.0
    images_next = np.stack(images_next).astype(np.float32) / 255.0
    actions = np.array(actions, np.int32)
    n = len(images_t)

    wm = LatentWorldModel(latent_dim=enc.latent_dim, action_dim=env.action_dim, seed=seed)

    losses = []
    for i in tqdm(range(train_updates), desc="Joint encoder+WM"):
        idx = rng.integers(0, n, size=batch_size)
        im_batch = images_t[idx].transpose(0, 3, 1, 2)
        nim_batch = images_next[idx].transpose(0, 3, 1, 2)

        next_latents = enc.forward(nim_batch).copy()
        latents = enc.forward(im_batch)
        loss, d_latent = wm.update_step_with_grad(latents, actions[idx], next_latents)

        if i % 3 == 0:  # encoder updates less often than WM, matches verify_encoder.py
            enc.backward(d_latent)
        losses.append(loss)

    # build the full latent dataset with the NOW-TRAINED encoder, for eval
    L = enc.forward(images_t.transpose(0, 3, 1, 2))
    L_next = enc.forward(images_next.transpose(0, 3, 1, 2))
    return L.astype(np.float32), actions, L_next.astype(np.float32), wm, losses


def main(n_steps=800, train_updates=400, batch_size=32, seed=0):
    print(f"Collecting {n_steps} transitions and jointly training encoder + world model...")
    L, A, L_next, wm, losses = collect_and_train(
        n_steps=n_steps, train_updates=train_updates, batch_size=batch_size, seed=seed)
    print(f"Done. L shape: {L.shape}")
    print(f"Joint-training loss: first 20 avg = {np.mean(losses[:20]):.5f} | "
          f"last 20 avg = {np.mean(losses[-20:]):.5f}")

    rng = np.random.default_rng(seed)
    n = len(L)

    print()
    print("=" * 60)
    print("VERIFICATION: real vs shuffled latents (post joint-training)")
    print("=" * 60)

    eval_idx = rng.integers(0, n, size=200)
    real_err = wm.prediction_error(L[eval_idx], A[eval_idx], L_next[eval_idx])

    shuffled_idx = rng.permutation(n)[:200]
    shuffled_err = wm.prediction_error(L[eval_idx], A[eval_idx], L_next[shuffled_idx])

    print(f"Real-transition prediction error:     {real_err.mean():.5f} +/- {real_err.std():.5f}")
    print(f"Shuffled-transition prediction error: {shuffled_err.mean():.5f} +/- {shuffled_err.std():.5f}")
    ratio = shuffled_err.mean() / (real_err.mean() + 1e-8)
    print(f"Ratio (shuffled / real):               {ratio:.2f}x")
    print()

    # Real bar: require a meaningful margin above 1.0x, not just "any ratio
    # over 1.0" (which the old 0.85-multiplier threshold effectively allowed
    # for near-1x results too close to call). 1.5x is still a modest bar --
    # the feature-vector world model (verify_world_model.py) achieves ~30x
    # on the same style of test, so 1.5x is intentionally conservative given
    # this is a harder, higher-dimensional, less mature pipeline.
    if ratio > 1.5:
        print("RESULT: LatentWorldModel IS learning real latent transition structure.")
        print(f"        (ratio {ratio:.2f}x clears the 1.5x margin)")
    else:
        print("RESULT: LatentWorldModel is NOT clearly learning transition structure.")
        print(f"        (ratio {ratio:.2f}x does not clear the 1.5x margin -- even with a")
        print("         jointly-trained encoder, this is not yet a strong result. Compare")
        print("         to the feature-vector world model's ~30x ratio on the same test")
        print("         style -- the visual pipeline is meaningfully behind it.)")

    return real_err, shuffled_err, losses


if __name__ == "__main__":
    main()
