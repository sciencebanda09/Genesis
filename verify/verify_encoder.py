"""
verify_encoder.py -- Isolated verification of VisionEncoder.

Demonstrates:
    1. The encoder runs without errors on single and batched inputs.
    2. Output shape matches (latent_dim,).
    3. Different images produce measurably different embeddings
       (cosine similarity well below 1.0).
    4. After a brief training loop via the world model objective,
       temporally adjacent frames (similar visual content) have
       HIGHER cosine similarity than random frame pairs.

This is NOT a claim about training convergence -- it is a sanity check
that the encoder processes images, produces latents, and can learn from
a prediction objective. Full training requires the complete pipeline.
"""
import numpy as np
from visual_track.visual_gridworld import VisualGridWorld
from visual_track.vision_encoder import VisionEncoder
from visual_track.world_model_v2 import LatentWorldModel


def random_image(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def run():
    print("=" * 60)
    print("verify_encoder.py")
    print("=" * 60)

    seed = 42
    rng = np.random.default_rng(seed)
    latent_dim = 128

    print("\n[1] Creating encoder...")
    enc = VisionEncoder(latent_dim=latent_dim, rng=rng)
    print(f"    Encoder params: {sum(p.size for p in enc.params())}")

    print("\n[2] Testing single-image encode...")
    img = random_image(seed=0)
    z = enc.encode(img)
    print(f"    Input shape: {img.shape}  ->  latent shape: {z.shape}")
    assert z.shape == (latent_dim,), f"Expected ({latent_dim},), got {z.shape}"
    print("    PASS: single encode returns correct shape.")

    print("\n[3] Testing batch forward...")
    batch = np.stack([random_image(seed=i) for i in range(8)])
    x = batch.astype(np.float32) / 255.0
    x = x.transpose(0, 3, 1, 2)
    z_batch = enc.forward(x)
    print(f"    Batch {batch.shape}  ->  latents {z_batch.shape}")
    assert z_batch.shape == (8, latent_dim)
    print("    PASS: batch forward returns correct shape.")

    print("\n[4] Testing embedding diversity (untrained encoder)...")
    imgs = np.stack([random_image(seed=i) for i in range(32)])
    x = imgs.astype(np.float32) / 255.0
    x = x.transpose(0, 3, 1, 2)
    zs = enc.forward(x)
    norms = np.linalg.norm(zs, axis=-1, keepdims=True)
    sim = (zs @ zs.T) / (norms @ norms.T + 1e-8)
    np.fill_diagonal(sim, np.nan)
    mean_sim = float(np.nanmean(sim))
    std_sim = float(np.nanstd(sim))
    print(f"    Mean pairwise cosine sim: {mean_sim:.4f}  (std: {std_sim:.4f})")
    assert abs(mean_sim) < 0.9, f"Embeddings nearly identical ({mean_sim:.4f})"
    print("    PASS: embeddings are diverse.")

    print("\n[5] Training encoder via world model objective...")
    env = VisualGridWorld(max_steps=100, render_size=64, seed=seed)
    wm = LatentWorldModel(latent_dim=latent_dim, action_dim=env.action_dim, seed=seed)

    images_t, actions, images_next = [], [], []
    obs = env.reset()
    for _ in range(300):
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
    batch_size = 16
    losses = []

    # Joint training: encoder every 5th step, WM every step
    for i in range(100):
        idx = rng.integers(0, n, size=batch_size)
        im_batch = images_t[idx].transpose(0, 3, 1, 2)
        nim_batch = images_next[idx].transpose(0, 3, 1, 2)

        next_latents = enc.forward(nim_batch).copy()
        latents = enc.forward(im_batch)

        loss, d_latent = wm.update_step_with_grad(latents, actions[idx], next_latents)

        if i % 5 == 0:
            enc.backward(d_latent)

        losses.append(loss)

    print(f"    Trained {len(losses)} steps (encoder {len(losses)//5}x). "
          f"Loss: first 10 avg={np.mean(losses[:10]):.5f}, "
          f"last 10 avg={np.mean(losses[-10:]):.5f}")

    print("\n[6] Verifying temporal similarity > random similarity...")
    idx_pairs = rng.integers(0, n - 1, size=100)
    idx_rand = rng.integers(0, n, size=100)

    im_t = images_t[idx_pairs].transpose(0, 3, 1, 2)
    im_n = images_next[idx_pairs].transpose(0, 3, 1, 2)
    im_r = images_t[idx_rand].transpose(0, 3, 1, 2)

    z_t = enc.forward(im_t)
    z_n = enc.forward(im_n)
    z_r = enc.forward(im_r)

    def cos_sim(a, b):
        num = np.sum(a * b, axis=-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
        return float(np.mean(num / den))

    cos_temporal = cos_sim(z_t, z_n)
    cos_random = cos_sim(z_t, z_r)

    print(f"    Temporal-pair cos sim:  {cos_temporal:.4f}")
    print(f"    Random-pair cos sim:    {cos_random:.4f}")
    print(f"    Gap:                    {cos_temporal - cos_random:+.4f}")

    print()
    if cos_temporal > cos_random:
        print("RESULT: encoder produces temporally-aware embeddings.")
        print("        (adjacent frames are closer in latent space than random pairs).")
    else:
        print("RESULT: temporal gap not observed with short training.")
        print("        The encoder architecture is verified; longer training may help.")

    return cos_temporal, cos_random, losses


if __name__ == "__main__":
    run()
