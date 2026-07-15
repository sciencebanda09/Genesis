"""
train_visual.py -- Phase 2 end-to-end training loop.

The organism now receives ONLY rendered RGB images. No symbolic state
vectors. The vision encoder learns a latent representation through the
world model prediction objective, and the D1 agent learns to act using
those latents.

Architecture:
    VisualGridWorld -> RGB image -> VisionEncoder -> latent (128-dim)
        -> D1Agent (GRU dueling-Q on latents)
        -> LatentWorldModel (predicts latent_{t+1})
        -> RNDModule (curiosity on latents)

Gradient flow during training:
    WM loss -> backprop -> LatentWorldModel params
                        -> VisionEncoder params (representation learning)
    D1 loss -> backprop -> GRU + trunk + Q-heads (policy learning)
    RND     -> backprop -> RND predictor only (no encoder gradient)

This separation is deliberate:
    - The encoder learns solely to produce predictable latents (WM obj).
    - The agent learns to use latents for value estimation.
    - Curiosity explores in latent space, not pixel space.
"""
import argparse
import numpy as np
from tqdm import tqdm

from core.agent import D1Agent
from core.rnd import RNDModule
from core.logger import JsonlLogger
from .visual_gridworld import VisualGridWorld, ACTIONS
from .visual_buffer import VisualReplayBuffer
from .vision_encoder import VisionEncoder
from .world_model_v2 import LatentWorldModel


def run(episodes=200, max_steps=200, seed=0, render_size=64,
        log_path="logs/phase2_run.jsonl", warmup_steps=500,
        update_every=1, print_every=10, batch_size=64,
        latent_dim=128, encoder_lr=1e-5):
    env = VisualGridWorld(max_steps=max_steps, render_size=render_size, seed=seed)
    rng = np.random.default_rng(seed)

    encoder = VisionEncoder(latent_dim=latent_dim, lr=encoder_lr, rng=rng)
    wm = LatentWorldModel(latent_dim=latent_dim, action_dim=env.action_dim, seed=seed)
    agent = D1Agent(state_dim=latent_dim, action_dim=env.action_dim, seed=seed)
    rnd = RNDModule(state_dim=latent_dim, seed=seed)

    vbuf = VisualReplayBuffer(
        capacity=20000, img_h=render_size, img_w=render_size,
        latent_dim=latent_dim, gru_dim=agent.gru_dim, seed=seed,
    )

    logger = JsonlLogger(log_path)
    global_step = 0
    episode_returns = []

    for ep in tqdm(range(episodes), desc="Training"):
        image = env.reset()
        agent.reset_hidden()
        ep_intrinsic_return = 0.0

        for t in range(max_steps):
            h_before = agent._h.copy()
            latent = encoder.encode(image)
            action = agent.select_action(latent)
            h_after = agent._h.copy()

            next_image, extrinsic_r, done, info = env.step(action)
            next_latent = encoder.encode(next_image)

            intrinsic_raw = rnd.intrinsic_reward(next_latent)
            intrinsic_r = rnd.normalize(np.array([intrinsic_raw]))[0]

            agent.store(latent, action, intrinsic_r, next_latent, done,
                        h_before, h_after)
            vbuf.add(image, latent, action, intrinsic_r,
                     next_image, next_latent, h_before, h_after, done)

            logger.log_step(
                episode=ep, step=t, global_step=global_step,
                obs=latent, action=action, action_name=ACTIONS[action],
                extrinsic_reward=extrinsic_r, intrinsic_reward=intrinsic_r,
                done=done, image=image, latent=latent,
            )

            ep_intrinsic_return += intrinsic_r
            image = next_image
            global_step += 1

            if global_step > warmup_steps and global_step % update_every == 0:
                rnd.update_step(np.array([next_latent]))
                stats = agent.update()

                if len(vbuf) >= batch_size:
                    vb = vbuf.sample(batch_size)
                    im_batch = (vb["images"].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
                    nim_batch = (vb["next_images"].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

                    next_latents_det = encoder.forward(nim_batch).copy()
                    latents = encoder.forward(im_batch)

                    wm_loss, d_latent = wm.update_step_with_grad(
                        latents, vb["actions"], next_latents_det)
                    encoder.backward(d_latent)

                    if stats is not None:
                        stats["wm_loss"] = wm_loss

                if stats is not None:
                    logger.log_update(global_step, stats)

            if done:
                break

        episode_returns.append(ep_intrinsic_return)
        if (ep + 1) % print_every == 0:
            recent = episode_returns[-print_every:]
            print(f"episode {ep+1:4d} | intrinsic_return {np.mean(recent):8.4f} "
                  f"| eps {agent.epsilon():.3f} | buffer {len(vbuf):5d} "
                  f"| global_step {global_step}")

    logger.close()
    return episode_returns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=64,
                        choices=[32, 64])
    parser.add_argument("--log-path", type=str, default="logs/phase2_run.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
        render_size=args.render_size, log_path=args.log_path,
        batch_size=args.batch_size, encoder_lr=args.encoder_lr,
        latent_dim=args.latent_dim)
