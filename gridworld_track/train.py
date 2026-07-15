"""
train.py — Phase 1 training loop.

Newborn agent: no task reward, pure RND curiosity, D1 delay-corrected value
learning, plus concept-formation steps 1 and 2:
  1. Forward world model on the GRU hidden state (world_model.py) --
     verified: predicts real transitions ~32x better than shuffled ones.
  2. Contrastive projector (contrastive.py) trained on the world model's
     INTERNAL hidden features, not raw h -- raw h was verified to lack
     separable structure (same/diff-action cosine sim both ~0.84, no gap).
     World-model features, by contrast, gave a genuine gap (+1.21 after
     training vs +0.004 before, in isolated verification).

Both are purely additive: they observe what D1/RND already produce and
train alongside, without altering D1's or RND's own updates.
"""
import argparse
import numpy as np

from core.rnd import RNDModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from core.contrastive import ContrastiveProjector
from core.logger import JsonlLogger
from .gridworld import GridWorld, ACTIONS


def run(episodes=200, max_steps=200, seed=0, log_path="logs/phase1_run.jsonl",
        warmup_steps=500, update_every=1, print_every=10,
        train_world_model=True, train_contrastive=True, contrastive_batch=64):
    env = GridWorld(max_steps=max_steps, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)
    wm = ForwardWorldModel(gru_dim=agent.gru_dim, action_dim=env.action_dim, seed=seed) \
        if train_world_model else None
    # contrastive projector's input dim = world model's hidden feature dim,
    # which is the trunk's last hidden layer size (hidden_dim in ForwardWorldModel)
    proj = ContrastiveProjector(gru_dim=wm.net.layers[-2].W.shape[1] if wm is not None else agent.gru_dim,
                                 seed=seed) if (train_contrastive and wm is not None) else None
    logger = JsonlLogger(log_path)

    global_step = 0
    episode_returns = []

    for ep in range(episodes):
        obs = env.reset()
        agent.reset_hidden()
        ep_intrinsic_return = 0.0

        for t in range(max_steps):
            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, extrinsic_r, done, info = env.step(action)

            intrinsic_r_raw = rnd.intrinsic_reward(next_obs)
            intrinsic_r = rnd.normalize(np.array([intrinsic_r_raw]))[0]

            h_after = agent._h.copy()
            agent.store(obs, action, intrinsic_r, next_obs, done, h_before, h_after)

            logger.log_step(
                episode=ep, step=t, global_step=global_step,
                obs=obs, action=action, action_name=ACTIONS[action],
                extrinsic_reward=extrinsic_r, intrinsic_reward=intrinsic_r,
                done=done, interacted_with=info.get("interacted_with"),
            )

            ep_intrinsic_return += intrinsic_r
            obs = next_obs
            global_step += 1

            if global_step > warmup_steps and global_step % update_every == 0:
                rnd.update_step(np.array([next_obs]))
                stats = agent.update()
                if wm is not None:
                    wm_loss = wm.update_step(h_before, [action], h_after)
                    if stats is not None:
                        stats["world_model_loss"] = wm_loss

                if proj is not None and len(agent.buffer) >= contrastive_batch:
                    batch = agent.buffer.sample(contrastive_batch)
                    onehot = np.zeros((contrastive_batch, env.action_dim), np.float32)
                    onehot[np.arange(contrastive_batch), batch["actions"]] = 1.0
                    x = np.concatenate([batch["hiddens"], onehot], axis=-1)
                    wm_feats = wm.net.hidden_features(x)
                    a_batch = batch["actions"]
                    positive_mask = a_batch[:, None] == a_batch[None, :]  # same-action pairing
                    c_loss = proj.update_step(wm_feats, positive_mask)
                    if stats is not None and c_loss is not None:
                        stats["contrastive_loss"] = c_loss

                if stats is not None:
                    logger.log_update(global_step, stats)

            if done:
                break

        episode_returns.append(ep_intrinsic_return)
        if (ep + 1) % print_every == 0:
            recent = episode_returns[-print_every:]
            print(f"episode {ep+1:4d} | intrinsic_return {np.mean(recent):8.4f} "
                  f"| eps {agent.epsilon():.3f} | buffer {len(agent.buffer):5d} "
                  f"| global_step {global_step}")

    logger.close()
    return episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", type=str, default="logs/phase1_run.jsonl")
    args = parser.parse_args()

    run(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
        log_path=args.log_path)
