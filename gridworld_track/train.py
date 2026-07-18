"""
train.py — Integrated training loop.

Phase 1 core (D1 + RND + ForwardWorldModel + ContrastiveProjector) plus
all cognitive modules from Phases 2.5–10 wired into the same loop:

  Executive Cortex (regulation)
  Adaptive memory (uniform + prioritized mixing)
  ObjectPermanence (slot tracking)
  MultiStepWorldModel (ensemble, multi-step)
  ConceptFormation (hierarchical clustering)
  EpisodicMemory + KnowledgeConsolidation
  SelfModel (metacognitive monitoring)

All new modules are additive: they observe what D1/RND produce and train
alongside, without altering D1's or RND's own updates.
"""
import argparse
import numpy as np
from tqdm import tqdm

from core.rnd import RNDModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from core.contrastive import ContrastiveProjector
from core.logger import JsonlLogger
from core.replay_buffer import PrioritizedReplay
from core.executive_cortex import ExecutiveCortex
from core.object_permanence import ObjectPermanence
from core.world_model_v3 import MultiStepWorldModel
from core.concept_formation import ConceptFormation
from core.memory import EpisodicMemory
from core.consolidation import KnowledgeConsolidation
from core.self_model import SelfModel
from .gridworld import GridWorld, ACTIONS


def run(episodes=200, max_steps=200, seed=0, log_path="logs/phase1_run.jsonl",
        warmup_steps=500, update_every=1, print_every=10,
        train_world_model=True, train_contrastive=True, contrastive_batch=64,
        use_cortex=True, use_adaptive_memory=True,
        train_multistep_wm=True, use_object_permanence=True,
        use_concept_formation=True, use_self_model=True,
        n_ensemble=3):
    env = GridWorld(max_steps=max_steps, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)

    wm = ForwardWorldModel(gru_dim=agent.gru_dim, action_dim=env.action_dim, seed=seed) \
        if train_world_model else None
    proj = ContrastiveProjector(
        gru_dim=wm.net.layers[-2].W.shape[1] if wm is not None else agent.gru_dim,
        seed=seed) if (train_contrastive and wm is not None) else None

    # Cognitive modules
    cortex = ExecutiveCortex() if use_cortex else None
    prio_buffer = PrioritizedReplay(
        agent.buffer.capacity, agent.state_dim, agent.gru_dim,
        seed=seed + 1000) if use_adaptive_memory else None
    # ponytail: fixed 8 slots. Add slot attention later.
    tracker = ObjectPermanence(n_slots=8, feature_dim=16, max_invisible=5) \
        if use_object_permanence else None
    mwm = MultiStepWorldModel(gru_dim=agent.gru_dim, action_dim=env.action_dim,
                               n_ensemble=n_ensemble, hidden_dim=64, seed=seed) \
        if train_multistep_wm else None
    cf = ConceptFormation(embed_dim=agent.gru_dim, n_concepts=16) \
        if use_concept_formation else None
    em = EpisodicMemory(max_episodes=200) if use_concept_formation else None
    cons = KnowledgeConsolidation(embed_dim=agent.gru_dim, n_concepts=16,
                                   consolidate_every=30,
                                   min_episodes_for_consolidation=3) \
        if use_concept_formation else None
    sm = SelfModel(window=200) if use_self_model else None

    logger = JsonlLogger(log_path)

    global_step = 0
    episode_returns = []

    for ep in tqdm(range(episodes), desc="Training"):
        obs = env.reset()
        agent.reset_hidden()
        ep_intrinsic_return = 0.0

        episode_hiddens, episode_actions, episode_rewards = [], [], []

        for t in range(max_steps):
            if cortex is not None:
                cortex.observe(global_step=global_step, episode=ep, coverage=env.coverage())

            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, extrinsic_r, done, info = env.step(action)

            intrinsic_r_raw = rnd.intrinsic_reward(next_obs)
            intrinsic_r = rnd.normalize(np.array([intrinsic_r_raw]))[0]

            h_after = agent._h.copy()
            agent.store(obs, action, intrinsic_r, next_obs, done, h_before, h_after)

            if prio_buffer is not None:
                prio_buffer.add(obs, h_before, action, intrinsic_r,
                                next_obs, h_after, done)

            if tracker is not None:
                visible = env.visible_objects()
                if visible:
                    tracker.update(visible, timestep=global_step)

            if cf is not None:
                slots = tracker.get_active_slots() if tracker is not None else None
                emb, meta = cf.embed(obs, object_slots=slots)
                cf.update(emb)

            logger.log_step(
                episode=ep, step=t, global_step=global_step,
                obs=obs, action=action, action_name=ACTIONS[action],
                extrinsic_reward=extrinsic_r, intrinsic_reward=intrinsic_r,
                done=done, interacted_with=info.get("interacted_with"),
            )

            ep_intrinsic_return += intrinsic_r
            episode_hiddens.append(h_after)
            episode_actions.append(action)
            episode_rewards.append(intrinsic_r)
            obs = next_obs
            global_step += 1

            if global_step > warmup_steps and global_step % update_every == 0:
                rnd.update_step(np.array([next_obs]))

                # Agent update with optional adaptive memory mixing
                combined = None
                uniform_batch = None
                prio_batch = None
                if prio_buffer is not None and cortex is not None:
                    mw = cortex.memory_weights
                    bs = agent.batch_size
                    n_u = max(1, int(bs * mw.get('uniform', 1.0)))
                    n_p = max(1, int(bs * mw.get('prioritized', 0.0)))
                    uniform_batch = agent.buffer.sample(n_u)
                    prio_batch = prio_buffer.sample(n_p) if len(prio_buffer) >= n_p else None
                    if uniform_batch is not None or prio_batch is not None:
                        if uniform_batch is not None and prio_batch is not None:
                            combined = {}
                            for k in uniform_batch:
                                if isinstance(uniform_batch[k], np.ndarray):
                                    combined[k] = np.concatenate(
                                        [uniform_batch[k], prio_batch[k]], axis=0)
                            combined['indices'] = (
                                list(uniform_batch.get('indices', []))
                                + list(prio_batch.get('indices', [])))
                            combined['weights'] = np.concatenate(
                                [uniform_batch['weights'], prio_batch['weights']])
                        elif uniform_batch is not None:
                            combined = uniform_batch
                        else:
                            combined = prio_batch
                    stats = agent.update(batch=combined) if combined is not None else None
                else:
                    stats = agent.update()

                if wm is not None:
                    wm_loss = wm.update_step(h_before, [action], h_after)
                    if stats is not None:
                        stats["world_model_loss"] = wm_loss

                if mwm is not None:
                    mwm_loss = float(np.mean(mwm.update_step(
                        h_before.reshape(1, -1), np.array([action]),
                        h_after.reshape(1, -1))))
                    if stats is not None:
                        stats["multistep_wm_loss"] = mwm_loss

                if proj is not None and len(agent.buffer) >= contrastive_batch:
                    batch = agent.buffer.sample(contrastive_batch)
                    onehot = np.zeros((contrastive_batch, env.action_dim), np.float32)
                    onehot[np.arange(contrastive_batch), batch["actions"]] = 1.0
                    x = np.concatenate([batch["hiddens"], onehot], axis=-1)
                    wm_feats = wm.net.hidden_features(x)
                    a_batch = batch["actions"]
                    positive_mask = a_batch[:, None] == a_batch[None, :]
                    c_loss = proj.update_step(wm_feats, positive_mask)
                    if stats is not None and c_loss is not None:
                        stats["contrastive_loss"] = c_loss

                # Adaptive memory: per-sample priority update + per-buffer cortex tracking
                if combined is not None and stats is not None and prio_buffer is not None:
                    per_sample = stats.get('td_error', None)
                    if per_sample is not None:
                        n_u = len(uniform_batch['states']) if uniform_batch is not None else 0
                        n_p = len(prio_batch['states']) if prio_batch is not None else 0
                        prio_errs = per_sample[n_u:n_u + n_p]
                        if 'indices' in combined and len(prio_errs) > 0:
                            prio_buffer.update_priorities(combined['indices'], prio_errs)
                        if n_u > 0 and cortex is not None:
                            cortex.observe(uniform_td_error=float(
                                np.mean(np.abs(per_sample[:n_u]))))
                        if n_p > 0 and cortex is not None:
                            cortex.observe(prioritized_td_error=float(
                                np.mean(np.abs(prio_errs))))

                if sm is not None and stats is not None:
                    sm.observe_loss("td_error", stats.get('td_error_mean', 0.0))
                    sm.observe_loss("delay_loss", stats.get('delay_loss', 0.0))
                    if wm is not None:
                        sm.observe_loss("wm_loss", wm_loss)

                if cortex is not None:
                    cortex.observe(
                        td_error_mean=stats.get('td_error_mean', 0.0) if stats else 0.0,
                        wm_loss=wm_loss if wm is not None else 0.0,
                        rnd_reward=float(np.mean(ep_intrinsic_return / (t + 1))),
                        rnd_loss=stats.get('delay_loss', 0.0) if stats else 0.0,
                        coverage=env.coverage(),
                        intrinsic_return=ep_intrinsic_return / (t + 1),
                    )
                    if sm is not None:
                        cortex.observe(policy_loss=stats.get('td_error_mean', 0.0) if stats else 0.0)
                    cortex.regulate()

                if stats is not None:
                    logger.log_update(global_step, stats)

            if done:
                break

        # End of episode: store in episodic memory + consolidate
        if em is not None and len(episode_hiddens) > 0:
            em.store_episode(
                np.array([obs] * len(episode_hiddens)) if len(episode_hiddens) > 0 else np.zeros((0, agent.state_dim)),
                np.array(episode_hiddens), np.array(episode_actions),
                np.array(episode_rewards), {"episode": ep})

        if cons is not None and em is not None:
            all_h, transitions = cons.replay_for_consolidation(em, agent.buffer if hasattr(agent, 'buffer') else None)
            if all_h is not None:
                cons.step(em, global_step, h_states=all_h, transitions=transitions)

        episode_returns.append(ep_intrinsic_return)
        if (ep + 1) % print_every == 0:
            recent = episode_returns[-print_every:]
            extras = []
            if cf is not None:
                extras.append(f"concepts {cf.n_active_concepts()}")
            if tracker is not None:
                extras.append(f"objects {tracker.get_object_count()}")
            if cortex is not None:
                extras.append(f"eps {cortex.epsilon:.3f}")
            else:
                extras.append(f"eps {agent.epsilon():.3f}")
            extra_str = " | ".join(extras)
            print(f"ep {ep+1:4d} | return {np.mean(recent):8.4f} "
                  f"| buffer {len(agent.buffer):5d} "
                  f"| step {global_step:6d} | {extra_str}")

    logger.close()
    return episode_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", type=str, default="logs/phase1_run.jsonl")
    parser.add_argument("--no-cortex", action="store_true")
    parser.add_argument("--no-adaptive-memory", action="store_true")
    parser.add_argument("--no-multistep-wm", action="store_true")
    parser.add_argument("--no-object-permanence", action="store_true")
    parser.add_argument("--no-concept-formation", action="store_true")
    parser.add_argument("--no-self-model", action="store_true")
    args = parser.parse_args()

    run(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
        log_path=args.log_path,
        use_cortex=not args.no_cortex,
        use_adaptive_memory=not args.no_adaptive_memory,
        train_multistep_wm=not args.no_multistep_wm,
        use_object_permanence=not args.no_object_permanence,
        use_concept_formation=not args.no_concept_formation,
        use_self_model=not args.no_self_model)
