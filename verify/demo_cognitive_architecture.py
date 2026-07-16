"""
demo_cognitive_architecture.py — Integration demo of all cognitive systems.

Wires together every new module in a mini training loop on GridWorld.
Demonstrates that the systems compose correctly.

This is NOT a full benchmark — it's a smoke test that all modules
can be instantiated and used together without errors.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from core.memory import WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
from core.consolidation import KnowledgeConsolidation
from core.object_permanence import ObjectPermanence
from core.world_model_v3 import MultiStepWorldModel
from core.planning import Planner
from core.concept_formation import ConceptFormation
from core.goals import GoalWeights, GoalDrivenReward
from core.reasoning import ReasoningEngine, Proposition
from core.self_model import SelfModel
from core.language import GroundedVocabulary
from core.executive_cortex import ExecutiveCortex

PASS_BAR = "all cognitive systems compose and execute without error"


def run_demo():
    print("=" * 60)
    print("Cognitive Architecture Integration Demo")
    print("=" * 60)

    # 1. All Systems Initialization
    print("\n[1/5] Initializing all cognitive systems...")
    
    state_dim, gru_dim, action_dim = 8, 32, 5
    
    wm = WorkingMemory(capacity=20, state_dim=state_dim, gru_dim=gru_dim)
    em = EpisodicMemory(max_episodes=50)
    sem = SemanticMemory(n_concepts=16, embed_dim=gru_dim)
    pm = ProceduralMemory(max_skills=10)
    
    cons = KnowledgeConsolidation(embed_dim=gru_dim, n_concepts=16,
                                   consolidate_every=30, min_episodes_for_consolidation=3)
    
    tracker = ObjectPermanence(n_slots=8, feature_dim=16, max_invisible=5)
    
    mwm = MultiStepWorldModel(gru_dim=gru_dim, action_dim=action_dim, 
                               n_ensemble=3, hidden_dim=64, seed=42)
    
    planner = Planner(mwm, horizon=8, n_candidates=32, n_elites=8)
    
    cf = ConceptFormation(embed_dim=gru_dim, n_concepts=16, use_objects=True)
    
    gw = GoalWeights()
    gdr = GoalDrivenReward(gw)
    
    reason = ReasoningEngine()
    
    sm = SelfModel(window=200)
    
    vocab = GroundedVocabulary()
    vocab.learn_from_concepts(cf, action_names=["up", "down", "left", "right", "interact"])
    
    cortex = ExecutiveCortex()
    
    print("   All systems initialized.")

    # 2. Simulate a mini training loop (20 steps, 3 episodes)
    print("\n[2/5] Simulating 3 episodes of interaction...")
    
    global_step = 0
    
    for episode in range(3):
        h = np.random.randn(gru_dim).astype(np.float32)
        obs = np.random.randn(state_dim).astype(np.float32)
        
        episode_hiddens, episode_actions, episode_rewards, episode_obs = [], [], [], []
        
        for t in range(30):
            action = np.random.randint(0, action_dim)
            next_obs = np.random.randn(state_dim).astype(np.float32)
            reward = float(np.random.randn())
            h_next = np.random.randn(gru_dim).astype(np.float32)
            
            # Working Memory
            wm.push(obs, h, action, reward, global_step)
            
            # Object Permanence (mock objects)
            if np.random.random() > 0.7:
                mock_objects = [
                    (np.random.randn(16).astype(np.float32), 
                     np.random.randn(2).astype(np.float32), 
                     np.random.randint(1, 4), 0.8)
                ]
                tracker.update(mock_objects, timestep=global_step)
            
            # Concept Formation
            slots = tracker.get_active_slots()
            emb, meta = cf.embed(next_obs, object_slots=slots)
            concept = cf.update(emb)
            
            # World Model
            mwm.update_step(h.reshape(1, -1), np.array([action]), h_next.reshape(1, -1))
            
            # Reasoning
            reason.tell(Proposition(f"concept_{concept}", "observed", None, 0.9))
            reason.infer()
            
            # Self Model
            sm.observe_loss("wm_loss", float(np.random.rand() * 0.1))
            sm.observe_loss("td_error", float(np.random.rand() * 2.0))
            sm.observe_loss("coverage", float(np.random.rand() * 0.3))
            
            # Goals
            total_reward, breakdown = gdr.compute(
                curiosity_r=float(np.random.rand()),
                exploration_r=float(np.random.rand()),
                prediction_r=float(1.0 / (1.0 + np.random.rand())),
            )
            
            # Executive Cortex
            cortex.observe(
                global_step=global_step,
                coverage=float(np.random.rand()),
                td_error_mean=float(np.random.rand()),
                wm_loss=float(np.random.rand() * 0.1),
                rnd_reward=float(np.random.rand()),
                policy_loss=float(np.random.rand() * 0.5),
            )
            params = cortex.regulate()
            if params:
                gdr.set_weights(**params.get('goal_weights', {}))
            
            # Store episode data
            episode_hiddens.append(h_next)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_obs.append(next_obs)
            
            h, obs = h_next, next_obs
            global_step += 1
        
        # Episodic Memory
        ep_id = em.store_episode(
            np.array(episode_obs), np.array(episode_hiddens),
            np.array(episode_actions), np.array(episode_rewards),
            {"episode": episode}
        )
        
        # Knowledge Consolidation
        all_h, transitions = cons.replay_for_consolidation(em, None)
        if all_h is not None:
            cons.step(em, global_step, h_states=all_h, transitions=transitions)
        
        # Semantic Memory
        if all_h is not None:
            sem.update_concepts(all_h[:100])
            for f, a, t in (transitions or [])[:50]:
                sem.observe_transition(f, a, t)
        
        # Procedural Memory (mock skill discovery)
        if np.random.random() > 0.8:
            pm.store_skill(f"explore_path_{ep_id}", 
                          np.random.randn(gru_dim), np.random.randn(gru_dim),
                          np.random.randint(0, action_dim, size=5))
        
        # Language (auto-name concepts)
        freqs = cf.concept_frequency()
        for c in range(len(freqs)):
            if freqs[c] > 0:
                vocab.auto_name_concept(c, freqs[c])
        
        # Self Model
        sm.encode_self_state(np.array(episode_hiddens[-10:]))
        
        print(f"   Episode {episode+1}: "
              f"concepts={cf.n_active_concepts()}, "
              f"objects={tracker.get_object_count()}, "
              f"episodic={len(em)}, "
              f"vocab={vocab.vocabulary_size()}, "
              f"skills={len(pm.list_skills())}, "
              f"goal_weights={len(gw.as_dict())}")
    
    # 3. Demonstrate Planning
    print("\n[3/5] Demonstrating planning pipeline...")
    h_current = np.random.randn(gru_dim).astype(np.float32)
    plan_actions, plan_traj, plan_score = planner.plan(h_current)
    print(f"   Plan: {len(plan_actions)} steps, score={plan_score:.4f}")
    action = planner.act(h_current)
    print(f"   Act: action={action}")
    
    # 4. Demonstrate Reasoning
    print("\n[4/5] Demonstrating reasoning...")
    derived = reason.infer()
    print(f"   KB size: {reason.get_state()['kb_size']}, "
          f"inferences: {reason.get_state()['inferences_made']}")
    
    # 5. Demonstrate Self-Reflection
    print("\n[5/5] Demonstrating self-reflection...")
    known = sm.know_what_i_know()
    gaps = sm.know_what_i_dont_know()
    console = cortex.get_summary()
    print(f"   Known modules: {list(known['module_confidence'].keys())}")
    print(f"   Confidences: { {k: f'{v:.2f}' for k, v in known['module_confidence'].items()} }")
    print(f"   Agent state: eps={cortex.epsilon:.2f}, "
          f"goals={ {k: f'{v:.2f}' for k, v in cortex.goal_weights.items()} }")
    print(f"   Gaps: {len(gaps['uncertain_modules'])} uncertain, "
          f"{len(gaps.get('rare_transitions', []))} rare transitions")
    
    print("\n" + "=" * 60)
    print(f"PASS: {PASS_BAR}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
