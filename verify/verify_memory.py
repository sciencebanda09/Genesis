"""
verify_memory.py — Verify the four-level memory hierarchy.

Tests:
  1. WorkingMemory: stores and retrieves recent context
  2. EpisodicMemory: stores whole episodes, recalls by recency/similarity/salience
  3. SemanticMemory: builds concept clusters and transition schemas
  4. ProceduralMemory: stores and retrieves skills

Pass bar: all four systems operate without error and produce expected outputs.
"""
import numpy as np
import sys

from core.memory import WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory

PASS_BAR = "all four memory subsystems operational"

def test_working_memory():
    wm = WorkingMemory(capacity=10, state_dim=4, gru_dim=8)
    for i in range(15):
        wm.push(np.array([i]*4, np.float32), np.array([i]*8, np.float32), i % 3, float(i), i)
    seq = wm.get_state_sequence()
    assert len(seq) == 10, f"expected 10 (capacity), got {len(seq)}"
    ctx = wm.get_recent_context(5)
    assert len(ctx["states"]) == 5, f"expected 5 recent, got {len(ctx['states'])}"
    print(f"  WorkingMemory: OK ({len(wm)} items, recent={len(ctx['states'])})")

def test_episodic_memory():
    em = EpisodicMemory(max_episodes=10)
    for ep_i in range(8):
        states = np.random.randn(20, 4).astype(np.float32)
        hiddens = np.random.randn(20, 8).astype(np.float32)
        actions = np.random.randint(0, 5, size=20).astype(np.int32)
        rewards = np.random.randn(20).astype(np.float32) * (ep_i + 1)
        em.store_episode(states, hiddens, actions, rewards, {"ep_type": ep_i % 2})
    assert len(em) == 8, f"expected 8 episodes, got {len(em)}"
    recent = em.recall_recent(3)
    assert len(recent) == 3
    salient = em.recall_by_salience(2)
    assert len(salient) == 2
    assert salient[0]["total_return"] >= salient[1]["total_return"]
    by_tag = em.recall_by_tag("ep_type", 0)
    assert len(by_tag) > 0
    sim = em.recall_by_similarity(np.random.randn(8))
    assert len(sim) > 0
    print(f"  EpisodicMemory: OK ({len(em)} episodes, salient={salient[0]['total_return']:.2f})")

def test_semantic_memory():
    sm = SemanticMemory(n_concepts=8, embed_dim=16)
    for _ in range(100):
        h = np.random.randn(16).astype(np.float32)
        h2 = np.random.randn(16).astype(np.float32)
        sm.update_concepts(h.reshape(1, -1))
        sm.observe_transition(h, np.random.randint(0, 5), h2)
    c = sm.assign_concept(np.random.randn(16))
    assert 0 <= c < 8
    tm = sm.transition_matrix()
    assert tm.shape[0] == 8
    state = sm.get_state()
    assert state["n_concepts"] == 8
    print(f"  SemanticMemory: OK ({state['n_transitions']} transitions, {len(sm.transition_counts)} schemas)")

def test_procedural_memory():
    pm = ProceduralMemory(max_skills=5)
    for i in range(3):
        pm.store_skill(f"skill_{i}", np.random.randn(8), np.random.randn(8),
                       np.random.randint(0, 5, size=10))
    skills = pm.list_skills()
    assert len(skills) == 3
    retrieved = pm.retrieve_skill(np.random.randn(8))
    assert retrieved is not None
    assert "name" in retrieved
    print(f"  ProceduralMemory: OK ({len(skills)} skills, retrieved={retrieved['name']})")

if __name__ == "__main__":
    print("verify_memory.py — Memory System Verification")
    test_working_memory()
    test_episodic_memory()
    test_semantic_memory()
    test_procedural_memory()
    print(f"\nPASS: {PASS_BAR}")
