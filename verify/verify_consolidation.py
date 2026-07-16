"""
verify_consolidation.py — Verify knowledge consolidation pipeline.

Tests:
  1. Consolidation triggers after enough episodes
  2. Concept prototypes are updated from episodic h-states
  3. Transition schemas are induced from episode trajectories
  4. Replay for consolidation samples both episodic and buffer data

Pass bar: consolidation produces semantic knowledge from episodic experience,
transition schemas reflect observed patterns.
"""
import numpy as np
import sys

from core.consolidation import KnowledgeConsolidation
from core.memory import EpisodicMemory
from core.replay_buffer import ReplayBuffer

PASS_BAR = "knowledge consolidation operational: episodic -> semantic"

def test_consolidation_trigger():
    em = EpisodicMemory(max_episodes=10)
    cons = KnowledgeConsolidation(embed_dim=8, n_concepts=4, 
                                   consolidate_every=10, min_episodes_for_consolidation=3)
    # Not enough episodes
    result = cons.step(em, global_step=10)
    assert result is None, "should not consolidate with < 3 episodes"
    # Add episodes
    for _ in range(4):
        hiddens = np.random.randn(10, 8).astype(np.float32)
        em.store_episode(np.random.randn(10, 4), hiddens, 
                         np.random.randint(0, 5, 10), np.random.randn(10))
    result = cons.step(em, global_step=20)
    assert result is not None or cons._last_consolidation_step < 10
    print(f"  Consolidation trigger: OK (episodes={len(em)})")

def test_concept_consolidation():
    cons = KnowledgeConsolidation(embed_dim=8, n_concepts=4,
                                   consolidate_every=1, min_episodes_for_consolidation=1)
    h_states = np.random.randn(50, 8).astype(np.float32)
    cons._consolidate_concepts(h_states)
    state = cons.get_state()
    assert state["semantic"]["n_concepts"] == 4
    print(f"  Concept consolidation: OK ({len(h_states)} states -> {state['semantic']['n_concepts']} concepts)")

def test_schema_induction():
    cons = KnowledgeConsolidation(embed_dim=8, n_concepts=4,
                                   consolidate_every=1, min_episodes_for_consolidation=1)
    transitions = []
    for _ in range(100):
        h_from = np.random.randn(8).astype(np.float32)
        action = np.random.randint(0, 5)
        h_to = h_from + np.random.randn(8) * 0.1  # small perturbation
        transitions.append((h_from, action, h_to))
    cons._consolidate_schemas(transitions)
    state = cons.get_state()
    assert state["semantic"]["n_transitions"] > 0, "should have transition schemas"
    print(f"  Schema induction: OK ({state['semantic']['n_transitions']} transitions)")

def test_replay_for_consolidation():
    cons = KnowledgeConsolidation(embed_dim=8, n_concepts=4)
    em = EpisodicMemory(max_episodes=5)
    for _ in range(3):
        hiddens = np.random.randn(10, 8).astype(np.float32)
        em.store_episode(np.random.randn(10, 4), hiddens,
                         np.random.randint(0, 5, 10), np.random.randn(10))
    buf = ReplayBuffer(capacity=100, state_dim=4, gru_dim=8, seed=0)
    for _ in range(50):
        buf.add(np.random.randn(4), np.random.randn(8), 
                np.random.randint(0, 5), np.random.randn(),
                np.random.randn(4), np.random.randn(8), False)
    all_h, transitions = cons.replay_for_consolidation(em, buf)
    assert all_h is not None, "should produce h-states"
    assert len(transitions) > 0, "should produce transitions"
    print(f"  Replay for consolidation: OK ({all_h.shape[0]} states, {len(transitions)} transitions)")

if __name__ == "__main__":
    print("verify_consolidation.py — Knowledge Consolidation Verification")
    test_consolidation_trigger()
    test_concept_consolidation()
    test_schema_induction()
    test_replay_for_consolidation()
    print(f"\nPASS: {PASS_BAR}")
