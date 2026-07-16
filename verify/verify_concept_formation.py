"""
verify_concept_formation.py — Verify hierarchical concept formation.

Tests:
  1. Embedding from raw observations
  2. Concept assignment produces valid cluster indices
  3. Concept updates and frequency tracking
  4. Object-aware embedding when ObjectPermanence slots provided

Pass bar: embeddings convert observations to concept space, assignments
are valid cluster IDs, concept frequencies sum to 1.
"""
import numpy as np
import sys

from core.concept_formation import ConceptFormation

PASS_BAR = "hierarchical concept formation operational"

def test_raw_embedding():
    cf = ConceptFormation(embed_dim=8, n_concepts=4, use_objects=False)
    obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], np.float32)
    emb, meta = cf.embed(obs)
    assert emb.shape == (8,), f"expected (8,), got {emb.shape}"
    assert meta["source"] == "raw"
    print(f"  Raw embedding: OK (shape={emb.shape})")

def test_padded_embedding():
    cf = ConceptFormation(embed_dim=16, n_concepts=4, use_objects=False)
    obs = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    emb, meta = cf.embed(obs)
    assert emb.shape == (16,), f"expected (16,), got {emb.shape}"
    print(f"  Padded embedding: OK (shape={emb.shape})")

def test_concept_assignment():
    cf = ConceptFormation(embed_dim=8, n_concepts=4, use_objects=False)
    for _ in range(50):
        emb, _ = cf.embed(np.random.randn(8))
        c = cf.update(emb)
        assert 0 <= c < 4, f"concept {c} out of range"
    freqs = cf.concept_frequency()
    assert freqs.shape == (4,), f"expected (4,), got {freqs.shape}"
    assert abs(freqs.sum() - 1.0) < 1e-6, "frequencies should sum to 1"
    n_active = cf.n_active_concepts()
    assert n_active > 0, "at least one concept should be active"
    print(f"  Concept assignment: OK ({n_active}/{cf.n_concepts} active, freqs={freqs})")

def test_object_aware_embedding():
    cf = ConceptFormation(embed_dim=8, n_concepts=4, use_objects=True)
    obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], np.float32)
    # Mock object slots
    class MockSlot:
        def __init__(self, features):
            self.features = features
    slots = [MockSlot(np.random.randn(8).astype(np.float32)) for _ in range(3)]
    emb, meta = cf.embed(obs, object_slots=slots)
    assert emb.shape == (8,), f"expected (8,), got {emb.shape}"
    assert meta["source"] == "objects"
    assert meta["n_objects"] == 3
    print(f"  Object-aware embedding: OK (source={meta['source']}, n_objects={meta['n_objects']})")

if __name__ == "__main__":
    print("verify_concept_formation.py — Concept Formation Verification")
    test_raw_embedding()
    test_padded_embedding()
    test_concept_assignment()
    test_object_aware_embedding()
    print(f"\nPASS: {PASS_BAR}")
