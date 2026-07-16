"""
verify_language.py — Verify grounded vocabulary.

Tests:
  1. Learn word: token→referent binding
  2. Understand: token lookup returns grounded meaning
  3. Produce: referent lookup returns best token
  4. Phrase understanding: simple compositional semantics
  5. Auto-populate from concepts

Pass bar: vocabulary learns/retrieves word-referent mappings,
composes simple agent-verb-object phrases.
"""
import numpy as np
import sys

from core.language import GroundedVocabulary

PASS_BAR = "grounded vocabulary operational"

def test_learn_and_understand():
    vocab = GroundedVocabulary()
    vocab.learn_word("apple", "object", 1)
    word = vocab.understand("apple")
    assert word is not None, "should understand 'apple'"
    assert word.referent_type == "object"
    assert word.referent_id == 1
    print(f"  Learn/Understand: OK (apple -> object {word.referent_id})")

def test_produce():
    vocab = GroundedVocabulary()
    vocab.learn_word("up", "action", 0)
    vocab.learn_word("right", "action", 3)
    token = vocab.produce("action", 0)
    assert token == "up", f"expected 'up', got {token}"
    token = vocab.produce("action", 3)
    assert token == "right", f"expected 'right', got {token}"
    print(f"  Produce: OK (action 0 -> {vocab.produce('action', 0)}, action 3 -> {vocab.produce('action', 3)})")

def test_phrase():
    vocab = GroundedVocabulary()
    vocab.learn_word("agent", "entity", 0)
    vocab.learn_word("see", "action", 2)
    vocab.learn_word("apple", "object", 1)
    phrase = vocab.understand_phrase("agent see apple")
    assert phrase is not None
    assert phrase["agent"].token == "agent"
    assert phrase["verb"].token == "see"
    assert phrase["object"].token == "apple"
    print(f"  Phrase: OK (agent={phrase['agent'].token}, verb={phrase['verb'].token}, object={phrase['object'].token})")

def test_auto_populate():
    vocab = GroundedVocabulary()
    actions = ["up", "down", "left", "right", "interact"]
    # Mock concept formation
    class MockCF:
        def __init__(self):
            self._n_active = 4
            self._n_concepts = 8
        def n_active_concepts(self):
            return self._n_active
    vocab.learn_from_concepts(MockCF(), action_names=actions)
    assert vocab.vocabulary_size() >= len(actions), "should learn action words"
    print(f"  Auto-populate: OK (vocab size={vocab.vocabulary_size()}, actions={actions})")

if __name__ == "__main__":
    print("verify_language.py — Language Verification")
    test_learn_and_understand()
    test_produce()
    test_phrase()
    test_auto_populate()
    print(f"\nPASS: {PASS_BAR}")
