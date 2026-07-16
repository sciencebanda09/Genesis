"""
verify_reasoning.py — Verify symbolic reasoning engine.

Tests:
  1. Proposition creation and equality
  2. Tell/Ask: store and query facts
  3. Inference: derive new facts from rules
  4. Explanation: show reasoning chain

Pass bar: KB stores facts, inference generates new propositions,
explanation traces reasoning.
"""
import numpy as np
import sys

from core.reasoning import Proposition, ReasoningEngine

PASS_BAR = "symbolic reasoning operational"

def test_proposition():
    p = Proposition("agent", "at", "location_A", truth=0.9)
    assert p.subject == "agent"
    assert p.predicate == "at"
    assert p.object == "location_A"
    assert abs(p.truth - 0.9) < 1e-6
    p2 = Proposition("agent", "at", "location_A")
    assert p == p2
    print(f"  Proposition: OK ({p})")

def test_tell_ask():
    eng = ReasoningEngine()
    eng.tell(Proposition("agent", "at", "location_A", 1.0))
    eng.tell(Proposition("location_A", "has", "object_B", 0.8))
    val = eng.ask("agent", "at", "location_A")
    assert abs(val - 1.0) < 1e-6
    val2 = eng.ask("location_A", "has", "object_B")
    assert abs(val2 - 0.8) < 1e-6
    val3 = eng.ask("agent", "at", "unknown")
    assert abs(val3) < 1e-6
    print(f"  Tell/Ask: OK (3 facts, known={len(eng.kb)})")

def test_inference():
    eng = ReasoningEngine()
    eng.tell(Proposition("if", "agent_at_A", "sees_B", 1.0))
    eng.tell(Proposition("agent_at_A", "sees_B", 1.0))
    derived = eng.infer()
    assert len(derived) > 0, "should derive at least one proposition"
    print(f"  Inference: OK ({len(derived)} derived propositions)")

def test_explain():
    eng = ReasoningEngine()
    eng.tell(Proposition("if", "raining", "wet", 1.0))
    eng.tell(Proposition("raining", "true", None, 1.0))
    eng.infer()
    explanation = eng.explain("raining", "inferred_wet")
    assert len(explanation) > 0, "should have explanation chain"
    print(f"  Explanation: OK ({explanation})")

if __name__ == "__main__":
    print("verify_reasoning.py — Reasoning Verification")
    test_proposition()
    test_tell_ask()
    test_inference()
    test_explain()
    print(f"\nPASS: {PASS_BAR}")
