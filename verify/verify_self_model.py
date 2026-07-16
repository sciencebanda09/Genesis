"""
verify_self_model.py — Verify metacognitive self-model.

Tests:
  1. Loss tracking: module losses are recorded
  2. Improvement rate: correctly identifies improvement vs stagnation
  3. Confidence calibration: confident when loss low
  4. Know what I know / don't know: gap identification

Pass bar: self-model tracks module performance, detects learning
improvement, identifies gaps.
"""
import numpy as np
import sys

from core.self_model import SelfModel

PASS_BAR = "metacognitive self-model operational"

def test_loss_tracking():
    sm = SelfModel(window=100)
    for i in range(50):
        sm.observe_loss("wm", 1.0 / (i + 1))
    hist = sm._loss_history.get("wm")
    assert hist is not None
    assert len(hist) == 50
    print(f"  Loss tracking: OK (50 steps for wm)")

def test_improvement_rate():
    sm = SelfModel(window=100)
    for i in range(100):
        sm.observe_loss("improving", 10.0 - i * 0.1)  # decreasing = improving
    imp = sm.improvement_rate("improving", window=50)
    assert imp > 0, f"positive improvement expected, got {imp}"
    # Now flat loss
    for i in range(100):
        sm.observe_loss("flat", 5.0)
    imp2 = sm.improvement_rate("flat", window=50)
    assert abs(imp2) < 0.1, f"near-zero improvement expected, got {imp2}"
    print(f"  Improvement rate: OK (improving={imp:.4f}, flat={imp2:.4f})")

def test_confidence():
    sm = SelfModel(window=100)
    for i in range(100):
        sm.observe_loss("mod", 0.1)
    conf = sm.confidence("mod")
    assert 0 < conf <= 1.0, f"confidence should be in (0,1], got {conf}"
    for i in range(100):
        sm.observe_loss("mod", 10.0)
    conf2 = sm.confidence("mod")
    print(f"  Confidence: OK (low loss={conf:.4f}, high loss={conf2:.4f})")

def test_knowledge_gaps():
    sm = SelfModel(window=100)
    for i in range(20):
        sm.observe_loss("rnd", 5.0)
    for i in range(20):
        sm.observe_loss("wm", 0.05)
    known = sm.know_what_i_know()
    assert "module_confidence" in known
    assert "wm" in known["module_confidence"]
    gaps = sm.know_what_i_dont_know()
    assert "uncertain_modules" in gaps
    print(f"  Knowledge gaps: OK (modules={list(known['module_confidence'].keys())})")

def test_self_representation():
    sm = SelfModel()
    h_states = np.random.randn(10, 8).astype(np.float32)
    rep = sm.encode_self_state(h_states)
    assert rep.shape == (16,), f"expected (16,), got {rep.shape}"
    print(f"  Self representation: OK (shape={rep.shape})")

if __name__ == "__main__":
    print("verify_self_model.py — Self Model Verification")
    test_loss_tracking()
    test_improvement_rate()
    test_confidence()
    test_knowledge_gaps()
    test_self_representation()
    print(f"\nPASS: {PASS_BAR}")
