"""
verify_goals.py — Verify multi-drive goals system.

Tests:
  1. GoalWeights: initialization and normalization
  2. GoalDrivenReward: composite reward computation
  3. Weighted combination: individual contributions add up

Pass bar: goal weights normalize correctly, composite rewards are
weighted sums, individual contributions trackable.
"""
import numpy as np
import sys

from core.goals import GoalWeights, GoalDrivenReward

PASS_BAR = "multi-drive goals system operational"

def test_goal_weights():
    gw = GoalWeights()
    d = gw.as_dict()
    assert len(d) == 7, f"expected 7 goals, got {len(d)}"
    assert d["curiosity"] == 1.0
    gw.set(curiosity=2.0, safety=1.0)
    gw.normalize()
    d = gw.as_dict()
    total = sum(d.values())
    assert abs(total - 1.0) < 1e-6, f"normalized weights should sum to 1, got {total}"
    print(f"  GoalWeights: OK (7 goals, sum={total:.4f})")

def test_composite_reward():
    gw = GoalWeights()
    gw.set(curiosity=1.0, exploration=1.0, efficiency=0.0, safety=0.0,
           knowledge=0.0, survival=0.0, prediction=0.0)
    gw.normalize()
    gdr = GoalDrivenReward(gw)
    total, breakdown = gdr.compute(curiosity_r=0.8, exploration_r=0.6, safety_r=0.0)
    w_cur = gdr.weights.curiosity
    w_exp = gdr.weights.exploration
    expected = w_cur * 0.8 + w_exp * 0.6
    assert abs(total - expected) < 1e-6, f"expected {expected}, got {total}"
    assert "curiosity" in breakdown
    assert "exploration" in breakdown
    print(f"  Composite reward: OK (total={total:.4f}, breakdown keys={list(breakdown.keys())})")

def test_set_weights():
    gdr = GoalDrivenReward()
    gdr.set_weights(curiosity=2.0, safety=0.5, exploration=1.5)
    w = gdr.weights.as_dict()
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-6, f"weights should auto-normalize to 1, got {total}"
    print(f"  Set weights: OK (normalized sum={total:.4f})")

if __name__ == "__main__":
    print("verify_goals.py — Goals System Verification")
    test_goal_weights()
    test_composite_reward()
    test_set_weights()
    print(f"\nPASS: {PASS_BAR}")
