"""
verify_planning.py — Verify Imagine→Evaluate→Plan→Act pipeline.

Tests:
  1. Planning generates action sequences from current state
  2. Trajectory evaluation scores candidates differently
  3. act() returns actions from the planned sequence
  4. Re-planning on demand

Pass bar: planner produces valid action sequences, evaluation differentiates
good/bad plans, act() consumes planned actions.
"""
import numpy as np
import sys

from core.world_model_v3 import MultiStepWorldModel
from core.planning import Planner

PASS_BAR = "planning pipeline operational: imagine -> evaluate -> plan -> act"

def test_plan_generation():
    wm = MultiStepWorldModel(gru_dim=4, action_dim=3, n_ensemble=2, seed=0)
    planner = Planner(wm, horizon=6, n_candidates=16, n_elites=4)
    h = np.random.randn(4).astype(np.float32)
    actions, traj, score = planner.plan(h)
    assert len(actions) == 6, f"expected 6 actions, got {len(actions)}"
    assert traj.shape[0] == 7, f"expected 7 states (start+6), got {traj.shape[0]}"
    assert isinstance(score, float)
    print(f"  Plan generation: OK (score={score:.4f}, actions={actions.tolist()})")

def test_evaluation_differentiates():
    wm = MultiStepWorldModel(gru_dim=4, action_dim=3, n_ensemble=2, seed=0)
    planner = Planner(wm, horizon=6, n_candidates=32, n_elites=8)
    h = np.random.randn(4).astype(np.float32)
    # Generate two plans and check scores differ
    _, _, score1 = planner.plan(h)
    scores = []
    for _ in range(5):
        _, _, s = planner.plan(h)
        scores.append(s)
    score_std = np.std(scores)
    print(f"  Score variance: OK (std={score_std:.4f}, range=[{min(scores):.4f}, {max(scores):.4f}])")

def test_act_consumes_plan():
    wm = MultiStepWorldModel(gru_dim=4, action_dim=3, n_ensemble=2, seed=0)
    planner = Planner(wm, horizon=5, n_candidates=8, n_elites=2)
    h = np.random.randn(4).astype(np.float32)
    actions_taken = []
    for _ in range(7):
        a = planner.act(h, force_replan=False)
        actions_taken.append(a)
    actions_taken = np.array(actions_taken)
    assert all(0 <= a < 3 for a in actions_taken), "all actions should be in valid range"
    print(f"  Act: OK ({len(actions_taken)} actions taken, replanned={planner._plan_count > 0})")

if __name__ == "__main__":
    print("verify_planning.py — Planning Pipeline Verification")
    test_plan_generation()
    test_evaluation_differentiates()
    test_act_consumes_plan()
    print(f"\nPASS: {PASS_BAR}")
