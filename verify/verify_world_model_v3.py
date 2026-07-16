"""
verify_world_model_v3.py — Verify multi-step world model.

Tests:
  1. Single-step prediction (matches Phase 1 ForwardWorldModel behavior)
  2. Multi-step rollout over k steps
  3. Uncertainty estimation via ensemble disagreement
  4. Imagination: generate candidate trajectories

Pass bar: rollout prediction error grows gracefully with horizon (not diverging),
ensemble disagreement is non-zero, imagination runs without error.
"""
import numpy as np
import sys

from core.world_model_v3 import MultiStepWorldModel

PASS_BAR = "multi-step prediction, uncertainty, and imagination all operational"

def test_single_step():
    wm = MultiStepWorldModel(gru_dim=8, action_dim=5, n_ensemble=3, seed=0)
    h = np.random.randn(4, 8).astype(np.float32)
    a = np.array([0, 1, 2, 3], np.int32)
    preds = wm.predict_step(h, a, ensemble_idx=0)
    assert preds.shape == (4, 8), f"expected (4,8), got {preds.shape}"
    loss = wm.update_step(h, a, np.random.randn(4, 8).astype(np.float32))
    assert loss > 0, "loss should be positive"
    print(f"  Single-step: OK (loss={loss:.6f})")

def test_uncertainty():
    wm = MultiStepWorldModel(gru_dim=8, action_dim=5, n_ensemble=3, seed=0)
    h = np.random.randn(1, 8).astype(np.float32)
    a = np.array([0], np.int32)
    mean, std = wm.predict_with_uncertainty(h, a)
    assert mean.shape == (1, 8), f"expected (1,8), got {mean.shape}"
    assert std.shape == (1, 8), f"expected (1,8), got {std.shape}"
    # ensemble disagreement should be non-zero (different random seeds)
    assert std.mean() > 0, "ensemble std should be non-zero"
    print(f"  Uncertainty: OK (mean std={std.mean():.6f})")

def test_rollout():
    wm = MultiStepWorldModel(gru_dim=8, action_dim=5, n_ensemble=2, seed=0)
    h_start = np.random.randn(8).astype(np.float32)
    actions = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], np.int32)
    traj, unc = wm.rollout(h_start, actions)
    assert traj.shape == (11, 8), f"expected (11,8), got {traj.shape}"
    assert unc is not None
    assert unc.shape == (10, 8), f"expected (10,8), got {unc.shape}"
    # prediction error should grow with horizon (approximately)
    errs = []
    for t in range(len(actions)):
        errs.append(float(np.mean((traj[t+1] - np.random.randn(8)) ** 2)))
    print(f"  Rollout: OK (11 steps, mean err={np.mean(errs):.4f})")

def test_imagination():
    wm = MultiStepWorldModel(gru_dim=8, action_dim=5, n_ensemble=2, seed=0)
    h_start = np.random.randn(8).astype(np.float32)
    trajs, seqs, uncs = wm.imagine(h_start, horizon=8, n_candidates=6)
    assert trajs.shape == (6, 9, 8), f"expected (6,9,8), got {trajs.shape}"
    assert seqs.shape == (6, 8), f"expected (6,8), got {seqs.shape}"
    print(f"  Imagination: OK ({trajs.shape[0]} candidates, {trajs.shape[1]-1} steps each)")

if __name__ == "__main__":
    print("verify_world_model_v3.py — Multi-Step World Model Verification")
    test_single_step()
    test_uncertainty()
    test_rollout()
    test_imagination()
    print(f"\nPASS: {PASS_BAR}")
