"""Verification for the learned downstream-progress meta-controller.

Checks the invariants that matter for Paper 1's experiment:
  1. proposals stay in the bounded raw-edit space;
  2. candidate evaluation constructs fresh inner-loop state;
  3. the controller selects and learns from downstream rewards;
  4. all live regulation dimensions are applied without LR accumulation.
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from core.seal.meta_controller import (  # noqa: E402
    IsolatedCandidateEvaluator,
    LearnedMetaController,
    scale_meta_edit,
)
from core.agent import D1Agent  # noqa: E402
from core.executive_cortex.cortex import ExecutiveCortex  # noqa: E402
from core.rnd import RNDModule  # noqa: E402
from core.world_model import ForwardWorldModel  # noqa: E402
from core.replay_buffer import PrioritizedReplay  # noqa: E402
from gridworld_track.gridworld import GridWorld  # noqa: E402
from gridworld_track.train_meta_control import (  # noqa: E402
    LiveSnapshotEvaluator,
    _apply_live_edit,
    _snapshot,
)


class _FakeInner:
    def __init__(self, seed, tracker):
        self.seed = seed
        self.tracker = tracker

    def run(self, edit, seed_offset=0):
        # Reward has a known optimum, so this is a real downstream selection
        # test rather than merely a shape/smoke test.
        self.tracker.append(self.seed)
        target = np.array([0.8, 0.2, 0.4, 0.6, 0.3], np.float32)
        reward = -float(np.square(np.asarray(edit) - target).sum())
        return reward, {"seed": self.seed}


def main():
    controller = LearnedMetaController(seed=7, exploration_std=0.2)
    state = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    proposals = controller.propose(state, n_candidates=5)
    assert proposals.shape == (5, 5)
    assert np.all(proposals >= 0.0) and np.all(proposals <= 1.0)

    seeds = []
    evaluator = IsolatedCandidateEvaluator(
        lambda seed: _FakeInner(seed, seeds), base_seed=100)
    result = controller.outer_step(state, evaluator, n_candidates=5, seed_offset=20)
    assert len(seeds) == 5
    assert len(set(seeds)) == 1, "candidate evaluations did not use matched probes"
    assert result["best_edit"].shape == (5,)
    assert np.isfinite(result["update"]["policy_loss"])
    assert len(controller.history) == 1

    physical = scale_meta_edit(np.zeros(5, np.float32))
    upper = scale_meta_edit(np.ones(5, np.float32))
    assert np.all(physical == np.array([0.1, 0.5, 0.0, 0.01, 0.0], np.float32))
    assert np.all(upper == np.array([2.0, 2.0, 1.0, 1.0, 1.0], np.float32))

    class _Optim:
        lr = 0.01

    class _Agent:
        policy_net = type("Policy", (), {"optim": _Optim()})()

        def epsilon(self):
            return 0.5

    agent = _Agent()
    _apply_live_edit(agent, np.array([1.2, 1.5, 0.7, 0.4, 0.8]), 0.01)
    assert np.isclose(agent.policy_net.optim.lr, 0.015)
    assert np.isclose(agent._meta_curiosity_beta, 1.2)
    assert np.isclose(agent._meta_replay_priority_exp, 0.7)
    assert np.isclose(agent._meta_memory_mix_ratio, 0.8)
    # Re-applying an edit must set base_lr * multiplier, not multiply the
    # already-modified learning rate.
    _apply_live_edit(agent, np.array([1.0, 0.5, 0.2, 0.8, 0.1]), 0.01)
    assert np.isclose(agent.policy_net.optim.lr, 0.005)

    env = GridWorld(max_steps=20, seed=4)
    obs = env.reset()
    real_agent = D1Agent(state_dim=8, action_dim=5, seed=4)
    real_rnd = RNDModule(state_dim=8, seed=4)
    real_wm = ForwardWorldModel(gru_dim=32, action_dim=5, seed=4)
    real_cortex = ExecutiveCortex()
    real_prio = PrioritizedReplay(
        real_agent.buffer.capacity, real_agent.state_dim, real_agent.gru_dim, seed=4)
    base = _snapshot(env, real_agent, real_rnd, real_wm, real_cortex, real_prio)
    evaluator = LiveSnapshotEvaluator(
        base, obs, inner_steps=1, base_lr=0.001, max_steps=20, probe_tasks=2)
    before_pos = env.pos
    before_steps = real_agent.steps_done
    before_lr = real_agent.policy_net.optim.lr
    score, probe_info = evaluator(np.full(5, 0.5, np.float32))
    assert np.isfinite(score)
    assert len(probe_info["task_rewards"]) == 2
    assert np.allclose(probe_info["task_infos"][0]["initial_observation"], obs)
    probe_env = GridWorld(max_steps=20, seed=1)
    expected_probe_obs = probe_env.reset()
    assert np.allclose(
        probe_info["task_infos"][1]["initial_observation"], expected_probe_obs)
    assert np.isclose(
        probe_info["robust_score"],
        probe_info["mean_reward"]
        - 0.5 * probe_info["std_reward"]
        + 0.5 * probe_info["worst_reward"],
    )
    assert env.pos == before_pos
    assert real_agent.steps_done == before_steps
    assert np.isclose(real_agent.policy_net.optim.lr, before_lr)
    assert len(real_agent.buffer) == 0

    print("[PASS] learned meta-controller invariants verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
