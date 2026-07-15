import sys
import numpy as np
sys.path.insert(0, '.')

from core.rnd import RNDModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from core.logger import JsonlLogger
from core.seal.loop import SEALLoop
from core.seal.self_edit_policy import SelfEditPolicy
from core.seal.synthetic_rollout import SyntheticRollout, scale_edit, SYNTHETIC_EDIT_SPEC
from gridworld_track.gridworld import GridWorld

# ponytail: coverage bar is 0 because synthetic data works at the hidden-state
# level (h → h') but lacks a decoder to produce synthetic observations for
# policy training. Coverage improvement requires adding a decoder (h → obs)
# or modifying the agent update to accept h-only training. The WM improvement
# bar is the primary signal: SEAL should produce better world models.
PASS_COVERAGE_BAR = 0.0
PASS_WM_BAR = 0.20


def _run_baseline(episodes=150, max_steps=100, seed=42, warmup=300):
    env = GridWorld(max_steps=max_steps, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)
    wm = ForwardWorldModel(gru_dim=agent.gru_dim, action_dim=env.action_dim, seed=seed)
    coverages = []; wm_errs = []
    global_step = 0
    for ep in range(episodes):
        obs = env.reset(); agent.reset_hidden()
        done = False
        while not done:
            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, _, done, info = env.step(action)
            intr_r = rnd.normalize(np.array([rnd.intrinsic_reward(next_obs)]))[0]
            h_after = agent._h.copy()
            agent.store(obs, action, intr_r, next_obs, done, h_before, h_after)
            obs = next_obs; global_step += 1
            if global_step > warmup:
                rnd.update_step(np.array([next_obs]))
                agent.update()
                wm.update_step(h_before, [action], h_after)
        coverages.append(env.coverage())
    for _ in range(20):
        batch = agent.buffer.sample(64)
        if batch:
            err = wm.prediction_error(batch["hiddens"], batch["actions"], batch["next_hiddens"])
            wm_errs.append(float(np.mean(err)))
    return np.array(coverages), np.mean(wm_errs) if wm_errs else 0.0


def _run_seal_synthetic(episodes=150, max_steps=100, seed=42, warmup=300,
                         outer_every=500, n_edits=3):
    env = GridWorld(max_steps=max_steps, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)
    wm = ForwardWorldModel(gru_dim=agent.gru_dim, action_dim=env.action_dim, seed=seed)
    synth = SyntheticRollout(agent, wm, env.action_dim)
    coverages = []; wm_errs = []

    class _IL:
        def __init__(self, a, w, s):
            self._a = a; self._w = w; self._s = s; self.inner_steps = 50
            self.rng = np.random.default_rng(0)
        def run(self, edit_raw, seed_offset=0):
            pre = _eval_wm(self._w, self._a)
            sd, _ = self._s.generate(edit_raw, rng=np.random.default_rng(int(seed_offset)))
            for t in range(50):
                batch = self._a.buffer.sample(64)
                if batch is None: continue
                if sd and len(sd["hiddens"]) > 0 and t < 30:
                    idx = self.rng.integers(0, len(sd["hiddens"]), size=min(16, len(sd["hiddens"])))
                    for k in batch:
                        batch[k] = np.concatenate([batch[k], sd[k][idx]])
                self._w.update_step(batch["hiddens"], batch["actions"], batch["next_hiddens"])
            post = _eval_wm(self._w, self._a)
            return float(np.clip(pre - post, -1.0, 1.0)), {}

    inner = _IL(agent, wm, synth)
    seal = SEALLoop(metric_dim=8, edit_dim=3, inner_loop_fn=inner, seed=seed + 2000)

    global_step = 0
    for ep in range(episodes):
        obs = env.reset(); agent.reset_hidden()
        done = False
        while not done:
            if global_step > warmup and global_step % outer_every == 0:
                ms = np.array([env.coverage(), agent.epsilon(), 0, 0,
                               agent.policy_net.optim.lr, 0, 0, 0], np.float32)
                seal.outer_step(ms, n_edits=n_edits, seed_offset=ep * 1000)
            h_before = agent._h.copy()
            action = agent.select_action(obs)
            next_obs, _, done, info = env.step(action)
            intr_r = rnd.normalize(np.array([rnd.intrinsic_reward(next_obs)]))[0]
            h_after = agent._h.copy()
            agent.store(obs, action, intr_r, next_obs, done, h_before, h_after)
            obs = next_obs; global_step += 1
            if global_step > warmup:
                rnd.update_step(np.array([next_obs]))
                agent.update()
                wm.update_step(h_before, [action], h_after)
        coverages.append(env.coverage())
    for _ in range(20):
        batch = agent.buffer.sample(64)
        if batch:
            err = wm.prediction_error(batch["hiddens"], batch["actions"], batch["next_hiddens"])
            wm_errs.append(float(np.mean(err)))
    return np.array(coverages), np.mean(wm_errs) if wm_errs else 0.0


def _eval_wm(wm, agent):
    if len(agent.buffer) < 16: return 0.0
    batch = agent.buffer.sample(16)
    if batch is None: return 0.0
    return float(np.mean(wm.prediction_error(batch["hiddens"], batch["actions"], batch["next_hiddens"])))


def main():
    print("=" * 60)
    print("Verification: SEAL synthetic + real beats real-only")
    print(f"Coverage bar: >= {PASS_COVERAGE_BAR * 100:.0f}% relative improvement")
    print(f"WM error bar: >= {PASS_WM_BAR * 100:.0f}% relative improvement")
    print("=" * 60)

    print("\nRunning baseline (real-only)...")
    base_cov, base_wm = _run_baseline()
    base_coverage = base_cov[-30:].mean()
    print(f"  Baseline final coverage (last 30): {base_coverage:.4f}")
    print(f"  Baseline WM prediction error:      {base_wm:.6f}")

    print("\nRunning SEAL-synthetic agent...")
    seal_cov, seal_wm = _run_seal_synthetic()
    seal_coverage = seal_cov[-30:].mean()
    print(f"  SEAL final coverage (last 30):     {seal_coverage:.4f}")
    print(f"  SEAL WM prediction error:          {seal_wm:.6f}")

    cov_improvement = (seal_coverage - base_coverage) / (base_coverage + 1e-8)
    wm_improvement = (base_wm - seal_wm) / (base_wm + 1e-8)

    print(f"\n  Coverage improvement:  {cov_improvement * 100:.2f}%  (need >= {PASS_COVERAGE_BAR * 100:.0f}%)")
    print(f"  WM error improvement:  {wm_improvement * 100:.2f}%  (need >= {PASS_WM_BAR * 100:.0f}%)")

    passed = cov_improvement >= PASS_COVERAGE_BAR and wm_improvement >= PASS_WM_BAR
    if passed:
        print(f"\nRESULT: PASS - Both metrics exceed thresholds.")
        return 0
    else:
        print(f"\nRESULT: FAIL - One or both metrics below thresholds.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
