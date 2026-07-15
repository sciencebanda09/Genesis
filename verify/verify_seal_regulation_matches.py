import sys
import numpy as np
sys.path.insert(0, '.')

from core.rnd import RNDModule
from core.agent import D1Agent
from core.world_model import ForwardWorldModel
from core.executive_cortex.cortex import ExecutiveCortex
from core.seal.self_edit_policy import SelfEditPolicy
from core.seal.regulation import scale_edit, REGULATION_EDIT_SPEC
from gridworld_track.gridworld import GridWorld

PASS_BAR = 0.90
INNER_STEPS = 50


def _run_ec(episodes=150, max_steps=100, seed=42, warmup=300):
    env = GridWorld(max_steps=max_steps, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)
    wm = ForwardWorldModel(gru_dim=agent.gru_dim, action_dim=env.action_dim, seed=seed)
    cortex = ExecutiveCortex()
    coverages = []; gs = 0
    for _ in range(episodes):
        obs = env.reset(); agent.reset_hidden()
        done = False
        while not done:
            hb = agent._h.copy(); a = agent.select_action(obs)
            ns, _, d, _ = env.step(a)
            ir = rnd.normalize(np.array([rnd.intrinsic_reward(ns)]))[0]
            ha = agent._h.copy()
            cw = cortex.curiosity_weights.get("rnd", 1.0)
            agent.store(obs, a, cw*ir, ns, d, hb, ha)
            obs = ns; gs += 1
            if gs > warmup:
                rnd.update_step(np.array([ns])); agent.update(); wm.update_step(hb, [a], ha)
                cortex.observe(global_step=gs, coverage=env.coverage(), rnd_reward=ir,
                               td_error_mean=0.0, wm_loss=0.0)
            if d: break
        coverages.append(env.coverage())
    return np.array(coverages)


def _run_seal(episodes=150, max_steps=100, seed=42, warmup=300):
    env = GridWorld(max_steps=max_steps, seed=seed)
    rnd = RNDModule(state_dim=env.state_dim, seed=seed)
    agent = D1Agent(state_dim=env.state_dim, action_dim=env.action_dim, seed=seed)
    wm = ForwardWorldModel(gru_dim=agent.gru_dim, action_dim=env.action_dim, seed=seed)
    cortex = ExecutiveCortex()
    coverages = []
    policy = SelfEditPolicy(metric_dim=8, edit_dim=5, seed=seed + 1000)
    buf_ms, buf_edits = [], []
    gs = 0
    window_start_cov = env.coverage()
    current_edit = policy.generate(np.zeros(8, np.float32))
    scaled_edit = scale_edit(current_edit, REGULATION_EDIT_SPEC)
    agent.policy_net.optim.lr *= np.clip(scaled_edit[1], 0.5, 2.0)
    seal_beta = float(scaled_edit[0])

    for ep in range(episodes):
        obs = env.reset(); agent.reset_hidden()
        done = False
        while not done:
            if gs > warmup and gs % INNER_STEPS == 0 and gs > 0:
                cov = env.coverage()
                reward = cov - window_start_cov
                ms = SelfEditPolicy.get_metric_state(None, cortex, agent, env, wm, rnd)
                if reward >= 0:
                    buf_ms.append(ms.copy()); buf_edits.append(current_edit.copy())
                if len(buf_ms) >= 4 and len(buf_ms) % 2 == 0:
                    policy.train_on_edits(np.array(buf_ms[-30:]), np.array(buf_edits[-30:]))
                current_edit = policy.generate(ms)
                scaled_edit = scale_edit(current_edit, REGULATION_EDIT_SPEC)
                agent.policy_net.optim.lr = 1e-3 * np.clip(scaled_edit[1], 0.5, 2.0)
                seal_beta = float(scaled_edit[0])
                window_start_cov = cov

            hb = agent._h.copy(); a = agent.select_action(obs)
            ns, _, d, _ = env.step(a)
            ir = rnd.normalize(np.array([rnd.intrinsic_reward(ns)]))[0]
            ha = agent._h.copy()
            agent.store(obs, a, seal_beta * ir, ns, d, hb, ha)
            obs = ns; gs += 1
            if gs > warmup:
                rnd.update_step(np.array([ns])); agent.update(); wm.update_step(hb, [a], ha)
                cortex.observe(global_step=gs, coverage=env.coverage(), rnd_reward=ir,
                               td_error_mean=0.0, wm_loss=0.0)
            if d: break
        coverages.append(env.coverage())
    return np.array(coverages)


def main():
    print("=" * 60)
    print("Verification: SEAL matches EC heuristic coverage")
    print("=" * 60)
    print("Running EC...")
    ec = _run_ec(); ec_f = ec[-30:].mean()
    print(f"  EC final: {ec_f:.4f}")
    print("Running SEAL...")
    se = _run_seal(); se_f = se[-30:].mean()
    print(f"  SEAL final: {se_f:.4f}")
    ratio = se_f / (ec_f + 1e-8)
    print(f"\n  SEAL/EC ratio: {ratio:.4f}  (need >= {PASS_BAR})")
    if ratio >= PASS_BAR:
        print("\nRESULT: PASS"); return 0
    else:
        print("\nRESULT: FAIL"); return 1

if __name__ == "__main__":
    sys.exit(main())
