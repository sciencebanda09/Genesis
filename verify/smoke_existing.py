"""
smoke_existing.py — Quick smoke test that existing systems still work
after adding new modules and editing ExecutiveCortex.
"""
import sys
import numpy as np

# Test all new imports
from core.memory import WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
from core.consolidation import KnowledgeConsolidation
from core.object_permanence import ObjectPermanence
from core.world_model_v3 import MultiStepWorldModel
from core.planning import Planner
from core.concept_formation import ConceptFormation
from core.goals import GoalWeights, GoalDrivenReward
from core.reasoning import ReasoningEngine, Proposition
from core.self_model import SelfModel
from core.language import GroundedVocabulary, GroundedWord
from core.executive_cortex import ExecutiveCortex, MetricBuffer
print("All imports OK")

# Test ExecutiveCortex still works with extended goals
cortex = ExecutiveCortex()
cortex.observe(coverage=0.5, td_error_mean=0.1, wm_loss=0.05, rnd_reward=0.3, global_step=100)
params = cortex.regulate()
assert 'goal_weights' in params
g = params['goal_weights']
assert isinstance(g, dict)
assert len(g) >= 5  # at minimum original drives + new ones
print(f"ExecutiveCortex regulate OK: {len(g)} goals, epsilon={cortex.epsilon:.3f}")

# Test MetricBuffer still works
mb = MetricBuffer(maxlen=50)
for i in range(100):
    mb.append(float(i))
assert abs(mb.mean() - 74.5) < 1.0
print(f"MetricBuffer OK: mean={mb.mean():.1f}, trend={mb.trend():.3f}")

# Test ForwardWorldModel still works (existing import from core.world_model)
from core.world_model import ForwardWorldModel
wm = ForwardWorldModel(gru_dim=8, action_dim=5)
h = np.random.randn(2, 8).astype(np.float32)
a = np.array([0, 1], np.int32)
pred = wm.predict(h, a)
assert pred.shape == (2, 8)
loss = wm.update_step(h, a, np.random.randn(2, 8).astype(np.float32))
print(f"ForwardWorldModel OK: loss={loss:.6f}")

# Test D1Agent still works (basic import and construction test)
from core.agent import D1Agent
agent = D1Agent(state_dim=8, action_dim=5)
print(f"D1Agent OK: gru_dim={agent.gru_dim}, buffer={len(agent.buffer)}")

# Test that MultiStepWorldModel and ForwardWorldModel coexist
mwm = MultiStepWorldModel(gru_dim=8, action_dim=5, n_ensemble=2)
mwm_loss = mwm.update_step(h, a, np.random.randn(2, 8).astype(np.float32))
print(f"MultiStepWorldModel OK: loss={mwm_loss:.6f}")

# Test contrastive projector still works
from core.contrastive import ContrastiveProjector
proj = ContrastiveProjector(gru_dim=8)
z = proj.embed(np.random.randn(4, 8))
assert z.shape == (4, 16)
print(f"ContrastiveProjector OK: embed shape={z.shape}")

# Test RNDModule
from core.rnd import RNDModule
rnd = RNDModule(state_dim=8)
r = rnd.intrinsic_reward(np.random.randn(8))
print(f"RNDModule OK: reward={r:.4f}")

# Test OnlineKMeans
from core.clustering import OnlineKMeans
km = OnlineKMeans(n_clusters=4, embed_dim=8)
dummy = np.random.randn(10, 8).astype(np.float32)
km.update_step(dummy)
labels = km.assign(np.random.randn(1, 8).astype(np.float32))
print(f"OnlineKMeans OK: labels={labels}")

# Test reconstruction
from core.recon_auxiliary import ReconstructionAuxiliary
ra = ReconstructionAuxiliary(gru_dim=8, obs_dim=4)
loss = ra.update_step(np.random.randn(4, 8).astype(np.float32), np.random.randn(4, 4).astype(np.float32))
print(f"ReconstructionAuxiliary OK: loss={loss:.6f}")

# Test logger
from core.logger import JsonlLogger
logger = JsonlLogger("logs/smoke_test.jsonl")
logger.log_step(episode=1, step=0, global_step=0, obs=np.zeros(8), action=0,
                action_name="up", extrinsic_reward=0.0, intrinsic_reward=0.0, done=False)
logger.log_update(0, {"td_error": 0.1})
logger.close()
print(f"Logger OK: wrote smoke_test.jsonl")

print("\n=== ALL SMOKE TESTS PASSED ===")
