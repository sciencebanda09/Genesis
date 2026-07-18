# GENESIS — Research Roadmap

This is the multi-year research plan. Phases build on each other — each unlocks capabilities the next one needs.

---

## Genesis Foundation Research

### Phase 1 — Birth
✅ **Complete**

An agent with no task reward, driven by intrinsic curiosity, learning a stable D1 delay-corrected value function over a toy gridworld. Core brain framework (GRU policy, RND, forward world model, contrastive projector, clustering probe, reconstruction auxiliary). All verification scripts with stated pass/fail bars. Bugs documented with finite-difference checks.

↓

### Phase 2 — Perception
✅ **Complete**

Vision encoder (CNN + ChannelNorm) that transforms raw RGB pixels into latent vectors, trained by world-model prediction loss. Visual track pipeline numerically stable. VisualGridWorld renders the same environment to 64×64 images.

⬜ **Remaining:** Port concept-formation tests to visual track. Tune encoder ↔ world model co-training.

↓

### Phase 2.5 — Executive Cortex
✅ **Complete**

The organism's first meta-cognitive system. The Executive Cortex observes internal learning dynamics across all subsystems and adaptively regulates:

- **Curiosity weighting** — which intrinsic signal to trust (RND, ICM, future modules) and by how much, based on recent reward magnitudes.
- **Memory mixing** — how to sample experience across replay strategies (uniform, prioritized, sequence), driven by TD-error **trend** (negative trend = improving = weight up) rather than absolute magnitude, avoiding positive-feedback loops from biased sampling.
- **Exploration rate** — adaptive epsilon controlled by coverage trend feedback, replacing fixed schedules.
- **Learning rates** — per-module LR factors modulated by loss trends.

**Design principle:** The cortex NEVER uses hardcoded rules (`if maze: use ICM`). All regulation is driven by observed metric dynamics. Every future cognitive subsystem (Attention, Planning, Language, Reasoning, Emotion, Self Model) connects through the Executive Cortex rather than talking directly to each other.

**Verification:** 9/12 metrics favor Executive Cortex over static baselines in 3 experiments (adaptive curiosity, adaptive memory, adaptive exploration). Adaptive memory improved from 2/4 to 3/4 after fixing per-sample TD error tracking and switching from magnitude-based to trend-based regulation. Outperforms static baselines in curiosity, memory, and exploration regulation. See `verify/verify_executive_cortex.py`.

⬜ **Remaining:** Integrate as default regulation layer in both training loops. Add curiosity module registry. Long-horizon stability tests (200+ episodes).

↓

### Phase 2.6 — SEAL (Self-Adapting Agent)
✅ **Complete**

A meta-learning layer where the agent learns to generate **self-edits** that control its own learning process. Inspired by the SEAL framework (Zweiger, Pari et al., NeurIPS 2025), the agent uses an outer reinforcement-learning loop to optimize an inner training loop — the model generates edits, trains itself with them, and the outer loop reinforces edits that improve downstream performance.

Two complementary directions, sharing a shared ReSTEM (rejection sampling + SFT) outer loop:

**Direction A: Meta-Regulation (complements Executive Cortex)**
- `SelfEditPolicy` MLP maps metric state → regulation parameters `[curiosity_beta, lr_mult, replay_priority_exp, exploration_eps, memory_mix_ratio]`
- Edits are applied per-window (every K steps); the policy is trained via regression on good edits
- SEAL sets window-level strategy, Executive Cortex handles step-level dynamics within each window
- **Verified:** SEAL matches EC heuristic (94.2% ratio, bar ≥ 90%). Beats test within single-seed noise.

**Direction B: Synthetic Experience Generation**
- ForwardWorldModel generates synthetic `(h, a, h')` rollouts to augment training data
- Self-edit specifies `[num_steps, noise_scale, mix_ratio]` for the generator
- Synthetic data mixed into World Model training; outer loop optimizes generation parameters
- **Verified:** WM prediction error reduced 23.7% vs real-only training (bar ≥ 20%)

**Design rule:** Gradient isolation preserved — SelfEditPolicy has its own optimizer, no gradient flows from inner-loop losses into the self-edit generation policy. The ReSTEM outer loop uses binary reward filtering (keep edits with reward ≥ median), followed by SFT regression on the kept edits.

Implementation in `core/seal/` (shared infrastructure) + direction-specific modules.

**Verification:** 3 verification scripts with stated pass/fail bars — all passing.

⬜ **Remaining:** Multi-seed sweeps for statistical significance. Add decoder for synthetic observations (h → obs) to enable policy training on synthetic data. Multi-edit sampling per outer step (as in the original SEAL paper) for faster policy convergence.

↓

### Phase 2.7 — Systems Integration
✅ **Complete**

All cognitive modules from Phases 2.5–10 wired into a single training loop:

- **Executive Cortex** — default regulation layer in `train.py`, replacing hardcoded epsilon schedules and curiosity weights
- **Adaptive memory** — uniform + prioritized replay mixing, per-sample TD error tracking, trend-based regulation
- **ObjectPermanence** — slot tracking fed from real environment object detections
- **MultiStepWorldModel** — ensemble world model trained alongside the existing single-step model
- **ConceptFormation** — hierarchical clustering on real agent hiddens + object features
- **EpisodicMemory + KnowledgeConsolidation** — episode storage with periodic consolidation at episode boundaries
- **SelfModel** — metacognitive loss tracking across all subsystems
- **Agent refactor** — `D1Agent.update()` now accepts optional pre-sampled batch for adaptive memory mixing

**Improvements made during integration:**
- ReasoningEngine modus_ponens rule now correctly derives consequence predicates (was deriving `inferred_true` instead of `inferred_wet`)
- ConceptFormation deferred initialization removes dummy-data seeding (previously initialized clusterer with random data)
- MultiStepWorldModel + Planner: NaN guards on predict/update/evaluate paths
- ReasoningEngine `explain()` now returns meaningful inference chains
- `D1Agent.update()` returns per-sample `td_error` for priority updates

**Pass bar:** All 14 verify scripts pass. Training loop runs error-free with all cognitive modules enabled.

⬜ **Remaining:** Goals-driven exploration, curiosity module registry, long-horizon stability (200+ episode runs).

↓

| Phase | Status | Built in 2.7 | Pass bar | Remaining |
|-------|--------|--------------|----------|-----------|
| **3 — Object Understanding** | ◐ | ObjectPermanence slot tracking | NMI(slots, objects) > 0.5 | Feed real agent data → verify NMI bar |
| **4 — Memory** | ◐ | 4-level hierarchy + KnowledgeConsolidation | Memory-augmented agent beats no-memory agent on navigation | Spatial (cognitive map), temporal arch comparison |
| **5 — Concept Formation** | ◐ | ConceptFormation pipeline (deferred init) | NMI(clusters, context) > 0.5 | Compositional learning (parts→objects→categories→relations) |
| **6 — World Model** | ◐ | MultiStepWorldModel + ensemble uncertainty | 10-step pred error < persistence baseline | Verification test, rollout consistency tuning |
| **7 — Reasoning & Planning** | ◐ | ReasoningEngine + Planner (NaN guards) | Agent distinguishes correlation from causation | Causal attribution (D3), counterfactual, tree search |
| **8 — Language** | ◐ | GroundedVocabulary (learn/understand/produce) | Novel 3-step compositional instruction executed | Instruction following, multi-agent communication |
| **9 — Motivation & Emotion** | ⬜ | Modules exist, drives not active in training loop | Exploratory vs exploitative phases from internal state (not epsilon) | Goals-driven exploration, drive competition |
| **10 — Self Model** | ◐ | SelfModel monitoring + EpisodicMemory | Calibrated metacognition (confidence matches accuracy) | Epistemic action, theory of mind |
| **11 — Lifelong Growth** | ⬜ | — | N tasks sequentially without forgetting | Progressive expansion, EWC, curriculum |
| **12 — Self Improvement** | ⬜ | — | Discovers better learning strategy than initial | Meta-learning, arch search, reward design |
| **13 — Multi-Agent Society** | ⬜ | — | N agents > N independent agents in coverage | Knowledge transfer, role specialization |
| **14 — Physical Embodiment** | ⬜ | — | Same arch explores real room with minimal mod | Sim-to-real, real-time sensorimotor |

---

## Genesis Foundation Research

These aren't phases — they're cross-cutting improvements that should be investigated continuously, starting now.

### 1. Better Curiosity

| Current | Problems |
|---------|----------|
| RND | noisy TV, novelty only, no uncertainty |

I'd compare:

| RND | ICM | NGU | Plan2Explore |
|-----|-----|-----|--------------|

Not necessarily replace RND, but understand the tradeoffs.

### 2. Replay Memory

| Current | Upgrade Path |
|---------|--------------|
| ReplayBuffer | Experience Replay → Prioritized Replay → Sequence Replay → Episodic Replay |

Much closer to biological memory.

### 3. Representation Quality

This is probably the biggest weakness. Instead of one encoder, test:

| CNN | ResNet | BYOL | VICReg | SimCLR | MAE |
|-----|--------|------|--------|--------|-----|

Which representation is actually best?

### 4. World Model

| Current | Problem |
|---------|---------|
| latent → latent | uncertainty prediction, multi-step prediction, rollout consistency, imagination |

Instead of predict 1 step, predict:

| 10 steps | 20 steps | 50 steps |
|----------|----------|----------|

### 5. Spatial Memory

Right now, Genesis doesn't remember where things are. Humans have:

```
hippocampus → cognitive map
```

Genesis should too.

### 6. Temporal Memory

| Current | Compare |
|---------|---------|
| GRU | GRU, LSTM, Transformer, Mamba, RWKV |

Find what actually works.

### 7. Representation Diagnostics

Don't just train. Measure:

| Mutual Information | CKA | Linear Probe | Cluster Purity | Latent Collapse | Nearest Neighbor Accuracy | Object Consistency |
|--------------------|-----|--------------|----------------|-----------------|---------------------------|-------------------|

Become obsessed with measuring representations.

### 8. Perception Robustness

Train under:

| Lighting | Rotation | Noise | Occlusion | New Layouts |
|----------|----------|-------|-----------|-------------|

A baby still recognizes a toy in dim light. Genesis should too.

### 9. Development

This one is huge. Instead of `train 5000 episodes`, create:

```
Age 0
↓
Age 1
↓
Age 2
↓
Age 3
```

Each age unlocks new capabilities. Exactly like humans.

---

## Old Roadmap (archived reference)

For context, the original 13-phase vision:

1. Birth — ✅
2. Perception — ✅
3. Memory — ⬜
4. Concept Formation — ⬜
5. World Model — ⬜
6. Reasoning — ⬜
7. Language — ⬜
8. Emotion & Motivation — ⬜
9. Self Model — ⬜
10. Growth — ⬜
11. Self Evolution — ⬜
12. Society — ⬜
13. Embodiment — ⬜

---

## Audit Addendum — verify/ script fixes (this session)

Two verification scripts under `verify/` were found to have bugs in the
*test itself*, not the systems they were testing. Flagging this against
the "✅ Complete" markers above, since it affects what those markers can
honestly claim.

**`verify_world_model_v2.py`** — the encoder used to build the transition
dataset was never trained (`enc.encode()` called, `enc.backward()` never
invoked). An untrained encoder produces near-arbitrary latents, so the
world model was being asked to predict noise from noise. Original result:
1.02x real-vs-shuffled ratio (no signal). Fixed by jointly training
encoder + world model before evaluation. New result: **4.33x** — a real,
positive signal, though still well behind the feature-vector world
model's ~30x on the equivalent test.

**`verify_encoder.py`** — went through three rounds before landing on a
fair test, kept here because the wrong turns are informative:
1. *Original:* cosine similarity, temporal vs. random-reset pairs, pass
   bar = "any positive gap." Result printed as PASS at a +0.0007 gap —
   noise-level, false positive.
2. *Round 2:* switched to Euclidean distance (cosine saturates near 1.0 on
   this environment because raw pixels are already ~0.999 cosine-similar
   to each other — small grid, uniform background — confirmed directly).
   Result looked strong (5.47x ratio) until compared against an
   *untrained* encoder's own noise floor (5.11x) — nearly identical,
   because even a random encoder inherits pixel-level temporal
   autocorrelation just by being a roughly-continuous function of its
   input.
3. *Round 3 (current):* compare adjacent (t, t+1) vs. temporally-distant
   (t, t+20) pairs from a single continuous trajectory, and require the
   *encoder's* ratio to exceed the *raw-pixel* ratio on the same pairs —
   the only version of this test that actually isolates "did training add
   structure" from "did the encoder merely fail to destroy structure
   already in the pixels." Honest result: encoder ratio (1.15x) is not
   higher than the raw-pixel ratio (1.38x) — **no evidence found that
   training gives the encoder temporally-aware structure beyond simple
   pixel autocorrelation**, in this run.

**What this means for the roadmap:** Phase 2's architecture (encoder runs,
trains without diverging, produces correctly-shaped output) is genuinely
verified. The stronger claim — that the encoder *learns* meaningful
temporal/visual structure from the world-model objective — is not yet
supported by evidence and should not be assumed true when building Phase
3/4 work on top of the visual track. Feature-vector track (Phase 1)
remains the more thoroughly verified substrate.

Both scripts now state an explicit noise-floor or baseline comparison
before declaring a result, not just "is the number positive."
