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

### Phase 3 — Object Understanding
⬜

Entities as persistent objects, not transient pixel clusters. The organism should recognize that an object moved, not just that a different pixel arrived.

- Object discovery via temporal coherence (objects move as wholes)
- Slot attention / object-centric representations
- Occlusion reasoning: an object behind a wall still exists
- Object permanence test: tracking through disappearance

**Pass bar:** Clusters in slot space correlate with ground-truth object identity at NMI > 0.5.

↓

### Phase 4 — Memory
⬜

Beyond the GRU's recurrent state — dedicated memory architecture.

- **Experience Replay → Prioritized Replay → Sequence Replay → Episodic Replay** — much closer to biological memory
- **Spatial Memory** — Genesis doesn't remember where things are. Humans have hippocampus → cognitive map. Genesis should too.
- **Temporal Memory** — Compare GRU vs LSTM vs Transformer vs Mamba vs RWKV. Find what actually works.
- Memory compression: abstract repeated patterns into schemas
- Memory consolidation: replay past experiences during "rest" periods

**Pass bar:** Agent with memory outperforms same-capacity agent without memory on a navigation task with a memorized layout.

↓

### Phase 5 — Concept Formation
⬜

Build abstract concepts from experience. This is where the contrastive learning and clustering probes from Phase 1 graduate into something that actually works.

- Combine reconstruction + contrastive + object-centric representations
- Compositional concept learning: "red square" = "red" ⊗ "square"
- Systematic generalization tests

**Pass bar:** NMI(clusters, context) > 0.5 on the Phase 1 gridworld test.

↓

### Phase 6 — World Model
⬜

Predict the future using objects and concepts — not just one-step latent prediction.

- **Uncertainty prediction** — know what you don't know
- **Multi-step prediction** — instead of predict 1 step, predict 10 / 20 / 50 steps
- Rollout consistency: long-horizon predictions stay coherent
- Imagination: use the world model for planning, not just auxiliary loss

**Pass bar:** World model predicts 10 steps ahead with lower error than a persistence baseline.

↓

### Phase 7 — Reasoning & Planning
⬜

Causal inference: the organism attributes effects to their causes. This is the D3 integration point.

- Causal history buffer (ICN) to track consequence delay
- Counterfactual reasoning: "what if I had taken a different action?"
- Goal-directed planning using the world model
- Tree search in latent space

**Pass bar:** Agent distinguishes correlation from causation in a controlled environment.

↓

### Phase 8 — Language
⬜

Not human-level NLU — grounded language: associating symbols with percepts, actions, and internal states.

- Grounded vocabulary: learn that "up" maps to a specific action vector
- Instruction following: "go to the blue object" as a compositional command
- Communicating learned concepts to another agent

**Pass bar:** Agent correctly executes a novel 3-step compositional instruction it has never seen before.

↓

### Phase 9 — Motivation & Emotion
⬜

Beyond pure curiosity — a richer drive system.

- Multiple intrinsic rewards: competence, novelty, surprise, predictability
- Drive competition: which signal to follow when they conflict?
- Emotional valence: tagging experiences with "good" / "bad" / "surprising"
- Mood as a persistent state that modulates learning rate and exploration

**Pass bar:** Agent shows distinct exploratory vs. exploitative phases modulated by internal state, not just epsilon decay.

↓

### Phase 10 — Self Model
⬜

The agent models itself: its own capabilities, limitations, current knowledge state, and uncertainty.

- Metacognitive monitoring: "do I know how to do this?"
- Epistemic humility: confidence-calibrated predictions
- Skill discovery: identifying reusable behavioral primitives
- Self-image: a persistent representation of "what kind of thing I am"

**Pass bar:** Agent that is uncertain about a task seeks information rather than guessing.

↓

### Phase 11 — Lifelong Growth
⬜

The organism accumulates knowledge across tasks without catastrophic forgetting.

- Progressive network expansion
- Elastic weight consolidation / synaptic intelligence
- Curriculum learning: self-directed ordering of challenges
- Transfer learning to novel environments

**Pass bar:** Agent learns N tasks sequentially with final performance on task 1 comparable to single-task agent.

↓

### Phase 12 — Self Improvement
⬜

The agent modifies its own architecture, learning algorithm, and reward function.

- Meta-learning: learning to learn
- Architecture search at agent level
- Reward design: agent tunes its own intrinsic motivation
- Self-modification with safety constraints

**Pass bar:** Agent discovers a more efficient learning strategy than its initial configuration.

↓

### Phase 13 — Multi-Agent Society
⬜

Multiple agents interact, share knowledge, specialize, and form the rudiments of an artificial culture.

- Multi-agent exploration: coverage vs. competition
- Knowledge transfer: one agent's learned policy seeded into another
- Role specialization: agents that develop complementary skills
- Cultural accumulation: knowledge that outlives any single agent

**Pass bar:** A society of N agents explores more efficiently than N independent agents.

↓

### Phase 14 — Physical Embodiment
⬜

The agent leaves the simulation and interacts with the physical world.

- Real-time sensorimotor loop
- Robustness to noisy, partial, delayed sensory input
- Sim-to-real transfer of learned representations
- Safe exploration in physical space

**Pass bar:** The same architecture that explored a gridworld can explore a real room with minimal modification.

---

## Foundation Research: what needs work right now

These aren't phases — they're cross-cutting improvements that should be investigated continuously, starting now.

### 1. Better Curiosity

| Current | Problem | Candidates to compare |
|---------|---------|----------------------|
| RND | noisy TV, novelty only, no uncertainty | RND, ICM, NGU, Plan2Explore |

Not necessarily replace RND — understand the tradeoffs first.

### 2. Representation Quality

This is probably the biggest weakness. Instead of one encoder, test:

| CNN | ResNet | BYOL | VICReg | SimCLR | MAE |
|-----|--------|------|--------|--------|-----|

Which representation is actually best for downstream tasks?

### 3. Representation Diagnostics

Don't just train. Measure:

- Mutual Information
- CKA (Centered Kernel Alignment)
- Linear Probe accuracy
- Cluster Purity
- Latent Collapse ratio
- Nearest Neighbor Accuracy
- Object Consistency

Become obsessed with measuring representations.

### 4. Perception Robustness

Train under:

- Lighting changes
- Rotation
- Noise
- Occlusion
- New layouts

A baby still recognizes a toy in dim light. Genesis should too.

### 5. Developmental Stages

Instead of `train 5000 episodes` flat:

```
Age 0:  random motor babbling
Age 1:  basic object tracking
Age 2:  goal-directed reaching
Age 3:  tool use
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
