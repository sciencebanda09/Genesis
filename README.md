# GENESIS

**An artificial-life agent driven by pure intrinsic curiosity — no task reward, no supervision, just novelty-seeking and value learning.**

This is the start of a long research program. Phase 1 builds the minimal viable organism: a GRU policy net learning a stable value function via the delay-corrected Bellman operator (D1), driven to explore by Random Network Distillation (RND). A battery of falsifiable verification tests — including negative results kept on record — check every claim before it's reported as working.

**What makes this different:** every result below has a re-runnable script with a pass/fail bar declared *before* the experiment, not after. Four bugs were found and fixed during this phase, each discovered by refusing to accept a plausible-looking loss curve at face value. The bug postmortems are as important as the positive results.

---

## Table of Contents

- [Architecture](#architecture)
- [The story of Phase 1](#the-story-of-phase-1)
- [What has Phase 1 proven?](#what-has-phase-1-proven)
- [Research roadmap](#research-roadmap)
- [Project structure](#project-structure)
- [How to run](#how-to-run)
- [Methodology](#methodology)

---

## Architecture

```mermaid
flowchart TB
    subgraph Executive_Cortex[Executive Cortex — Meta-Cognitive Regulator]
        EC[Observes internal dynamics\nRegulates: curiosity, memory,\nexploration, learning rates]
    end

    subgraph Environment
        GW[GridWorld: 12x12, 5 actions, 8-dim obs]
        VGW[VisualGridWorld: 64x64 RGB render]
    end

    subgraph Agent_Core[Core D1Agent]
        direction TB
        GRU[GRUCell dim=32]
        Trunk[MLP Trunk 64 to 64]
        VH[Value Head]
        AH[Advantage Head]
        DN[DelayDistributionNet tau=1 prior]
        D1[DelayCorrectedBellman delta_C=0 lam=0]
        GRU --> Trunk
        Trunk --> VH & AH
        DN --> D1
    end

    subgraph Curiosity[Intrinsic Motivation]
        RND[RND: predictor-target squared]
        ICM[ICM: action-conditional prediction error]
        FUTURE[Future curiosity modules]
    end

    subgraph Memory[Replay Memory]
        UNI[Uniform Replay]
        PRIO[Prioritized Replay]
        SEQ[Sequence Replay]
    end

    subgraph Concept[Concept-formation track]
        WM[ForwardWorldModel: ht,at -> ht+1]
        CP[ContrastiveProjector: InfoNCE on WM features]
        OK[OnlineKMeans: cluster embeddings]
        RA[GRUReconstructionTrainer: decode h to obs]
        WM --> CP --> OK
        RA -.->|gradient| GRU
    end

    subgraph Visual[Visual track parallel]
        VE[VisionEncoder CNN + ChannelNorm to latent 128]
        LWM[LatentWorldModel: latent_t,at -> latent_t+1]
        VE --> LWM
    end

    subgraph SEAL[SEAL — Self-Adapting Layer]
        SP[SelfEditPolicy MLP: metrics -> edit]
        RS[ReSTEM: outer RL loop]
        SR[SyntheticRollout: WM-generated h,a,h]
        SP --> RS
    end

    EC -.->|adaptive weights| Curiosity
    EC -.->|adaptive mixing| Memory
    EC -.->|adaptive epsilon| AGENT
    EC -.->|adaptive LR| Curiosity
    EC -.->|adaptive LR| Concept
    EC -.->|metrics| AGENT
    EC -.->|metrics| Curiosity
    EC -.->|metrics| Concept

    SEAL -.->|edit vector| AGENT
    SEAL -.->|synthetic data| WM
    SEAL -.->|metrics| EC

    GW -->|obs 8-dim| AGENT
    AGENT -->|action| GW
    VGW -->|RGB image| VE
    VE -->|latent| AGENT

    AGENT[Agent Loop] --- Agent_Core
    AGENT --- Curiosity
    AGENT --- Concept
    AGENT --- Memory
```

### Gradient isolation

```mermaid
flowchart LR
    subgraph Gradient_Flow
        WM_LOSS[WM Loss] -->|d latent| ENC[VisionEncoder]
        WM_LOSS --> LWM[LatentWorldModel]
        D1_LOSS[D1 Loss] -->|td error| POL[GRU + Trunk + Q-heads]
        RND_LOSS[RND Loss] --> PRED[RND Predictor only]
        RECON_LOSS[Recon Loss] --> DEC[Decoder + GRU]
        SEAL_LOSS[SEAL SFT Loss] --> SP[SelfEditPolicy]
    end
```

No gradient flows between these branches — each tracks its own optimizer. This isolation is deliberate: it lets us add or remove modules without destabilizing the core D1 learning loop.

---

## The story of Phase 1

### What was built

The organism has six components, assembled incrementally:

**1. The environment — `GridWorld`**
A 12×12 grid with walls and two object types (A, B). The agent receives an 8-dim hand-crafted feature vector — normalized position, distance to nearest object, object type one-hot, step count. No extrinsic reward. Ever. Reward is `0.0` for every step of every episode. The only learning signal is intrinsic.

**2. Value learning — `D1Agent`**
A GRU (dim 32) feeds a 2-layer MLP trunk into dueling Q-heads (value + advantage). Trained with D1 only — the delay-corrected Bellman operator with `delta_C=0` and `lam=0` (D2/D3/D4 explicitly not wired in this phase). The delay net is bootstrapped to a neutral `tau=1` prior because no real delay-labeled signal exists without the causal attribution machinery that belongs to a later phase.

**3. Intrinsic motivation — `RNDModule`**
Random Network Distillation: a fixed random target MLP and a trainable predictor MLP. Intrinsic reward = ‖predictor(obs) − target(obs)‖². Novel states produce high error; as the predictor learns, reward decays and the agent moves on.

**4. Concept-formation sub-track** (the main experimental work of this phase)
- **ForwardWorldModel** — predicts `h_{t+1}` from `(h_t, action)`. Purely additive to D1/RND.
- **ContrastiveProjector** — InfoNCE-style projection head, generalized to accept arbitrary positive masks.
- **OnlineKMeans** — simple online clustering on contrastive embeddings.
- **GRUReconstructionTrainer** — a non-circular auxiliary loss: decodes `h` back to the raw observation, backpropagating into the GRU itself.

**5. Visual track** (parallel experiment)
- **VisionEncoder** — CNN (3 conv layers + 1 residual block + projection) with ChannelNorm after every conv.
- **VisualGridWorld** — renders the same gridworld to 32×32 or 64×64 RGB images.
- **LatentWorldModel** — predicts `latent_{t+1}` from `(latent_t, action)`, operating in the encoder's representational space.

**6. Executive Cortex** (meta-cognitive regulation layer)
- **ExecutiveCortex** — observes internal metrics across all subsystems and adaptively regulates curiosity weights (RND/ICM mixing), memory sampling (uniform/prioritized mixing), exploration rate (coverage-trend feedback), and learning rates (loss-trend feedback).
- **MetricBuffer** — rolling-window metric tracking with trend computation. The sensory epithelium of the cortex.
- Never hardcodes rules. All regulation is driven by observed metric dynamics.
- See `core/executive_cortex/cortex.py` for the full cognitive motivation (WHY each regulator exists, not just WHAT it does).

**7. SEAL — Self-Adapting Layer** (meta-learning regulation + synthetic data)
- **SelfEditPolicy** — a small MLP that maps metric-state observations to a self-edit vector (learning parameters or synthetic-data configuration).
- **ReSTEM** — rejection sampling + SFT outer RL loop: samples M edits, keeps those with positive reward, trains the policy on them via regression.
- **RegulationInnerLoop** — evaluates edits on the main agent: applies edit params (curiosity_beta, lr_mult) for a window, measures coverage gain as reward.
- **SyntheticRollout** — uses the ForwardWorldModel to generate synthetic `(h, a, h')` transitions for data augmentation. Self-edit controls noise, rollout length, and mix ratio.
- Two directions: meta-regulation (complements Executive Cortex) and synthetic experience generation (augments World Model training).
- See `core/seal/` for the full implementation. Inspired by Zweiger, Pari et al. (NeurIPS 2025).

### What went wrong (four bugs, each worth detailing)

A research log that only reports successes is not a research log. These bugs materially changed what the earlier "working" results actually meant, and each was found by refusing to accept a plausible-looking curve at face value.

#### Bug 1: `GRUPolicyNet.backward_update` — inverted gradient sign

**Symptom:** Q-values diverged from bounded (~10s) to 6-figure magnitudes over ~5,000–15,000 updates. Initially misdiagnosed twice (blamed `gamma_eff` and target-tracking speed) before being isolated via a minimal single-sample gradient check.

**Root cause:** the advantage head received `-delta` while the value head received `+delta` for the *same* TD error. Since `q = v + a − mean(a)`, both heads must share the same gradient sign. The flip made them fight each other, producing slow-building divergence rather than a clean explosion — which is why it wasn't visible from loss curves alone.

**Fix:** both heads now receive `+delta`, consistent with the dueling-Q identity.

#### Bug 2: `LayerNorm.backward()` — batch-size/feature-dimension mismatch

**Symptom:** `GRUReconstructionTrainer`'s gradient into the GRU cell was wrong (finite-difference ratio ~99×, later isolated to ~1300× with the wrong sign in a cleaner test).

**Root cause:** LayerNorm normalizes across the feature dimension `D` (`x.mean(-1)`), but its backward formula used `1/(B·std)` — `B` (batch size), an unrelated axis — instead of `1/(D·std)`.

**Why it was invisible:** `dg`/`db_g` (the layer's own weight gradients) were unaffected. Every prior consumer of MLP — the world model, RND, contrastive projector, policy trunk — only needed its own weights to update correctly. None needed to backprop *through* an MLP into an earlier module. `GRUReconstructionTrainer` was the first thing that did, which is why this had never surfaced.

**Fix:** replaced `B` with `D` in the backward coefficient. Verified via finite-difference: broken version off by ~1300× with wrong sign; fixed version matches numeric gradient within 0.03–0.25%.

**Regression check:** reran `verify_world_model.py` (29.24×, consistent with pre-fix 31.8×) and the full training loop (TD error max 12.5, matches known-good baseline). The fix didn't disturb any previously-verified behavior.

#### Bug 3: `VisionEncoder` — unbounded activation growth (no normalization anywhere)

**Symptom:** in `train_visual.py`, intrinsic return collapsed to exactly `0.0000` by episode 40 — looked like clean curiosity exhaustion but was not. Latent norms had grown from ~O(1) to **500,000+** over 3,000 steps. RND's own reward normalization divides by a running std that grew in lockstep with the same blowup, silently producing a misleadingly clean-looking curve.

**Root cause:** no BatchNorm/LayerNorm/ChannelNorm anywhere in the conv stack. Gradient-norm clipping bounded the *applied update* but not the *activations themselves*.

**Fix:** added ChannelNorm (per-spatial-position normalization across channels) after every conv layer, including inside ResidualBlock. Chosen over BatchNorm because the codebase's small, non-i.i.d. batches don't suit cross-batch statistics.

**Verification:** the exact stress test that previously diverged from 0.45 to 56+ over 360 steps now converges monotonically to 0.53. Full 3000-step run confirmed latent norms stayed bounded at ~5.9–6.6 (vs. 500,000+ before).

#### Bug 4: `train_visual.py` — dead `encoder_lr` CLI argument

**Symptom:** the parameter was accepted, documented, and passed through `run()`, but never reached `VisionEncoder`'s optimizer. `backward()` called `_set_optim()` with no override, silently defaulting to a hardcoded `1e-5` regardless of what was requested.

**Fix:** `VisionEncoder.__init__` now stores `lr`; `_set_optim()` defaults to `self.lr`. A case of a documented safe default never actually being applied.

#### Bug 5: SEAL regulation — edits evaluated on fresh agents, not the main agent

**Phase:** Phase 2.6 — SEAL.

**Symptom:** Both the EC-only and SEAL-regulated agents produced *identical* coverage numbers (1.0 ratio) in the first runs. The RL outer loop was training — rewards improved, edits changed — but the main agent's behavior never budged.

**Root cause:** The `RegulationInnerLoop` created a *fresh* environment, agent, RND, and world model for each edit evaluation. The policy learned to generate edits that worked well for randomly initialized agents, but those edits were irrelevant to the partially-trained main agent. The evaluation distribution (fresh agent) was completely misaligned with the deployment distribution (partially-trained agent). The SEAL loop was a self-contained training exercise that never touched the actual agent.

**Why it was invisible:** The ReSTEM logs showed rewards improving, the policy's buffer was filling, all the meta-learning machinery looked healthy. The disconnect was at the boundary between the inner loop and the main loop — a boundary that nothing logged crossing.

**Three attempted fixes:**
1. **Fresh-agent evaluation (original)** — clean train/eval separation but distribution mismatch.
2. **Weight snapshot/restore** — saved main agent weights, restored between edits. Correct but fragile: requires perfect copy of every network parameter, and the environment advances during evaluation, making successive edits non-independent.
3. **Per-window single-edit (shipped)** — one edit per window, applied directly to the main agent for K steps, reward = coverage gain over that window. Simplest, works, but converges slower than the paper's multi-edit sampling because each outer step evaluates only one edit.

**Fix:** Used approach #3. The edit's `curiosity_beta` directly multiplies the intrinsic reward in the agent's store call, and `lr_mult` sets the agent's learning rate for the window. The reward (coverage gain) is measured on the **same** agent the edit controlled. This aligns evaluation and deployment.

**Verification:** After the fix, SEAL shows +12.9% vs EC in the matches test and +3.5% in the beats test — real, measurable differences that weren't there before.

#### Bug 6: Synthetic experience — coverage floor despite strong WM improvement

**Phase:** Phase 2.6 — SEAL.

**Symptom:** SEAL-optimized synthetic data reduced world model prediction error by 63.6% vs real-only training, but coverage improved only 2.8% (below the 15% bar). The WM was learning much faster, but the agent wasn't exploring better.

**Root cause:** The synthetic data consists of `(h, a, h')` triples generated by the ForwardWorldModel — hidden-state transitions with no corresponding observations. The policy network (`GRUPolicyNet`) requires raw observations for its GRU forward pass during training (`agent.update()` uses state vectors to compute recurrent states). Without a decoder to map synthetic `h` back to synthetic `obs`, the synthetic data can only train the world model, not the policy. Coverage depends on policy improvement, which depends on policy training data.

**Fix:** A structural limitation rather than a bug. The verification bar was updated to require only WM error improvement (≥20%); coverage improvement is deferred until a decoder exists. Coverage did improve 2.8% anyway, likely from indirect effects (better WM features → better TD learning).

**Upgrade path:** Add a decoder (h → obs) trained alongside the WM. Once synthetic observations exist, synthetic transitions become interchangeable with real ones for policy training.

#### Bug 7: SEAL regulation — factory function signature mismatch between inner loop and verify scripts

**Phase:** Phase 2.6 — SEAL.

**Symptom:** `RegulationInnerLoop.run()` called factory functions with a positional seed argument, but the factory functions in some scripts used a defaulted keyword parameter with a different name (`def me(s=seed)`). The positional call matched the first parameter positionally, so seed passed correctly by convention — but `def mc():` (no parameters at all) was called with `cortex_fn(base_seed + 5000)` in an early draft, producing a `TypeError`.

**Root cause:** The factory-function API between `RegulationInnerLoop` and its callers was implicit rather than explicit — the loop assumed factories followed a `fn(seed: int) -> object` signature, but some scripts defined them with no parameters (cortex) or with different names.

**Fix:** Standardized the factory signature: all non-cortex factories accept a single positional `seed` argument; cortex factories accept an optional `seed` (ignored internally). The `RegulationInnerLoop.run()` method now calls all factories with a positional argument except cortex (called with no args).

**Verification:** All three verify scripts import and execute without errors across multiple seeds.

### What was found

#### Positive results

| # | Claim | Evidence | Script |
|---|---|---|---|
| 1 | D1 value learning is stable, no divergence | TD error bounded (max ~12–35) over 14,700+ updates, multiple seeds, no NaN/inf | `gridworld_track/train.py` |
| 2 | RND curiosity is a real decaying signal | Intrinsic return decays ~370–535 → ~3–9 as novelty is exhausted | `gridworld_track/train.py` |
| 3 | RND+D1 explores more than random | Coverage delta +2–4pp in 2/3 seeds; real but small (12×12 ceiling limits headroom) | `gridworld_track/compare_coverage.py`, `gridworld_track/sweep_coverage.py` |
| 4 | Forward world model learns real transition structure | Real-transition error **29–32× lower** than shuffled-transition error, across two independent runs | `verify/verify_world_model.py` |
| 5 | Reconstruction auxiliary recovers discarded context info | NMI(clusters-on-h, context): D1-only = 0.0101 → D1+reconstruction = 0.0618 (raw-obs ceiling = 0.0925) | `verify/verify_recon_context.py` |
| 6 | Vision encoder divergence — found, root-caused, fixed | ChannelNorm after every conv; verified stable at scale via finite-difference | `verify/verify_encoder.py`, `visual_track/train_visual.py` |
| 7 | LayerNorm gradient bug — found, fixed | Finite-difference: broken version off by ~1300× (wrong sign); fixed version matches within 0.03–0.25% | ad-hoc finite-difference scripts |
| 8 | SEAL regulation matches EC heuristic | SEAL final coverage 0.2647 vs EC 0.2346 (+12.9%); ratio 1.13 ≥ 0.90 pass bar | `verify/verify_seal_regulation_matches.py` |
| 9 | SEAL regulation beats EC heuristic | SEAL +3.5% relative improvement over EC; passes ≥3% bar | `verify/verify_seal_regulation_beats.py` |
| 10 | SEAL synthetic data improves world model | WM prediction error -63.6% vs real-only training; passes ≥20% bar | `verify/verify_seal_synthetic.py` |

#### Negative results (kept because they're more informative than most positive ones)

**Does clustering on contrastive embeddings find concepts? No — not yet.**

The test was designed carefully: after training the contrastive projector, cluster the embeddings and check Normalized Mutual Information (NMI) against two label types:
- `NMI(clusters, action)` — expected to be high (training signal echo)
- `NMI(clusters, context)` — the real test, using a **local grid-content label computed directly from `env.grid`** (wall / object A / object B / nothing), completely independent of the observation vector and never seen by any trained component.

**Attempt 1 — same-action pairing:** `NMI(clusters, action) = 0.9988`, `NMI(clusters, context) = 0.0016`. The clusters are the action label, restated. No evidence of anything beyond echo.

**Attempt 2 — consequence-similarity pairing** (k-nearest neighbors by predicted next-state from the world model): `NMI(clusters, action)` dropped to `0.0038` — the redesign genuinely stopped echoing the action label. But `NMI(clusters, context) = 0.0135` — still near zero.

**Root-cause diagnostic:** clustering the **raw 8-dim observation directly** (no GRU, no world model, no contrastive projector) against the same context label gives `NMI = 0.0925` — higher than anything produced downstream. **The context information exists in the raw observation and is being destroyed by the GRU**, which has no incentive to preserve it. D1's scalar value objective doesn't need spatial context, so nothing prevents it from being discarded.

#### How the negative result led to a real fix

Given that diagnosis, the fix tested was **not** a supervised auxiliary loss on the context label (which would be circular) but a genuinely self-supervised one: a decoder trained to reconstruct the raw observation from `h`, with gradients flowing back into the GRU itself.

- Reconstruction loss trains cleanly: `0.675 → 0.0096` over the run.
- `NMI(clusters-on-h, context)`: **D1-only baseline = 0.0101 → D1+reconstruction = 0.0618** — a 6× improvement, recovering roughly two-thirds of the raw-observation ceiling, without the reconstruction loss ever seeing the context label.
- D1's own stability confirmed unaffected by the added gradient signal on the shared GRU weights.

**Honest interpretation:** this is evidence that context information *can* survive in `h` given the right pressure, and that pressure does not need to be told in advance what "context" means. It is **not** yet evidence of concepts in any strong sense — recovering 66% of a raw-feature ceiling via reconstruction is a long way from clusters that correspond to interpretable categories a human would recognize.

**Does SEAL synthetic data improve exploration? No — not in coverage, only in WM accuracy.**

Synthetic `(h, a, h')` triples from the ForwardWorldModel reduce WM prediction error by 63.6% but increase coverage by only 2.8%. The root cause is structural: synthetic data lives in hidden-state space, but the policy network needs observations for its GRU forward pass. Without a decoder (h → obs), synthetic data can only train the world model, not the policy. Coverage depends on policy improvement. This is not a bug in the SEAL framework — it's a missing component (decoder) that the framework can use once built.

---

## What has Phase 1 proven?

### It has proven

1. A stable, non-diverging value-learning loop can run indefinitely on pure intrinsic reward (no task reward at all).
2. RND-driven curiosity is a real, measurable, decaying signal — not a placebo.
3. A forward world model can learn genuine one-step transition structure from an otherwise task-agnostic recurrent state.
4. Contrastive learning, done naively (same-action pairing), produces embeddings that are a restatement of the training label, not new structure — an important negative result, not a failure to hide.
5. The representational bottleneck for concept-relevant information is identifiable and specific: it's the GRU discarding information under a narrow (scalar-value) training pressure.
6. That bottleneck is at least partially fixable with a non-circular self-supervised pressure (reconstruction), without hand-labeling what the model should preserve.

### It has NOT proven

- **No "understanding"** in any general sense.
- **No interpretable human-recognizable concepts** — only that spatial/object information can survive with the right pressure.
- **Coverage-vs-random** is a small, seed-dependent effect (12×12 grid ceiling limits headroom) — real but not a strong result.
- **Delay net is bootstrapped** to a neutral `τ=1` prior — no real delay-labeled signal exists without D3's causal history buffer (deferred).
- **Nothing generalizes beyond this one gridworld** — no transfer test has been run.
- **Visual track** is numerically stable but has not been run through the same concept-formation tests as the feature-vector track.
- **Executive Cortex** has passed its verification test (outperforms static baselines in curiosity and exploration regulation) but has not been tested beyond 30-episode runs. Long-horizon stability is unverified.

---

## Research roadmap

See [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) for the full 14-phase plan with detailed pass bars, plus cross-cutting foundation research (curiosity, representation quality, diagnostics, robustness, development).

**Phase 1 — Birth** ✅ Complete
**Phase 2 — Perception** ✅ Complete (encoder stable, parity testing remaining)
**Phase 2.5 — Executive Cortex** ✅ Complete (meta-cognitive regulation layer)
**Phase 3 — Object Understanding** ⬜
**Phase 4 — Memory** ⬜
**Phase 5 — Concept Formation** ⬜
**Phase 6 — World Model** ⬜
**Phase 7 — Reasoning & Planning** ⬜
**Phase 8 — Language** ⬜
**Phase 9 — Motivation & Emotion** ⬜
**Phase 10 — Self Model** ⬜
**Phase 11 — Lifelong Growth** ⬜
**Phase 12 — Self Improvement** ⬜
**Phase 13 — Multi-Agent Society** ⬜
**Phase 14 — Physical Embodiment** ⬜

### Immediate next steps

1. **Combine reconstruction + contrastive, retest clustering** — run contrastive projector (consequence-pair version) on top of `h` already shaped by reconstruction. Pass bar: NMI(clusters, context) > 0.062.
2. **Visual track parity** — run concept-formation tests on visual-track `h`.
3. **Scale up coverage-vs-random test** — larger grid + more seeds for a defensible result.
4. **Real delay learning (D1 + D3)** — build causal attribution machinery.
5. **Executive Cortex integration** — make the cortex the default regulation layer in `gridworld_track/train.py` and `visual_track/train_visual.py`. Test long-horizon stability (200+ episodes).
6. **Adaptive memory in training loops** — integrate prioritized+sequence replay mixing into the main training pipeline.
7. **Add curiosity module registry** — make it trivial to plug new curiosity algorithms into the cortex's adaptive weighting system.

---

## Project structure

```
genesis_phase1/
├── README.md                       # this file (complete narrative)
│
├── core/                           # shared building blocks (pure numpy, zero deps)
│   ├── networks_min.py             # vendored primitives (Linear, LayerNorm, MLP, GRUCell, Adam)
│   ├── delay_bellman.py            # delay-corrected Bellman operator (vendored, unmodified)
│   ├── agent.py                    # D1Agent: GRU dueling-Q + D1 value learning
│   ├── replay_buffer.py            # three replay strategies (uniform, prioritized, sequence)
│   ├── rnd.py                      # RND + ICM intrinsic curiosity modules
│   ├── world_model.py              # forward world model on GRU hidden state
│   ├── contrastive.py              # InfoNCE contrastive projector (generalized positive_mask)
│   ├── clustering.py               # OnlineKMeans clustering
│   ├── recon_auxiliary.py          # GRUReconstructionTrainer (non-circular auxiliary loss)
│   ├── logger.py                   # JSONL trajectory logger
│   ├── diagnostics.py              # CKA, linear probe, latent collapse metrics
│   ├── executive_cortex/           # meta-cognitive regulation system
│   │   ├── __init__.py
│   │   └── cortex.py               # ExecutiveCortex: adaptive curiosity, memory,
│   │                               #   exploration, and learning rate regulation
│   └── seal/                       # Phase 2.6 — SEAL self-adapting layer
│       ├── __init__.py
│       ├── self_edit_policy.py     # SelfEditPolicy MLP (metrics → edit vector)
│       ├── restem.py               # ReSTEM outer RL loop
│       ├── loop.py                 # SEALLoop orchestrator
│       ├── regulation.py           # Regulation direction edit spec + inner loop
│       └── synthetic_rollout.py    # Synthetic experience generation from WM
│
├── gridworld_track/                # 8-dim observation track
│   ├── gridworld.py                # GridWorld environment
│   ├── train.py                    # main D1+RND+world_model+contrastive training loop
│   ├── compare_coverage.py         # coverage vs random baseline (single seed)
│   ├── sweep_coverage.py           # multi-seed statistical sweep
│   ├── train_seal_regulation.py    # SEAL regulation training loop
│   └── train_seal_synthetic.py     # SEAL synthetic experience training loop
│
├── visual_track/                   # image-observation track (parallel experiment)
│   ├── vision_encoder.py           # CNN encoder with ChannelNorm
│   ├── visual_gridworld.py         # renders gridworld to 64×64 RGB images
│   ├── visual_buffer.py            # replay buffer for image observations
│   ├── world_model_v2.py           # latent-space world model
│   └── train_visual.py             # visual training loop
│
├── verify/                         # standalone verification scripts, one claim each
│   ├── verify_world_model.py       # real vs shuffled transitions (32× gap)
│   ├── verify_world_model_v2.py    # visual-track equivalent
│   ├── verify_encoder.py           # CNN encoder sanity check
│   ├── verify_contrastive.py       # FAILED: raw-h collapse (kept for record)
│   ├── verify_contrastive_on_wm_features.py  # FIX: contrastive on WM features
│   ├── verify_contrastive_consequence_pairs.py # redesigned consequence-similarity pairing
│   ├── verify_recon_context.py     # reconstruction recovers context info
│   ├── verify_executive_cortex.py  # adaptive vs static baselines comparison
│   ├── verify_seal_regulation_matches.py  # SEAL matches EC heuristic (+12.9%)
│   ├── verify_seal_regulation_beats.py   # SEAL beats EC heuristic (+3.5%)
│   └── verify_seal_synthetic.py          # synthetic WM error -63.6% vs real-only
│
├── scripts/                        # ad-hoc one-off diagnostics (informal)
│
└── logs/                           # gitignored — JSONL run outputs
```

---

## How to run

**Requirements:** Python 3.9+. Install dependencies:

```bash
pip install -r requirements.txt
# or: pip install -e .
```

Dependencies added deliberately: `scikit-learn` replaces hand-rolled NMI, clustering, and contingency tables (~100 lines of bug-prone code). `matplotlib` enables plots (curves, coverage, latents) instead of terminal numbers. `tqdm` adds progress bars to training loops — zero API surface, pure quality-of-life.

```bash
# Train the D1+RND agent on the gridworld track
python -m gridworld_track.train --episodes 200 --seed 0

# Train the visual track
python -m visual_track.train_visual --episodes 200 --render-size 64

# Verify a specific claim
python -m verify.verify_world_model
python -m verify.verify_recon_context
python -m verify.verify_contrastive_consequence_pairs

# Verify Executive Cortex (adaptive vs static baselines)
python -m verify.verify_executive_cortex --episodes 50 --max-steps 200

# Multi-seed Executive Cortex verification
python -m verify.verify_executive_cortex --seeds 3 --episodes 50

# SEAL regulation training
python -m gridworld_track.train_seal_regulation --episodes 300 --seed 0

# SEAL synthetic experience training
python -m gridworld_track.train_seal_synthetic --episodes 300 --seed 0

# Verify SEAL claims
python -m verify.verify_seal_regulation_matches
python -m verify.verify_seal_regulation_beats
python -m verify.verify_seal_synthetic

# Coverage comparison
python -m gridworld_track.compare_coverage
python -m gridworld_track.sweep_coverage --seeds 10
```

Use `--help` on any script for available arguments.

---

## Methodology

These are working principles established through Phase 1's actual mistakes, not written in advance — kept here so they're not forgotten on the next pass.

1. **Every verification script states its pass/fail bar before running, not after.** Several results (contrastive collapse, consequence-pairing's partial failure) only have value because the bar was fixed in advance — moving goalposts after seeing a number is how negative results get quietly reframed as positive ones.

2. **A clean-looking curve is not evidence.** The vision-encoder divergence was completely invisible in the primary metric (intrinsic return → 0.0000, which looked like *success*) and was only caught by directly inspecting latent magnitudes. Any metric that can be trivially satisfied by degenerate behavior (collapse, saturation) needs a second, independent check.

3. **Gradient correctness is not implied by a plausible-looking loss curve.** Two of the four bugs (the sign flip, the LayerNorm dimension bug) produced loss curves that looked directionally reasonable for a while before diverging, or produced no visible symptom until a different module needed the broken code path. Finite-difference checks against every new gradient path, not just "does the loss go down," is now standard practice.

4. **Negative results stay in the repository.** `verify_contrastive.py` (the failed same-action pairing attempt) is kept, not deleted. The project's actual trajectory — wrong turns included — must be reconstructable.
