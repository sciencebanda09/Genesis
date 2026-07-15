# Project Genesis — Phase 1 Research Log

**Status:** Phase 1 in progress. Core learning loop stable and verified. Concept-formation sub-track (steps 1-4) partially successful, with one confirmed negative result and one confirmed positive result.

**Scope of Phase 1 (per the original plan):** birth of a minimal-knowledge agent — digital genome, core brain framework, basic drives (curiosity/exploration), time awareness, random exploration, simple sensors/actions. This document covers only what has actually been built and empirically checked, not the aspirational full architecture in the original vision doc.

---

## 1. What "Phase 1" means here, concretely

The original 13-phase vision document describes an artificial-life research program, most of which (self-evolution, society, embodiment) are unsolved, lab-scale research problems, not buildable milestones. Phase 1 was scoped down to something falsifiable:

> An agent with no task reward, driven by intrinsic curiosity, learning a stable value function over a toy environment, with an explicit test of whether anything resembling *concepts* emerges from that process.

Every claim below was checked against a stated pass/fail bar **before** being reported as working, and negative results are kept in the record rather than discarded.

---

## 2. Architecture as built

### 2.1 Environment — `GridWorld`
- 12x12 grid, discrete 5-action space (up/down/left/right/interact), walls + two object types (A, B).
- Observation: hand-crafted 8-dim feature vector — `[y_norm, x_norm, nearest_obj_dy, nearest_obj_dx, onehot(3), step_norm]`. Not a raw patch.
- No extrinsic reward. Ever. Reward is 0.0 for the entire episode, by design — this is a pure-curiosity agent.

### 2.2 Value learning — `D1Agent`
- `GRUPolicyNet`: GRU (dim 32) to 2-layer MLP trunk to dueling Q-head (value + advantage).
- Trained with **D1 only** — CCPL's delay-corrected Bellman operator (`delay_bellman.py`, vendored unmodified from an external project), with `delta_C=0` and `lam=0` (D2/D3/D4 explicitly not wired in this phase).
- Delay net bootstrapped to a neutral `tau=1` prior (no real delay-labeled signal exists without D3's causal history buffer — documented as an honest limitation, not hidden).
- Hyperparameters: `gamma=0.99`, `gru_dim=32`, `hidden_dim=64`, `tau_max=10`, `batch_size=64`, `buffer_capacity=20,000`, `target_update_tau=0.01`, `eps` decays `1.0->0.05` over 3000 steps.

### 2.3 Intrinsic motivation — `RNDModule`
- Random Network Distillation: fixed random target MLP + trainable predictor MLP, reward = prediction error, batch-normalized.

### 2.4 Concept-formation sub-track (steps 1-4, this session's main work)
1. **`ForwardWorldModel`** — predicts `h_{t+1}` from `(h_t, action)`. Purely additive to D1/RND.
2. **`ContrastiveProjector`** — InfoNCE-style projection head; generalized to accept an arbitrary `positive_mask` (originally hardcoded to same-action pairing).
3. **`OnlineKMeans`** — simple online clustering on contrastive embeddings.
4. **`GRUReconstructionTrainer`** — non-circular auxiliary loss: decodes `h` back to the raw observation, backprops into the GRU itself.

### 2.5 Visual track (parallel experiment)
- `VisionEncoder`: CNN (3 conv layers + 1 residual block + projection), now with `ChannelNorm` after every conv.
- `visual_gridworld.py` renders the same gridworld to 32x32/64x64 RGB images.
- `world_model_v2.py`, `train_visual.py`: same D1+RND+world-model loop, running on encoded image latents instead of hand-crafted features.

---

## 3. What is verified and working

Every result below has a corresponding script in `verify/` that can be re-run.

| # | Claim | Evidence | Script |
|---|---|---|---|
| 1 | D1 value learning is stable, no divergence | TD error bounded (max ~12-35) over 14,700+ updates, multiple seeds, no NaN/inf | `train.py` |
| 2 | RND curiosity is a real learning signal | Intrinsic return decays ~370-535 -> ~3-9 as novelty is exhausted, consistent across seeds | `train.py` |
| 3 | RND+D1 explores more than random | Coverage delta +2-4pp in 2/3 single seeds; directionally real but small-effect (12x12 grid ceiling limits headroom) | `compare_coverage.py`, `sweep_coverage.py` |
| 4 | Forward world model learns real transition structure | Real-transition error **29-32x lower** than shuffled-transition error, across two independent runs | `verify_world_model.py` |
| 5 | Contrastive learning mechanically works (InfoNCE correct) | On a synthetic, class-balanced input: same-class similarity 0.94, different-class similarity -0.29 after training | isolated diagnostic (not a checked-in script) |
| 6 | Vision encoder divergence — found, root-caused, fixed | See section 4.1 | `verify_encoder.py`, `train_visual.py` |
| 7 | `LayerNorm.backward()` gradient bug — found, fixed | Finite-difference check: broken version off by ~1300x (wrong sign); fixed version matches numeric gradient within 0.03-0.25% | ad-hoc finite-difference scripts (see section 4.2) |
| 8 | Reconstruction auxiliary recovers discarded context info | NMI(clusters-on-*h*, independent context label): D1-only = 0.0101 -> D1+reconstruction = 0.0618 (raw-obs ceiling = 0.0925) | `verify_recon_context.py` |

---

## 4. Bugs found and fixed (in order of discovery)

This section exists because a research log that only reports successes is not a research log — these bugs materially changed what the earlier "working" results actually meant, and each was found by refusing to accept a plausible-looking curve at face value.

### 4.1 `GRUPolicyNet.backward_update` — inverted gradient sign
- **Symptom:** Q-values diverged from bounded (~10s) to 6-figure magnitudes over ~5,000-15,000 updates. Initially misdiagnosed twice (blamed `gamma_eff`/target-tracking speed) before being isolated via a minimal single-sample gradient check.
- **Root cause:** the advantage head received `-delta` while the value head received `+delta` for the *same* TD error. Since `q = v + a - mean(a)`, both heads must share the same gradient sign — the flip made them fight each other, producing slow-building divergence rather than a clean explosion (which is why it wasn't visible from loss curves alone).
- **Scope:** present in the original vendored source, inherited via vendoring. A second, related but distinct bug was also found in the source project's `QNetwork.backward_update` (used by the DQN baseline) — both heads there get `-delta` *consistently*, which is internally self-consistent but uniformly backwards. **This QNetwork bug was reported but never patched in the source project.** This is flagged as high-priority technical debt: it plausibly explains prior "CCPL underperforms baselines" findings from earlier audit work, since the DQN baseline itself may never have been learning correctly.

### 4.2 `LayerNorm.backward()` — batch-size/feature-dimension mismatch
- **Symptom:** `GRUReconstructionTrainer`'s gradient into the GRU cell was wrong (finite-difference ratio ~99x, later isolated to ~1300x with the wrong sign in a cleaner test).
- **Root cause:** `LayerNorm` normalizes across the feature dimension `D` (`x.mean(-1)`), but its backward formula's coefficient used `1/(B * std)` — `B` (batch size), an unrelated axis — instead of `1/(D * std)`.
- **Why it was invisible until now:** `dg`/`db_g` (the layer's own weight gradients) were unaffected and summed correctly over the batch axis by design. Every prior consumer of `MLP` (`ForwardWorldModel`, `RNDModule`, `ContrastiveProjector`, `GRUPolicyNet`'s trunk) only needed its own weights to update correctly — none of them needed to backprop `dx` *through* the MLP into an earlier module. `GRUReconstructionTrainer` was the first thing in the entire project that did, which is why this had never surfaced.
- **Verification:** confirmed broken and then confirmed fixed via finite-difference at three different indices with a non-degenerate (non-zero-mean) target, since a zero-target test produces floating-point cancellation artifacts that look like a false negative.
- **Regression check:** reran `verify_world_model.py` (29.24x, consistent with the pre-fix 31.8x) and the full `train.py` loop (TD error max 12.5, matches known-good baseline) — confirms the fix didn't disturb any previously-verified behavior.

### 4.3 `VisionEncoder` — unbounded activation growth (no normalization anywhere)
- **Symptom:** in `train_visual.py`, intrinsic return collapsed to exactly `0.0000` by episode 40 — looked like clean curiosity exhaustion but was not. Latent norms had actually grown from ~O(1) to **500,000+** over 3,000 steps; RND's own reward normalization divides by a running std that grew in lockstep with the same blowup, silently producing a misleadingly clean-looking curve.
- **Root cause:** no BatchNorm/LayerNorm/ChannelNorm anywhere in the conv stack. Gradient-norm clipping (already present) bounded the *applied update* but not the *activations themselves* — confirmed via isolated stress tests at multiple learning rates (`1e-3` diverged in ~15 steps to 5 million; the documented "safe" `1e-5` also diverged, just over ~250-360 steps instead).
- **Fix:** added `ChannelNorm` (per-spatial-position normalization across channels, chosen over BatchNorm because this codebase's small, non-i.i.d. batches don't suit cross-batch statistics well) after every conv layer, including inside `ResidualBlock`.
- **Verification:** finite-difference check confirmed gradients remained correct after the fix (ratio 0.9996). The exact stress test that previously diverged from 0.45 to 56+ over 360 steps now converges monotonically to 0.53; the same test at `lr=1e-3` (previously exploded to 5 million by step 30) now converges to loss `2e-5` by step 135. Full 3000-step `train_visual.py` run confirmed latent norms stayed bounded at ~5.9-6.6 (vs. 500,000+ before).

### 4.4 `train_visual.py` — dead `encoder_lr` CLI argument
- **Symptom:** the parameter was accepted, documented, and passed through `run()`, but never reached `VisionEncoder`'s optimizer — `backward()` called `_set_optim()` with no override, silently defaulting to a hardcoded `1e-5` regardless of what was requested.
- **Fix:** `VisionEncoder.__init__` now accepts and stores `lr`; `_set_optim()` defaults to `self.lr` instead of a hardcoded literal.
- **Follow-on finding:** once wiring was fixed, the *previous default value* (`1e-3`) was tested and found to still diverge at real training scale even with `ChannelNorm` in place (latent norms reached ~2,000). The default was corrected to `1e-5`, matching what the module's own original comments had already (correctly) identified as the safe value — this was a case of a documented safe default never actually being applied.

---

## 5. Confirmed negative result: does clustering find concepts? (No — not yet.)

This is the most important honest finding of Phase 1 and is reported in full rather than summarized away.

**Test design:** after training the contrastive projector, cluster the resulting embeddings and check Normalized Mutual Information (NMI) against two label types:
- `NMI(clusters, action)` — expected to be high, since action identity is what the contrastive loss was trained on. Not interesting on its own.
- `NMI(clusters, context)` — the real test. `context` is a **local grid-content label computed directly from `env.grid`** (what's physically adjacent: wall / object A / object B / nothing), completely independent of the observation vector and never seen by D1, RND, the world model, or the contrastive projector.

**Attempt 1 — same-action pairing (original design):**
- `NMI(clusters, action) = 0.9988` — clusters essentially *are* the action label, restated.
- `NMI(clusters, context) = 0.0016` — statistically zero. No evidence of anything beyond a label echo.
- *(Note: an earlier version of this test used a broken context label — `gridworld._nearest_object()` never considers walls and has no distance cutoff, so "wall" and "nothing nearby" could structurally never occur. This was caught by inspecting the label distribution, found to be exactly zero in two categories, and fixed by computing context directly from grid-adjacency instead. The negative result above is from the corrected, trustworthy label.)*

**Attempt 2 — consequence-similarity pairing (redesign):** positive pairs = k-nearest neighbors by predicted next-state (from the world model), regardless of action identity.
- `NMI(clusters, action)` dropped to `0.0038` — confirms the redesign genuinely stopped echoing the action label.
- `NMI(clusters, context) = 0.0135` — still near zero. The redesign fixed one problem (trivial label echo) without solving the actual one (no context-relevant structure emerging).

**Root-cause diagnostic:** clustered the **raw 8-dim observation directly** (no GRU, no world model, no contrastive projector) against the same context label: `NMI = 0.0925`. This is higher than anything produced downstream. **Conclusion: the context information genuinely exists in the raw observation and is being destroyed by the GRU**, which has no incentive to preserve it — D1's scalar value objective doesn't need it, so nothing prevents it from being discarded.

## 6. Confirmed positive result: reconstruction pressure recovers it

Given the diagnosis in section 5, the fix tested was **not** a supervised auxiliary loss on the context label (which would be circular and would prove nothing about emergence) but a genuinely self-supervised one: a decoder trained to reconstruct the raw observation from `h`, with gradients flowing back into the GRU itself.

- Reconstruction loss trains cleanly: `0.675 -> 0.0096` over the run.
- `NMI(clusters-on-h, context)`: **D1-only baseline = 0.0101 -> D1+reconstruction = 0.0618** — a 6x improvement, recovering roughly two-thirds of the raw-observation ceiling (0.0925), without the reconstruction loss ever seeing the context label.
- D1's own stability confirmed unaffected by the added gradient signal on the shared GRU weights (TD error max 12.4, matching the no-reconstruction baseline exactly).

**Honest interpretation:** this is evidence that context information *can* survive in `h` given the right pressure, and that pressure does not need to be told in advance what "context" means. It is **not** yet evidence of concepts in any strong sense — recovering 66% of a raw-feature ceiling via reconstruction is a long way from clusters that correspond to interpretable categories a human would recognize. The next real test (not yet run) is whether reclustering *this* reconstruction-shaped `h` — combined with contrastive learning on top of it — produces a stronger context NMI than either intervention alone.

---

## 7. What Phase 1 has actually proven, stated plainly

1. A stable, non-diverging value-learning loop can run indefinitely on pure intrinsic reward (no task reward at all).
2. RND-driven curiosity is a real, measurable, decaying signal — not a placebo.
3. A forward world model can learn genuine one-step transition structure from an otherwise task-agnostic recurrent state.
4. Contrastive learning, done naively (same-action pairing), produces embeddings that are a restatement of the training label, not new structure — an important negative result, not a failure to hide.
5. The representational bottleneck for concept-relevant information is identifiable and specific: it's the GRU discarding information under a narrow (scalar-value) training pressure, not an absence of usable structure in the raw environment.
6. That bottleneck is at least partially, genuinely fixable with a non-circular self-supervised pressure (reconstruction), without hand-labeling what the model should preserve.

## 8. What Phase 1 has NOT proven (explicit, so nothing is overclaimed)

- No evidence of "understanding" in any general sense.
- No evidence of interpretable, human-recognizable concepts — only that spatial/object information can survive with the right pressure. Whether that survivorship organizes into anything resembling discrete categories is untested.
- The state-coverage-vs-random result (section 3, item 3) is a small, seed-dependent effect (12x12 grid ceiling limits headroom) — real but not a strong result.
- The delay net (D1's own mechanism) has not learned anything substantive yet — it's bootstrapped to a neutral tau=1 prior because no true delay-labeled signal exists without D3's causal history buffer. This is explicitly deferred, not solved.
- Nothing here generalizes beyond this one gridworld. No transfer test has been run.
- The visual track (image observations via `VisionEncoder`) is now numerically stable but has not yet been tested for the same concept-formation questions as the feature-vector track — it's a parallel, less-mature experiment.

---

## 9. Next phase: concrete plan

Ordered by dependency, not by ambition. Each item states what would specifically be tested and what would count as a pass/fail, matching the discipline used throughout Phase 1 rather than restating the original vision doc's phase list.

### 9.1 Immediate next step — combine reconstruction + contrastive, retest clustering
**What:** run `ContrastiveProjector` (consequence-pair version) on top of `h` that has *already* been shaped by `GRUReconstructionTrainer`, then recluster.
**Pass bar:** `NMI(clusters, context)` should exceed both individual interventions' results (0.0618 for reconstruction alone, 0.0135 for consequence-pairing alone) — ideally approaching the 0.0925 raw-observation ceiling.
**Fail condition:** if the combination doesn't beat the better of the two individual results, that's evidence the two pressures aren't complementary and a different combination strategy is needed.

### 9.2 Port the `QNetwork` sign-bug fix to the source project (separate from this repo)
**What:** the `QNetwork.backward_update` bug found in section 4.1 is still live in the vendored source project, affecting the DQN baseline used in CCPL's own benchmarks.
**Why this matters more than it looks:** if the DQN baseline was never learning correctly, prior "CCPL underperforms baselines" conclusions from earlier audit work may be invalid — this could change actual paper-facing results, not just this side project.
**Action:** apply the equivalent fix in a **separate, explicit patch to the source project** (not this repo), rerun the existing CCPL benchmark suite, compare before/after.

### 9.3 Scale up the coverage-vs-random test
**What:** the current result (section 3, item 3) is small and seed-dependent because the 12x12 grid gives random exploration too high a coverage floor.
**Plan:** rerun `sweep_coverage.py` with a larger grid (e.g. 20x20 or 30x30) and more seeds (10-20) for a statistically defensible mean+/-std, not 3 anecdotal seeds.

### 9.4 Visual track parity
**What:** the visual track (`VisionEncoder` -> `world_model_v2.py`) is now numerically stable but has never been run through the same concept-formation gauntlet (contrastive, clustering, NMI-against-independent-context) as the feature-vector track.
**Plan:** repeat sections 5-6's methodology on visual-track `h`, using pixel-level ground truth (e.g. is a wall/object visible in a fixed region of the rendered frame) as the independent label instead of `local_context_label()`.

### 9.5 Real delay learning (D1 + D3 integration)
**What:** the delay net is currently bootstrapped to a neutral prior because no real delay-labeled signal exists. This was explicitly deferred at the start of Phase 1 and remains deferred.
**Plan:** this is the actual bridge to Phase 5 (world model / cause-and-effect) in the original vision document. Requires building D3's causal attribution machinery (ICN, causal history buffer) — a substantially larger scope than anything in Phase 1, and should be scoped as its own phase with its own pass/fail bar before starting, not folded in incrementally.

### 9.6 Housekeeping
- Restructure the flat file layout into `core/` / `gridworld_track/` / `visual_track/` / `verify/` (proposed structure already written up separately).
- Add a `README.md` that states the current verified claims (section 3) and known limitations (section 8) up front, so the project's actual state is legible without reading this whole log.

---

## 10. Methodological notes (how this project stays honest)

These are working principles that were established through this session's actual mistakes, not written in advance — kept here so they're not forgotten on the next pass.

1. **Every verification script states its pass/fail bar before running, not after.** Several results in this log (contrastive collapse, consequence-pairing's partial failure) only have value because the bar was fixed in advance — moving goalposts after seeing a number is how negative results get quietly reframed as positive ones.
2. **A clean-looking curve is not evidence.** The vision-encoder divergence was completely invisible in the primary metric (intrinsic return -> 0.0000, which looked like *success*) and was only caught by directly inspecting latent magnitudes. Any metric that can be trivially satisfied by degenerate behavior (collapse, saturation) needs a second, independent check.
3. **Gradient correctness is not implied by a plausible-looking loss curve.** Two of the four bugs in section 4 (the sign flip, the LayerNorm dimension bug) produced loss curves that looked directionally reasonable for a while before diverging, or in one case produced no visible symptom at all until a different module needed the broken code path. Finite-difference checks against every new gradient path, not just "does the loss go down," is now standard practice for this project.
4. **Negative results stay in the repository and in this log.** `verify_contrastive.py` (the failed same-action pairing attempt) is kept, not deleted, specifically so the project's actual trajectory — including the wrong turns — is reconstructable.
