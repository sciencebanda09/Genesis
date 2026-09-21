"""Learned meta-control for the Genesis inner learning loop.

This module is deliberately separate from :mod:`executive_cortex.cortex`.
The Executive Cortex is an interpretable, hand-designed controller.  The
controller here learns a mapping from learning dynamics to regulation edits
using downstream progress measured by isolated candidate evaluations.

The important experimental invariant is that candidate evaluation is
counterfactual: an evaluator must construct fresh state (or an equivalent
copy) for every candidate.  No candidate is allowed to train the live agent.
"""

import numpy as np

from core.seal.regulation import REGULATION_EDIT_SPEC
from core.seal.self_edit_policy import SelfEditPolicy


class IsolatedCandidateEvaluator:
    """Adapter that makes fresh inner loops for every candidate.

    Parameters
    ----------
    inner_loop_factory:
        Callable receiving a seed and returning an object with ``run``.
        ``run`` receives ``(edit, seed_offset=0)`` and returns
        ``(reward, info)``.  The factory is called once per candidate.

    This small adapter exists to make the no-gradient-residue/no-shared-state
    rule explicit in experiment code and easy to test.
    """

    def __init__(self, inner_loop_factory, base_seed=0):
        self.inner_loop_factory = inner_loop_factory
        self.base_seed = int(base_seed)

    def __call__(self, edit, seed_offset=0):
        seed = self.base_seed + int(seed_offset)
        inner = self.inner_loop_factory(seed)
        return inner.run(np.asarray(edit, np.float32), seed_offset=0)


class LearnedMetaController:
    """A contextual, learned controller for inner-loop regulation.

    The policy proposes bounded regulation vectors.  Several noisy proposals
    are evaluated on isolated inner learners; proposals at or above the
    median downstream-progress reward become ReSTEM-style supervised targets.
    This is a lightweight contextual policy-improvement loop that works with
    Genesis's NumPy-only neural-network primitives.
    """

    def __init__(self, metric_dim=8, edit_dim=None, hidden_dim=32,
                 policy_lr=1e-3, exploration_std=0.12,
                 keep_top_fraction=0.5, seed=0):
        if edit_dim is None:
            edit_dim = REGULATION_EDIT_SPEC["dims"]
        if not 0.0 < keep_top_fraction <= 1.0:
            raise ValueError("keep_top_fraction must be in (0, 1]")
        self.metric_dim = int(metric_dim)
        self.edit_dim = int(edit_dim)
        self.exploration_std = float(exploration_std)
        self.keep_top_fraction = float(keep_top_fraction)
        self.rng = np.random.default_rng(seed)
        self.policy = SelfEditPolicy(
            metric_dim=self.metric_dim,
            edit_dim=self.edit_dim,
            hidden_dim=hidden_dim,
            lr=policy_lr,
            seed=seed,
        )
        self.history = []

    def propose(self, metric_state, n_candidates=5, exploration_std=None,
                incumbent=None):
        """Return candidate raw edits in the policy's [0, 1] parameter space.

        When an incumbent edit is supplied it is included verbatim as the
        first candidate.  This creates a trust-region safeguard: noisy
        candidate search cannot discard the currently deployed controller
        solely because a short evaluation window was unlucky.
        """
        state = np.asarray(metric_state, np.float32).reshape(-1)
        if state.size != self.metric_dim:
            raise ValueError(
                f"metric_state has {state.size} values; expected {self.metric_dim}")
        n_candidates = int(n_candidates)
        if n_candidates < 1:
            raise ValueError("n_candidates must be positive")
        sigma = self.exploration_std if exploration_std is None else float(exploration_std)
        center = self.policy.generate(state)
        edits = np.repeat(center[None, :], n_candidates, axis=0)
        if incumbent is not None:
            incumbent = np.asarray(incumbent, np.float32).reshape(-1)
            if incumbent.size != self.edit_dim:
                raise ValueError("incumbent has the wrong edit dimension")
            edits[0] = np.clip(incumbent, 0.0, 1.0)
        if n_candidates > 1 and sigma > 0.0:
            edits[1:] += self.rng.normal(0.0, sigma, size=(n_candidates - 1, self.edit_dim))
        return np.clip(edits, 0.0, 1.0).astype(np.float32)

    def update(self, metric_state, edits, rewards):
        """Train on above-median proposals and return update diagnostics."""
        state = np.asarray(metric_state, np.float32).reshape(-1)
        edits = np.asarray(edits, np.float32)
        rewards = np.asarray(rewards, np.float32).reshape(-1)
        if edits.ndim != 2 or edits.shape[1] != self.edit_dim:
            raise ValueError("edits must have shape (N, edit_dim)")
        if len(edits) != len(rewards) or len(rewards) == 0:
            raise ValueError("edits and rewards must be non-empty and aligned")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("rewards must be finite")

        n_keep = max(1, int(np.ceil(len(rewards) * self.keep_top_fraction)))
        order = np.argsort(rewards)[::-1]
        keep = order[:n_keep]
        states = np.repeat(state[None, :], len(keep), axis=0)
        loss = self.policy.train_on_edits(states, edits[keep])
        result = {
            "kept": int(len(keep)),
            "total": int(len(rewards)),
            "threshold": float(np.min(rewards[keep])),
            "policy_loss": float(loss),
            "best_reward": float(np.max(rewards)),
            "mean_reward": float(np.mean(rewards)),
        }
        return result

    def outer_step(self, metric_state, evaluator, n_candidates=5,
                   seed_offset=0, exploration_std=None, incumbent=None,
                   min_improvement=0.0):
        """Propose, isolate-evaluate, learn, and return the best intervention."""
        edits = self.propose(metric_state, n_candidates, exploration_std, incumbent)
        rewards = []
        infos = []
        for index, edit in enumerate(edits):
            reward, info = evaluator(edit, seed_offset=int(seed_offset) + index * 1000)
            rewards.append(float(reward))
            infos.append(info)
        rewards = np.asarray(rewards, np.float32)
        update = self.update(metric_state, edits, rewards)
        best = int(np.argmax(rewards))
        # A short rollout is a noisy estimate of downstream progress.  If an
        # incumbent was evaluated in slot zero, require a real improvement
        # before replacing it; otherwise the controller can thrash despite
        # seeing no evidence that the new edit is better.
        if incumbent is not None and rewards[best] < rewards[0] + float(min_improvement):
            best = 0
        result = {
            "best_index": best,
            "best_reward": float(rewards[best]),
            "mean_reward": float(rewards.mean()),
            "best_edit": edits[best].copy(),
            "rewards": rewards.copy(),
            "edits": edits.copy(),
            "infos": infos,
            "update": update,
        }
        self.history.append(result)
        return result


def scale_meta_edit(raw_vector, spec=None):
    """Scale a raw [0, 1] edit into the physical regulation ranges."""
    if spec is None:
        spec = REGULATION_EDIT_SPEC
    raw = np.asarray(raw_vector, np.float32)
    if raw.shape != (spec["dims"],):
        raise ValueError(f"raw edit must have shape ({spec['dims']},)")
    scaled = np.zeros_like(raw)
    for i, (lo, hi) in enumerate(spec["ranges"]):
        scaled[i] = lo + (hi - lo) * float(np.clip(raw[i], 0.0, 1.0))
    return scaled
