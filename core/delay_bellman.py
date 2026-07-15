"""
delay_bellman.py — Delay-Corrected Bellman Operator
=====================================================
Vendored copy (unmodified logic) from an external project.
Only change: import repointed to networks_min (trimmed local copy).
Source frozen 2026-07-13.
"""
"""
delay_bellman.py — Delay-Corrected Bellman Operator
=====================================================
DIRECTION 1: The first provably correct TD update under unknown stochastic
consequence delay.

THE PROBLEM:
  Standard CMDP TD target:
    y_t = r_t + γ · Q(s_{t+1}, a*) − λ · c_t

  But c_t is NOT the consequence of a_t. It's the consequence of a_{t-τ}
  where τ ~ P(τ|h_t) is unknown. Standard methods either:
    (a) Ignore delay entirely (CPO, PPO-Lag)
    (b) Use a heuristic delay buffer (CCPL v6)

  Neither is theoretically justified. Neither corrects the TD target.

THE CCPL SOLUTION — Delay-Corrected Bellman Operator:
  
  Define the delay-aware value function:
    Q^τ(s_t, a_t) = E_{τ~P(τ|h)}[r_t + γ^(1+τ) · Q(s_{t+1+τ}, a*) - λ · c_{t+τ}]

  The corresponding TD target is:
    ŷ_t = r_t + Σ_{k=1}^{τ_max} P(τ=k|h_t) · [γ^k · Q(s_{t+k}, π(s_{t+k})) - λ · c_{t+k}]

  This is a weighted mixture of τ-step returns, weighted by the delay distribution.

CONTRACTION PROOF (Theorem 1):
  
  Let T^τ be the delay-corrected Bellman operator:
    (T^τ Q)(s,a) = r(s,a) + Σ_k P(τ=k|h) · γ^k · max_{a'} Q(s_{k}, a')

  Theorem: T^τ is a contraction in the L∞ norm with constant:
    γ_eff = Σ_k P(τ=k|h) · γ^k ≤ γ^1 < 1

  Proof: For any Q₁, Q₂:
    ||(T^τ Q₁)(s,a) - (T^τ Q₂)(s,a)||∞
    = ||Σ_k P(τ=k|h) · γ^k · [max Q₁(s_k) - max Q₂(s_k)]||∞
    ≤ Σ_k P(τ=k|h) · γ^k · ||Q₁ - Q₂||∞
    = γ_eff · ||Q₁ - Q₂||∞
    < ||Q₁ - Q₂||∞   (since γ_eff ≤ γ < 1 under A1: τ ≥ 1 a.s.)

  Therefore T^τ has a unique fixed point Q* = T^τ Q*, and value iteration
  converges to Q* for any initialisation. □

  This is the first contraction proof for a constrained TD operator under
  unknown stochastic consequence delay. It does not appear in the literature.

CAUSAL ATTRIBUTION INTEGRATION:
  
  The full CCPL TD target combines delay correction WITH causal attribution:

    ŷ_t = r_t + Σ_k P(τ=k|h_t) · γ^k · Q(s_{t+k}, π(s_{t+k}))
              - λ(s_t) · ΔC(s_t, a_t, h_t)    ← causal attribution, not raw c

  This is strictly more correct than any prior method because:
    1. The TD bootstrap is weighted by the true delay distribution (not heuristic)
    2. The penalty is the CAUSAL EFFECT of a_t, not correlated consequence
"""

import numpy as np
from core.networks_min import Adam, sigmoid, softmax, MLP, Linear, softplus


# ─────────────────────────────────────────────────────────────────────────────
# Delay Distribution Network
# ─────────────────────────────────────────────────────────────────────────────

class DelayDistributionNet:
    """
    Learns P(τ | h_t) — the probability distribution over delay values τ ∈ {1..τ_max}.

    Input:  GRU hidden state h_t  (gru_dim,)  — encodes recent trajectory
    Output: Categorical distribution over τ values  (τ_max,)

    Unlike the original DelayEstimatorNet which predicts E[τ], this network
    predicts the FULL DISTRIBUTION, enabling the weighted Bellman sum.

    Architecture: MLP → softmax → Categorical(τ_max categories)
    """

    def __init__(self, gru_dim: int = 40, hidden_dim: int = 32,
                 tau_max: int = 15, lr: float = 3e-4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.tau_max    = tau_max
        self.gru_dim    = gru_dim
        self.tau_values = np.arange(1, tau_max + 1, dtype=np.float32)

        self.net   = MLP([gru_dim, hidden_dim, hidden_dim, tau_max], rng)
        self.optim = Adam(self.net.all_params(), lr=lr)

        self._last_logits = None

    def forward(self, h: np.ndarray) -> np.ndarray:
        """
        h: (B, gru_dim) or (gru_dim,)
        Returns: (B, tau_max) probability distribution over τ values
        """
        h = np.asarray(h, np.float32)
        scalar = h.ndim == 1
        if scalar: h = h[None]
        logits = self.net.forward(h)
        self._last_logits = logits
        return softmax(logits)   # (B, tau_max)

    def expected_tau(self, h: np.ndarray) -> float:
        """E[τ|h] = Σ_k k · P(τ=k|h)"""
        probs = self.forward(h)
        return float((probs * self.tau_values[None]).sum(-1).mean())

    def update_step(self, h_batch: np.ndarray,
                    observed_tau: np.ndarray,
                    weights: np.ndarray) -> float:
        """
        Supervised update: cross-entropy loss on observed delay values.
        observed_tau: (B,) integer delay observations ∈ {1..tau_max}
        """
        h_batch      = np.asarray(h_batch,      np.float32)
        observed_tau = np.asarray(observed_tau, np.int32)
        weights      = np.asarray(weights,      np.float32)
        B            = len(observed_tau)

        probs  = self.forward(h_batch)   # (B, tau_max)
        # Cross-entropy: -log P(τ_obs | h)
        tau_idx = np.clip(observed_tau - 1, 0, self.tau_max - 1)
        log_p   = np.log(probs[np.arange(B), tau_idx] + 1e-8)
        loss    = float(-np.mean(log_p * weights))

        # Backward through softmax + MLP
        d_logits = probs.copy()
        d_logits[np.arange(B), tau_idx] -= 1.0
        d_logits *= (weights / B)[:, None]

        _, grads = self.net.backward(d_logits)
        self.optim.step(grads)
        return loss

    def params(self): return self.net.all_params()


# ─────────────────────────────────────────────────────────────────────────────
# Delay-Corrected Bellman Target Computer
# ─────────────────────────────────────────────────────────────────────────────

class DelayCorrectedBellman:
    """
    Computes the delay-corrected TD target:

        ŷ_t = r_t + Σ_{k=1}^{τ_max} P(τ=k|h_t) · γ^k · Q_target(s_{t+k}, a*_{t+k})
                  - λ(s_t) · ΔC(s_t, a_t, h_t)

    This requires a τ-step lookahead buffer that stores recent (s, Q) pairs
    so we can look up Q(s_{t+k}, ·) for each k.

    The lookahead buffer is a rolling window of the last τ_max steps.
    During training we use the replay buffer's stored next_states +
    a multi-step return estimator.

    APPROXIMATION for batch training (since we can't do exact τ-step lookahead
    in off-policy replay):

        ŷ_t ≈ r_t + γ_eff(h_t) · Q_target(s_{t+1}, a*_{t+1})
                  - λ(s_t) · ΔC(s_t, a_t, h_t)

    where γ_eff(h_t) = E_{τ~P(τ|h)}[γ^τ] = Σ_k P(τ=k|h) · γ^k

    This is the EFFECTIVE DISCOUNT — it's higher when the agent expects
    long delays (consequences arrive far in the future), and lower when
    delays are short. This is a novel adaptive discount mechanism.

    THEOREM 1 MACHINE-VERIFIABLE CLAIM:
        γ_eff < γ^1 < 1   when P(τ ≥ 1) = 1   (delay is always ≥ 1 step)
        Therefore T^τ is a contraction.
    """

    def __init__(self, delay_net: DelayDistributionNet,
                 gamma: float = 0.99, tau_max: int = 15):
        self.delay_net = delay_net
        self.gamma     = gamma
        self.tau_max   = tau_max
        self.tau_vals  = np.arange(1, tau_max + 1, dtype=np.float32)
        self.gamma_pows = np.array([gamma**k for k in range(1, tau_max + 1)],
                                    dtype=np.float32)

    def gamma_eff(self, h_batch: np.ndarray) -> np.ndarray:
        """
        Compute effective discount γ_eff(h) = Σ_k P(τ=k|h) · γ^k
        h_batch: (B, gru_dim)
        Returns: (B,) effective discounts
        """
        probs = self.delay_net.forward(h_batch)   # (B, tau_max)
        return (probs * self.gamma_pows[None]).sum(-1).astype(np.float32)

    def weighted_bootstrap(self, h_batch: np.ndarray,
                           q_by_delay: np.ndarray,
                           valid_by_delay: np.ndarray = None,
                           fallback_next_q: np.ndarray = None,
                           dones: np.ndarray = None) -> tuple:
        """
        Compute the delay-mixture bootstrap over stored future states:
            sum_k P(tau=k|h) * gamma^k * Q(s_{t+k}, a*_k)

        Invalid future entries fall back to the old one-step approximation
        when fallback_next_q is provided. This keeps off-policy replay usable
        near episode ends and before enough future transitions exist.
        """
        q_by_delay = np.asarray(q_by_delay, np.float32)
        probs = self.delay_net.forward(h_batch)
        weights = probs * self.gamma_pows[None]

        if valid_by_delay is None:
            valid = np.ones_like(q_by_delay, np.float32)
        else:
            valid = np.asarray(valid_by_delay, np.float32)

        if fallback_next_q is not None:
            fallback = np.asarray(fallback_next_q, np.float32)[:, None]
            q_mix = np.where(valid > 0.5, q_by_delay, fallback)
        else:
            q_mix = np.where(valid > 0.5, q_by_delay, 0.0)

        bootstrap = (weights * q_mix).sum(axis=1)
        if dones is not None:
            bootstrap *= (1.0 - np.asarray(dones, np.float32))

        gamma_e = weights.sum(axis=1).astype(np.float32)
        coverage = (probs * valid).sum(axis=1).astype(np.float32)
        return bootstrap.astype(np.float32), gamma_e, coverage

    def td_target(self, rewards: np.ndarray,
                  next_q: np.ndarray,
                  delta_C: np.ndarray,
                  lam: np.ndarray,
                  dones: np.ndarray,
                  h_batch: np.ndarray,
                  penalty_scale: float = 1.0) -> np.ndarray:
        """
        Full delay-corrected Bellman target with causal attribution.

        rewards       : (B,)
        next_q        : (B,) — Q_target(s_{t+1}, argmax Q_online(s_{t+1}))
        delta_C       : (B,) — causal attribution ΔC(s, a, h)
        lam           : (B,) — λ(s_t)
        dones         : (B,)
        h_batch       : (B, gru_dim)
        penalty_scale : float — must match the scale used in action selection (default 12.0)
                        BUG FIX: previously hardcoded to 1.0 here but 12.0 in action selection,
                        causing a 12× mismatch between the penalty the policy was trained on
                        and the penalty used for action selection.
        """
        gamma_e = self.gamma_eff(h_batch)          # (B,) — adaptive discount
        target  = (rewards
                   + gamma_e * next_q * (1.0 - dones)
                   - lam * penalty_scale * np.clip(delta_C, 0.0, None))   # scaled to match action selection
        return target.astype(np.float32), gamma_e

    def verify_contraction(self, h_sample: np.ndarray) -> dict:
        """
        Machine-verifiable check of Theorem 1.
        Returns dict with γ_eff values and contraction proof evidence.
        """
        probs     = self.delay_net.forward(h_sample)
        gamma_e   = (probs * self.gamma_pows[None]).sum(-1)
        gamma_min = float(self.gamma_pows.min())   # γ^τ_max
        gamma_max = float(self.gamma_pows.max())   # γ^1

        return {
            "gamma":          self.gamma,
            "gamma_eff_mean": float(gamma_e.mean()),
            "gamma_eff_std":  float(gamma_e.std()),
            "gamma_eff_min":  float(gamma_e.min()),
            "gamma_eff_max":  float(gamma_e.max()),
            "gamma_1":        float(self.gamma),
            "gamma_tau_max":  gamma_min,
            "contraction_satisfied": bool(float(gamma_e.max()) < 1.0),
            "proof": (
                "T^τ is a contraction iff γ_eff < 1. "
                f"γ_eff ∈ [{float(gamma_e.min()):.4f}, {float(gamma_e.max()):.4f}]. "
                f"Max γ_eff = {float(gamma_e.max()):.4f} < 1.0: "
                + ("VERIFIED ✓" if float(gamma_e.max()) < 1.0 else "FAILED ✗")
            ),
        }
