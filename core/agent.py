"""
agent.py — Phase 1 "newborn": GRU dueling-Q policy trained with CCPL's D1
(delay-corrected Bellman operator) against a pure intrinsic (RND) reward.

Explicitly NOT included (deferred to a later integration phase):
    D2 (state-conditioned lambda)  -- no constraint signal exists yet
    D3 (causal attribution / ICN)  -- delta_C is hardcoded to 0 below
    D4 (dual Q-functions)          -- single Q-head, no separate cost Q

delta_C=0, lam=0 means D1's td_target() call degrades to:
    y_t = r_t + gamma_eff(h_t) * Q_target(s_{t+1}, a*) * (1 - done)

which is exactly "delay-corrected DQN with adaptive discount, no penalty
term" -- i.e. genuinely D1 in isolation, not D1 wrapped in dead code paths.
"""
import numpy as np
from core.networks_min import GRUPolicyNet, gru_batch_forward
from core.delay_bellman import DelayDistributionNet, DelayCorrectedBellman
from core.replay_buffer import ReplayBuffer


class D1Agent:
    def __init__(self, state_dim, action_dim, gru_dim=32, hidden_dim=64,
                 tau_max=10, lr_policy=1e-3, lr_delay=3e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=3000,
                 batch_size=64, buffer_capacity=20_000,
                 target_update_tau=0.01, td_target_clip=50.0, seed=0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gru_dim = gru_dim
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.batch_size = batch_size
        self.target_update_tau = target_update_tau
        self.td_target_clip = td_target_clip

        self.rng = np.random.default_rng(seed)
        self.steps_done = 0

        self.policy_net = GRUPolicyNet(state_dim, action_dim, gru_dim, hidden_dim,
                                        n_layers=2, lr=lr_policy, seed=seed)
        self.target_net = GRUPolicyNet(state_dim, action_dim, gru_dim, hidden_dim,
                                        n_layers=2, lr=lr_policy, seed=seed)
        self.target_net.copy_weights_from(self.policy_net)

        # D1: delay distribution + delay-corrected Bellman target
        self.delay_dist = DelayDistributionNet(gru_dim=gru_dim, hidden_dim=32,
                                                tau_max=tau_max, lr=lr_delay, seed=seed)
        self.bellman = DelayCorrectedBellman(self.delay_dist, gamma=gamma, tau_max=tau_max)

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, gru_dim, seed=seed)
        self._h = self.policy_net.zero_state(1)

    def epsilon(self):
        frac = min(1.0, self.steps_done / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def reset_hidden(self):
        self._h = self.policy_net.zero_state(1)

    def select_action(self, obs: np.ndarray) -> int:
        eps = self.epsilon()
        if self.rng.random() < eps:
            action = int(self.rng.integers(self.action_dim))
            # still advance hidden state so h stays consistent with the
            # trajectory actually taken, even on random actions
            _, self._h = self.policy_net.forward(obs, self._h)
            return action
        q, h_new = self.policy_net.forward(obs, self._h)
        self._h = h_new
        return int(np.argmax(q[0]))

    def store(self, state, action, reward, next_state, done, h_before, h_after):
        self.buffer.add(state, h_before, action, reward, next_state, h_after, done)

    def update(self, batch=None):
        if batch is None:
            batch = self.buffer.sample(self.batch_size)
            if batch is None:
                return None

        S, NS = batch["states"], batch["next_states"]
        A, R, D = batch["actions"], batch["rewards"], batch["dones"]
        H, NH = batch["hiddens"], batch["next_hiddens"]
        W = batch["weights"]
        B = len(A)

        # current hidden states used for delay-net conditioning
        H_cur = gru_batch_forward(self.policy_net.gru, S, H)

        # target Q at next state, using target net's GRU + trunk
        H_next_pol = gru_batch_forward(self.policy_net.gru, NS, NH)
        H_next_tgt = gru_batch_forward(self.target_net.gru, NS, NH)

        feat_pol = self.policy_net.trunk.forward(H_next_pol)
        q_next_online = (self.policy_net.val_head.forward(feat_pol) +
                          self.policy_net.adv_head.forward(feat_pol))
        q_next_online -= q_next_online.mean(axis=-1, keepdims=True)
        next_actions = q_next_online.argmax(axis=-1)

        feat_tgt = self.target_net.trunk.forward(H_next_tgt)
        q_next_tgt = (self.target_net.val_head.forward(feat_tgt) +
                      self.target_net.adv_head.forward(feat_tgt))
        q_next_tgt -= q_next_tgt.mean(axis=-1, keepdims=True)
        next_q = q_next_tgt[np.arange(B), next_actions]
        # ROOT CAUSE of the divergence found in smoke testing: clipping the
        # assembled target (previous attempt) did NOT stop divergence,
        # because the runaway quantity is next_q itself. Loop: an inflated
        # target pushes policy_net up -> soft-update carries that into
        # target_net -> next_q grows -> next target grows further. This is
        # the standard DQN overestimation bootstrap trap; D1's gamma_eff
        # (~0.94-0.99) isn't low enough on its own to damp it against a
        # fixed Adam LR. Clipping next_q (the actual bootstrapped quantity)
        # breaks the loop at its source instead of patching the symptom.
        next_q = np.clip(next_q, -self.td_target_clip, self.td_target_clip)

        # D1: delay-corrected TD target. delta_C=0, lam=0 -> no D3/D2 term.
        delta_C = np.zeros(B, np.float32)
        lam = np.zeros(B, np.float32)
        target, gamma_e = self.bellman.td_target(
            rewards=R, next_q=next_q, delta_C=delta_C, lam=lam,
            dones=D, h_batch=H_cur, penalty_scale=1.0,
        )

        # current Q(s,a) for TD error
        feat_cur = self.policy_net.trunk.forward(H_cur)
        q_cur = (self.policy_net.val_head.forward(feat_cur) +
                 self.policy_net.adv_head.forward(feat_cur))
        q_cur -= q_cur.mean(axis=-1, keepdims=True)
        q_sa = q_cur[np.arange(B), A]

        td_error = q_sa - target

        # update policy net (Q-head + GRU + trunk)
        self.policy_net.backward_update(S, H, A, td_error, W)

        # update delay distribution net: supervised on observed 1-step "delay"
        # NOTE: without a real delay-labeled signal (that's what D3's causal
        # history buffer would normally supply), we can't yet train the delay
        # net on *true* consequence delay. For a pure-curiosity D1-only phase,
        # we bootstrap it toward tau=1 (immediate credit) as a neutral prior --
        # this keeps gamma_eff well-defined and the contraction guarantee
        # intact, but it means the delay net isn't learning anything
        # substantive yet. Flagging this honestly: real delay learning is a
        # D1+D3 integration task, not something this phase can do standalone.
        observed_tau = np.ones(B, np.int32)
        delay_loss = self.delay_dist.update_step(H_cur, observed_tau, W)

        self.target_net.soft_update_from(self.policy_net, tau=self.target_update_tau)
        self.steps_done += 1

        return {
            "td_error_mean": float(np.mean(np.abs(td_error))),
            "td_error": td_error,
            "gamma_eff_mean": float(gamma_e.mean()),
            "delay_loss": delay_loss,
            "q_mean": float(q_sa.mean()),
        }
