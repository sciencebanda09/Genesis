"""
contrastive.py — Step 2 of concept formation: pull together states whose
predicted futures are similar, push apart states whose predicted futures
diverge.

WHY THIS DEPENDS ON STEP 1 (world_model.py), not raw h:
Contrastive learning needs a notion of "these two states are alike" that
isn't circular. Using raw distance in h-space to decide what to pull
together would be meaningless -- h hasn't been shaped to have distance
mean anything yet at this point. The forward world model DOES give us a
real, verified (31x real-vs-shuffled gap) signal: predicted h_{t+1} under
the SAME action encodes something about transition dynamics. So: two
states h_i, h_j are treated as a "positive pair" if, under the same
action, the world model predicts similar next-states. This is a real,
non-circular basis for similarity, not an assumption.

MECHANISM (SimCLR-style, adapted for this small-scale setting):
  1. Project h through a small trainable projection head -> z (embedding).
  2. For a batch of (h_i, action_i), compute predicted h_next via the
     world model, then bucket transitions with the SAME action together.
  3. Within an action-bucket, treat all pairs as positives (their
     predicted-future similarity is implicit in sharing an action from
     similar starting conditions); everything outside the bucket in the
     batch is a negative.
  4. InfoNCE loss: pull positive z's together, push negative z's apart,
     via cosine similarity + temperature-scaled softmax cross-entropy.

WHAT "WORKING" MEANS HERE (checked in verify_contrastive.py before wiring
into train.py): embeddings z for states sharing an action-context should
have HIGHER cosine similarity on average than z's from different
action-contexts, and this gap should be larger after training than at
initialization. If there's no gap, the projection head isn't learning
anything the raw h didn't already have (or lacked).
"""
import numpy as np
from core.networks_min import MLP, Adam


class ContrastiveProjector:
    def __init__(self, gru_dim, embed_dim=16, hidden_dim=32, lr=1e-3,
                 temperature=0.2, seed=0):
        rng = np.random.default_rng(seed)
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.net = MLP([gru_dim, hidden_dim, embed_dim], rng)
        self.optim = Adam(self.net.all_params(), lr=lr)

    def embed(self, h_batch):
        """h_batch: (B, gru_dim) -> normalized embeddings (B, embed_dim)"""
        h_batch = np.asarray(h_batch, np.float32)
        z = self.net.forward(h_batch)
        norm = np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8
        return z / norm

    def update_step(self, h_batch, positive_mask):
        """
        InfoNCE-style contrastive update.

        positive_mask: (B, B) boolean matrix, positive_mask[i,j] = True if
        j is a positive pair for anchor i. Caller decides what "positive"
        means. Originally this was hardcoded to same-action pairing; that
        was tested and found to only rediscover the action label (NMI
        against action = 0.9988) with ZERO correlation to spatial/object
        context (NMI = 0.0016) -- i.e. no evidence of concept formation,
        just an echo of the training signal. Generalized so callers can
        supply predicted-consequence similarity (from the world model)
        instead, which is the principled fix: states are "alike" if their
        predicted futures are alike, not if the raw action label matches.
        Diagonal is ignored regardless of what's passed in. Returns None
        if no anchor has any positive pair in this batch.
        """
        h_batch = np.asarray(h_batch, np.float32)
        positive_mask = np.asarray(positive_mask, dtype=bool).copy()
        B = len(h_batch)
        np.fill_diagonal(positive_mask, False)

        z_raw = self.net.forward(h_batch)
        norm = np.linalg.norm(z_raw, axis=-1, keepdims=True) + 1e-8
        z = z_raw / norm

        sim = (z @ z.T) / self.temperature  # (B, B) cosine similarity matrix

        if not positive_mask.any():
            return None  # no positive pairs available in this batch

        # InfoNCE: for each anchor i, positives are positive_mask[i], all
        # others (excluding self) are negatives/denominator.
        mask_self = ~np.eye(B, dtype=bool)
        exp_sim = np.exp(sim - sim.max(axis=-1, keepdims=True))  # numerical stability
        exp_sim = exp_sim * mask_self

        denom = exp_sim.sum(axis=-1) + 1e-8
        pos_sum = (exp_sim * positive_mask).sum(axis=-1)

        valid = positive_mask.any(axis=-1)
        if not valid.any():
            return None

        log_prob = np.log(pos_sum[valid] + 1e-8) - np.log(denom[valid])
        loss = float(-log_prob.mean())

        # gradient of InfoNCE loss w.r.t. z (standard SimCLR-style form)
        n_pos = positive_mask.sum(axis=-1, keepdims=True).astype(np.float32)
        n_pos_safe = np.where(n_pos > 0, n_pos, 1.0)
        p_ij = exp_sim / denom[:, None]  # softmax probabilities
        target = positive_mask.astype(np.float32) / n_pos_safe
        d_sim = (p_ij - target) * mask_self
        d_sim = d_sim * valid[:, None]  # zero out invalid anchors
        d_sim = d_sim / max(valid.sum(), 1)

        d_z = (d_sim @ z + d_sim.T @ z) / self.temperature  # (B, embed_dim)

        # backprop through normalization: dz_raw = (I - z z^T)/norm * dz
        d_z_raw = (d_z - z * (z * d_z).sum(-1, keepdims=True)) / norm

        _, grads = self.net.backward(d_z_raw)
        self.optim.step(grads)
        return loss
