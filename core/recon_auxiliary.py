"""
recon_auxiliary.py — self-supervised auxiliary loss for h.

WHY THIS IS DIFFERENT FROM A CIRCULAR AUXILIARY: it does NOT use the
hand-labeled local_context_label() signal anywhere. It only uses the raw
observation the agent already receives (state_dim=8 for GridWorld),
which is the SAME data the GRU already consumes as input -- nothing new
is fed to the model, no extra supervision is invented. The pressure this
adds is: "h must retain enough information to reconstruct the observation
that produced it," which is a general-purpose, unsupervised objective, not
"h must predict wall-vs-object" (which would be circular given how the
verification test is built).

WHAT THIS TESTS: h was diagnosed as the bottleneck -- raw obs carried
context signal (NMI 0.093 via direct clustering) but h destroyed nearly
all of it (NMI 0.0016-0.0135 downstream), because D1's scalar value loss
gives h no reason to preserve anything except what's useful for value
prediction. A reconstruction loss is a genuinely different, orthogonal
pressure: it doesn't tell h WHAT to keep (context, action, or anything
else specifically) -- it just penalizes throwing information away in
general. If context survives under this pressure, that's real evidence
it emerges from information-preservation, not from being told to.

Architecture: small MLP decoder, h -> reconstructed obs (state_dim).
Loss: MSE(decoder(h), original_obs). Backprops into decoder only by
default (recon_only_decoder=True) OR into the GRU too if you want the
recurrent state itself to be shaped by this pressure (recon_only_decoder
=False) -- both modes are provided since it's a real design choice, not
obviously one-or-the-other.
"""
import numpy as np
from core.networks_min import MLP, Adam


class ReconstructionAuxiliary:
    def __init__(self, gru_dim, obs_dim, hidden_dim=32, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.gru_dim = gru_dim
        self.obs_dim = obs_dim
        self.decoder = MLP([gru_dim, hidden_dim, obs_dim], rng)
        self.optim = Adam(self.decoder.all_params(), lr=lr)

    def reconstruct(self, h_batch):
        return self.decoder.forward(np.asarray(h_batch, np.float32))

    def update_step(self, h_batch, obs_batch):
        """
        Decoder-only update: trains the decoder to reconstruct obs_batch
        from h_batch, but does NOT backprop into h itself (h_batch is
        treated as a fixed input here). This mode only tells you whether
        h ALREADY contains recoverable context info post-hoc -- it can't
        change what h encodes. Useful as a probe, not as a training signal
        for h. See GRUReconstructionTrainer for the version that actually
        shapes h during agent training.
        """
        h_batch = np.asarray(h_batch, np.float32)
        obs_batch = np.asarray(obs_batch, np.float32)

        pred = self.decoder.forward(h_batch)
        diff = pred - obs_batch
        loss = float(np.mean(diff ** 2))

        d_out = (2.0 / diff.shape[-1]) * diff / len(h_batch)
        _, grads = self.decoder.backward(d_out)
        self.optim.step(grads)
        return loss

    def probe_reconstruction_error(self, h_batch, obs_batch):
        pred = self.reconstruct(h_batch)
        obs_batch = np.asarray(obs_batch, np.float32)
        return np.mean((pred - obs_batch) ** 2, axis=-1)


class GRUReconstructionTrainer:
    """
    The version that actually shapes h, not just probes it.

    Calls gru.forward(obs, h_prev) directly (NOT the stateless
    gru_batch_forward helper used elsewhere) so GRUCell caches what it
    needs for backward(). Backprops the reconstruction loss through the
    decoder AND into the GRU cell's own weights, additively alongside
    whatever other objective (D1's value loss) is also training that same
    GRU. Kept as a fully separate module with its own optimizer -- does
    NOT touch D1Agent.update()'s existing, verified TD-learning path.
    This is deliberate: the two losses are independent pressures on the
    same shared weights, not fused into one combined loss function, so
    each can be checked/disabled independently.
    """
    def __init__(self, gru_cell, obs_dim, hidden_dim=32, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.gru = gru_cell
        self.obs_dim = obs_dim
        self.decoder = MLP([gru_cell.Wr.shape[1], hidden_dim, obs_dim], rng)
        all_params = self.decoder.all_params() + self.gru.params()
        self.optim = Adam(all_params, lr=lr)

    def update_step(self, obs_batch, h_prev_batch):
        """
        obs_batch: (B, obs_dim) -- the observation to reconstruct.
        h_prev_batch: (B, gru_dim) -- the GRU hidden state BEFORE this obs
        was consumed (i.e. h_before, matching what D1Agent.store() saves).

        Runs the GRU forward on (obs_batch, h_prev_batch) to get h_new,
        decodes h_new back to obs_batch, and backprops MSE reconstruction
        error through the decoder AND through the GRU cell.
        """
        obs_batch = np.asarray(obs_batch, np.float32)
        h_prev_batch = np.asarray(h_prev_batch, np.float32)

        h_new = self.gru.forward(obs_batch, h_prev_batch)  # caches internally
        pred = self.decoder.forward(h_new)
        diff = pred - obs_batch
        loss = float(np.mean(diff ** 2))

        d_pred = (2.0 / diff.shape[-1]) * diff / len(obs_batch)
        d_h_new, decoder_grads = self.decoder.backward(d_pred)
        _, _, gru_grads = self.gru.backward(d_h_new)

        all_grads = decoder_grads + gru_grads
        self.optim.step(all_grads)
        return loss
