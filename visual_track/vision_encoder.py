"""
vision_encoder.py -- Phase 2 vision encoder.

Transforms raw RGB images into 128-dimensional latent vectors that the
rest of the organism consumes. This is the equivalent of an infant's
early visual cortex: it must learn to extract spatial structure from
pixels without any semantic labels or coordinates.

Architecture:
    Input:  (C, H, W) float32 in [0, 1]
    Conv1:   3 -> 32, 5x5 stride 2, ReLU
    Conv2:   32 -> 64, 3x3 stride 2, ReLU
    ResBlock(64) -> 64
    Conv3:   64 -> 128, 3x3 stride 2, ReLU
    GlobalAvgPool -> 128
    Linear: 128 -> latent_dim (default 128)

Gradient flow (during training):
    WM loss -> d_latent -> encoder.backward() -> encoder weights update
    RND and D1 do NOT backprop into the encoder (RND would pull latents
    toward predictability, undermining exploration; D1's hidden states
    become inconsistent with latents computed by a different encoder
    version, and the WM signal is already sufficient for representation
    learning).
"""
import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def d_relu(x):
    return (x > 0).astype(np.float32)


# -- im2col-based Conv2d -------------------------------------------------------
# Chosen because it keeps the entire vision encoder in pure numpy without
# external dependencies. im2col is the standard CPU-path for conv backprop.
# Memory: for 64x64 input, B=64, the largest intermediate column buffer
# is ~38 MB (ResBlock), well within typical limits.


class Conv2d:
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, rng=None):
        self.in_c = in_c
        self.out_c = out_c
        self.kh = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.kw = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        self.sy = stride if isinstance(stride, int) else stride[0]
        self.sx = stride if isinstance(stride, int) else stride[1]
        self.py = padding if isinstance(padding, int) else padding[0]
        self.px = padding if isinstance(padding, int) else padding[1]

        if rng is None:
            rng = np.random.default_rng()
        scale = np.sqrt(2.0 / (in_c * self.kh * self.kw))
        self.W = rng.normal(0, scale, (out_c, in_c, self.kh, self.kw)).astype(np.float32)
        self.b = np.zeros(out_c, dtype=np.float32)

    def _im2col(self, x):
        B, C, H, W = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (self.py, self.py), (self.px, self.px)), mode='constant')
        Hp, Wp = H + 2 * self.py, W + 2 * self.px
        Ho = (Hp - self.kh) // self.sy + 1
        Wo = (Wp - self.kw) // self.sx + 1

        strides = (xp.strides[0], xp.strides[1],
                   xp.strides[2] * self.sy, xp.strides[3] * self.sx,
                   xp.strides[2], xp.strides[3])
        shape = (B, C, Ho, Wo, self.kh, self.kw)
        cols = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(B, C * self.kh * self.kw, Ho * Wo)
        return cols, Ho, Wo

    def forward(self, x):
        self._x = x
        B, C, H, W = x.shape
        cols, Ho, Wo = self._im2col(x)
        self._cols = cols
        self._oshape = (B, Ho, Wo)
        Wf = self.W.reshape(self.out_c, -1)
        y = Wf @ cols + self.b[:, None]
        return y.reshape(self.out_c, B, Ho, Wo).transpose(1, 0, 2, 3)

    def backward(self, d_out):
        B, C, H, W = self._x.shape
        Ho, Wo = self._oshape[1], self._oshape[2]
        d_flat = d_out.transpose(1, 0, 2, 3).reshape(self.out_c, -1)

        self._dW = (d_flat @ self._cols.transpose(0, 2, 1).reshape(-1, C * self.kh * self.kw)).reshape(self.W.shape)
        self._db = d_flat.sum(axis=1)

        Wf = self.W.reshape(self.out_c, -1)
        d_cols = (Wf.T @ d_flat).reshape(B, self.kh, self.kw, C, Ho, Wo).transpose(0, 3, 1, 2, 4, 5)

        xp = np.pad(self._x, ((0, 0), (0, 0), (self.py, self.py), (self.px, self.px)),
                    mode='constant')
        dxp = np.zeros_like(xp)

        for kh_ in range(self.kh):
            for kw_ in range(self.kw):
                y_start = kh_
                x_start = kw_
                dxp[:, :, y_start:y_start + Ho * self.sy:self.sy,
                    x_start:x_start + Wo * self.sx:self.sx] += d_cols[:, :, kh_, kw_]

        if self.py > 0 and self.px > 0:
            return dxp[:, :, self.py:-self.py, self.px:-self.px]
        elif self.py > 0:
            return dxp[:, :, self.py:-self.py, :]
        elif self.px > 0:
            return dxp[:, :, :, self.px:-self.px]
        return dxp

    def params(self):
        return [self.W, self.b]


class ChannelNorm:
    """
    Normalizes across the channel dimension at each spatial position:
    for input (B, C, H, W), each (b, :, h, w) vector is normalized to
    zero mean / unit variance across C, then scaled+shifted by learned
    per-channel gain/bias (broadcast over H, W).

    ADDED DURING VISION-PIPELINE AUDIT (2026-07-14): VisionEncoder had no
    normalization anywhere in its conv stack. Verified via finite-difference
    check that Conv2d's gradients are individually correct, and ResidualBlock
    alone is stable -- but the FULL encoder diverges under sustained training
    regardless of target (zero or random), batch (fixed or fresh each step),
    or LR (1e-3 diverges in ~15 steps, 1e-5 diverges more slowly but still
    diverges by ~step 250-360). Confirmed this also happens in the real
    train_visual.py loop: latent norms reached ~500,000+ by step 3000 in a
    60-episode run, silently masked by RND's own reward normalization
    (which divides by a running std that grows alongside the same blowup,
    making normalized intrinsic reward look like clean curiosity decay --
    "0.0000" -- when it's actually representational collapse/explosion).

    Root cause: nothing bounded activation magnitude directly at any layer;
    only the applied gradient step was small, which slows divergence but
    doesn't prevent it over enough steps. ChannelNorm after each conv fixes
    this at the source, the standard remedy for this class of instability
    in conv stacks (this codebase avoids BatchNorm because its cross-batch
    statistics don't suit the small, non-i.i.d. batches used here; per-
    position channel normalization needs no batch statistics and is a
    standard alternative for this setting).
    """
    def __init__(self, channels, eps=1e-5):
        self.g = np.ones((1, channels, 1, 1), np.float32)
        self.b = np.zeros((1, channels, 1, 1), np.float32)
        self.eps = eps
        self._cache = None

    def forward(self, x):
        mu = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_hat = (x - mu) / std
        self._cache = (x_hat, std, x.shape[1])
        return self.g * x_hat + self.b

    def backward(self, d_out):
        x_hat, std, C = self._cache
        dg = (d_out * x_hat).sum(axis=(0, 2, 3), keepdims=True)
        db = d_out.sum(axis=(0, 2, 3), keepdims=True)
        dx_hat = d_out * self.g
        dx = (1.0 / (C * std)) * (
            C * dx_hat
            - dx_hat.sum(axis=1, keepdims=True)
            - x_hat * (dx_hat * x_hat).sum(axis=1, keepdims=True)
        )
        return dx, dg, db

    def params(self):
        return [self.g, self.b]


class ResidualBlock:
    """Conv -> Norm -> ReLU -> Conv -> Norm -> skip-add -> ReLU. Preserves spatial dims."""
    def __init__(self, channels, rng=None):
        self.c1 = Conv2d(channels, channels, 3, stride=1, padding=1, rng=rng)
        self.n1 = ChannelNorm(channels)
        self.c2 = Conv2d(channels, channels, 3, stride=1, padding=1, rng=rng)
        self.n2 = ChannelNorm(channels)

    def forward(self, x):
        a = self.c1.forward(x)
        a = self.n1.forward(a)
        self._c1_out = a
        b = relu(a)
        c = self.c2.forward(b)
        c = self.n2.forward(c)
        self._c2_out = c
        self._sum_x_c2 = x + c
        return relu(self._sum_x_c2)

    def backward(self, d_out):
        d_pre = d_out * d_relu(self._sum_x_c2)
        d_skip = d_pre
        d_c = d_pre
        d_c, dg2, db2 = self.n2.backward(d_c)
        d_b = self.c2.backward(d_c)
        d_a = d_b * d_relu(self._c1_out)
        d_a, dg1, db1 = self.n1.backward(d_a)
        d_x_conv = self.c1.backward(d_a)
        self._dg1, self._db1, self._dg2, self._db2 = dg1, db1, dg2, db2
        return d_skip + d_x_conv

    def params(self):
        return (self.c1.params() + self.n1.params() +
                self.c2.params() + self.n2.params())


class VisionEncoder:
    """CNN encoder: image (C,H,W) float32 [0,1] -> latent vector.

    The encoder is trained by the LatentWorldModel prediction loss.
    RND and D1 gradients do NOT flow back through the encoder.
    """
    def __init__(self, img_channels=3, latent_dim=128, lr=1e-5, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.latent_dim = latent_dim
        self.lr = lr  # BUG FIX (audit 2026-07-14): train_visual.py's encoder_lr
        # argument was accepted by run() but never reached the encoder --
        # backward() called self._set_optim() with no lr override, which
        # silently defaulted to 1e-5 regardless of what the caller passed.
        # Storing lr here and having _set_optim() default to self.lr closes
        # that gap; VisionEncoder(lr=...) or train_visual.py's encoder_lr
        # now actually takes effect.
        self.conv1 = Conv2d(img_channels, 32, 5, stride=2, padding=2, rng=rng)
        self.norm1 = ChannelNorm(32)
        self.conv2 = Conv2d(32, 64, 3, stride=2, padding=1, rng=rng)
        self.norm2 = ChannelNorm(64)
        self.res = ResidualBlock(64, rng=rng)
        self.conv3 = Conv2d(64, 128, 3, stride=2, padding=1, rng=rng)
        self.norm3 = ChannelNorm(128)
        self.proj = self._Linear(128, latent_dim, rng)
        self.optim = None

    class _Linear:
        def __init__(self, in_dim, out_dim, rng, scale=None):
            scale = scale or np.sqrt(2.0 / in_dim)
            self.W = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
            self.b = np.zeros(out_dim, np.float32)

        def forward(self, x):
            self._x = x
            return x @ self.W + self.b

        def backward(self, d_out):
            dW = self._x.T @ d_out
            db = d_out.sum(axis=0)
            dx = d_out @ self.W.T
            return dx, dW, db

        def params(self):
            return [self.W, self.b]

    def _set_optim(self, lr=None):
        """Lazy init for the optimizer (avoids circular import at module level).

        LR is 100x lower than the WM's LR (1e-3) by default. The joint training
        of encoder + world model suffers from a moving-target problem: every
        encoder update shifts ALL latent representations, which invalidates
        the WM's training targets for all future batches. A low encoder LR
        forces the latent space to change slowly enough for the WM to track.

        Even with this LR, the encoder has 185K params vs the WM's 34K, so the
        encoder still has more total representational capacity changing per step.
        """
        if self.optim is not None:
            return
        from core.networks_min import Adam
        self.optim = Adam(self.params(), lr=lr if lr is not None else self.lr)



    def params(self):
        return (self.conv1.params() + self.norm1.params() +
                self.conv2.params() + self.norm2.params() +
                self.res.params() +
                self.conv3.params() + self.norm3.params() +
                self.proj.params())

    def encode(self, image):
        """Encode a single (H, W, C) uint8 image -> (latent_dim,) latent."""
        if image.ndim == 3:
            image = image[None]
        x = image.astype(np.float32) / 255.0
        x = x.transpose(0, 3, 1, 2)
        return self.forward(x)[0]

    def forward(self, x):
        """x: (B, C, H, W) float32 [0, 1] -> (B, latent_dim) latent.

        Stores pre-activation outputs (post-norm, pre-ReLU) for correct
        ReLU backprop.
        """
        c1 = self.conv1.forward(x)
        c1 = self.norm1.forward(c1)
        self._c1_out = c1
        c1r = relu(c1)
        c2 = self.conv2.forward(c1r)
        c2 = self.norm2.forward(c2)
        self._c2_out = c2
        c2r = relu(c2)
        c3 = self.res.forward(c2r)
        c3 = self.conv3.forward(c3)
        c3 = self.norm3.forward(c3)
        self._c3_out = c3
        c3r = relu(c3)
        pooled = c3r.mean(axis=(2, 3))
        return self.proj.forward(pooled)

    def backward(self, d_latent):
        """Backprop through encoder, compute gradients, update weights.

        d_latent: (B, latent_dim) gradient w.r.t. encoder output.

        Gradient normalization: d_latent is scaled so its L2 norm (averaged over
        the batch) is bounded by MAX_GRAD_NORM. This prevents the 4 conv backward
        passes from amplifying small WM gradients into destructive updates.

        ChannelNorm after every conv (added during vision-pipeline audit, see
        ChannelNorm docstring) bounds activation magnitude directly at each
        layer, fixing the root cause of the divergence found during testing --
        gradient clipping alone slowed divergence but did not prevent it.
        """
        self._set_optim()
        MAX_GRAD_NORM = 1.0
        gn = float(np.sqrt(np.mean(d_latent ** 2))) + 1e-8
        if gn > MAX_GRAD_NORM:
            d_latent = d_latent * (MAX_GRAD_NORM / gn)

        dx, dWp, dbp = self.proj.backward(d_latent)
        B, C_, H3, W3 = self._c3_out.shape
        dx = dx.reshape(B, -1, 1, 1) / (H3 * W3)
        dx = np.broadcast_to(dx, (B, C_, H3, W3)).copy()
        dx = dx * d_relu(self._c3_out)
        dx, dg3, db3 = self.norm3.backward(dx)
        dx = self.conv3.backward(dx)
        dx = self.res.backward(dx)
        _, C2, H2, W2 = self._c2_out.shape
        dx = dx * d_relu(self._c2_out)
        dx, dg2, db2 = self.norm2.backward(dx)
        dx = self.conv2.backward(dx)
        _, C1, H1, W1 = self._c1_out.shape
        dx = dx * d_relu(self._c1_out)
        dx, dg1, db1 = self.norm1.backward(dx)
        dx = self.conv1.backward(dx)

        all_grads = [self.conv1._dW, self.conv1._db, dg1, db1,
                     self.conv2._dW, self.conv2._db, dg2, db2,
                     self.res.c1._dW, self.res.c1._db, self.res._dg1, self.res._db1,
                     self.res.c2._dW, self.res.c2._db, self.res._dg2, self.res._db2,
                     self.conv3._dW, self.conv3._db, dg3, db3,
                     dWp, dbp]
        self.optim.step(all_grads)

    def update(self, images, d_latent):
        """Convenience: forward then backward on a batch."""
        self.forward(images)
        self.backward(d_latent)
