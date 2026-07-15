"""
clustering.py — Step 3 of concept formation: discretize the contrastive
embedding space into stable, inspectable cluster prototypes.

WHY THIS DEPENDS ON STEP 2 (contrastive.py) working on world-model
features, not raw h: clustering needs an embedding space where distance
is meaningful. Raw h failed that bar (collapsed, no separable structure).
Contrastive-on-world-model-features passed it (verified gap +1.16 to
+1.21 across runs). Clustering here operates on THOSE embeddings.

WHAT WOULD MAKE THIS A REAL "CONCEPT," NOT JUST A RESTATEMENT OF THE
TRAINING SIGNAL: the contrastive projector was trained using ACTION
identity as its positive-pair signal. If clustering just rediscovers
5 clusters == 5 actions, that's not a new discovery -- it's the label
reflected back. The real test (done in verify_clustering.py) is whether
clusters correlate with something the projector was NEVER trained on:
spatial/object context (nearest_obj_type from the raw observation, which
never entered the contrastive loss). If clusters line up with THAT, it's
evidence of an emergent, non-trivial grouping -- something resembling a
concept rather than a label echo.

Mechanism: simple online k-means (no external deps). Not VQ-VAE-style
(no gradient through cluster assignment) -- kept simple and inspectable
for this phase; a differentiable VQ bottleneck is a reasonable later
upgrade, not required to test the core question above.
"""
import numpy as np


class OnlineKMeans:
    def __init__(self, n_clusters, embed_dim, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.n_clusters = n_clusters
        self.centers = rng.normal(0, 0.3, (n_clusters, embed_dim)).astype(np.float32)
        # re-normalize since embeddings from ContrastiveProjector are unit-norm
        self.centers /= (np.linalg.norm(self.centers, axis=-1, keepdims=True) + 1e-8)
        self.lr = lr
        self.counts = np.zeros(n_clusters, np.int64)

    def assign(self, z_batch):
        """z_batch: (B, embed_dim) unit-norm embeddings -> (B,) cluster indices."""
        z_batch = np.asarray(z_batch, np.float32)
        sims = z_batch @ self.centers.T  # cosine sim since both are (roughly) unit-norm
        return np.argmax(sims, axis=-1)

    def update_step(self, z_batch):
        """One online k-means update step (mini-batch, moving-average centers)."""
        z_batch = np.asarray(z_batch, np.float32)
        assignments = self.assign(z_batch)
        for k in range(self.n_clusters):
            mask = assignments == k
            if not mask.any():
                continue
            cluster_mean = z_batch[mask].mean(axis=0)
            self.counts[k] += mask.sum()
            # moving average toward this batch's cluster mean
            self.centers[k] = (1 - self.lr) * self.centers[k] + self.lr * cluster_mean
            norm = np.linalg.norm(self.centers[k]) + 1e-8
            self.centers[k] /= norm
        return assignments

    def cluster_sizes(self):
        return self.counts.copy()
