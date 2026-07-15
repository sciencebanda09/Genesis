"""
clustering.py — Step 3 of concept formation: discretize the embedding
space into stable, inspectable cluster prototypes.

Wraps sklearn.cluster.MiniBatchKMeans for a consistent interface with
the rest of the Genesis pipeline. Replaces an earlier hand-rolled
OnlineKMeans that did moving-average centroid updates — sklearn handles
edge cases (empty clusters, convergence) correctly.
"""
from sklearn.cluster import MiniBatchKMeans
import numpy as np


class OnlineKMeans:
    """Thin wrapper around MiniBatchKMeans with assign/update_step API."""
    def __init__(self, n_clusters, embed_dim, lr=0.05, seed=0, batch_size=256):
        self.n_clusters = n_clusters
        self.embed_dim = embed_dim
        self._km = MiniBatchKMeans(
            n_clusters=n_clusters, init="random", n_init=1,
            batch_size=batch_size, random_state=seed, max_iter=1,
            tol=0.0, reassignment_ratio=0.0,
        )
        self._initialized = False

    def assign(self, z_batch):
        z_batch = np.asarray(z_batch, np.float32)
        return self._km.predict(z_batch)

    def update_step(self, z_batch):
        z_batch = np.asarray(z_batch, np.float32)
        self._km.partial_fit(z_batch)
        self._initialized = True
        return self._km.labels_

    def cluster_sizes(self):
        return np.bincount(self._km.labels_, minlength=self.n_clusters)
