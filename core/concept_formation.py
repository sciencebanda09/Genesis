"""
concept_formation.py — Hierarchical concept learning pipeline.

Builds on Phase 1's clustering (which found NMI~0.01 on raw h) with:

    Pixels → Objects → Concepts → Knowledge

    pixels:   raw observation (8-dim state or 64x64 RGB image)
    objects:  object-centric representations (ObjectPermanence slots)
    concepts: discrete symbolic concepts (OnlineKMeans on object features)
    knowledge: concept-relation graph (transition schemas, affordances)

This replaces the flat h→clustering approach with a layered abstraction
pipeline that mirrors cognitive development.
"""
import numpy as np


class ConceptFormation:
    """Hierarchical concept pipeline.

    Works with or without an object-permanence tracker. When the tracker
    is present, concepts are formed over OBJECT features rather than raw
    h — this is the key improvement over Phase 1's direct h-clustering.

    Architecture:
        observation → [optional: object detection] → feature embedding
        → clustering → concept assignment → schema update

    ponytail: single-level clustering on object features. A full hierarchy
    (parts → objects → categories → relations) would use recursive
    clustering at multiple abstraction levels.
    """
    def __init__(self, embed_dim=32, n_concepts=16, use_objects=True):
        self.embed_dim = embed_dim
        self.n_concepts = n_concepts
        self.use_objects = use_objects
        self._concept_counter = np.zeros(n_concepts, np.float32)
        self._total_assignments = 0
        self._update_buffer = []
        self._update_batch_size = n_concepts * 2
        self._initialized = False
        self.rng = np.random.default_rng(42)
        from core.clustering import OnlineKMeans
        self.clusterer = OnlineKMeans(n_clusters=n_concepts, embed_dim=embed_dim)

    def embed(self, observation, object_slots=None):
        """Convert observation + optional object info into concept-space features.

        When object_slots are available, uses object features as the
        primary embedding signal. Otherwise falls back to raw observation.

        Returns: (embedding, metadata)
        """
        obs = np.asarray(observation, np.float32).ravel()
        if self.use_objects and object_slots is not None and len(object_slots) > 0:
            slot_features = [s.features for s in object_slots]
            obj_feats = np.stack(slot_features)
            concept_vec = obj_feats.mean(axis=0)
            if len(concept_vec) != self.embed_dim:
                pad = np.zeros(self.embed_dim, np.float32)
                pad[:min(len(concept_vec), self.embed_dim)] = concept_vec[:self.embed_dim]
                concept_vec = pad
            return concept_vec, {"source": "objects", "n_objects": len(object_slots)}
        if len(obs) < self.embed_dim:
            pad = np.zeros(self.embed_dim, np.float32)
            pad[:len(obs)] = obs
            return pad, {"source": "raw", "n_objects": 0}
        return obs[:self.embed_dim], {"source": "raw", "n_objects": 0}

    def assign_concept(self, embedding):
        """Assign an embedding to a discrete concept."""
        if not self._initialized:
            return int(self.rng.integers(self.n_concepts)) if hasattr(self, 'rng') else 0
        embedding = np.asarray(embedding, np.float32).reshape(1, -1)
        c = int(self.clusterer.assign(embedding)[0])
        self._concept_counter[c] += 1.0
        self._total_assignments += 1
        return c

    def update(self, embedding):
        """Update concept prototypes with a new embedding.
        
        Batches samples internally. First update initializes the clusterer
        once enough real data has been collected.
        """
        embedding = np.asarray(embedding, np.float32).ravel()
        self._update_buffer.append(embedding)
        if not self._initialized and len(self._update_buffer) >= self._update_batch_size:
            batch = np.stack(self._update_buffer)
            self.clusterer.update_step(batch)
            self._initialized = True
            self._update_buffer = []
        elif self._initialized and len(self._update_buffer) >= self._update_batch_size:
            batch = np.stack(self._update_buffer)
            self.clusterer.update_step(batch)
            self._update_buffer = []
        c = self.assign_concept(embedding)
        self._concept_counter[c] += 1.0
        self._total_assignments += 1
        return c

    def concept_frequency(self):
        """Normalized frequency of each concept assignment."""
        total = max(self._total_assignments, 1)
        return self._concept_counter / total

    def n_active_concepts(self):
        """Number of concepts that have been assigned at least once."""
        return int((self._concept_counter > 0).sum())

    def get_state(self):
        return {
            "n_active_concepts": self.n_active_concepts(),
            "n_concepts": self.n_concepts,
            "embed_dim": self.embed_dim,
            "total_assignments": self._total_assignments,
            "use_objects": self.use_objects,
        }
