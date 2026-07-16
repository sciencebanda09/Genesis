"""
object_permanence.py — Object identity tracking across time and occlusion.

Cognitive parallel: Piaget's object permanence — the understanding that
objects continue to exist even when not directly observed.

Mechanism:
  - Slot-based object representations (slots = tracked objects)
  - Identity binding: match observed objects to existing slots via
    feature similarity + location prediction
  - Occlusion handling: slots persist for N steps without observation
  - Birth/death: slot creation for new objects, slot retirement for gone ones

This is the bridge: pixels → objects → concepts → knowledge.
Pixels feed in, object slots come out, concepts abstract over slots.
"""
import numpy as np


class ObjectSlot:
    """A single tracked object with identity."""
    def __init__(self, slot_id, features, position, confidence=1.0):
        self.id = slot_id
        self.features = np.asarray(features, np.float32).copy()
        self.position = np.asarray(position, np.float32).copy()
        self.velocity = np.zeros_like(position, np.float32)
        self.confidence = confidence
        self.age = 0
        self.last_seen = 0
        self.most_recent_type = None

    def predict_position(self):
        """Estimate current position based on last known position + velocity."""
        return self.position + self.velocity

    def update(self, features, position, confidence, timestep):
        old_pos = self.position.copy()
        self.features = np.asarray(features, np.float32).copy()
        self.position = np.asarray(position, np.float32).copy()
        self.velocity = self.position - old_pos
        self.confidence = confidence
        self.age += 1
        self.last_seen = timestep


class ObjectPermanence:
    """Tracks objects across observations, maintaining identity.

    Usage:
        tracker = ObjectPermanence(n_slots=10)
        for each observation:
            detected_objects = detect_objects(obs)  # [(features, position, type), ...]
            tracker.update(detected_objects, timestep)
            slots = tracker.get_active_slots()
    """
    def __init__(self, n_slots=10, feature_dim=16, max_invisible=5,
                 match_threshold=0.7, slot_inertia=0.3):
        self.n_slots = n_slots
        self.feature_dim = feature_dim
        self.max_invisible = max_invisible
        self.match_threshold = match_threshold
        self.slot_inertia = slot_inertia
        self.slots = []
        self._next_id = 0
        self._timestep = 0

    def _compute_similarity(self, features, slot_features):
        """Cosine similarity between observation features and slot features."""
        f = np.asarray(features, np.float32).ravel()
        s = np.asarray(slot_features, np.float32).ravel()
        fn = f / (np.linalg.norm(f) + 1e-8)
        sn = s / (np.linalg.norm(s) + 1e-8)
        return float(fn @ sn)

    def _position_distance(self, pos, slot):
        pred = slot.predict_position()
        return float(np.linalg.norm(np.asarray(pos, np.float32) - pred))

    def update(self, detected_objects, timestep=None):
        """Match detected objects to existing slots, create new ones as needed.

        Args:
            detected_objects: list of (features, position, type, confidence)
                features: (feature_dim,) ndarray
                position: (2,) ndarray (x, y) or similar
                type: int or str object type label
                confidence: float detection confidence
            timestep: int current time step
        """
        self._timestep = timestep if timestep is not None else self._timestep + 1
        matched_slots = set()
        unmatched_objects = []

        for feat, pos, obj_type, conf in detected_objects:
            best_slot = None
            best_score = -float("inf")

            for slot in self.slots:
                if slot.id in matched_slots:
                    continue
                sim = self._compute_similarity(feat, slot.features)
                pos_dist = self._position_distance(pos, slot)
                combined = sim - self.slot_inertia * pos_dist
                if combined > best_score and combined > self.match_threshold - 0.3:
                    best_score = combined
                    best_slot = slot

            if best_slot is not None and best_score > self.match_threshold:
                best_slot.update(feat, pos, conf, self._timestep)
                matched_slots.add(best_slot.id)
            else:
                unmatched_objects.append((feat, pos, obj_type, conf))

        # Create new slots for unmatched objects
        for feat, pos, obj_type, conf in unmatched_objects:
            if len(self.slots) < self.n_slots:
                slot = ObjectSlot(self._next_id, feat, pos, conf)
                slot.most_recent_type = obj_type
                slot.last_seen = self._timestep
                self.slots.append(slot)
                self._next_id += 1

        # Mark unmatched active slots as "unseen" — they persist via inertia
        for slot in self.slots:
            if slot.id not in matched_slots:
                slot.velocity *= 0.9

        # Retire slots that haven't been seen for too long
        self.slots = [s for s in self.slots 
                      if self._timestep - s.last_seen <= self.max_invisible]

    def get_active_slots(self):
        """Return slots with high confidence, sorted by age."""
        return sorted([s for s in self.slots if s.confidence > 0.3], 
                      key=lambda s: s.age, reverse=True)

    def get_object_count(self):
        return len(self.get_active_slots())

    def is_known_object(self, features, threshold=None):
        """Check if features match any existing object identity."""
        if threshold is None:
            threshold = self.match_threshold
        for slot in self.slots:
            sim = self._compute_similarity(features, slot.features)
            if sim > threshold:
                return True, slot.id
        return False, None

    def get_state(self):
        return {
            "n_active": self.get_object_count(),
            "n_slots": len(self.slots),
            "slots": [{"id": s.id, "age": s.age, "confidence": s.confidence,
                       "position": s.position.tolist()} for s in self.slots],
        }
