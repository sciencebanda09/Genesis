"""
verify_object_permanence.py — Verify object identity tracking.

Tests:
  1. Object creation: new objects get new slots
  2. Identity persistence: same object re-observed reuses its slot
  3. Occlusion: object persists when temporarily unseen
  4. Slot retirement: gone objects eventually despawn

Pass bar: object identities are preserved across observations,
occluded objects are remembered for their grace period.
"""
import numpy as np
import sys

from core.object_permanence import ObjectPermanence

PASS_BAR = "object identity tracking, persistence, and occlusion handling all operational"

def test_object_creation():
    tracker = ObjectPermanence(n_slots=5, feature_dim=4, max_invisible=3)
    # Observe two objects
    obj1 = (np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.5, 0.5]), 1, 0.9)
    obj2 = (np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.2, 0.8]), 2, 0.8)
    tracker.update([obj1, obj2], timestep=0)
    assert tracker.get_object_count() == 2, f"expected 2 objects, got {tracker.get_object_count()}"
    print(f"  Object creation: OK ({tracker.get_object_count()} objects)")

def test_identity_persistence():
    tracker = ObjectPermanence(n_slots=5, feature_dim=4, max_invisible=3)
    features = np.array([1.0, 0.0, 0.0, 0.0])
    obj = (features, np.array([0.5, 0.5]), 1, 0.9)
    tracker.update([obj], timestep=0)
    slot_id_0 = tracker.slots[0].id
    # Same object, slightly different position
    obj2 = (features + np.random.randn(4) * 0.05, np.array([0.55, 0.48]), 1, 0.85)
    tracker.update([obj2], timestep=1)
    is_known, matched_id = tracker.is_known_object(features)
    assert is_known, "should recognize same object"
    assert matched_id == slot_id_0, f"should reuse slot {slot_id_0}, got {matched_id}"
    print(f"  Identity persistence: OK (slot {slot_id_0} reused)")

def test_occlusion():
    tracker = ObjectPermanence(n_slots=5, feature_dim=4, max_invisible=5)
    features = np.array([1.0, 0.0, 0.0, 0.0])
    obj = (features, np.array([0.5, 0.5]), 1, 0.9)
    tracker.update([obj], timestep=0)
    # Object disappears for 3 steps
    tracker.update([], timestep=1)
    tracker.update([], timestep=2)
    tracker.update([], timestep=3)
    assert tracker.get_object_count() >= 1, "object should persist during occlusion"
    # Object reappears
    tracker.update([(features, np.array([0.51, 0.49]), 1, 0.85)], timestep=4)
    assert tracker.get_object_count() == 1, "object should be recognized on reappearance"
    print(f"  Occlusion: OK (persisted through 3 invisible steps)")

def test_slot_retirement():
    tracker = ObjectPermanence(n_slots=5, feature_dim=4, max_invisible=2)
    features = np.array([1.0, 0.0, 0.0, 0.0])
    obj = (features, np.array([0.5, 0.5]), 1, 0.9)
    tracker.update([obj], timestep=0)
    # Object gone for 3 steps (exceeds max_invisible=2)
    tracker.update([], timestep=1)
    tracker.update([], timestep=2)
    tracker.update([], timestep=3)
    assert tracker.get_object_count() == 0, "object should be retired"
    print(f"  Slot retirement: OK (object removed after {tracker.max_invisible} unseen steps)")

if __name__ == "__main__":
    print("verify_object_permanence.py — Object Permanence Verification")
    test_object_creation()
    test_identity_persistence()
    test_occlusion()
    test_slot_retirement()
    print(f"\nPASS: {PASS_BAR}")
