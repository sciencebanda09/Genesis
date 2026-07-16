"""
self_model.py — Metacognitive self-model for Genesis.

What does the agent know? What doesn't it know? What is it trying to do?
How well is it learning?

Cognitive parallel: the prefrontal cortex's metacognitive layer — the
ability to reflect on one's own cognitive processes and form a coherent
self-representation.

Components:
  - Epistemic state: calibrated uncertainty per module
  - Performance tracking: learning curves, improvement rates
  - Self-description: compact representation of own capabilities
  - Gap detection: what the agent doesn't yet know or can't do
"""
import numpy as np
from collections import deque


class SelfModel:
    """Metacognitive self-monitoring for Genesis.

    Answers:
      - What do I know?   (confident concept assignments, mastered transitions)
      - What don't I know? (high-uncertainty states, rare transitions)
      - What am I trying to achieve? (active goals and their weights)
      - How well am I learning? (improvement rates, saturation)
    """
    def __init__(self, window=500):
        self.window = window
        self._loss_history = {}  # module_name -> deque of losses
        self._knowledge_boundary = {}  # (concept, action) -> exploration_count
        self._competence = {}  # skill_name -> success_rate
        self._performance_trend = {}  # module_name -> trend
        self._self_representation = None  # compact embedding of self-state

    def observe_loss(self, module_name, loss_value):
        """Track loss history for a module."""
        if module_name not in self._loss_history:
            self._loss_history[module_name] = deque(maxlen=self.window)
        self._loss_history[module_name].append(float(loss_value))

    def improvement_rate(self, module_name, window=100):
        """Rate of loss improvement. Positive = improving."""
        hist = self._loss_history.get(module_name)
        if hist is None or len(hist) < window + 1:
            return 0.0
        recent = list(hist)[-window:]
        return (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-8)

    def saturation_level(self, module_name, window=100):
        """How saturated a module is. 0 = still learning, 1 = plateaued."""
        hist = self._loss_history.get(module_name)
        if hist is None or len(hist) < window:
            return 0.0
        recent = list(hist)[-window:]
        if len(recent) < 2:
            return 0.0
        std = float(np.std(recent))
        mean = float(np.mean(recent))
        return float(np.clip(1.0 - std / max(mean, 1e-8), 0.0, 1.0))

    def confidence(self, module_name):
        """Calibrated confidence: 1 - normalized loss."""
        hist = self._loss_history.get(module_name)
        if hist is None or len(hist) < 10:
            return 0.5
        recent = list(hist)[-50:]
        normalized = float(np.clip(1.0 / (1.0 + np.mean(recent)), 0.0, 1.0))
        return normalized

    def know_what_i_know(self, semantic_memory=None):
        """Generate a self-description of known concepts and skills."""
        knowledge = {}
        if semantic_memory is not None:
            knowledge["n_concepts"] = semantic_memory.n_concepts
            knowledge["n_transitions"] = len(semantic_memory.transition_counts)
        knowledge["module_confidence"] = {
            k: self.confidence(k) for k in self._loss_history
        }
        knowledge["module_improvement"] = {
            k: self.improvement_rate(k) for k in self._loss_history
        }
        knowledge["module_saturation"] = {
            k: self.saturation_level(k) for k in self._loss_history
        }
        return knowledge

    def know_what_i_dont_know(self, semantic_memory=None, top_k=5):
        """Identify gaps in knowledge — things to learn next.

        Returns:
          - Under-explored concept-action pairs
          - Low-confidence module predictions
          - Rare or missing transitions
        """
        gaps = {"uncertain_modules": [], "rare_transitions": []}
        for module_name in self._loss_history:
            conf = self.confidence(module_name)
            if conf < 0.3:
                gaps["uncertain_modules"].append((module_name, conf))
        if semantic_memory is not None:
            rare = [(str(k), v) for k, v in 
                    semantic_memory.transition_counts.items() 
                    if sum(v.values()) < 3]
            gaps["rare_transitions"] = rare[:top_k]
        return gaps

    def update_competence(self, skill_name, success):
        """Track success/failure for a skill."""
        if skill_name not in self._competence:
            self._competence[skill_name] = deque(maxlen=50)
        self._competence[skill_name].append(float(success))

    def skill_mastery(self, skill_name):
        """Fraction of successful attempts for a skill."""
        hist = self._competence.get(skill_name)
        if hist is None or len(hist) < 5:
            return 0.0
        return float(np.mean(hist))

    def encode_self_state(self, recent_h_states):
        """Compact self-representation from recent hidden states.

        Returns a fixed-dim embedding summarizing the agent's internal state.
        """
        h = np.asarray(recent_h_states, np.float32)
        if len(h) == 0:
            return np.zeros(16, np.float32)
        rep = np.concatenate([
            h.mean(axis=0),
            h.std(axis=0),
        ])
        if len(rep) > 16:
            rep = rep[:16]
        elif len(rep) < 16:
            rep = np.pad(rep, (0, 16 - len(rep)))
        self._self_representation = rep
        return rep

    def current_goal_state(self, executive_cortex):
        """Summarize what the agent is currently trying to achieve."""
        return {
            "epsilon": executive_cortex.epsilon,
            "curiosity_weights": executive_cortex.curiosity_weights,
            "lr_factors": executive_cortex.lr_factors,
        }

    def get_state(self):
        return {
            "modules_tracked": list(self._loss_history.keys()),
            "skills": {k: self.skill_mastery(k) for k in self._competence},
        }
