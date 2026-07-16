"""
goals.py — Multi-drive motivational system for Genesis.

Expands beyond Phase 1's single curiosity drive to a full set of
potentially competing goals that the Executive Cortex must balance:

    Curiosity  — explore novel/unpredictable states
    Safety     — avoid dangerous/harmful states
    Efficiency — minimize steps/resources to achieve outcomes
    Knowledge  — reduce model uncertainty / seek informative experiences
    Survival   — avoid termination / stay in viable states
    Exploration— cover the state space widely
    Prediction — minimize world model prediction error

Cognitive parallel: the basal ganglia / limbic system's multiple
reinforcement signals — not one reward, but a weighted ensemble.

The Executive Cortex adapts goal weights based on internal state
and environmental demands. No hardcoded priority scheme.
"""
import numpy as np


class GoalWeights:
    """Current goal weights — the motivational palette.

    Updated by the Executive Cortex through regulation cycles.
    """
    def __init__(self):
        self.curiosity = 1.0
        self.safety = 0.0
        self.efficiency = 0.0
        self.knowledge = 1.0
        self.survival = 0.0
        self.exploration = 1.0
        self.prediction = 1.0

    def as_dict(self):
        return {
            "curiosity": self.curiosity,
            "safety": self.safety,
            "efficiency": self.efficiency,
            "knowledge": self.knowledge,
            "survival": self.survival,
            "exploration": self.exploration,
            "prediction": self.prediction,
        }

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    def normalize(self):
        total = sum(self.as_dict().values())
        if total > 0:
            for k in vars(self):
                setattr(self, k, getattr(self, k) / total)


class GoalDrivenReward:
    """Combines multiple motivational signals into a composite reward.

    Each sub-system produces a reward signal in [0, 1] range.
    The composite reward = weighted sum of all active goals.

    Usage:
        gdr = GoalDrivenReward()
        reward = gdr.compute(
            curiosity_r=rnd_reward,
            safety_r=safety_signal,
            knowledge_r=uncertainty_reduction,
            ...
        )
    """
    def __init__(self, goal_weights=None):
        self.weights = goal_weights or GoalWeights()

    def compute(self, **goal_rewards):
        """Compute weighted sum of goal rewards.

        Args:
            **goal_rewards: {goal_name: reward_value} for active goals.
                Missing goals contribute 0.

        Returns:
            composite_reward: float weighted sum
            breakdown: dict per-goal weighted contributions
        """
        w = self.weights
        goal_map = {
            "curiosity_r": ("curiosity", w.curiosity),
            "safety_r": ("safety", w.safety),
            "efficiency_r": ("efficiency", w.efficiency),
            "knowledge_r": ("knowledge", w.knowledge),
            "survival_r": ("survival", w.survival),
            "exploration_r": ("exploration", w.exploration),
            "prediction_r": ("prediction", w.prediction),
        }
        total = 0.0
        breakdown = {}
        for key, (goal_name, weight) in goal_map.items():
            raw = goal_rewards.get(key, 0.0)
            contribution = weight * raw
            breakdown[goal_name] = {"weight": weight, "raw": raw, "contribution": contribution}
            total += contribution
        return total, breakdown

    def set_weights(self, **kwargs):
        self.weights.set(**kwargs)
        self.weights.normalize()

    def get_state(self):
        return {"weights": self.weights.as_dict()}
