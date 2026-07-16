"""
planning.py — Imagine → Evaluate → Plan → Act pipeline.

Cognitive architecture for model-based planning:
  1. Observe — take current state (h)
  2. Imagine — use world model to simulate candidate futures
  3. Evaluate — score each imagined trajectory by value/goals
  4. Plan — select the best action sequence
  5. Act — execute first action, re-plan at next step

Replaces the current Observe → Act reflex with deliberation.

ponytail: random shooting for candidate generation. Cross-entropy method
(CEM) or MCTS if action space grows beyond 5 discrete actions.
"""
import numpy as np
from collections import deque


class Planner:
    """Imagine → Evaluate → Plan → Act loop.

    Uses the world model to simulate trajectories and the value function
    to evaluate them.
    """
    def __init__(self, world_model, value_fn=None, 
                 horizon=10, n_candidates=64, n_elites=8,
                 discount=0.99, replan_freq=1):
        self.wm = world_model
        self.value_fn = value_fn  # callable(h) -> value estimate
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.discount = discount
        self.replan_freq = replan_freq
        self._plan = deque()
        self._plan_count = 0
        self._rng = np.random.default_rng()

    def _evaluate_trajectory(self, h_trajectory, actions):
        """Score a candidate trajectory by cumulative discounted reward proxy.

        Uses value function if available, otherwise entropy/uncertainty as
        exploration bonus.

        ponytail: sum of predicted value at each step. A proper implementation
        would use TD(n) returns or Monte Carlo rollouts with the actual reward
        model.
        """
        if self.value_fn is not None:
            values = np.array([self.value_fn(h) for h in h_trajectory])
            discounts = self.discount ** np.arange(len(values))
            return float(np.sum(values * discounts))
        uncertainty = 0.0
        for t in range(len(h_trajectory) - 1):
            _, std = self.wm.predict_with_uncertainty(
                h_trajectory[t].reshape(1, -1), 
                np.array([actions[t]], np.int32))
            uncertainty += float(std.mean())
        return uncertainty

    def plan(self, h_current):
        """Generate a plan from current state.

        Returns (best_actions, best_trajectory, best_score)
        """
        h_current = np.asarray(h_current, np.float32).ravel()

        candidates_action_seqs = self._rng.integers(
            0, self.wm.action_dim, 
            size=(self.n_candidates, self.horizon)).astype(np.int32)

        scores = []
        trajectories = []
        for seq in candidates_action_seqs:
            traj, _ = self.wm.rollout(h_current, seq)
            score = self._evaluate_trajectory(traj, seq)
            scores.append(score)
            trajectories.append(traj)

        scores = np.array(scores)
        elite_idx = np.argsort(scores)[-self.n_elites:]
        elite_actions = candidates_action_seqs[elite_idx]
        elite_scores = scores[elite_idx]

        best_idx = int(elite_idx[-1])
        best_actions = candidates_action_seqs[best_idx]
        best_trajectory = trajectories[best_idx]

        # ponytail: return best of random shooting. CEM would iteratively
        # refit a distribution over the elite set for better samples.
        return best_actions, best_trajectory, float(elite_scores[-1])

    def act(self, h_current, force_replan=False):
        """Return the next action from the current plan, re-planning if needed.

        Usage:
            action = planner.act(agent_h)
            next_obs, _, done, _ = env.step(action)
        """
        self._plan_count += 1
        if force_replan or len(self._plan) == 0 or self._plan_count >= self.replan_freq:
            best_actions, traj, score = self.plan(h_current)
            self._plan = deque(best_actions)
            self._plan_count = 0

        if len(self._plan) > 0:
            return int(self._plan.popleft())
        return 0

    def get_state(self):
        return {
            "plan_length": len(self._plan),
            "horizon": self.horizon,
            "n_candidates": self.n_candidates,
        }
