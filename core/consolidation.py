"""
consolidation.py — Knowledge consolidation: Episodic → Semantic memory.

Cognitive parallel: hippocampal replay during rest/SWY waves. The agent
replays episodic traces offline, distilling regularities into semantic
knowledge (clusters, transition schemas, concept prototypes).

This is where the SEAL paper's outer-loop insight applies: consolidation
is itself a learnable process — which episodes to replay, how to compress
them, when to update schemas vs. keep exemplars.
"""
import numpy as np
from collections import deque, Counter
from core.clustering import OnlineKMeans


class KnowledgeConsolidation:
    """Consolidates episodic experiences into semantic knowledge.

    Pipeline:
        Experience → EpisodicMemory → Replay Buffer → Compression → SemanticMemory

    Three consolidation strategies:
      1. Prototype extraction — cluster h-states from recent episodes
      2. Schema induction — extract transition regularities
      3. Pruning — identify redundant/outdated knowledge

    ponytail: uniform sampling from recent episodes for consolidation.
    Future: importance-weighted replay based on prediction error or
    novelty (directly inspired by SEAL's adaptive edit policy).
    """
    def __init__(self, embed_dim=32, n_concepts=16, 
                 consolidate_every=50, replay_batch_size=128,
                 min_episodes_for_consolidation=5):
        self.embed_dim = embed_dim
        self.n_concepts = n_concepts
        self.consolidate_every = consolidate_every
        self.replay_batch_size = replay_batch_size
        self.min_episodes_for_consolidation = min_episodes_for_consolidation

        self.semantic = __import__('core.memory', fromlist=['SemanticMemory']).SemanticMemory(
            n_concepts=n_concepts, embed_dim=embed_dim)
        self._last_consolidation_step = 0
        self._consolidation_stats = deque(maxlen=100)

    def step(self, episodic_memory, global_step, h_states=None, transitions=None):
        """Run one consolidation cycle if enough steps have passed."""
        if global_step - self._last_consolidation_step < self.consolidate_every:
            return None
        if len(episodic_memory) < self.min_episodes_for_consolidation:
            return None

        self._last_consolidation_step = global_step
        stats = {}

        if h_states is not None:
            self._consolidate_concepts(h_states)
            stats["n_concepts_updated"] = self.n_concepts

        if transitions is not None:
            self._consolidate_schemas(transitions)
            stats["n_transitions_observed"] = len(transitions)

        self._consolidation_stats.append(stats)
        return stats

    def _consolidate_concepts(self, h_states):
        """Extract concept prototypes from hidden states."""
        h_states = np.asarray(h_states, np.float32).reshape(-1, self.embed_dim)
        if len(h_states) < 10:
            return
        self.semantic.update_concepts(h_states)

    def _consolidate_schemas(self, transitions):
        """Extract transition regularities as concept-level schemas."""
        for h_from, action, h_to in transitions:
            self.semantic.observe_transition(h_from, action, h_to)

    def replay_for_consolidation(self, episodic_memory, agent_buffer=None):
        """Replay a mix of episodic and buffer experience for consolidation.

        Returns (h_states, transitions) for concept and schema updates.
        """
        h_states = []
        transitions = []

        recent_eps = episodic_memory.recall_recent(k=5)
        for ep in recent_eps:
            if len(ep["hiddens"]) > 0:
                h_states.append(ep["hiddens"])
                for i in range(len(ep["hiddens"]) - 1):
                    transitions.append((ep["hiddens"][i], ep["actions"][i], ep["hiddens"][i + 1]))

        if agent_buffer is not None and len(agent_buffer) >= self.replay_batch_size:
            batch = agent_buffer.sample(self.replay_batch_size)
            if batch is not None:
                h_states.append(batch["hiddens"])
                for i in range(min(self.replay_batch_size, len(batch["hiddens"]) - 1)):
                    transitions.append((
                        batch["hiddens"][i], batch["actions"][i], batch["next_hiddens"][i]
                    ))

        all_h = np.concatenate([h.reshape(-1, self.embed_dim) for h in h_states]) if h_states else None
        return all_h, transitions

    def get_state(self):
        return {
            "semantic": self.semantic.get_state(),
            "last_consolidation_step": self._last_consolidation_step,
        }
