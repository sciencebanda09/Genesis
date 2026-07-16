"""
memory.py — Cognitive memory system for Genesis.

Four-level hierarchy replacing flat replay buffers:

    Working Memory    — immediate context, last N steps (attention focus)
    Episodic Memory   — auto-biographical trajectories with time-based indexing
    Semantic Memory   — abstracted concepts, regularities, schemas
    Procedural Memory — skill policies, action sequences

Replay buffers are storage. This is a cognitive architecture.
"""
import numpy as np
from collections import deque, defaultdict
from core.clustering import OnlineKMeans


class WorkingMemory:
    """Immediate context buffer — the last N observations and actions.

    Cognitive parallel: the contents of consciousness / attentional focus.
    Limited capacity (~10-20 items). Used for current-state understanding
    and short-range temporal coherence.
    """
    def __init__(self, capacity=20, state_dim=8, gru_dim=32):
        self.capacity = capacity
        self.state_dim = state_dim
        self.gru_dim = gru_dim
        self.reset()

    def reset(self):
        self.states = deque(maxlen=self.capacity)
        self.hiddens = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.timesteps = deque(maxlen=self.capacity)

    def push(self, state, hidden, action, reward, timestep):
        self.states.append(state.copy() if isinstance(state, np.ndarray) else state)
        self.hiddens.append(hidden.copy() if isinstance(hidden, np.ndarray) else hidden)
        self.actions.append(action)
        self.rewards.append(reward)
        self.timesteps.append(timestep)

    def get_state_sequence(self):
        return np.array(self.states, np.float32) if self.states else np.zeros((0, self.state_dim), np.float32)

    def get_hidden_sequence(self):
        return np.array(self.hiddens, np.float32) if self.hiddens else np.zeros((0, self.gru_dim), np.float32)

    def get_recent_context(self, n=None):
        if n is None:
            n = self.capacity
        n = min(n, len(self.states))
        if n == 0:
            return {}
        return {
            "states": np.array(list(self.states)[-n:], np.float32),
            "hiddens": np.array(list(self.hiddens)[-n:], np.float32),
            "actions": np.array(list(self.actions)[-n:], np.int32),
            "rewards": np.array(list(self.rewards)[-n:], np.float32),
        }

    def __len__(self):
        return len(self.states)


class EpisodicMemory:
    """Auto-biographical trajectory store with time-indexed retrieval.

    Stores complete episode trajectories. Supports recall by:
      - recency (most recent episodes)
      - similarity (episodes with similar hidden-state trajectories)
      - salience (high-reward or high-novelty episodes)

    ponytail: linear scan for similarity retrieval. KD-tree or HNSW if
    scale exceeds 10k episodes.
    """
    def __init__(self, max_episodes=500, similarity_top_k=5):
        self.max_episodes = max_episodes
        self.similarity_top_k = similarity_top_k
        self.episodes = deque(maxlen=max_episodes)
        self._episode_id = 0
        # ponytail: per-episode summary stats. Proper hippocampal indexing
        # (Tolman-Eichenbaum machine) if relational structure matters.
        self._summaries = deque(maxlen=max_episodes)

    def store_episode(self, states, hiddens, actions, rewards, context_tags=None):
        """Store a complete episode trajectory."""
        ep = {
            "id": self._episode_id,
            "states": np.asarray(states, np.float32),
            "hiddens": np.asarray(hiddens, np.float32),
            "actions": np.asarray(actions, np.int32),
            "rewards": np.asarray(rewards, np.float32),
            "length": len(states),
            "total_return": float(np.sum(rewards)),
            "context_tags": context_tags or {},
        }
        self.episodes.append(ep)
        summary = {
            "id": self._episode_id,
            "mean_h": float(np.mean(hiddens)) if len(hiddens) else 0.0,
            "total_return": ep["total_return"],
            "length": ep["length"],
            "final_state": hiddens[-1].copy() if len(hiddens) else None,
        }
        self._summaries.append(summary)
        self._episode_id += 1
        return ep["id"]

    def recall_recent(self, k=5):
        """Most recent k episodes."""
        return list(self.episodes)[-k:]

    def recall_by_similarity(self, query_h, k=None):
        """Episodes whose mean hidden state is closest to query_h."""
        if k is None:
            k = self.similarity_top_k
        if not self.episodes:
            return []
        query_h = np.asarray(query_h, np.float32).ravel()
        scored = []
        for ep in self.episodes:
            if len(ep["hiddens"]) == 0:
                continue
            ep_mean = ep["hiddens"].mean(axis=0)
            sim = -np.linalg.norm(query_h - ep_mean)
            scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def recall_by_salience(self, k=5):
        """Highest-return episodes."""
        sorted_eps = sorted(self.episodes, key=lambda e: e["total_return"], reverse=True)
        return sorted_eps[:k]

    def recall_by_tag(self, tag_key, tag_value):
        """Episodes matching a context tag."""
        return [ep for ep in self.episodes if ep["context_tags"].get(tag_key) == tag_value]

    def __len__(self):
        return len(self.episodes)


class SemanticMemory:
    """Abstracted knowledge — clusters, schemas, and regularities.

    Where EpisodicMemory stores "what happened when", SemanticMemory stores
    "what tends to happen". Built through consolidation from episodes.

    Stores:
      - Prototype vectors (cluster centroids in h-space)
      - Transition schemas (given state type S, action A tends to lead to S')
      - Object-concept bindings
    """
    def __init__(self, n_concepts=16, embed_dim=32):
        self.n_concepts = n_concepts
        self.embed_dim = embed_dim
        self.clusterer = OnlineKMeans(n_clusters=n_concepts, embed_dim=embed_dim)
        self.transition_counts = defaultdict(lambda: defaultdict(float))
        self.transition_probs = {}
        self.concept_labels = {}
        self.schema_cache = {}
        dummy = np.random.randn(max(n_concepts * 2, 10), embed_dim).astype(np.float32)
        self.clusterer.update_step(dummy)
        self._initialized = True

    def update_concepts(self, h_states):
        """Update concept prototypes from hidden states."""
        h_states = np.asarray(h_states, np.float32).reshape(-1, self.embed_dim)
        if len(h_states) < 2:
            return
        self.clusterer.update_step(h_states)
        self._initialized = True

    def assign_concept(self, h):
        """Assign a single hidden state to its concept cluster."""
        h = np.asarray(h, np.float32).reshape(1, -1)
        return int(self.clusterer.assign(h)[0])

    def observe_transition(self, h_from, action, h_to):
        c_from = self.assign_concept(h_from)
        c_to = self.assign_concept(h_to)
        key = (c_from, int(action))
        self.transition_counts[key][c_to] += 1.0
        self.transition_probs = {}  # invalidate cache

    def predict_next_concept(self, concept, action):
        """Most likely next concept given current concept and action."""
        key = (int(concept), int(action))
        if key in self.transition_counts:
            counts = self.transition_counts[key]
            total = sum(counts.values())
            return max(counts, key=counts.get), counts[max(counts, key=counts.get)] / total
        return concept, 0.0

    def transition_matrix(self):
        """Build concept x action x concept transition probability tensor.

        ponytail: computed on demand, cached until invalidated. For large
        concept spaces, precompute during consolidation instead.
        """
        if self.transition_probs:
            return self.transition_probs
        n = self.n_concepts
        result = np.zeros((n, 1, n), np.float32)  # action dim inferred, placeholder
        for (c_from, a), targets in self.transition_counts.items():
            total = sum(targets.values())
            for c_to, count in targets.items():
                if c_from < n and c_to < n:
                    if result.shape[1] <= a:
                        new_shape = (n, a + 1, n)
                        new_result = np.zeros(new_shape, np.float32)
                        new_result[:, :result.shape[1], :] = result
                        result = new_result
                    result[c_from, a, c_to] = count / total
        self.transition_probs = result
        return result

    def get_state(self):
        return {
            "n_concepts": self.n_concepts,
            "n_transitions": sum(len(v) for v in self.transition_counts.values()),
            "schemas": {str(k): dict(v) for k, v in list(self.transition_counts.items())[:20]},
        }


class ProceduralMemory:
    """Skill policies and action sequences.

    Stores reusable action sequences (options / skills) discovered through
    experience, indexed by initiation and termination conditions.
    """
    def __init__(self, max_skills=50):
        self.max_skills = max_skills
        self.skills = {}  # name -> {init_condition, term_condition, actions, embedding}
        self._next_skill_id = 0

    def store_skill(self, name, init_h, term_h, action_sequence, embedding=None):
        """Store a discovered skill."""
        self.skills[name] = {
            "id": self._next_skill_id,
            "init_h": np.asarray(init_h, np.float32).copy(),
            "term_h": np.asarray(term_h, np.float32).copy(),
            "actions": np.asarray(action_sequence, np.int32).copy(),
            "embedding": np.asarray(embedding, np.float32).copy() if embedding is not None else None,
            "n_actions": len(action_sequence),
        }
        self._next_skill_id += 1
        # evict oldest if over capacity
        if len(self.skills) > self.max_skills:
            oldest = min(self.skills.keys(), key=lambda k: self.skills[k]["id"])
            del self.skills[oldest]

    def retrieve_skill(self, current_h, action=None):
        """Find skill whose initiation condition best matches current state."""
        if not self.skills:
            return None
        current_h = np.asarray(current_h, np.float32).ravel()
        best_name, best_score = None, float("inf")
        for name, skill in self.skills.items():
            score = np.linalg.norm(current_h - skill["init_h"])
            if score < best_score:
                best_score = score
                best_name = name
        return {**self.skills[best_name], "name": best_name}

    def list_skills(self):
        return list(self.skills.keys())
