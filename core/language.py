"""
language.py — Grounded vocabulary and simple language understanding.

Ground zero for language in Genesis. Maps between:
  - Symbolic tokens (words) and grounded referents (concepts, objects, actions)
  - Simple compositional semantics (agent-verb-object)

This is NOT a large language model. It is a grounded vocabulary that
binds tokens to the agent's existing internal representations.

Cognitive parallel: early language acquisition — mapping sounds to
sensorimotor experience, not statistical text prediction.

ponytail: lookup-table grounding, no grammar. Full compositional language
requires a phrase-structure grammar and a parser.
"""
import numpy as np


class GroundedWord:
    """A word grounded in the agent's experience."""
    def __init__(self, token, referent_type, referent_id, confidence=1.0):
        self.token = token  # string symbol
        self.referent_type = referent_type  # "concept", "object", "action", "state"
        self.referent_id = referent_id  # int or string identifier
        self.confidence = confidence
        self.usage_count = 0

    def use(self):
        self.usage_count += 1


class GroundedVocabulary:
    """Maps tokens to grounded referents in the agent's internal world.

    Supports:
      - Learn: associate a token with a referent (concept, object, action)
      - Understand: given a token, find its grounded meaning
      - Produce: given a referent, find its token
      - Compose: simple agent-verb-object phrases
    """
    def __init__(self):
        self._token_to_ref = {}  # token -> GroundedWord
        self._ref_to_tokens = {}  # (type, id) -> list of tokens

    def learn_word(self, token, referent_type, referent_id, confidence=1.0):
        """Learn a grounded word."""
        word = GroundedWord(token, referent_type, referent_id, confidence)
        self._token_to_ref[token] = word
        key = (referent_type, referent_id)
        if key not in self._ref_to_tokens:
            self._ref_to_tokens[key] = []
        self._ref_to_tokens[key].append(token)

    def understand(self, token):
        """Given a word, return its grounded meaning."""
        return self._token_to_ref.get(token)

    def produce(self, referent_type, referent_id):
        """Given a referent, produce the most confident word for it."""
        key = (referent_type, referent_id)
        tokens = self._ref_to_tokens.get(key, [])
        if not tokens:
            return None
        best = max(tokens, key=lambda t: self._token_to_ref[t].confidence)
        return best

    def understand_phrase(self, phrase):
        """Simple compositional understanding: "agent verb object".

        Returns a dict {agent, verb, object} with grounded meanings.
        """
        tokens = phrase.lower().split()
        if len(tokens) == 3:
            agent = self.understand(tokens[0])
            verb = self.understand(tokens[1])
            obj = self.understand(tokens[2])
            return {"agent": agent, "verb": verb, "object": obj}
        if len(tokens) == 2:
            verb = self.understand(tokens[0])
            obj = self.understand(tokens[1])
            return {"agent": None, "verb": verb, "object": obj}
        return None

    def learn_from_concepts(self, concept_formation, action_names=None):
        """Auto-populate vocabulary from discovered concepts and actions."""
        if action_names:
            for i, name in enumerate(action_names):
                self.learn_word(name, "action", i)
        for c in range(min(16, concept_formation.n_active_concepts() or 
                          concept_formation.n_concepts)):
            token = f"concept_{c}"
            self.learn_word(token, "concept", c, confidence=0.5)

    def auto_name_concept(self, concept_id, frequency):
        """Generate a human-readable name for a concept based on its frequency."""
        if frequency > 0.3:
            name = f"common_thing_{concept_id}"
        elif frequency > 0.1:
            name = f"thing_{concept_id}"
        else:
            name = f"rare_thing_{concept_id}"
        self.learn_word(name, "concept", concept_id)
        return name

    def vocabulary_size(self):
        return len(self._token_to_ref)

    def get_state(self):
        return {
            "vocabulary_size": self.vocabulary_size(),
            "tokens": list(self._token_to_ref.keys()),
        }
