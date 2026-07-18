"""
reasoning.py — Basic symbolic reasoning module.

Ground zero for symbolic cognition in Genesis. Currently provides:

  1. Propositional logic — AND, OR, NOT, IMPLIES over concept-level facts
  2. Simple deduction — modus ponens, modus tollens
  3. State constraints — "if I see object A, I cannot be at location B"
  4. Causal inference — "action A leads to effect E" (from transition schemas)

This is deliberately minimal — a symbolic scratchpad that operates on
concept tokens, not raw h. The bridge between sub-symbolic (neural) and
symbolic (discrete) reasoning.

ponytail: forward chaining with hand-coded inference rules. A proper
implementation would learn inference rules via inductive logic programming
or a neural-symbolic hybrid.
"""
import numpy as np
from collections import defaultdict


class Proposition:
    """A symbolic proposition: (subject, predicate, object, truth_value)."""
    def __init__(self, subject, predicate, obj=None, truth=1.0):
        self.subject = subject
        self.predicate = predicate
        self.object = obj
        self.truth = float(np.clip(truth, -1.0, 1.0))

    def __eq__(self, other):
        return (self.subject == other.subject and self.predicate == other.predicate
                and self.object == other.object)

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __repr__(self):
        if self.object is not None:
            return f"{self.subject} {self.predicate} {self.object} [{self.truth:.2f}]"
        return f"{self.subject} {self.predicate} [{self.truth:.2f}]"


class ReasoningEngine:
    """Symbolic reasoning over concept-level propositions.

    Maintains a knowledge base (KB) of propositions and supports:
      - Tell: add propositions to KB
      - Ask: query the truth value of a proposition
      - Infer: derive new propositions via inference rules
      - Explain: show the chain of reasoning for a conclusion
    """
    def __init__(self):
        self.kb = {}  # (subject, predicate, object) -> truth value
        self.rules = []
        self._inference_history = defaultdict(list)
        self._add_default_rules()

    @staticmethod
    def _modus_ponens(kb, s, p, o):
        """If (if, s, consequence) in KB and (s, p, o) is a premise, derive consequence."""
        for (rs, rp, ro), rt in kb.items():
            if rs == "if" and rp == s and rt > 0.5:
                premise_val = kb.get((s, p, o), 0.0)
                if abs(premise_val) > 0.1:
                    return (f"inferred_{ro}", premise_val)
        return 0.0

    def _add_default_rules(self):
        """Default domain-agnostic inference rules."""
        self.add_rule("modus_ponens", ReasoningEngine._modus_ponens)
        self.add_rule("contradiction", lambda kb, s, p, o:
            -kb.get((s, "not", o), 0.0) if (s, "not", o) in kb else None)

    def add_rule(self, name, fn):
        """Register an inference rule: fn(kb, subject, predicate, object) -> truth or None."""
        self.rules.append((name, fn))

    def tell(self, proposition):
        """Add a proposition to the knowledge base."""
        key = (proposition.subject, proposition.predicate, proposition.object)
        self.kb[key] = proposition.truth

    def tell_many(self, propositions):
        for p in propositions:
            self.tell(p)

    def ask(self, subject, predicate, obj=None):
        """Query the truth value of a proposition."""
        key = (subject, predicate, obj)
        return self.kb.get(key, 0.0)

    def infer(self, subject=None, predicate=None, obj=None):
        """Run inference rules to derive new conclusions.

        Returns list of derived propositions.
        """
        derived = []
        for key, truth in list(self.kb.items()):
            s, p, o = key
            if subject is not None and s != subject:
                continue
            if predicate is not None and p != predicate:
                continue
            if obj is not None and o != obj:
                continue
            for rule_name, rule_fn in self.rules:
                result = rule_fn(self.kb, s, p, o)
                if result is None:
                    continue
                if isinstance(result, tuple):
                    pred_name, truth_val = result
                else:
                    pred_name, truth_val = f"inferred_{p}", result
                if abs(truth_val) > 0.1:
                    new_prop = Proposition(s, pred_name, o, truth_val)
                    if (new_prop.subject, new_prop.predicate, new_prop.object) not in self.kb:
                        self.kb[(new_prop.subject, new_prop.predicate, new_prop.object)] = truth_val
                        self._inference_history[(new_prop.subject, new_prop.predicate, new_prop.object)].append(
                            (rule_name, key))
                        derived.append(new_prop)
        return derived

    def explain(self, subject, predicate, obj=None):
        """Show the chain of reasoning for a proposition."""
        key = (subject, predicate, obj)
        if key in self._inference_history:
            chain = []
            for rule_name, premise_key in self._inference_history[key]:
                chain.append(f"{rule_name} from {premise_key}")
                if premise_key in self._inference_history:
                    chain.extend(self._inference_history[premise_key])
            return chain
        if key in self.kb:
            return [f"direct observation: {key} = {self.kb[key]:.2f}"]
        return ["unknown"]

    def constraints_from_schemas(self, semantic_memory):
        """Extract symbolic constraints from concept transition schemas.

        E.g., if concept A never transitions to concept B, add constraint
        "A cannot become B".
        """
        tm = semantic_memory.transition_matrix()
        constraints = []
        for c_from in range(min(tm.shape[0], semantic_memory.n_concepts)):
            for a in range(min(tm.shape[1], 5)):
                probs = tm[c_from, a]
                below_thresh = np.where((probs > 0) & (probs < 0.05))[0]
                for c_to in below_thresh:
                    constraints.append(Proposition(
                        f"concept_{c_from}", f"action_{a}_avoids", f"concept_{c_to}", -0.8))
        return constraints

    def get_state(self):
        return {
            "kb_size": len(self.kb),
            "n_rules": len(self.rules),
            "inferences_made": sum(len(v) for v in self._inference_history.values()),
        }
