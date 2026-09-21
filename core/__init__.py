"""Genesis cognitive modules.

Exports are lazy because some optional cognitive modules depend on heavier
third-party packages such as scikit-learn, while the core D1/RND and
meta-control tracks only require NumPy.  Eagerly importing every module made
``import core.agent`` fail when optional dependencies were absent.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "WorkingMemory": ("core.memory", "WorkingMemory"),
    "EpisodicMemory": ("core.memory", "EpisodicMemory"),
    "SemanticMemory": ("core.memory", "SemanticMemory"),
    "ProceduralMemory": ("core.memory", "ProceduralMemory"),
    "KnowledgeConsolidation": ("core.consolidation", "KnowledgeConsolidation"),
    "ObjectPermanence": ("core.object_permanence", "ObjectPermanence"),
    "ObjectSlot": ("core.object_permanence", "ObjectSlot"),
    "MultiStepWorldModel": ("core.world_model_v3", "MultiStepWorldModel"),
    "Planner": ("core.planning", "Planner"),
    "ConceptFormation": ("core.concept_formation", "ConceptFormation"),
    "GoalWeights": ("core.goals", "GoalWeights"),
    "GoalDrivenReward": ("core.goals", "GoalDrivenReward"),
    "ReasoningEngine": ("core.reasoning", "ReasoningEngine"),
    "Proposition": ("core.reasoning", "Proposition"),
    "SelfModel": ("core.self_model", "SelfModel"),
    "GroundedVocabulary": ("core.language", "GroundedVocabulary"),
    "GroundedWord": ("core.language", "GroundedWord"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    """Load an exported cognitive module only when that symbol is used."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'core' has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
