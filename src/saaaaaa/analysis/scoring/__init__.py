"""
Scoring Module

Implements TYPE_A through TYPE_F scoring modalities with strict validation
and reproducible results.

NOTE: This is a transition module during refactoring.
Some symbols come from ../scoring.py (legacy), some from ./scoring.py (new).
"""

# Import from NEW refactored scoring module
from .scoring import (
    Evidence,
    EvidenceStructureError,
    ModalityConfig,
    ModalityValidationError,
    QualityLevel,
    ScoredResult,
    ScoringError,
    ScoringModality,
    ScoringValidator,
    apply_scoring,
    determine_quality_level,
)

# Import from LEGACY scoring_legacy.py (renamed to avoid namespace shadowing)
try:
    from importlib import import_module

    # Import the standalone scoring_legacy.py file
    legacy_scoring = import_module('saaaaaa.analysis.scoring_legacy')
    if hasattr(legacy_scoring, 'MicroQuestionScorer'):
        MicroQuestionScorer = legacy_scoring.MicroQuestionScorer
    else:
        MicroQuestionScorer = None  # type: ignore
except (ImportError, AttributeError):
    MicroQuestionScorer = None  # type: ignore

__all__ = [
    "Evidence",
    "EvidenceStructureError",
    "MicroQuestionScorer",
    "ModalityConfig",
    "ModalityValidationError",
    "QualityLevel",
    "ScoredResult",
    "ScoringError",
    "ScoringModality",
    "ScoringValidator",
    "apply_scoring",
    "determine_quality_level",
]
