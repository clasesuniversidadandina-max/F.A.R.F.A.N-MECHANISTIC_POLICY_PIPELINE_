"""
Compatibility shim for macro_prompts module.

DEPRECATED: This module has been relocated to saaaaaa.analysis.macro_prompts
This compatibility layer will be removed in a future version.

Migration guide:
    OLD: from saaaaaa.processing.macro_prompts import ...
    NEW: from saaaaaa.analysis.macro_prompts import ...

All exports are re-exported from the new location with deprecation warnings.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on module import
warnings.warn(
    "saaaaaa.processing.macro_prompts is deprecated. "
    "Use saaaaaa.analysis.macro_prompts instead. "
    "This compatibility shim will be removed in version 0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all public symbols from new location
from saaaaaa.analysis.macro_prompts import (
    BayesianPortfolio,
    BayesianPortfolioComposer,
    ContradictionReport,
    ContradictionScanner,
    CoverageAnalysis,
    CoverageGapStressor,
    ImplementationRoadmap,
    MacroPromptsOrchestrator,
    PeerNormalization,
    PeerNormalizer,
    RoadmapOptimizer,
)

__all__ = [
    "CoverageAnalysis",
    "ContradictionReport",
    "BayesianPortfolio",
    "ImplementationRoadmap",
    "PeerNormalization",
    "CoverageGapStressor",
    "ContradictionScanner",
    "BayesianPortfolioComposer",
    "RoadmapOptimizer",
    "PeerNormalizer",
    "MacroPromptsOrchestrator",
]
