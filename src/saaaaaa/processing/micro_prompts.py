"""
Compatibility shim for micro_prompts module.

DEPRECATED: This module has been relocated to saaaaaa.analysis.micro_prompts
This compatibility layer will be removed in a future version.

Migration guide:
    OLD: from saaaaaa.processing.micro_prompts import ...
    NEW: from saaaaaa.analysis.micro_prompts import ...

All exports are re-exported from the new location with deprecation warnings.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on module import
warnings.warn(
    "saaaaaa.processing.micro_prompts is deprecated. "
    "Use saaaaaa.analysis.micro_prompts instead. "
    "This compatibility shim will be removed in version 0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all public symbols from new location
from saaaaaa.analysis.micro_prompts import (
    AntiMilagroStressTester,
    AuditResult,
    BayesianPosteriorExplainer,
    CausalChain,
    PosteriorJustification,
    ProportionalityPattern,
    ProvenanceAuditor,
    ProvenanceDAG,
    ProvenanceNode,
    QMCMRecord,
    Signal,
    StressTestResult,
    create_posterior_explainer,
    create_provenance_auditor,
    create_stress_tester,
)

__all__ = [
    "QMCMRecord",
    "ProvenanceNode",
    "ProvenanceDAG",
    "AuditResult",
    "ProvenanceAuditor",
    "Signal",
    "PosteriorJustification",
    "BayesianPosteriorExplainer",
    "CausalChain",
    "ProportionalityPattern",
    "StressTestResult",
    "AntiMilagroStressTester",
    "create_provenance_auditor",
    "create_posterior_explainer",
    "create_stress_tester",
]
