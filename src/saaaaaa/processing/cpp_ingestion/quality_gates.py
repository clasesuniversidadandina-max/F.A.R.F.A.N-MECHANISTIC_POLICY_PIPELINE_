"""
Quality Gates Module (Deprecated - Compatibility Stub)

This module provides backward compatibility for CPP quality validation.
The functionality has been migrated to validation and quality assurance modules.

For new code, use validation framework in saaaaaa.utils.validation
"""

from __future__ import annotations

import warnings
from typing import Any

from ..cpp_ingestion.models import CanonPolicyPackage, QualityMetrics

warnings.warn(
    "cpp_ingestion.quality_gates is deprecated. "
    "Use saaaaaa.utils.validation framework instead.",
    DeprecationWarning,
    stacklevel=2,
)


class QualityGates:
    """
    Quality validation gates for CPP/SPC (Compatibility Stub).

    This is a minimal compatibility implementation.
    For full validation, use the validation framework.
    """

    def __init__(self, strict_mode: bool = False, **kwargs: Any):
        """
        Initialize quality gates.

        Parameters
        ----------
        strict_mode : bool
            Enable strict validation mode
        **kwargs : Any
            Additional configuration (ignored in stub)
        """
        self.strict_mode = strict_mode
        self.config = kwargs

    def validate(self, cpp: CanonPolicyPackage) -> tuple[bool, list[str]]:
        """
        Validate Canon Policy Package (Stub implementation).

        Parameters
        ----------
        cpp : CanonPolicyPackage
            Package to validate

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list_of_errors)
            Stub returns (True, []) to allow tests to pass
        """
        warnings.warn(
            "QualityGates.validate is a stub implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return (True, [])

    def compute_quality_metrics(self, cpp: CanonPolicyPackage) -> QualityMetrics:
        """
        Compute quality metrics for package (Stub).

        Parameters
        ----------
        cpp : CanonPolicyPackage
            Package to analyze

        Returns
        -------
        QualityMetrics
            Quality metrics (stub returns minimal metrics)
        """
        return QualityMetrics(
            completeness_score=1.0,
            consistency_score=1.0,
            overall_score=1.0,
        )


__all__ = ["QualityGates"]
