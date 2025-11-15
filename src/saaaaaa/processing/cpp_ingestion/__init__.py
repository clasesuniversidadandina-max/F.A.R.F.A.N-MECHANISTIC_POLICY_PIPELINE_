"""
CPP Ingestion Package (Deprecated - Use SPC Ingestion)

This package provides backward compatibility for Canon Policy Package (CPP) ingestion.
The terminology has been migrated to Smart Policy Chunks (SPC).

For new code, use: saaaaaa.processing.spc_ingestion
"""

import warnings

from .models import (
    Budget,
    BudgetInfo,
    CanonPolicyPackage,
    Chunk,
    ChunkGraph,
    ChunkResolution,
    Confidence,
    Entity,
    GeoFacet,
    IntegrityIndex,
    KPI,
    KPIInfo,
    PolicyFacet,
    PolicyManifest,
    ProvenanceMap,
    QualityMetrics,
    TextSpan,
    TimeFacet,
)

# Re-export CPPIngestionPipeline from spc_ingestion for backward compatibility
try:
    from ..spc_ingestion import CPPIngestionPipeline
except ImportError:
    warnings.warn(
        "CPPIngestionPipeline not available. Install SPC ingestion dependencies.",
        ImportWarning,
        stacklevel=2,
    )
    CPPIngestionPipeline = None  # type: ignore

__all__ = [
    "Budget",
    "BudgetInfo",
    "CanonPolicyPackage",
    "Chunk",
    "ChunkGraph",
    "ChunkResolution",
    "Confidence",
    "CPPIngestionPipeline",
    "Entity",
    "GeoFacet",
    "IntegrityIndex",
    "KPI",
    "KPIInfo",
    "PolicyFacet",
    "PolicyManifest",
    "ProvenanceMap",
    "QualityMetrics",
    "TextSpan",
    "TimeFacet",
]
