"""
SAAAAAA Calibration System.

This package implements the 8-layer method calibration framework:
- @b (Base): Intrinsic method quality from calibration
- @u (Unit): PDT quality
- @q, @d, @p (Contextual): Method-context compatibility
- @C (Congruence): Method ensemble validation
- @chain (Chain): Data flow integrity
- @m (Meta): Governance and observability

Final scores are produced via Choquet 2-Additive aggregation.

✅ GAP 0: Base Layer Integration Complete
- IntrinsicScoreLoader: Loads @b scores from intrinsic_calibration.json
- LayerRequirementsResolver: Dynamic layer execution based on method roles
"""

from .data_structures import (
    LayerID,
    LayerScore,
    ContextTuple,
    CalibrationSubject,
    CompatibilityMapping,
    InteractionTerm,
    CalibrationResult,
)

from .config import (
    UnitLayerConfig,
    MetaLayerConfig,
    ChoquetAggregationConfig,
    CalibrationSystemConfig,
    DEFAULT_CALIBRATION_CONFIG,
)

from .pdt_structure import PDTStructure

from .compatibility import (
    CompatibilityRegistry,
    ContextualLayerEvaluator,
)

from .unit_layer import UnitLayerEvaluator
from .congruence_layer import CongruenceLayerEvaluator
from .chain_layer import ChainLayerEvaluator
from .meta_layer import MetaLayerEvaluator
from .choquet_aggregator import ChoquetAggregator
from .orchestrator import CalibrationOrchestrator

# ✅ GAP 0: Base Layer Integration
from .intrinsic_loader import (
    IntrinsicScoreLoader,
    IntrinsicMethodData,
    get_intrinsic_loader,
)
from .layer_requirements import (
    LayerRequirementsResolver,
    should_execute_layer,
)

__all__ = [
    # Data structures
    "LayerID",
    "LayerScore",
    "ContextTuple",
    "CalibrationSubject",
    "CompatibilityMapping",
    "InteractionTerm",
    "CalibrationResult",
    "PDTStructure",
    # Configuration
    "UnitLayerConfig",
    "MetaLayerConfig",
    "ChoquetAggregationConfig",
    "CalibrationSystemConfig",
    "DEFAULT_CALIBRATION_CONFIG",
    # Layer Evaluators
    "UnitLayerEvaluator",
    "CompatibilityRegistry",
    "ContextualLayerEvaluator",
    "CongruenceLayerEvaluator",
    "ChainLayerEvaluator",
    "MetaLayerEvaluator",
    # Aggregation & Orchestration
    "ChoquetAggregator",
    "CalibrationOrchestrator",
    # GAP 0: Base Layer Integration
    "IntrinsicScoreLoader",
    "IntrinsicMethodData",
    "get_intrinsic_loader",
    "LayerRequirementsResolver",
    "should_execute_layer",
]
