"""
Test fixtures and compatibility layer for calibration system tests.

Re-exports production calibration API with graceful fallbacks for missing symbols.
"""

# Try to import what we can from the production calibration system
try:
    from saaaaaa.core.calibration import (
        CalibrationOrchestrator as CalibrationEngine,
        ContextTuple as Context,
        LayerID as LayerType,
    )
except ImportError:
    CalibrationEngine = None  # type: ignore
    Context = None  # type: ignore
    LayerType = None  # type: ignore

# Try to import functions that may have missing dependencies
try:
    from saaaaaa.core.calibration.engine import calibrate
except ImportError:
    calibrate = None  # type: ignore

try:
    from saaaaaa.core.calibration.validators import validate_config_files
except ImportError:
    validate_config_files = None  # type: ignore

# Placeholder types for tests that may not be implemented yet
class ComputationGraph:
    """Placeholder for computation graph type."""
    pass


class EvidenceStore:
    """Placeholder for evidence store type."""
    pass


class MethodRole:
    """Placeholder for method role type."""
    pass


__all__ = [
    "calibrate",
    "CalibrationEngine",
    "validate_config_files",
    "Context",
    "ComputationGraph",
    "EvidenceStore",
    "LayerType",
    "MethodRole",
]
