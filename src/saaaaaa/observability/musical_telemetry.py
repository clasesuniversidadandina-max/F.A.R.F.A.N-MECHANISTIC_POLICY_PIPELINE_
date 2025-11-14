"""
Musical Telemetry - Observability through the lens of orchestral performance.

This module provides musical metaphors for system observability, transforming
technical telemetry into a symphonic narrative.

THE MUSICAL MAPPING:
--------------------
- Phase Execution → Musical Movement
- Method Execution → Instrumental Performance
- Calibration Score → Dynamic Marking (pp, mp, mf, ff)
- Timeout → Tempo Violation
- Error → Dissonance
- Success → Consonance / Resolution
- Parallel Execution → Polyphonic Texture
- Sequential Execution → Homophonic Texture
- Provenance DAG → Harmonic Progression
- Signal Registry → Acoustic Resonance

TRINITY INTEGRATION:
--------------------
Each event participates in the three-person pattern:
1. Design (Metaclass): What event type SHOULD exist
2. Definition (Class): What the event IS (structure)
3. Manifestation (Instance): The actual event occurrence

Example Usage:
-------------
```python
from saaaaaa.observability.musical_telemetry import (
    MusicalTelemetry,
    record_movement_start,
    record_instrumental_performance,
    record_dissonance,
    record_perfect_cadence
)

# Initialize the musical observer
telemetry = MusicalTelemetry()

# Record a phase (movement) starting
record_movement_start("ingest", movement_number=1, tempo="Allegro")

# Record a method execution (instrumental performance)
record_instrumental_performance(
    method_name="analyze_coherence",
    instrument="First Violin",
    dynamic="mf",
    duration_ms=150.5,
    quality_score=0.87
)

# Record an error (dissonance)
record_dissonance(
    phase="chunk",
    error_type="ValueError",
    message="Invalid chunk size",
    severity="diminished_fifth"  # Very dissonant!
)

# Record success (perfect cadence)
record_perfect_cadence(
    phase="report",
    resolution_type="authentic_cadence",
    final_score=0.92
)
```

VERSION: 1.0.0
AUTHOR: The Musician-God
CREATED: 2025-11-14
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ============================================================================
# MUSICAL ENUMERATIONS - The Vocabulary of Performance
# ============================================================================

class DynamicMarking(str, Enum):
    """Dynamic markings from softest to loudest.

    Maps calibration scores (0.0-1.0) to musical dynamics.
    """
    PPPP = "pppp"  # Pianississimo - 0.00-0.15 (nearly silent)
    PPP = "ppp"    # Pianissimo - 0.15-0.30 (very soft)
    PP = "pp"      # Piano - 0.30-0.45 (soft)
    P = "p"        # Piano - 0.45-0.60 (moderately soft)
    MP = "mp"      # Mezzo-piano - 0.60-0.70 (medium soft)
    MF = "mf"      # Mezzo-forte - 0.70-0.80 (medium loud)
    F = "f"        # Forte - 0.80-0.85 (loud)
    FF = "ff"      # Fortissimo - 0.85-0.90 (very loud)
    FFF = "fff"    # Fortississimo - 0.90-0.95 (extremely loud)
    FFFF = "ffff"  # Fortissississimo - 0.95-1.00 (as loud as possible)


class TempoMarking(str, Enum):
    """Tempo markings for phase execution speed.

    Maps execution duration to musical tempo.
    """
    GRAVE = "Grave"           # < 40 BPM - Very slow (> 10s per phase)
    LARGO = "Largo"           # 40-60 BPM - Slow (5-10s per phase)
    ADAGIO = "Adagio"         # 60-80 BPM - Leisurely (3-5s per phase)
    ANDANTE = "Andante"       # 80-100 BPM - Walking pace (1-3s per phase)
    MODERATO = "Moderato"     # 100-120 BPM - Moderate (0.5-1s per phase)
    ALLEGRO = "Allegro"       # 120-140 BPM - Fast (200-500ms per phase)
    VIVACE = "Vivace"         # 140-160 BPM - Lively (100-200ms per phase)
    PRESTO = "Presto"         # 160-200 BPM - Very fast (50-100ms per phase)
    PRESTISSIMO = "Prestissimo"  # > 200 BPM - Extremely fast (< 50ms per phase)


class DissonanceLevel(str, Enum):
    """Musical intervals representing error severity.

    More dissonant intervals = more severe errors.
    """
    PERFECT_UNISON = "perfect_unison"      # No error (consonant)
    PERFECT_FIFTH = "perfect_fifth"        # No error (very consonant)
    MAJOR_THIRD = "major_third"            # Warning - mild dissonance
    MINOR_SEVENTH = "minor_seventh"        # Error - moderate dissonance
    TRITONE = "tritone"                    # Critical error - maximum dissonance!
    MINOR_SECOND = "minor_second"          # Fatal error - extreme dissonance


class CadenceType(str, Enum):
    """Musical cadences representing completion types.

    Different cadences convey different types of resolution.
    """
    AUTHENTIC = "authentic_cadence"        # V → I - Perfect resolution
    PLAGAL = "plagal_cadence"              # IV → I - Amen cadence (gentle)
    HALF = "half_cadence"                  # Any → V - Incomplete (more to come)
    DECEPTIVE = "deceptive_cadence"        # V → vi - Surprise ending
    PICARDY_THIRD = "picardy_third"        # Minor → Major - Hopeful ending


class TextureType(str, Enum):
    """Musical texture types for execution patterns."""
    MONOPHONIC = "monophonic"              # Single sequential execution
    HOMOPHONIC = "homophonic"              # Sequential with accompaniment
    POLYPHONIC = "polyphonic"              # Multiple parallel executions
    HETEROPHONIC = "heterophonic"          # Variations on same theme


# ============================================================================
# MUSICAL EVENT DATACLASSES - The Trinity Pattern
# ============================================================================

@dataclass(frozen=True)
class MusicalEvent:
    """Base class for all musical events.

    Participates in the Trinity:
    - Metaclass: Event type definitions
    - Class: This structure specification
    - Instance: Actual logged event
    """
    timestamp: str
    event_type: str
    phase: str | None
    correlation_id: str | None = None
    policy_unit_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "phase": self.phase,
            "correlation_id": self.correlation_id,
            "policy_unit_id": self.policy_unit_id,
        }


@dataclass(frozen=True)
class MovementEvent(MusicalEvent):
    """A musical movement (phase execution) event.

    Maps to: Phase execution in the pipeline
    Musical form: A movement in a symphony
    """
    movement_number: int | None = None
    movement_name: str | None = None
    tempo: TempoMarking | None = None
    key_signature: str | None = None
    time_signature: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "movement_number": self.movement_number,
            "movement_name": self.movement_name,
            "tempo": self.tempo.value if self.tempo else None,
            "key_signature": self.key_signature,
            "time_signature": self.time_signature,
            "duration_ms": self.duration_ms,
        })
        return base


@dataclass(frozen=True)
class InstrumentalEvent(MusicalEvent):
    """An instrumental performance (method execution) event.

    Maps to: Individual method execution
    Musical form: A single instrument playing its part
    """
    method_name: str | None = None
    instrument: str | None = None
    dynamic: DynamicMarking | None = None
    articulation: str | None = None
    duration_ms: float | None = None
    quality_score: float | None = None
    notes_played: int | None = None  # Number of operations performed

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "method_name": self.method_name,
            "instrument": self.instrument,
            "dynamic": self.dynamic.value if self.dynamic else None,
            "articulation": self.articulation,
            "duration_ms": self.duration_ms,
            "quality_score": self.quality_score,
            "notes_played": self.notes_played,
        })
        return base


@dataclass(frozen=True)
class DissonanceEvent(MusicalEvent):
    """A dissonance (error) event.

    Maps to: Errors, exceptions, validation failures
    Musical form: Dissonant intervals that need resolution
    """
    error_type: str | None = None
    error_message: str | None = None
    dissonance_level: DissonanceLevel | None = None
    interval: str | None = None
    requires_resolution: bool = True

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "error_type": self.error_type,
            "error_message": self.error_message,
            "dissonance_level": self.dissonance_level.value if self.dissonance_level else None,
            "interval": self.interval,
            "requires_resolution": self.requires_resolution,
        })
        return base


@dataclass(frozen=True)
class CadenceEvent(MusicalEvent):
    """A cadence (completion) event.

    Maps to: Successful phase/pipeline completion
    Musical form: Harmonic cadences that provide closure
    """
    cadence_type: CadenceType | None = None
    resolution_quality: float | None = None
    final_chord: str | None = None
    satisfying: bool = True

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "cadence_type": self.cadence_type.value if self.cadence_type else None,
            "resolution_quality": self.resolution_quality,
            "final_chord": self.final_chord,
            "satisfying": self.satisfying,
        })
        return base


# ============================================================================
# UTILITY FUNCTIONS - Musical Transformations
# ============================================================================

def score_to_dynamic(score: float) -> DynamicMarking:
    """Convert a calibration score (0.0-1.0) to a dynamic marking.

    Args:
        score: Calibration score between 0.0 and 1.0

    Returns:
        Appropriate dynamic marking
    """
    if score >= 0.95:
        return DynamicMarking.FFFF
    elif score >= 0.90:
        return DynamicMarking.FFF
    elif score >= 0.85:
        return DynamicMarking.FF
    elif score >= 0.80:
        return DynamicMarking.F
    elif score >= 0.70:
        return DynamicMarking.MF
    elif score >= 0.60:
        return DynamicMarking.MP
    elif score >= 0.45:
        return DynamicMarking.P
    elif score >= 0.30:
        return DynamicMarking.PP
    elif score >= 0.15:
        return DynamicMarking.PPP
    else:
        return DynamicMarking.PPPP


def duration_to_tempo(duration_ms: float) -> TempoMarking:
    """Convert execution duration to tempo marking.

    Args:
        duration_ms: Execution time in milliseconds

    Returns:
        Appropriate tempo marking
    """
    if duration_ms < 50:
        return TempoMarking.PRESTISSIMO
    elif duration_ms < 100:
        return TempoMarking.PRESTO
    elif duration_ms < 200:
        return TempoMarking.VIVACE
    elif duration_ms < 500:
        return TempoMarking.ALLEGRO
    elif duration_ms < 1000:
        return TempoMarking.MODERATO
    elif duration_ms < 3000:
        return TempoMarking.ANDANTE
    elif duration_ms < 5000:
        return TempoMarking.ADAGIO
    elif duration_ms < 10000:
        return TempoMarking.LARGO
    else:
        return TempoMarking.GRAVE


def error_to_dissonance(error_type: str) -> DissonanceLevel:
    """Convert error type to dissonance level.

    Args:
        error_type: Type of error (e.g., "ValueError", "TimeoutError")

    Returns:
        Appropriate dissonance level
    """
    # Fatal errors → Extreme dissonance
    if "Fatal" in error_type or "Critical" in error_type:
        return DissonanceLevel.MINOR_SECOND

    # Timeout/system errors → Maximum dissonance
    if "Timeout" in error_type or "System" in error_type:
        return DissonanceLevel.TRITONE

    # Runtime errors → Moderate dissonance
    if "Runtime" in error_type or "Exception" in error_type:
        return DissonanceLevel.MINOR_SEVENTH

    # Validation errors → Mild dissonance
    if "Validation" in error_type or "Type" in error_type:
        return DissonanceLevel.MAJOR_THIRD

    # Default → Moderate dissonance
    return DissonanceLevel.MINOR_SEVENTH


# ============================================================================
# MUSICAL TELEMETRY CLASS - The Conductor
# ============================================================================

class MusicalTelemetry:
    """The conductor of the observability orchestra.

    This class provides the Trinity pattern at the system level:
    - Metaclass: The concept of musical telemetry
    - Class: This implementation specification
    - Instance: Actual telemetry collection
    """

    def __init__(self) -> None:
        """Initialize the musical telemetry system."""
        self.logger = logging.getLogger("musical_telemetry")
        self.events: list[MusicalEvent] = []
        self.current_movement: str | None = None
        self.start_time: float = time.time()

    def record_event(self, event: MusicalEvent) -> None:
        """Record a musical event.

        Args:
            event: The musical event to record
        """
        self.events.append(event)
        self.logger.info(
            f"🎵 {event.event_type}",
            extra=event.to_dict()
        )

    def movement_start(
        self,
        phase: str,
        movement_number: int | None = None,
        tempo: TempoMarking | None = None,
        correlation_id: str | None = None,
        policy_unit_id: str | None = None,
    ) -> None:
        """Record the start of a movement (phase).

        Args:
            phase: Phase name
            movement_number: Optional movement number
            tempo: Optional tempo marking
            correlation_id: Optional correlation ID
            policy_unit_id: Optional policy unit ID
        """
        self.current_movement = phase
        event = MovementEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="movement_start",
            phase=phase,
            movement_number=movement_number,
            movement_name=phase.upper(),
            tempo=tempo,
            correlation_id=correlation_id,
            policy_unit_id=policy_unit_id,
        )
        self.record_event(event)

    def movement_end(
        self,
        phase: str,
        duration_ms: float,
        correlation_id: str | None = None,
        policy_unit_id: str | None = None,
    ) -> None:
        """Record the end of a movement (phase).

        Args:
            phase: Phase name
            duration_ms: Duration in milliseconds
            correlation_id: Optional correlation ID
            policy_unit_id: Optional policy unit ID
        """
        tempo = duration_to_tempo(duration_ms)
        event = MovementEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="movement_end",
            phase=phase,
            tempo=tempo,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            policy_unit_id=policy_unit_id,
        )
        self.record_event(event)
        self.current_movement = None

    def instrumental_performance(
        self,
        method_name: str,
        instrument: str | None = None,
        quality_score: float | None = None,
        duration_ms: float | None = None,
        phase: str | None = None,
        correlation_id: str | None = None,
        policy_unit_id: str | None = None,
    ) -> None:
        """Record an instrumental performance (method execution).

        Args:
            method_name: Name of the method executed
            instrument: Instrument name (or module name)
            quality_score: Quality/calibration score (0.0-1.0)
            duration_ms: Execution duration in milliseconds
            phase: Phase name (defaults to current movement)
            correlation_id: Optional correlation ID
            policy_unit_id: Optional policy unit ID
        """
        dynamic = score_to_dynamic(quality_score) if quality_score is not None else None
        event = InstrumentalEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="instrumental_performance",
            phase=phase or self.current_movement,
            method_name=method_name,
            instrument=instrument,
            dynamic=dynamic,
            quality_score=quality_score,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            policy_unit_id=policy_unit_id,
        )
        self.record_event(event)

    def dissonance(
        self,
        error_type: str,
        error_message: str | None = None,
        phase: str | None = None,
        correlation_id: str | None = None,
        policy_unit_id: str | None = None,
    ) -> None:
        """Record a dissonance (error).

        Args:
            error_type: Type of error
            error_message: Optional error message
            phase: Phase name (defaults to current movement)
            correlation_id: Optional correlation ID
            policy_unit_id: Optional policy unit ID
        """
        dissonance_level = error_to_dissonance(error_type)
        event = DissonanceEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="dissonance",
            phase=phase or self.current_movement,
            error_type=error_type,
            error_message=error_message,
            dissonance_level=dissonance_level,
            correlation_id=correlation_id,
            policy_unit_id=policy_unit_id,
        )
        self.record_event(event)

    def cadence(
        self,
        cadence_type: CadenceType,
        resolution_quality: float | None = None,
        phase: str | None = None,
        correlation_id: str | None = None,
        policy_unit_id: str | None = None,
    ) -> None:
        """Record a cadence (successful completion).

        Args:
            cadence_type: Type of cadence
            resolution_quality: Quality of resolution (0.0-1.0)
            phase: Phase name (defaults to current movement)
            correlation_id: Optional correlation ID
            policy_unit_id: Optional policy unit ID
        """
        event = CadenceEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="cadence",
            phase=phase or self.current_movement,
            cadence_type=cadence_type,
            resolution_quality=resolution_quality,
            satisfying=resolution_quality is None or resolution_quality >= 0.7,
            correlation_id=correlation_id,
            policy_unit_id=policy_unit_id,
        )
        self.record_event(event)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a summary of the entire performance.

        Returns:
            Dictionary containing performance statistics
        """
        total_duration = time.time() - self.start_time

        movements = [e for e in self.events if isinstance(e, MovementEvent)]
        performances = [e for e in self.events if isinstance(e, InstrumentalEvent)]
        dissonances = [e for e in self.events if isinstance(e, DissonanceEvent)]
        cadences = [e for e in self.events if isinstance(e, CadenceEvent)]

        return {
            "total_duration_s": total_duration,
            "total_events": len(self.events),
            "movements": len(movements),
            "instrumental_performances": len(performances),
            "dissonances": len(dissonances),
            "cadences": len(cadences),
            "overall_tempo": duration_to_tempo(total_duration * 1000).value,
            "average_performance_quality": (
                sum(p.quality_score for p in performances if p.quality_score is not None) /
                len([p for p in performances if p.quality_score is not None])
                if performances else 0.0
            ),
            "dissonance_rate": len(dissonances) / len(self.events) if self.events else 0.0,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS - Quick Musical Recording
# ============================================================================

# Global singleton instance
_telemetry: MusicalTelemetry | None = None


def get_musical_telemetry() -> MusicalTelemetry:
    """Get the global musical telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = MusicalTelemetry()
    return _telemetry


def record_movement_start(
    phase: str,
    movement_number: int | None = None,
    tempo: TempoMarking | str | None = None,
    **kwargs: Any,
) -> None:
    """Convenience function to record movement start."""
    if isinstance(tempo, str):
        tempo = TempoMarking(tempo)
    get_musical_telemetry().movement_start(
        phase=phase,
        movement_number=movement_number,
        tempo=tempo,
        **kwargs
    )


def record_movement_end(phase: str, duration_ms: float, **kwargs: Any) -> None:
    """Convenience function to record movement end."""
    get_musical_telemetry().movement_end(phase=phase, duration_ms=duration_ms, **kwargs)


def record_instrumental_performance(
    method_name: str,
    instrument: str | None = None,
    quality_score: float | None = None,
    duration_ms: float | None = None,
    **kwargs: Any,
) -> None:
    """Convenience function to record instrumental performance."""
    get_musical_telemetry().instrumental_performance(
        method_name=method_name,
        instrument=instrument,
        quality_score=quality_score,
        duration_ms=duration_ms,
        **kwargs
    )


def record_dissonance(
    error_type: str,
    error_message: str | None = None,
    **kwargs: Any,
) -> None:
    """Convenience function to record dissonance."""
    get_musical_telemetry().dissonance(
        error_type=error_type,
        error_message=error_message,
        **kwargs
    )


def record_cadence(
    cadence_type: CadenceType | str,
    resolution_quality: float | None = None,
    **kwargs: Any,
) -> None:
    """Convenience function to record cadence."""
    if isinstance(cadence_type, str):
        cadence_type = CadenceType(cadence_type)
    get_musical_telemetry().cadence(
        cadence_type=cadence_type,
        resolution_quality=resolution_quality,
        **kwargs
    )


__all__ = [
    # Enums
    "DynamicMarking",
    "TempoMarking",
    "DissonanceLevel",
    "CadenceType",
    "TextureType",
    # Events
    "MusicalEvent",
    "MovementEvent",
    "InstrumentalEvent",
    "DissonanceEvent",
    "CadenceEvent",
    # Main class
    "MusicalTelemetry",
    # Utility functions
    "score_to_dynamic",
    "duration_to_tempo",
    "error_to_dissonance",
    # Convenience functions
    "get_musical_telemetry",
    "record_movement_start",
    "record_movement_end",
    "record_instrumental_performance",
    "record_dissonance",
    "record_cadence",
]
