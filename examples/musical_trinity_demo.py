#!/usr/bin/env python3
"""
Musical Trinity Demo - A showcase of the divine musical-computational duality.

This example demonstrates:
1. Musical telemetry for observability
2. Trinity validation for architectural integrity
3. The synthesis of music and computation

Run with:
    python examples/musical_trinity_demo.py

VERSION: 1.0.0
AUTHOR: The Musician-God
CREATED: 2025-11-14
"""

import sys
import time
from pathlib import Path

# Add src to path for development
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from dataclasses import dataclass
from saaaaaa.observability.musical_telemetry import (
    get_musical_telemetry,
    record_movement_start,
    record_movement_end,
    record_instrumental_performance,
    record_dissonance,
    record_cadence,
    TempoMarking,
    CadenceType,
)
from saaaaaa.utils.trinity_validator import (
    print_trinity_report,
    validate_trinity,
    is_trinitarian,
    blessed_by_trinity,
)


# ============================================================================
# DEMO: Create a blessed dataclass
# ============================================================================

@blessed_by_trinity
@dataclass
class PolicyEvidence:
    """Example evidence class blessed by the Trinity.

    This demonstrates the three-person pattern:
    - Metaclass: type (via type(PolicyEvidence))
    - Class: PolicyEvidence (this dataclass definition)
    - Instance: actual evidence objects
    """
    method_name: str
    question_id: str
    score: float
    confidence: float
    timestamp: str


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_musical_telemetry():
    """Demonstrate musical telemetry in action."""
    print("\n" + "=" * 70)
    print("🎼 DEMO 1: Musical Telemetry")
    print("=" * 70)
    print("\nLet's orchestrate a simple pipeline with musical observability...\n")

    telemetry = get_musical_telemetry()

    # Movement 1: Data Ingestion (Allegro - fast!)
    print("🎵 Movement I: Ingest (Allegro)")
    record_movement_start("ingest", movement_number=1, tempo=TempoMarking.ALLEGRO)
    time.sleep(0.1)  # Simulate work

    # Some instrumental performances (method executions)
    record_instrumental_performance(
        method_name="load_document",
        instrument="First Violin",
        quality_score=0.92,
        duration_ms=50.5
    )
    record_instrumental_performance(
        method_name="validate_format",
        instrument="Viola",
        quality_score=0.88,
        duration_ms=30.2
    )

    record_movement_end("ingest", duration_ms=100.0)
    print("  ✓ Movement I complete!")

    # Movement 2: Normalization (Andante - moderate pace)
    print("\n🎵 Movement II: Normalize (Andante)")
    record_movement_start("normalize", movement_number=2, tempo=TempoMarking.ANDANTE)
    time.sleep(0.15)

    record_instrumental_performance(
        method_name="normalize_text",
        instrument="Cello",
        quality_score=0.85,
        duration_ms=120.0
    )

    # Oh no! A dissonance (error)!
    print("  ⚠ Dissonance detected!")
    record_dissonance(
        error_type="ValidationError",
        error_message="Missing required field 'municipality'"
    )

    record_movement_end("normalize", duration_ms=150.0)
    print("  ○ Movement II complete (with dissonance)")

    # Movement 3: Analysis (Presto - very fast!)
    print("\n🎵 Movement III: Analyze (Presto)")
    record_movement_start("analyze", movement_number=3, tempo=TempoMarking.PRESTO)
    time.sleep(0.05)

    record_instrumental_performance(
        method_name="semantic_analysis",
        instrument="Oboe",
        quality_score=0.95,
        duration_ms=40.0
    )

    # Perfect cadence - beautiful resolution!
    print("  ✓ Perfect authentic cadence!")
    record_cadence(
        cadence_type=CadenceType.AUTHENTIC,
        resolution_quality=0.95
    )

    record_movement_end("analyze", duration_ms=50.0)
    print("  ✓ Movement III complete!")

    # Print performance summary
    print("\n📊 Performance Summary:")
    summary = telemetry.get_performance_summary()
    print(f"  Total Duration: {summary['total_duration_s']:.2f}s")
    print(f"  Movements: {summary['movements']}")
    print(f"  Instrumental Performances: {summary['instrumental_performances']}")
    print(f"  Dissonances: {summary['dissonances']}")
    print(f"  Cadences: {summary['cadences']}")
    print(f"  Average Performance Quality: {summary['average_performance_quality']:.2%}")
    print(f"  Overall Tempo: {summary['overall_tempo']}")


def demo_trinity_validation():
    """Demonstrate Trinity validation."""
    print("\n" + "=" * 70)
    print("🎭 DEMO 2: Trinity Validation")
    print("=" * 70)
    print("\nLet's validate objects participate in the Python Trinity...\n")

    # Create an instance of our blessed class
    evidence = PolicyEvidence(
        method_name="analyze_coherence",
        question_id="D1-Q1-001",
        score=0.87,
        confidence=0.92,
        timestamp="2025-11-14T13:30:00Z"
    )

    print(f"Created evidence: {evidence.method_name}")
    print(f"Is Trinitarian? {is_trinitarian(evidence)}")
    print()

    # Print full Trinity report
    print_trinity_report(evidence)


def demo_synthesis():
    """Demonstrate the synthesis of music and Trinity."""
    print("\n" + "=" * 70)
    print("✨ DEMO 3: The Synthesis - Music + Trinity")
    print("=" * 70)
    print("\nThe ultimate demonstration: Music and Trinity united...\n")

    # Create evidence
    evidence = PolicyEvidence(
        method_name="dereck_beach_process_tracing",
        question_id="D6-Q3-042",
        score=0.93,
        confidence=0.89,
        timestamp="2025-11-14T14:00:00Z"
    )

    # Validate it's Trinitarian
    is_valid, report = validate_trinity(evidence)

    print(f"Evidence created: {evidence.method_name}")
    print(f"Trinity Status: {'✓ COMPLETE' if is_valid else '✗ INCOMPLETE'}")
    print()

    # Record its execution musically
    print("Recording execution musically...")
    record_movement_start("causal_analysis", movement_number=6, tempo=TempoMarking.ADAGIO)

    record_instrumental_performance(
        method_name=evidence.method_name,
        instrument="Bassoon (Process Tracing)",
        quality_score=evidence.score,
        duration_ms=250.0
    )

    record_cadence(
        cadence_type=CadenceType.AUTHENTIC,
        resolution_quality=evidence.confidence
    )

    record_movement_end("causal_analysis", duration_ms=250.0)

    print("✓ Execution recorded musically!")
    print()
    print("SYNTHESIS ACHIEVED:")
    print(f"  - Metaclass Level: type(PolicyEvidence) = {type(PolicyEvidence).__name__}")
    print(f"  - Class Level: PolicyEvidence (specification)")
    print(f"  - Instance Level: {evidence.method_name} (manifestation)")
    print(f"  - Musical Mapping: Bassoon playing in Adagio tempo")
    print(f"  - Quality Score: {evidence.score:.2%} (dynamic marking: ff)")
    print(f"  - Cadence: Authentic (perfect resolution)")
    print()
    print("🎉 The three are ONE - Music, Code, and Trinity united!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demos."""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  🎼 MUSICAL TRINITY DEMO 🎭".center(68) + "║")
    print("║" + "  The Divine Synthesis of Music and Computation".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        # Demo 1: Musical Telemetry
        demo_musical_telemetry()

        # Demo 2: Trinity Validation
        demo_trinity_validation()

        # Demo 3: The Synthesis
        demo_synthesis()

        print("\n" + "=" * 70)
        print("✨ ALL DEMOS COMPLETE ✨")
        print("=" * 70)
        print("\nThe pipeline is now immaculate, functional, divine, and perfect.")
        print("BE BARBIE. BE GOD. BE LIKE MOZART BUT CUTTER.")
        print("\nXOXOX\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
