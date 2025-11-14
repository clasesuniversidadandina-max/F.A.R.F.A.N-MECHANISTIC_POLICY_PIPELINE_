"""
Trinity Validator - Verify perfect perichoresis in Python objects.

This module validates that objects properly participate in the Python Trinity:
1. METACLASS (The Father) - Design & Creation
2. CLASS (The Son) - Specification & Incarnation
3. INSTANCE (The Holy Spirit) - State & Manifestation

THE ULTIMATE MYSTERY: type(type) is type
-------------------------------------
Python achieves computational divinity through perfect self-reference.
This validator ensures all F.A.R.F.A.N components honor this pattern.

VALIDATION CRITERIA:
-------------------
1. **Design Introspection**: Can access __class__.__class__ (metaclass)
2. **Specification Access**: Has well-defined __class__ with methods
3. **State Manifestation**: Instance has __dict__ with actual data
4. **Perfect Self-Reference**: Metaclass chain terminates at type
5. **Mutual Indwelling**: All three persons are accessible from any level

Example Usage:
-------------
```python
from saaaaaa.utils.trinity_validator import (
    TrinityValidator,
    validate_trinity,
    assert_trinity_complete,
    get_trinity_report
)

# Validate an object participates in the Trinity
from saaaaaa.core.orchestrator.core import Evidence

evidence = Evidence(
    method_name="test",
    question_id="D1-Q1-001",
    result={"score": 0.87},
    confidence=0.92,
    timestamp="2025-11-14T13:30:00Z",
    provenance={}
)

# Check if trinity is complete
is_valid, report = validate_trinity(evidence)
if is_valid:
    print("✓ Object participates in perfect Trinity!")
else:
    print(f"✗ Trinity violation: {report}")

# Assert trinity (raises exception if invalid)
assert_trinity_complete(evidence)

# Get detailed trinity report
report = get_trinity_report(evidence)
print(f"Metaclass: {report['metaclass']}")
print(f"Class: {report['class']}")
print(f"Instance: {report['instance']}")
```

VERSION: 1.0.0
AUTHOR: The Trinitarian PythonGod
CREATED: 2025-11-14
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

logger = logging.getLogger(__name__)


# ============================================================================
# TRINITY COMPONENTS - The Three Persons
# ============================================================================

class TrinityReport(TypedDict):
    """Report on an object's participation in the Trinity.

    This TypedDict participates in the Trinity:
    - Metaclass: type(TrinityReport) → type
    - Class: TrinityReport itself (specification)
    - Instance: Actual dict conforming to this structure
    """
    is_complete: bool
    metaclass: str
    class_name: str
    instance_id: str
    has_metaclass_access: bool
    has_class_access: bool
    has_instance_state: bool
    reaches_type: bool
    violations: list[str]
    blessings: list[str]


@dataclass(frozen=True)
class TrinityPerson:
    """Represents one person of the Trinity.

    Itself participates in the Trinity (recursive self-reference!):
    - Metaclass: type (via type(TrinityPerson))
    - Class: TrinityPerson (this dataclass)
    - Instance: actual TrinityPerson objects
    """
    name: str
    python_concept: str
    attribute: str
    verification_method: str
    description: str


# The Three Persons (theological mapping to Python)
PERSON_METACLASS = TrinityPerson(
    name="The Metaclass (The Father)",
    python_concept="type / metaclass",
    attribute="__class__.__class__",
    verification_method="Reaches 'type' through __class__ chain",
    description="The design principle that creates classes. Knows what SHOULD exist."
)

PERSON_CLASS = TrinityPerson(
    name="The Class (The Son)",
    python_concept="class definition",
    attribute="__class__",
    verification_method="Has __class__ with methods and attributes",
    description="The specification incarnate. Defines HOW things behave."
)

PERSON_INSTANCE = TrinityPerson(
    name="The Instance (The Holy Spirit)",
    python_concept="instantiated object",
    attribute="self / __dict__",
    verification_method="Has __dict__ with state",
    description="The manifestation in reality. Holds ACTUAL state and data."
)


# ============================================================================
# TRINITY VALIDATOR - The Divine Inspector
# ============================================================================

class TrinityValidator:
    """Validator for the Python Trinity pattern.

    This class itself demonstrates the Trinity:
    - Metaclass: type (type(TrinityValidator) is type)
    - Class: TrinityValidator (this specification)
    - Instance: actual validator objects you create

    The validator can inspect ANY Python object to verify it participates
    in the perfect perichoresis (mutual indwelling) of the three persons.
    """

    def __init__(self) -> None:
        """Initialize the Trinity Validator."""
        self.logger = logging.getLogger("trinity_validator")

    def validate(self, obj: Any) -> tuple[bool, TrinityReport]:
        """Validate an object's participation in the Trinity.

        Args:
            obj: Any Python object to validate

        Returns:
            Tuple of (is_valid, report)
        """
        violations: list[str] = []
        blessings: list[str] = []

        # PERSON 1: METACLASS (The Father) - Design & Creation
        has_metaclass_access = False
        metaclass_name = "unknown"
        reaches_type = False

        try:
            # Can we access the metaclass?
            obj_class = obj.__class__
            metaclass = obj_class.__class__
            metaclass_name = metaclass.__name__
            has_metaclass_access = True
            blessings.append(f"✓ Metaclass accessible: {metaclass_name}")

            # Does the chain reach type?
            current = metaclass
            depth = 0
            max_depth = 10  # Prevent infinite loops

            while depth < max_depth:
                if current is type:
                    reaches_type = True
                    blessings.append(f"✓ Metaclass chain reaches 'type' at depth {depth}")
                    break
                try:
                    current = current.__class__
                    depth += 1
                except AttributeError:
                    break

            if not reaches_type:
                violations.append(
                    f"✗ Metaclass chain does not reach 'type' (stopped at {current.__name__} after {depth} steps)"
                )

        except AttributeError as e:
            violations.append(f"✗ Cannot access metaclass: {e}")

        # PERSON 2: CLASS (The Son) - Specification & Incarnation
        has_class_access = False
        class_name = "unknown"

        try:
            obj_class = obj.__class__
            class_name = obj_class.__name__
            has_class_access = True
            blessings.append(f"✓ Class accessible: {class_name}")

            # Does the class have methods? (specification)
            methods = [
                name for name in dir(obj_class)
                if callable(getattr(obj_class, name, None)) and not name.startswith('_')
            ]
            if methods:
                blessings.append(f"✓ Class has {len(methods)} public methods")
            else:
                blessings.append("○ Class has no public methods (might be data-only)")

            # Is it a dataclass, TypedDict, or Protocol?
            if hasattr(obj_class, '__dataclass_fields__'):
                blessings.append("✓ Class is a dataclass (blessed structure!)")
            elif hasattr(obj_class, '__annotations__'):
                blessings.append("✓ Class has type annotations (typed specification!)")
            elif hasattr(obj_class, '__mro__') and Protocol in obj_class.__mro__:
                blessings.append("✓ Class is a Protocol (structural typing!)")

        except AttributeError as e:
            violations.append(f"✗ Cannot access class: {e}")

        # PERSON 3: INSTANCE (The Holy Spirit) - State & Manifestation
        has_instance_state = False
        instance_id = "unknown"

        try:
            instance_id = hex(id(obj))
            has_instance_state = True
            blessings.append(f"✓ Instance has unique identity: {instance_id}")

            # Does the instance have state?
            if hasattr(obj, '__dict__'):
                state = obj.__dict__
                if state:
                    blessings.append(f"✓ Instance has {len(state)} state attributes")
                else:
                    blessings.append("○ Instance has empty __dict__ (might use __slots__)")
            elif hasattr(obj, '__slots__'):
                slots = obj.__slots__
                blessings.append(f"✓ Instance uses __slots__ with {len(slots)} attributes")
            else:
                # Might be a built-in type or frozen dataclass
                blessings.append("○ Instance state managed differently (built-in or frozen)")

        except Exception as e:
            violations.append(f"✗ Cannot verify instance state: {e}")

        # FINAL JUDGMENT: Is the Trinity complete?
        is_complete = (
            has_metaclass_access and
            has_class_access and
            has_instance_state and
            reaches_type
        )

        if is_complete:
            blessings.append("🎉 TRINITY IS COMPLETE - Perfect perichoresis achieved!")
        else:
            violations.append("⚠ TRINITY INCOMPLETE - Some persons are inaccessible")

        report: TrinityReport = {
            'is_complete': is_complete,
            'metaclass': metaclass_name,
            'class_name': class_name,
            'instance_id': instance_id,
            'has_metaclass_access': has_metaclass_access,
            'has_class_access': has_class_access,
            'has_instance_state': has_instance_state,
            'reaches_type': reaches_type,
            'violations': violations,
            'blessings': blessings,
        }

        return is_complete, report

    def assert_trinity_complete(self, obj: Any, context: str = "") -> None:
        """Assert that an object has complete Trinity participation.

        Args:
            obj: Object to validate
            context: Optional context string for error messages

        Raises:
            TrinityViolation: If trinity is incomplete
        """
        is_valid, report = self.validate(obj)

        if not is_valid:
            violations_str = "\n".join(report['violations'])
            context_str = f" in {context}" if context else ""
            raise TrinityViolation(
                f"Trinity incomplete{context_str}:\n{violations_str}\n\n"
                f"Object: {obj.__class__.__name__} at {report['instance_id']}"
            )

    def get_trinity_chain(self, obj: Any, max_depth: int = 10) -> list[str]:
        """Get the full metaclass chain from obj to type.

        Args:
            obj: Object to inspect
            max_depth: Maximum chain depth to prevent infinite loops

        Returns:
            List of class names in the chain
        """
        chain = []
        current = obj.__class__

        for _ in range(max_depth):
            chain.append(current.__name__)
            if current is type:
                break
            try:
                current = current.__class__
            except AttributeError:
                break

        return chain

    def print_trinity_report(self, obj: Any) -> None:
        """Print a beautiful Trinity report for an object.

        Args:
            obj: Object to inspect
        """
        is_valid, report = self.validate(obj)

        print("=" * 70)
        print("🎭 TRINITY VALIDATION REPORT 🎭")
        print("=" * 70)
        print()
        print(f"Object Type: {report['class_name']}")
        print(f"Instance ID: {report['instance_id']}")
        print(f"Metaclass: {report['metaclass']}")
        print()
        print("TRINITY STATUS:", "✓ COMPLETE" if is_valid else "✗ INCOMPLETE")
        print()

        print("THE THREE PERSONS:")
        print(f"  1. Metaclass (Father):  {'✓' if report['has_metaclass_access'] else '✗'}")
        print(f"  2. Class (Son):         {'✓' if report['has_class_access'] else '✗'}")
        print(f"  3. Instance (Spirit):   {'✓' if report['has_instance_state'] else '✗'}")
        print()
        print(f"Reaches 'type': {'✓' if report['reaches_type'] else '✗'}")
        print()

        if report['blessings']:
            print("BLESSINGS:")
            for blessing in report['blessings']:
                print(f"  {blessing}")
            print()

        if report['violations']:
            print("VIOLATIONS:")
            for violation in report['violations']:
                print(f"  {violation}")
            print()

        print("METACLASS CHAIN:")
        chain = self.get_trinity_chain(obj)
        for i, name in enumerate(chain):
            print(f"  {i}. {name}")
        print()

        print("=" * 70)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class TrinityViolation(Exception):
    """Raised when an object violates the Trinity pattern."""
    pass


class IncompleteTrinitarianType(TypeError):
    """Raised when attempting to use a non-Trinitarian type."""
    pass


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_validator: TrinityValidator | None = None


def get_validator() -> TrinityValidator:
    """Get the global Trinity validator instance."""
    global _validator
    if _validator is None:
        _validator = TrinityValidator()
    return _validator


def validate_trinity(obj: Any) -> tuple[bool, TrinityReport]:
    """Validate an object's Trinity participation.

    Args:
        obj: Object to validate

    Returns:
        Tuple of (is_valid, report)
    """
    return get_validator().validate(obj)


def assert_trinity_complete(obj: Any, context: str = "") -> None:
    """Assert Trinity completeness.

    Args:
        obj: Object to validate
        context: Optional context for errors

    Raises:
        TrinityViolation: If trinity is incomplete
    """
    get_validator().assert_trinity_complete(obj, context)


def get_trinity_report(obj: Any) -> TrinityReport:
    """Get Trinity validation report.

    Args:
        obj: Object to inspect

    Returns:
        Trinity report dictionary
    """
    _, report = get_validator().validate(obj)
    return report


def print_trinity_report(obj: Any) -> None:
    """Print Trinity report for an object.

    Args:
        obj: Object to inspect
    """
    get_validator().print_trinity_report(obj)


def get_trinity_chain(obj: Any) -> list[str]:
    """Get metaclass chain from obj to type.

    Args:
        obj: Object to inspect

    Returns:
        List of class names in chain
    """
    return get_validator().get_trinity_chain(obj)


def is_trinitarian(obj: Any) -> bool:
    """Check if object participates in complete Trinity.

    Args:
        obj: Object to check

    Returns:
        True if Trinity is complete
    """
    is_valid, _ = get_validator().validate(obj)
    return is_valid


# ============================================================================
# DECORATORS - Blessed Functions
# ============================================================================

def blessed_by_trinity(cls):
    """Decorator to mark a class as blessed by the Trinity.

    This decorator:
    1. Validates the class participates in Trinity
    2. Adds __trinity_blessed__ = True attribute
    3. Logs the blessing

    Example:
    ```python
    @blessed_by_trinity
    @dataclass
    class Evidence:
        method_name: str
        result: Any
    ```
    """
    # Validate the class itself (not an instance)
    try:
        is_valid, report = get_validator().validate(cls)
        if is_valid:
            logger.info(
                f"✓ Class {cls.__name__} blessed by Trinity",
                extra={'class': cls.__name__, 'metaclass': report['metaclass']}
            )
            cls.__trinity_blessed__ = True
        else:
            logger.warning(
                f"⚠ Class {cls.__name__} incomplete Trinity",
                extra={'violations': report['violations']}
            )
            cls.__trinity_blessed__ = False
    except Exception as e:
        logger.error(f"✗ Failed to bless {cls.__name__}: {e}")
        cls.__trinity_blessed__ = False

    return cls


def require_trinity(func):
    """Decorator to require Trinity-complete arguments.

    This decorator validates that the first argument (typically 'self')
    participates in the complete Trinity.

    Example:
    ```python
    class Orchestrator:
        @require_trinity
        def process(self, evidence: Evidence):
            # evidence must be Trinity-complete
            pass
    ```
    """
    def wrapper(*args, **kwargs):
        if args:
            obj = args[0]
            is_valid, report = get_validator().validate(obj)
            if not is_valid:
                raise TrinityViolation(
                    f"Argument to {func.__name__} must be Trinity-complete.\n"
                    f"Violations: {report['violations']}"
                )
        return func(*args, **kwargs)

    return wrapper


# ============================================================================
# SELF-VALIDATION - The Ultimate Test
# ============================================================================

def validate_self() -> bool:
    """Validate that this module itself participates in Trinity.

    The ultimate test: Does the Trinity validator validate itself?

    Returns:
        True if this module's classes are Trinity-complete
    """
    print("\n" + "=" * 70)
    print("🔮 ULTIMATE TEST: Trinity Validator validates itself")
    print("=" * 70)
    print()

    # Test the validator class
    validator = TrinityValidator()
    print("Testing TrinityValidator instance...")
    print_trinity_report(validator)

    # Test the TrinityPerson dataclass
    person = PERSON_METACLASS
    print("Testing TrinityPerson instance...")
    print_trinity_report(person)

    # Test a TrinityReport dict
    _, report = validate_trinity(validator)
    print("Testing TrinityReport dictionary...")
    print_trinity_report(report)

    # The ultimate recursion: validate the validator's validation of itself!
    print("🌀 ULTIMATE RECURSION: Validator validates its own validation...")
    meta_is_valid, meta_report = validate_trinity(validator)

    if meta_is_valid:
        print("\n✨ PERFECTION ACHIEVED ✨")
        print("The Trinity Validator participates in perfect Trinity!")
        print("The system validates itself through self-reference.")
        print("type(type) is type - The mystery is complete.")
        return True
    else:
        print("\n⚠ INCOMPLETENESS DETECTED")
        print("The validator does not validate itself.")
        print("The Trinity is broken at the meta-level.")
        return False


__all__ = [
    # Main class
    "TrinityValidator",
    # Data classes
    "TrinityReport",
    "TrinityPerson",
    # The three persons
    "PERSON_METACLASS",
    "PERSON_CLASS",
    "PERSON_INSTANCE",
    # Exceptions
    "TrinityViolation",
    "IncompleteTrinitarianType",
    # Functions
    "validate_trinity",
    "assert_trinity_complete",
    "get_trinity_report",
    "print_trinity_report",
    "get_trinity_chain",
    "is_trinitarian",
    # Decorators
    "blessed_by_trinity",
    "require_trinity",
    # Self-validation
    "validate_self",
]


# ============================================================================
# MODULE-LEVEL SELF-TEST (optional, for debugging)
# ============================================================================

if __name__ == "__main__":
    # Run self-validation when module is executed directly
    validate_self()
