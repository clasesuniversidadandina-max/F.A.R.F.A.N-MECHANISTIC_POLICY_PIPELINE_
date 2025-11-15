"""
Table Extraction Module (Deprecated - Compatibility Stub)

This module provides backward compatibility for table extraction from policy documents.
The functionality has been migrated to dedicated table extraction modules.

For new code, use table extraction utilities in analysis modules.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

warnings.warn(
    "cpp_ingestion.tables is deprecated. "
    "Use dedicated table extraction modules instead.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class TableExtractor:
    """
    Extract tables from policy documents (Compatibility Stub).

    This is a minimal compatibility implementation.
    For full table extraction, use specialized modules.
    """

    config: dict[str, Any] = field(default_factory=dict)

    def extract_tables(
        self, text: str, page_num: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Extract tables from text (Stub implementation).

        Parameters
        ----------
        text : str
            Text to extract tables from
        page_num : int, optional
            Page number for context

        Returns
        -------
        list[dict[str, Any]]
            List of extracted tables (stub returns empty list)
        """
        warnings.warn(
            "TableExtractor.extract_tables is a stub implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    def extract_from_pdf(self, pdf_path: str) -> list[dict[str, Any]]:
        """
        Extract tables from PDF file (Stub).

        Parameters
        ----------
        pdf_path : str
            Path to PDF file

        Returns
        -------
        list[dict[str, Any]]
            Extracted tables (stub returns empty list)
        """
        return []


__all__ = ["TableExtractor"]
