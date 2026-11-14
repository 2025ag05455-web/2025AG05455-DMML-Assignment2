"""Exceptions raised by the validation stage."""
from __future__ import annotations

from typing import Any


class DataValidationError(Exception):
    """A data-quality gate failed: the data must not flow to downstream stages.

    ``details`` carries a JSON-serialisable context (gate values, failing checks, ...)
    so orchestration code can log or persist the reason without parsing the message.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details or {}
