"""Stage 2 - data validation (raw landing CSVs -> typed, validated interim layer).

Modules:
    schemas     entity specs, Pandera schemas (diagnostic + strict contract) and the check catalogue
    checks      vectorised custom checks (null tokens, type coercion, duplicates) and the IssueLog
    validate    stage entrypoint ``run(cfg)`` orchestrating users -> items -> interactions
    report      data profile, ground-truth comparison and JSON / Markdown report rendering
"""
from src.validation.exceptions import DataValidationError

__all__ = ["DataValidationError"]
