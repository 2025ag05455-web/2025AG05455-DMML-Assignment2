"""Uniform logging to console and logs/pipeline.log."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from src.utils.config import PROJECT_ROOT

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DIR_ENV = "RECSYS_LOG_DIR"  # overrides the log directory (the test suite points it at a temp dir)


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Return logger ``name`` writing to the console and ``<log_dir>/pipeline.log``.

    ``log_dir`` defaults to ``$RECSYS_LOG_DIR`` or ``<project root>/logs``. Handlers are attached
    only once per logger name, so repeated calls never duplicate output.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_recsys_configured", False):
        return logger
    logger.setLevel(logging.INFO)
    log_dir = Path(log_dir or os.environ.get(LOG_DIR_ENV) or PROJECT_ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger._recsys_configured = True  # type: ignore[attr-defined]
    return logger
