"""Make the project root importable as ``src`` for all tests.

Stage loggers are redirected to a temporary directory so that test runs never append to the
pipeline's own ``logs/pipeline.log``.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("RECSYS_LOG_DIR", tempfile.mkdtemp(prefix="recsys-test-logs-"))
