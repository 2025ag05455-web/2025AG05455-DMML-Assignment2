"""Small, dependency-free I/O helpers shared by all pipeline stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet(df: pd.DataFrame, p: str | Path) -> Path:
    p = Path(p)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)
    return p


def read_parquet(p: str | Path) -> pd.DataFrame:
    return pd.read_parquet(p)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def write_json(obj: Any, p: str | Path) -> Path:
    p = Path(p)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return p


def read_json(p: str | Path) -> Any:
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(p: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
