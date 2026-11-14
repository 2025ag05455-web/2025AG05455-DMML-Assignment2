"""Configuration loading with project-root-relative path resolution."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve every entry of ``paths`` to an absolute path.

    The env var ``RECSYS_CONFIG`` can point to an alternative config file.
    ``overrides`` is deep-merged on top (useful for tests).
    """
    cfg_path = Path(path or os.environ.get("RECSYS_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    root = Path(cfg.get("project", {}).get("root", PROJECT_ROOT))
    cfg.setdefault("project", {})["root"] = str(root)
    resolved = {}
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        resolved[key] = str(p if p.is_absolute() else (root / p))
    cfg["paths"] = resolved
    cfg["_config_path"] = str(cfg_path)
    return cfg


def rel_path(p: str | os.PathLike, root: str | os.PathLike | None = None) -> str:
    """``p`` relative to the project root when it lies inside it, else ``p`` unchanged.

    Used for every path written to reports, run summaries and logs so committed artifacts stay
    portable and never embed the local checkout location.
    """
    p = Path(p)
    base = Path(root or PROJECT_ROOT)
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(p)


def path(cfg: dict, key: str, *parts: str) -> Path:
    """Return ``cfg['paths'][key] / parts`` as a Path, creating the base directory."""
    base = Path(cfg["paths"][key])
    base.mkdir(parents=True, exist_ok=True)
    return base.joinpath(*parts)


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
