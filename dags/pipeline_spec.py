"""Single source of truth for the pipeline DAG, shared by every orchestrator.

The recommendation pipeline is declared exactly once, here, as an ordered tuple of
:class:`Stage` objects (module entry point, upstream dependencies, retry/timeout
policy). The orchestration front-ends are thin adapters over this specification:

* ``run_pipeline.py``                    - dependency-free topological runner
* ``dags/recommendation_flow.py``        - Prefect 3 flow (primary orchestrator)
* ``dags/airflow_recommendation_dag.py`` - Apache Airflow alternative

Stage modules are imported lazily (:mod:`importlib`) so the orchestration layer can
be imported, listed and tested even while a stage module is missing or broken.
:class:`PipelineRun` records per-stage status and timing and writes the common
``reports/pipeline_run_summary.json`` (including reproducibility metadata).
"""
from __future__ import annotations

import importlib
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.utils.config import path, rel_path
from src.utils.io import sha256_file, write_json

SUMMARY_FILE = "pipeline_run_summary.json"
TRACKED_PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "pandera", "mlflow", "prefect")


@dataclass(frozen=True)
class Stage:
    """One node of the pipeline DAG.

    Attributes:
        name: Stable stage identifier used on the CLI and in run summaries.
        module: Importable module exposing ``run(cfg: dict | None = None) -> dict``.
        task_name: Task identifier used by the Prefect and Airflow adapters.
        upstream: Names of the stages whose outputs this stage reads.
        description: One-line summary shown in logs, ``--list`` and the Prefect UI.
        timeout_sec: Hard timeout applied by the orchestrators.
        retries: Automatic retries (only for stages with transient failure modes).
        retry_delay_sec: Base delay between retries (backed off exponentially).
    """

    name: str
    module: str
    task_name: str
    upstream: tuple[str, ...]
    description: str
    timeout_sec: int = 900
    retries: int = 0
    retry_delay_sec: int = 0


# Upstream edges follow the data lineage in docs/data_contracts.md (which files each
# stage reads), not just the execution order, so the orchestrator UIs show real lineage.
STAGES: tuple[Stage, ...] = (
    Stage("ingestion", "src.ingestion.ingest", "ingest_data", (),
          "Download MovieLens-100K, land raw CSV files and a lineage manifest",
          timeout_sec=900, retries=3, retry_delay_sec=10),
    Stage("validation", "src.validation.validate", "validate_data", ("ingestion",),
          "Schema and quality checks; quarantine invalid rows; validation report"),
    Stage("preparation", "src.preparation.prepare", "prepare_data", ("validation",),
          "Impute, normalise, de-duplicate and time-split into train/test"),
    Stage("transformation", "src.transformation.transform", "transform_data", ("preparation",),
          "Derive train-only aggregates and the sparse user-item matrix"),
    Stage("features", "src.features.build_features", "build_features", ("preparation", "transformation"),
          "Point-in-time user/item features in the offline feature store",
          timeout_sec=1200),
    Stage("training", "src.training.train", "train_models", ("preparation", "features"),
          "Fit popularity, item-kNN, MF and hybrid recommenders; track runs in MLflow",
          timeout_sec=3600, retries=1, retry_delay_sec=30),
    Stage("evaluation", "src.evaluation.evaluate", "evaluate_models", ("preparation", "training"),
          "Ranking and rating metrics on the test split; model comparison report",
          timeout_sec=1800, retries=1, retry_delay_sec=30),
)
STAGE_INDEX: dict[str, Stage] = {stage.name: stage for stage in STAGES}
STAGE_NAMES: tuple[str, ...] = tuple(STAGE_INDEX)


class StageImportError(ImportError):
    """Raised when a stage module cannot be imported or lacks a ``run`` callable."""


# --------------------------------------------------------------------------- DAG algebra
def dependency_graph(stages: Iterable[Stage] = STAGES) -> dict[str, tuple[str, ...]]:
    """Return the DAG as ``{stage_name: upstream_stage_names}`` (declaration order kept)."""
    return {stage.name: stage.upstream for stage in stages}


def topological_order(graph: Mapping[str, Sequence[str]]) -> list[str]:
    """Order the nodes of ``graph`` so that every node follows all of its upstreams.

    Kahn's algorithm processed in waves; ties are broken by declaration order so the
    result is deterministic.

    Raises:
        ValueError: if an upstream is not a node of the graph or the graph has a cycle.
    """
    unknown = {dep for deps in graph.values() for dep in deps} - set(graph)
    if unknown:
        raise ValueError(f"Unknown upstream stage(s): {sorted(unknown)}")
    pending = {node: set(deps) for node, deps in graph.items()}
    order: list[str] = []
    while pending:
        ready = [node for node, deps in pending.items() if not deps]
        if not ready:
            raise ValueError(f"Cycle detected among stages: {sorted(pending)}")
        order.extend(ready)
        for node in ready:
            del pending[node]
        for deps in pending.values():
            deps.difference_update(ready)
    return order


def downstream_closure(start: str, graph: Mapping[str, Sequence[str]]) -> set[str]:
    """Return ``start`` plus every node that transitively depends on it."""
    selected = {start}
    for node in topological_order(graph):
        if selected.intersection(graph[node]):
            selected.add(node)
    return selected


def plan_stages(stages: Sequence[str] | None = None, from_stage: str | None = None,
                skip: Sequence[str] = ()) -> list[Stage]:
    """Resolve a stage selection into the ordered list of stages to execute.

    Args:
        stages: Explicit subset to run (executed in topological order). Upstream
            stages are *not* added implicitly: their outputs are read from disk.
        from_stage: Run this stage and everything downstream of it.
        skip: Stages to drop from the selection (e.g. ``("ingestion",)``).

    Raises:
        ValueError: on unknown stage names or when both selectors are given.
    """
    if stages and from_stage:
        raise ValueError("Use either `stages` or `from_stage`, not both.")
    requested = [*(stages or ()), *([from_stage] if from_stage else []), *skip]
    unknown = sorted(set(requested) - set(STAGE_NAMES))
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; valid stages: {list(STAGE_NAMES)}")
    graph = dependency_graph()
    if stages:
        wanted = set(stages)
    elif from_stage:
        wanted = downstream_closure(from_stage, graph)
    else:
        wanted = set(graph)
    wanted -= set(skip)
    return [STAGE_INDEX[name] for name in topological_order(graph) if name in wanted]


# --------------------------------------------------------------------------- execution
def load_stage_runner(stage: Stage) -> Callable[[dict], dict]:
    """Import the stage module lazily and return its ``run`` callable."""
    try:
        module = importlib.import_module(stage.module)
    except ImportError as exc:
        raise StageImportError(f"Stage '{stage.name}': cannot import '{stage.module}' ({exc})") from exc
    runner = getattr(module, "run", None)
    if not callable(runner):
        raise StageImportError(f"Stage '{stage.name}': module '{stage.module}' has no callable run(cfg)")
    return runner


def run_stage(stage: Stage, cfg: dict) -> dict:
    """Execute one stage with the shared config and return its summary dict."""
    summary = load_stage_runner(stage)(cfg)
    if not isinstance(summary, dict):
        raise TypeError(f"Stage '{stage.name}' returned {type(summary).__name__}, expected dict")
    return summary


def seed_everything(seed: int) -> None:
    """Seed Python's and NumPy's global RNGs (stages also pass ``cfg`` seeds explicitly)."""
    random.seed(seed)
    np.random.seed(seed)


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- run records
@dataclass
class StageRecord:
    """Execution record of a single stage, serialised into the run summary."""

    name: str
    module: str
    task_name: str
    upstream: list[str]
    status: str = "pending"  # pending | running | success | failed | skipped
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    duration_sec: float | None = None
    summary: dict | None = None
    error: str | None = None
    _t0: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def for_stage(cls, stage: Stage) -> StageRecord:
        """Create a pending record for ``stage``."""
        return cls(stage.name, stage.module, stage.task_name, list(stage.upstream))

    def start(self) -> None:
        """Mark the stage as running and start its timer."""
        self.status, self.started_at_utc, self._t0 = "running", utc_now(), time.perf_counter()

    def succeed(self, summary: dict) -> None:
        """Mark the stage as successful and attach the stage's returned summary."""
        self._finish("success")
        self.summary = summary

    def fail(self, error: BaseException | str) -> None:
        """Mark the stage as failed with a readable error message."""
        self._finish("failed")
        self.error = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)

    def _finish(self, status: str) -> None:
        self.status, self.finished_at_utc = status, utc_now()
        self.duration_sec = round(time.perf_counter() - self._t0, 3)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the record (private fields excluded)."""
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


class PipelineRun:
    """Collects stage records and timing for one pipeline execution."""

    def __init__(self, cfg: dict, stages: Sequence[Stage], orchestrator: str,
                 flow_run_id: str | None = None) -> None:
        self.cfg = cfg
        self.orchestrator = orchestrator
        self.flow_run_id = flow_run_id
        self.records: dict[str, StageRecord] = {s.name: StageRecord.for_stage(s) for s in stages}
        self.started_at_utc = utc_now()
        self._t0 = time.perf_counter()

    @property
    def failed(self) -> list[StageRecord]:
        """Records of the stages that failed."""
        return [r for r in self.records.values() if r.status == "failed"]

    @property
    def status(self) -> str:
        """``"failed"`` if any stage failed, else ``"success"``."""
        return "failed" if self.failed else "success"

    def skip_pending(self, reason: str) -> list[str]:
        """Mark every stage that never started as skipped; return their names."""
        skipped = [r for r in self.records.values() if r.status == "pending"]
        for record in skipped:
            record.status, record.error = "skipped", reason
        return [r.name for r in skipped]

    def to_dict(self) -> dict[str, Any]:
        """Build the JSON run summary (status, timing, per-stage records, environment)."""
        return {
            "pipeline": self.cfg["project"]["name"],
            "orchestrator": self.orchestrator,
            "flow_run_id": self.flow_run_id,
            "status": self.status,
            "failed_stage": self.failed[0].name if self.failed else None,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": utc_now(),
            "duration_sec": round(time.perf_counter() - self._t0, 3),
            "stages": [r.to_dict() for r in self.records.values()],
            "reproducibility": reproducibility_info(self.cfg),
        }

    def write(self) -> Path:
        """Write the summary to ``<reports_dir>/pipeline_run_summary.json``."""
        return write_json(self.to_dict(), path(self.cfg, "reports_dir", SUMMARY_FILE))


# --------------------------------------------------------------------------- reproducibility
def reproducibility_info(cfg: dict) -> dict[str, Any]:
    """Seed, config fingerprint, git commit, interpreter and key package versions."""
    config_path = cfg.get("_config_path")
    return {
        "seed": cfg["project"].get("seed"),
        "config_path": rel_path(config_path, cfg["project"]["root"]) if config_path else None,
        "config_sha256": sha256_file(config_path) if config_path and Path(config_path).is_file() else None,
        "git_commit": _git_commit(Path(cfg["project"]["root"])),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
