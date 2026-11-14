"""MLflow experiment-tracking helpers shared by training, evaluation and prediction.

The tracking store and the artifact root from ``cfg['training']['mlflow']`` are resolved to
absolute locations under the project root, so runs land in the same store regardless of the
working directory the pipeline is started from.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import sqlalchemy.pool
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

log = get_logger("tracking")

SQLITE_PREFIX = "sqlite:///"


def ensure_sqlalchemy_compat() -> None:
    """Let MLflow 2.22's SQL store import under SQLAlchemy >= 2.1.

    ``mlflow.store.db.utils`` imports ``sqlalchemy.pool.FallbackAsyncAdaptedQueuePool``, which
    SQLAlchemy 2.1 removed. MLflow only uses the name to translate the optional
    ``MLFLOW_SQLALCHEMYSTORE_POOLCLASS`` setting, so aliasing it to its parent class
    ``AsyncAdaptedQueuePool`` does not change the sqlite store's behaviour. No-op on SQLAlchemy 2.0.
    """
    if not hasattr(sqlalchemy.pool, "FallbackAsyncAdaptedQueuePool"):
        sqlalchemy.pool.FallbackAsyncAdaptedQueuePool = sqlalchemy.pool.AsyncAdaptedQueuePool


@contextmanager
def preserved_root_logging() -> Iterator[None]:
    """Restore the root logger's handlers and level after MLflow initialises a new store.

    Creating a fresh SQL tracking store runs MLflow's Alembic migrations, whose ``env.py`` calls
    ``logging.config.fileConfig``: it swaps the root handlers (e.g. Prefect's console handler)
    for Alembic's plain console handler for the rest of the process.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def resolve_tracking_uri(uri: str, root: Path) -> str:
    """Absolute tracking URI: relative ``sqlite:///x.db`` or plain paths are anchored at ``root``."""
    if uri.startswith(SQLITE_PREFIX):
        db = Path(uri[len(SQLITE_PREFIX):])
        return uri if db.is_absolute() else SQLITE_PREFIX + str((root / db).resolve())
    if "://" in uri:
        return uri
    return (root / uri).resolve().as_uri()


def resolve_artifact_location(location: str, root: Path) -> str:
    """Artifact root as a URI; relative paths become ``file://`` URIs under ``root``."""
    if "://" in location:
        return location
    return (root / location).resolve().as_uri()


def configure_mlflow(cfg: dict) -> str:
    """Point MLflow at the configured store and return the experiment id (created if missing).

    A deleted experiment of the same name is restored rather than re-created, because MLflow
    names stay reserved until the deleted experiment is purged.
    """
    ensure_sqlalchemy_compat()
    mlflow_cfg = cfg["training"]["mlflow"]
    root = Path(cfg["project"]["root"])
    mlflow.set_tracking_uri(resolve_tracking_uri(mlflow_cfg["tracking_uri"], root))
    name = mlflow_cfg["experiment_name"]
    artifact_location = resolve_artifact_location(mlflow_cfg["artifact_location"], root)
    with preserved_root_logging():  # opening the store may create and migrate a new database
        client = MlflowClient()
        experiment = client.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = client.create_experiment(name, artifact_location=artifact_location)
        log.info("Created MLflow experiment %r (id=%s, artifacts=%s)", name, experiment_id, artifact_location)
    else:
        experiment_id = experiment.experiment_id
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment_id)
        if experiment.artifact_location != artifact_location:
            log.warning("Experiment %r keeps its original artifact location %s (config: %s)",
                        name, experiment.artifact_location, artifact_location)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id
