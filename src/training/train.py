"""Stage 6 - Model training with MLflow experiment tracking.

For every model enabled in ``cfg['training']['models']`` one MLflow run (named after the model)
is opened in the configured experiment. Each run records

* params  - the model hyperparameters and the data lineage (split cutoff, training-set size and
  checksum, feature-set version / as_of, raw archive sha256, config sha256),
* tags    - ``model_type``, ``stage``, ``pipeline``, ``pipeline_version``,
* metrics - ``fit_time_sec``, ``train_rmse``, ``train_mae`` (+ ``train_rmse_epoch`` per SGD epoch),
* artifacts - the pickled model, its metadata and the exact config file used.

Every model is fitted from the transformation stage's train user-item matrix
(``user_item_matrix.npz`` + ``matrix_index.json``), which is first checked against
``train.parquet`` (non-zeros, ids and values) so a stale matrix fails the stage before any run is
opened. Models read their features through :class:`src.features.store.FeatureStore` - the same
access layer used by evaluation and serving (``src.training.predict``). Before training, the
store's point-in-time ``as_of`` is checked against the split cutoff so features can never include
information from the test period. The hybrid reuses the base model (``mf_svd``) fitted earlier in
the same run instead of training an identical copy; it fits its own only when that model is disabled.

Inputs:
    processed_dir/train.parquet, split_metadata.json (preparation),
    processed_dir/user_item_matrix.npz, matrix_index.json (transformation),
    features_dir/* via FeatureStore (features), raw_dir/ingestion_manifest.json (lineage only)

Outputs (models_dir):
    <model>/model.pkl       pickled :class:`src.training.models.BaseRecommender`
    <model>/metadata.json   run_id, experiment_id, params, lineage, trained_at, train metrics
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
from scipy import sparse

from src.evaluation.metrics import mae, rmse
from src.features.store import FeatureStore
from src.training.models import BaseRecommender, HybridRecommender, InteractionMatrix, create_model
from src.training.tracking import configure_mlflow
from src.transformation.transform import MATRIX_FILE, MATRIX_INDEX_FILE
from src.utils.config import load_config, path, rel_path
from src.utils.io import read_json, read_parquet, sha256_file, write_json
from src.utils.logger import get_logger

log = get_logger("training")

MODEL_FILE = "model.pkl"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class TrainingInputs:
    """Inputs shared by every model of one training run (loaded once).

    Attributes:
        train: ``train.parquet`` interactions.
        interactions: the transformation stage's user-item matrix, verified against ``train``.
        store: feature access layer.
        lineage: MLflow lineage parameters (:func:`data_lineage`).
    """

    train: pd.DataFrame
    interactions: InteractionMatrix
    store: FeatureStore
    lineage: dict


def enabled_models(cfg: dict) -> list[str]:
    """Names of the models switched on in the config (config order)."""
    return [name for name, params in cfg["training"]["models"].items() if params.get("enabled", True)]


def training_order(names: list[str]) -> list[str]:
    """``names`` with the hybrid last, so the base model it reuses is fitted before it."""
    return sorted(names, key=lambda name: name == HybridRecommender.name)


def load_interaction_matrix(cfg: dict) -> InteractionMatrix:
    """Load the transformation stage's train user-item matrix and its row / column index.

    Raises:
        FileNotFoundError: if the matrix or its index is missing.
        ValueError: if the index does not fit the matrix.
    """
    files = [path(cfg, "processed_dir", MATRIX_FILE), path(cfg, "processed_dir", MATRIX_INDEX_FILE)]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Transformation outputs not found: {missing}. Run the transformation stage "
                                "first (python -m src.transformation.transform) before training.")
    index = read_json(files[1])
    return InteractionMatrix(sparse.load_npz(files[0]).tocsr(), np.asarray(index["user_ids"], dtype=np.int64),
                             np.asarray(index["item_ids"], dtype=np.int64))


def data_lineage(cfg: dict, store: FeatureStore) -> dict:
    """Lineage parameters shared by every run; checks feature/split point-in-time consistency.

    Raises:
        ValueError: if the feature store's ``as_of`` differs from the split cutoff.
    """
    split = read_json(path(cfg, "processed_dir", "split_metadata.json"))
    features = store.metadata()
    if features["as_of_timestamp"] != split["cutoff_timestamp"]:
        raise ValueError(f"feature as_of {features['as_of_timestamp']} != split cutoff {split['cutoff_timestamp']}; "
                         "rebuild features (src.features.build_features) after re-splitting")
    manifest = read_json(path(cfg, "raw_dir", "ingestion_manifest.json"))
    return {
        "data.split_method": split["method"],
        "data.split_cutoff_timestamp": split["cutoff_timestamp"],
        "data.split_cutoff_utc": split["cutoff_time_utc"],
        "data.n_train": split["n_train"],
        "data.n_train_users": split["n_train_users"],
        "data.n_train_items": split["n_train_items"],
        "data.train_sha256": sha256_file(path(cfg, "processed_dir", "train.parquet")),
        "data.user_item_matrix_sha256": sha256_file(path(cfg, "processed_dir", MATRIX_FILE)),
        "data.feature_set_version": features["version"],
        "data.feature_as_of": features["as_of"],
        "data.raw_archive_sha256": manifest["archive_sha256"],
        "config_sha256": sha256_file(cfg["_config_path"]),
    }


def run_tags(cfg: dict, model: BaseRecommender) -> dict[str, str]:
    """MLflow tags identifying the model type and the pipeline release."""
    return {
        "model_type": type(model).__name__,
        "model_name": model.name,
        "stage": "training",
        "pipeline": cfg["project"]["name"],
        "pipeline_version": str(cfg["training"]["mlflow"]["pipeline_version"]),
    }


def train_metrics(model: BaseRecommender, train: pd.DataFrame) -> dict[str, float]:
    """In-sample rating error (optimistic by construction; test metrics come from evaluation)."""
    predictions = model.predict(train["user_id"], train["item_id"])
    return {"train_rmse": rmse(train["rating"], predictions), "train_mae": mae(train["rating"], predictions)}


def train_model(name: str, cfg: dict, inputs: TrainingInputs, experiment_id: str,
                fitted: Mapping[str, tuple[BaseRecommender, str]]) -> tuple[dict, BaseRecommender]:
    """Fit one model inside its own MLflow run, persist it and return its run summary and the model.

    ``fitted`` maps the names of the models already trained in this run to ``(model, run_id)``;
    a hybrid whose base model is among them reuses it (recorded as tag / metadata
    ``base_model_run_id``) instead of fitting an identical copy.
    """
    model = create_model(name, cfg["training"]["models"], seed=int(cfg["project"]["seed"]))
    tags = run_tags(cfg, model)
    reused = isinstance(model, HybridRecommender) and model.base_model in fitted
    if reused:
        base, tags["base_model_run_id"] = fitted[model.base_model]
        model.use_fitted_base(base)
    model_dir = path(cfg, "models_dir", name)
    with mlflow.start_run(experiment_id=experiment_id, run_name=name, tags=tags) as active:
        params = model.get_params()
        mlflow.log_params({**params, **inputs.lineage})
        started = time.perf_counter()
        model.fit(inputs.train, feature_store=inputs.store, interactions=inputs.interactions)
        metrics = {"fit_time_sec": round(time.perf_counter() - started, 3), **train_metrics(model, inputs.train)}
        mlflow.log_metrics(metrics)
        for epoch, value in enumerate(model.epoch_rmse_, start=1):
            mlflow.log_metric("train_rmse_epoch", value, step=epoch)
        model_path = model.save(model_dir / MODEL_FILE)
        metadata = {
            "model": name,
            "model_class": type(model).__name__,
            "run_id": active.info.run_id,
            "experiment_id": experiment_id,
            "tracking_uri": mlflow.get_tracking_uri(),
            "params": params,
            "lineage": inputs.lineage,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_metrics": metrics,
            "epoch_train_rmse": model.epoch_rmse_,
            "n_users": len(model.user_ids_),
            "n_items": len(model.item_ids_),
            **({"base_model_run_id": tags["base_model_run_id"]} if reused else {}),
        }
        metadata_path = write_json(metadata, model_dir / METADATA_FILE)
        for artifact in (model_path, metadata_path):
            mlflow.log_artifact(str(artifact), artifact_path="model")
        mlflow.log_artifact(cfg["_config_path"], artifact_path="config")
    log.info("Trained %-10s run_id=%s fit=%.1fs train_rmse=%.4f%s", name, active.info.run_id,
             metrics["fit_time_sec"], metrics["train_rmse"],
             f" (reused {model.base_model} run {tags['base_model_run_id']})" if reused else "")
    return {"run_id": active.info.run_id, "model_path": rel_path(model_path, cfg["project"]["root"]), **metrics}, model


def run(cfg: dict | None = None) -> dict:
    """Train every enabled model and return ``{stage, experiment_id, tracking_uri, models}``."""
    cfg = cfg or load_config()
    train = read_parquet(path(cfg, "processed_dir", "train.parquet"))
    interactions = load_interaction_matrix(cfg)
    interactions.check_matches(train)  # fail before any MLflow run is opened
    store = FeatureStore(cfg)
    inputs = TrainingInputs(train, interactions, store, data_lineage(cfg, store))
    experiment_id = configure_mlflow(cfg)
    names = enabled_models(cfg)
    log.info("Training %s on %d interactions from the %s x %s user-item matrix (experiment %s)", names, len(train),
             *interactions.matrix.shape, experiment_id)
    results: dict[str, dict] = {}
    fitted: dict[str, tuple[BaseRecommender, str]] = {}
    for name in training_order(names):
        results[name], model = train_model(name, cfg, inputs, experiment_id, fitted)
        fitted[name] = (model, results[name]["run_id"])
    results = {name: results[name] for name in names}  # report in config order
    return {
        "stage": "training",
        "experiment_id": experiment_id,
        "experiment_name": cfg["training"]["mlflow"]["experiment_name"],
        "tracking_uri": cfg["training"]["mlflow"]["tracking_uri"],  # as configured (project-relative)
        "n_train": int(len(train)),
        "models": results,
    }


if __name__ == "__main__":
    run()
