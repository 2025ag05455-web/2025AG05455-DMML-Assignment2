#!/usr/bin/env python
"""Prefect 3 orchestration of the end-to-end recommendation pipeline.

One Prefect task per stage (``ingest_data`` ... ``evaluate_models``). Upstream
dependencies come from :mod:`dags.pipeline_spec` and are passed to every task via
``wait_for`` so the lineage DAG is visible in the Prefect UI. Retries and timeouts
are also taken from the spec (network-bound ``ingest_data`` retries with
exponential back-off). Validation results and the final model comparison are
published as Prefect artifacts, and every run writes
``reports/pipeline_run_summary.json``.

Usage::

    python dags/recommendation_flow.py                    # run once (temporary local Prefect API)
    python dags/recommendation_flow.py --skip-ingestion   # reuse data/raw
    python dags/recommendation_flow.py --serve            # daily deployment, 02:00 UTC (long-running)

To see runs in the UI start ``make prefect-ui`` (binds 127.0.0.1:4200) and export
``PREFECT_API_URL=http://127.0.0.1:4200/api`` before running the flow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python dags/recommendation_flow.py`
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact, create_table_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402
from prefect.runtime import flow_run  # noqa: E402
from prefect.states import State  # noqa: E402
from prefect.tasks import exponential_backoff  # noqa: E402

from dags.pipeline_spec import STAGE_INDEX, PipelineRun, plan_stages, run_stage, seed_everything  # noqa: E402
from src.utils.config import load_config, path, rel_path  # noqa: E402
from src.utils.io import read_json  # noqa: E402

DEPLOYMENT_NAME = "daily-recsys-pipeline"
DAILY_CRON = "0 2 * * *"
MAX_ARTIFACT_ROWS = 200
MAX_MARKDOWN_CHARS = 50_000
#: Top-level sections of reports/validation_report.json shown in the headline table artifact.
VALIDATION_HEADLINE_SECTIONS = ("status", "summary", "quality_gate", "post_condition_gate", "ground_truth")


def _task_options(stage_name: str) -> dict[str, Any]:
    """Prefect ``@task`` keyword arguments derived from the shared stage spec."""
    stage = STAGE_INDEX[stage_name]
    return {
        "name": stage.task_name,
        "description": stage.description,
        "tags": ["recsys", stage.name],
        "timeout_seconds": stage.timeout_sec,
        "retries": stage.retries,
        "retry_delay_seconds": exponential_backoff(stage.retry_delay_sec) if stage.retries else 0,
        "retry_jitter_factor": 0.2 if stage.retries else None,
        # Stages have file side effects, so results must never be served from a cache.
        "cache_policy": NO_CACHE,
    }


def _execute(stage_name: str, cfg: dict) -> dict:
    """Run one stage inside a Prefect task with task-level logging."""
    logger = get_run_logger()
    stage = STAGE_INDEX[stage_name]
    logger.info("Starting stage '%s' (%s): %s", stage.name, stage.module, stage.description)
    summary = run_stage(stage, cfg)
    logger.info("Stage '%s' summary: %s", stage.name, json.dumps(summary, default=str)[:1000])
    return summary


# --------------------------------------------------------------------------- tasks
@task(**_task_options("ingestion"))
def ingest_data(cfg: dict) -> dict:
    """Stage 1: download/reuse MovieLens-100K and land raw CSVs (retried: network I/O)."""
    return _execute("ingestion", cfg)


@task(**_task_options("validation"))
def validate_data(cfg: dict) -> dict:
    """Stage 2: validate raw data and publish the validation report as an artifact."""
    summary = _execute("validation", cfg)
    publish_validation_artifact(cfg)
    return summary


@task(**_task_options("preparation"))
def prepare_data(cfg: dict) -> dict:
    """Stage 3: clean, impute, de-duplicate and split train/test by time."""
    return _execute("preparation", cfg)


@task(**_task_options("transformation"))
def transform_data(cfg: dict) -> dict:
    """Stage 4: build train-only aggregates and the user-item matrix."""
    return _execute("transformation", cfg)


@task(**_task_options("features"))
def build_features(cfg: dict) -> dict:
    """Stage 5: materialise point-in-time features into the offline feature store."""
    return _execute("features", cfg)


@task(**_task_options("training"))
def train_models(cfg: dict) -> dict:
    """Stage 6: train all enabled recommenders and log them to MLflow."""
    return _execute("training", cfg)


@task(**_task_options("evaluation"))
def evaluate_models(cfg: dict) -> dict:
    """Stage 7: evaluate the models and publish the model comparison as an artifact."""
    summary = _execute("evaluation", cfg)
    publish_model_comparison_artifact(cfg)
    return summary


TASKS = {
    "ingestion": ingest_data,
    "validation": validate_data,
    "preparation": prepare_data,
    "transformation": transform_data,
    "features": build_features,
    "training": train_models,
    "evaluation": evaluate_models,
}


# --------------------------------------------------------------------------- artifacts
def flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dicts into ``(dotted.key, value)`` pairs; lists are summarised."""
    if isinstance(obj, dict):
        return [pair for k, v in obj.items() for pair in flatten(v, f"{prefix}.{k}" if prefix else str(k))]
    if isinstance(obj, list):
        nested = any(isinstance(x, (dict, list)) for x in obj)
        return [(prefix, f"[{len(obj)} entries]" if nested else ", ".join(map(str, obj))[:200])]
    return [(prefix, obj)]


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    """Render a list of homogeneous dicts as a GitHub-flavoured markdown table."""
    if not rows:
        return "_no rows_"
    headers = list(rows[0])

    def fmt(value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |" for row in rows]
    return "\n".join(lines)


def load_model_comparison(cfg: dict) -> list[dict[str, Any]]:
    """Rows of ``reports/model_comparison.csv`` (empty list if it does not exist)."""
    csv_path = path(cfg, "reports_dir", "model_comparison.csv")
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    return df.round(4).astype(object).where(df.notna(), "").to_dict(orient="records")


def validation_rows(report: dict) -> list[dict[str, Any]]:
    """Headline rows of the validation report (status, counts, gates, recall) for a table artifact.

    Falls back to the whole flattened report if none of the headline sections exist.
    """
    rows = [{"check": key, "value": value} for key, value in flatten(report)]
    headline = [r for r in rows if r["check"].split(".")[0] in VALIDATION_HEADLINE_SECTIONS]
    return (headline or rows)[:MAX_ARTIFACT_ROWS]


def publish_validation_artifact(cfg: dict) -> None:
    """Publish the validation results: headline table (from JSON) + full markdown report."""
    logger = get_run_logger()
    json_path = path(cfg, "reports_dir", "validation_report.json")
    md_path = json_path.with_suffix(".md")
    if not json_path.is_file():
        logger.warning("No validation report at %s; artifact skipped", json_path)
        return
    create_table_artifact(table=validation_rows(read_json(json_path)), key="validation-summary",
                          description=f"Headline data-validation results ({json_path.name})")
    if md_path.is_file():
        create_markdown_artifact(markdown=md_path.read_text(encoding="utf-8")[:MAX_MARKDOWN_CHARS],
                                 key="validation-report", description=f"Full report ({md_path.name})")


def publish_model_comparison_artifact(cfg: dict) -> None:
    """Publish ``reports/model_comparison.csv`` as a Prefect table artifact."""
    rows = load_model_comparison(cfg)
    if not rows:
        get_run_logger().warning("No model_comparison.csv found; artifact skipped")
        return
    create_table_artifact(table=rows, key="model-comparison",
                          description="Offline evaluation of all recommenders on the time-based test split")


def publish_run_summary_artifact(cfg: dict, summary: dict) -> None:
    """Publish a markdown overview of the run: stage status/timing and model comparison."""
    stage_rows = [{"stage": s["name"], "task": s["task_name"], "status": s["status"],
                   "duration_sec": s["duration_sec"] if s["duration_sec"] is not None else "-"}
                  for s in summary["stages"]]
    sections = [
        f"# Recommendation pipeline run: {summary['status'].upper()}",
        f"Flow run `{summary['flow_run_id']}` | started {summary['started_at_utc']} | "
        f"duration {summary['duration_sec']:.1f} s | seed {summary['reproducibility']['seed']}",
        "## Stages", markdown_table(stage_rows),
    ]
    comparison = load_model_comparison(cfg)
    if comparison:
        sections += ["## Model comparison", markdown_table(comparison)]
    create_markdown_artifact(markdown="\n\n".join(sections), key="pipeline-run-summary",
                             description="Per-stage status, timing and final model comparison")


# --------------------------------------------------------------------------- flow
@flow(name="recommendation-pipeline", log_prints=True, timeout_seconds=4 * 3600,
      description="MovieLens-100K: ingestion -> validation -> preparation -> transformation -> "
                  "features -> training -> evaluation")
def recommendation_pipeline(config_path: str | None = None, skip_ingestion: bool = False) -> dict:
    """Run every stage in dependency order and write ``reports/pipeline_run_summary.json``.

    Args:
        config_path: Alternative YAML config (default ``configs/config.yaml`` / ``$RECSYS_CONFIG``).
        skip_ingestion: Reuse the existing ``data/raw`` landing files instead of re-ingesting.

    Raises:
        RuntimeError: if any stage fails (the summary is still written first).
    """
    logger = get_run_logger()
    cfg = load_config(config_path)
    seed_everything(int(cfg["project"]["seed"]))
    stages = plan_stages(skip=("ingestion",) if skip_ingestion else ())
    run = PipelineRun(cfg, stages, orchestrator="prefect", flow_run_id=flow_run.id)
    logger.info("Execution plan: %s", " -> ".join(s.name for s in stages))

    states: dict[str, State] = {}
    for stage in stages:
        record = run.records[stage.name]
        record.start()
        upstream = [states[name] for name in stage.upstream if name in states]
        state = TASKS[stage.name](cfg, wait_for=upstream, return_state=True)
        if not state.is_completed():
            record.fail(state.message or state.name)
            logger.error("Stage '%s' ended in state %s: %s", stage.name, state.name, record.error)
            break
        record.succeed(state.result())
        states[stage.name] = state

    skipped = run.skip_pending(reason="not run: an upstream stage failed")
    out = run.write()
    summary = run.to_dict()
    publish_run_summary_artifact(cfg, summary)
    logger.info("Run summary written to %s", rel_path(out, cfg["project"]["root"]))
    if run.failed:
        raise RuntimeError(f"Stage '{run.failed[0].name}' failed ({run.failed[0].error}); "
                           f"skipped: {skipped or 'none'}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Run or schedule the Prefect recommendation pipeline.")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/config.yaml)")
    parser.add_argument("--skip-ingestion", action="store_true", help="reuse existing data/raw files")
    parser.add_argument("--serve", action="store_true",
                        help=f"serve deployment '{DEPLOYMENT_NAME}' on cron '{DAILY_CRON}' (blocks)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: run the flow once (exit status 0/1), or serve it on a daily schedule."""
    args = parse_args(argv)
    params = {"config_path": args.config, "skip_ingestion": args.skip_ingestion}
    if args.serve:
        recommendation_pipeline.serve(name=DEPLOYMENT_NAME, cron=DAILY_CRON, parameters=params,
                                      tags=["recsys", "daily"],
                                      description="Daily retrain and evaluation of the MovieLens recommenders")
        return 0
    state = recommendation_pipeline(**params, return_state=True)
    if not state.is_completed():
        print(f"ERROR: flow run ended in state {state.name}: {state.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
