#!/usr/bin/env python
"""Dependency-free workflow runner for the recommendation pipeline.

A plain-Python alternative to the Prefect flow (``dags/recommendation_flow.py``):
it executes the stage DAG declared in :mod:`dags.pipeline_spec` in topological
order, times every stage, stops at the first failure (downstream stages are marked
``skipped``) and writes the same ``reports/pipeline_run_summary.json``.

Examples::

    python run_pipeline.py                              # full pipeline
    python run_pipeline.py --stages ingestion           # a single stage
    python run_pipeline.py --stages training evaluation # a subset (DAG order kept)
    python run_pipeline.py --from-stage features        # features + everything downstream
    python run_pipeline.py --config configs/config.yaml --list   # show the plan only

Exit status: 0 on success, 1 if a stage failed, 2 on invalid arguments.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from dags.pipeline_spec import (
    STAGE_NAMES,
    PipelineRun,
    Stage,
    dependency_graph,
    plan_stages,
    run_stage,
    seed_everything,
)
from src.utils.config import load_config, rel_path
from src.utils.logger import get_logger

log = get_logger("orchestration")

#: The pipeline DAG as ``{stage: upstream stages}`` (shared with the Prefect/Airflow adapters).
PIPELINE_DAG: dict[str, tuple[str, ...]] = dependency_graph()


def execute(stages: Sequence[Stage], cfg: dict) -> PipelineRun:
    """Run ``stages`` sequentially (already topologically ordered), failing fast."""
    run = PipelineRun(cfg, stages, orchestrator="run_pipeline")
    for position, stage in enumerate(stages, start=1):
        record = run.records[stage.name]
        log.info("[%d/%d] %s - %s", position, len(stages), stage.name, stage.description)
        record.start()
        try:
            record.succeed(run_stage(stage, cfg))
        except Exception as exc:  # noqa: BLE001 - any stage error must stop the run cleanly
            record.fail(exc)
            log.exception("Stage '%s' failed after %.2fs: %s", stage.name, record.duration_sec, record.error)
            break
        log.info("[%d/%d] %s finished in %.2fs", position, len(stages), stage.name, record.duration_sec)
    skipped = run.skip_pending(reason="not run: an upstream stage failed")
    if skipped:
        log.error("Fail-fast: skipped downstream stage(s) %s", ", ".join(skipped))
    return run


def run_pipeline(config_path: str | None = None, stages: Sequence[str] | None = None,
                 from_stage: str | None = None) -> dict:
    """Programmatic entry point: plan, execute and persist one pipeline run.

    Returns the run summary (also written to ``reports/pipeline_run_summary.json``).
    """
    cfg = load_config(config_path)
    seed_everything(int(cfg["project"]["seed"]))
    plan = plan_stages(stages=stages, from_stage=from_stage)
    log.info("Execution plan: %s", " -> ".join(s.name for s in plan))
    run = execute(plan, cfg)
    out = run.write()
    summary = run.to_dict()
    log.info("Pipeline %s in %.2fs; summary written to %s",
             summary["status"].upper(), summary["duration_sec"], rel_path(out, cfg["project"]["root"]))
    return summary


def format_plan(plan: Sequence[Stage]) -> str:
    """Human-readable execution plan used by ``--list``."""
    rows = [f"{i}. {s.name:<15} module={s.module:<30} upstream={','.join(s.upstream) or '-'}"
            for i, s in enumerate(plan, start=1)]
    return "\n".join(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/config.yaml or $RECSYS_CONFIG)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stages", nargs="+", choices=STAGE_NAMES, metavar="STAGE",
                           help=f"run only these stages, in DAG order; choices: {', '.join(STAGE_NAMES)}")
    selection.add_argument("--from-stage", choices=STAGE_NAMES, metavar="STAGE",
                           help="run this stage and every stage downstream of it")
    parser.add_argument("--list", action="store_true", help="print the execution plan and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit status."""
    args = parse_args(argv)
    if args.list:
        print(format_plan(plan_stages(stages=args.stages, from_stage=args.from_stage)))
        return 0
    summary = run_pipeline(args.config, stages=args.stages, from_stage=args.from_stage)
    if summary["status"] != "success":
        failed = next(s for s in summary["stages"] if s["status"] == "failed")
        print(f"ERROR: stage '{failed['name']}' failed - {failed['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
