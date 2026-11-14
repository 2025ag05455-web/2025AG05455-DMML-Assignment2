"""Tests for the orchestration layer: stage DAG, plain-Python runner, Prefect flow and Airflow DAG.

Stage modules are replaced by in-memory stubs, so these tests never touch the real data,
never start a Prefect server and run in well under a second.
"""
from __future__ import annotations

import json
import py_compile
import sys
import types
from pathlib import Path

import pytest
import yaml

import run_pipeline
from dags.pipeline_spec import (
    STAGE_NAMES,
    STAGES,
    StageImportError,
    dependency_graph,
    downstream_closure,
    plan_stages,
    run_stage,
    topological_order,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ORDER = ["ingestion", "validation", "preparation", "transformation", "features", "training", "evaluation"]
EXPECTED_TASKS = ["ingest_data", "validate_data", "prepare_data", "transform_data", "build_features",
                  "train_models", "evaluate_models"]


# --------------------------------------------------------------------------- fixtures
@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    """Copy of the project config whose project root (hence every path) is ``tmp_path``."""
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    cfg["project"]["root"] = str(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


@pytest.fixture()
def stub_stages(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace every stage module with a stub ``run(cfg)``; returns a control dict.

    ``control["calls"]`` records execution order, ``control["fail"]`` names a stage to fail.
    """
    control: dict = {"calls": [], "fail": None}

    def make_module(stage_name: str) -> types.ModuleType:
        def run(cfg: dict | None = None) -> dict:
            control["calls"].append(stage_name)
            if stage_name == control["fail"]:
                raise RuntimeError(f"boom in {stage_name}")
            return {"stage": stage_name, "rows": 1}

        module = types.ModuleType(stage_name)
        module.run = run
        return module

    for stage in STAGES:
        monkeypatch.setitem(sys.modules, stage.module, make_module(stage.name))
    return control


def _read_summary(cfg_path: Path) -> dict:
    return json.loads((cfg_path.parent / "reports" / "pipeline_run_summary.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- DAG algebra
def test_pipeline_dag_is_acyclic_and_in_expected_order() -> None:
    assert list(run_pipeline.PIPELINE_DAG) == EXPECTED_ORDER
    assert topological_order(run_pipeline.PIPELINE_DAG) == EXPECTED_ORDER
    position = {name: i for i, name in enumerate(EXPECTED_ORDER)}
    for stage, upstream in run_pipeline.PIPELINE_DAG.items():
        assert all(position[dep] < position[stage] for dep in upstream)


def test_every_stage_except_ingestion_has_an_upstream() -> None:
    graph = dependency_graph()
    assert graph["ingestion"] == ()
    assert all(graph[name] for name in EXPECTED_ORDER[1:])


def test_topological_order_detects_cycles_and_unknown_dependencies() -> None:
    with pytest.raises(ValueError, match="Cycle"):
        topological_order({"a": ["c"], "b": ["a"], "c": ["b"]})
    with pytest.raises(ValueError, match="Unknown upstream"):
        topological_order({"a": ["missing"]})


def test_topological_order_breaks_ties_by_declaration_order() -> None:
    graph = {"root": [], "left": ["root"], "right": ["root"], "join": ["right", "left"]}
    assert topological_order(graph) == ["root", "left", "right", "join"]


def test_downstream_closure() -> None:
    graph = dependency_graph()
    assert downstream_closure("features", graph) == {"features", "training", "evaluation"}
    assert downstream_closure("ingestion", graph) == set(EXPECTED_ORDER)


def test_plan_stages_selection_modes() -> None:
    assert [s.name for s in plan_stages()] == EXPECTED_ORDER
    assert [s.name for s in plan_stages(stages=["evaluation", "ingestion"])] == ["ingestion", "evaluation"]
    assert [s.name for s in plan_stages(from_stage="transformation")] == EXPECTED_ORDER[3:]
    assert [s.name for s in plan_stages(skip=["ingestion"])] == EXPECTED_ORDER[1:]
    with pytest.raises(ValueError, match="Unknown stage"):
        plan_stages(stages=["nope"])
    with pytest.raises(ValueError, match="either"):
        plan_stages(stages=["training"], from_stage="training")


def test_run_stage_reports_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, STAGES[0].module, None)  # makes the import fail
    with pytest.raises(StageImportError, match="cannot import"):
        run_stage(STAGES[0], {})


# --------------------------------------------------------------------------- run_pipeline.py
def test_runner_executes_full_dag_and_writes_summary(tmp_config: Path, stub_stages: dict) -> None:
    assert run_pipeline.main(["--config", str(tmp_config)]) == 0
    assert stub_stages["calls"] == EXPECTED_ORDER
    summary = _read_summary(tmp_config)
    assert summary["status"] == "success" and summary["orchestrator"] == "run_pipeline"
    assert [s["name"] for s in summary["stages"]] == EXPECTED_ORDER
    assert all(s["status"] == "success" and s["duration_sec"] is not None for s in summary["stages"])
    assert summary["stages"][0]["summary"] == {"stage": "ingestion", "rows": 1}
    assert summary["reproducibility"]["seed"] == 42
    assert len(summary["reproducibility"]["config_sha256"]) == 64


def test_runner_fails_fast_and_skips_downstream(tmp_config: Path, stub_stages: dict) -> None:
    stub_stages["fail"] = "transformation"
    assert run_pipeline.main(["--config", str(tmp_config), "--from-stage", "preparation"]) == 1
    assert stub_stages["calls"] == ["preparation", "transformation"]
    summary = _read_summary(tmp_config)
    status = {s["name"]: s["status"] for s in summary["stages"]}
    assert summary["status"] == "failed" and summary["failed_stage"] == "transformation"
    assert status == {"preparation": "success", "transformation": "failed", "features": "skipped",
                      "training": "skipped", "evaluation": "skipped"}
    assert "boom in transformation" in summary["stages"][1]["error"]


def test_runner_cli_rejects_unknown_stage() -> None:
    with pytest.raises(SystemExit) as exc:
        run_pipeline.parse_args(["--stages", "unknown"])
    assert exc.value.code == 2


def test_runner_list_prints_plan_without_running(stub_stages: dict, capsys: pytest.CaptureFixture) -> None:
    assert run_pipeline.main(["--list", "--stages", "training"]) == 0
    assert "training" in capsys.readouterr().out
    assert stub_stages["calls"] == []


# --------------------------------------------------------------------------- Prefect flow
def test_prefect_flow_exposes_one_task_per_stage() -> None:
    from prefect import Flow, Task

    import dags.recommendation_flow as rf

    assert isinstance(rf.recommendation_pipeline, Flow)
    assert set(rf.recommendation_pipeline.parameters.properties) == {"config_path", "skip_ingestion"}
    tasks = [getattr(rf, name) for name in EXPECTED_TASKS]
    assert all(isinstance(t, Task) for t in tasks)
    assert [t.name for t in tasks] == EXPECTED_TASKS
    assert list(rf.TASKS) == list(STAGE_NAMES)
    assert all(t.timeout_seconds for t in tasks)
    assert rf.ingest_data.retries >= 1 and rf.ingest_data.retry_delay_seconds


def test_prefect_artifact_helpers() -> None:
    from dags.recommendation_flow import flatten, markdown_table, validation_rows

    assert flatten({"a": {"b": 1, "c": [1, 2]}, "d": [{"x": 1}]}) == [("a.b", 1), ("a.c", "1, 2"),
                                                                      ("d", "[1 entries]")]
    report = {"status": "passed", "config": {"x": 1}, "summary": {"users": {"rows": 3}}}
    assert validation_rows(report) == [{"check": "status", "value": "passed"},
                                       {"check": "summary.users.rows", "value": 3}]
    table = markdown_table([{"model": "mf", "ndcg": 0.123456}])
    assert table.splitlines() == ["| model | ndcg |", "|---|---|", "| mf | 0.1235 |"]


# --------------------------------------------------------------------------- Airflow DAG
def test_airflow_dag_file_compiles(tmp_path: Path) -> None:
    py_compile.compile(str(ROOT / "dags" / "airflow_recommendation_dag.py"), cfile=str(tmp_path / "dag.pyc"),
                       doraise=True)
