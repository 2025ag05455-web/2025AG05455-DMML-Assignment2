"""Apache Airflow DAG - alternative to the Prefect flow, for teams standardised on Airflow.

**Not used by default** (Airflow is not a project dependency, so this file is only
syntax-checked by the test suite). It schedules exactly the same stage DAG as
``dags/recommendation_flow.py`` because both are generated from
:mod:`dags.pipeline_spec`: one ``PythonOperator`` per stage, the same lineage edges,
the same per-stage retries/timeouts, a daily 02:00 UTC schedule and no catch-up.

Usage (Airflow 2.4+ or 3.x)::

    pip install "apache-airflow>=2.9"
    export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
    export RECSYS_CONFIG=$PWD/configs/config.yaml   # optional alternative config
    airflow dags test recommendation_pipeline

Each task returns the stage's JSON summary, which Airflow stores as an XCom.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # make `src` and `dags` importable from the Airflow DAG folder
    sys.path.insert(0, str(ROOT))

from airflow import DAG  # noqa: E402

try:  # Airflow 3.x
    from airflow.providers.standard.operators.python import PythonOperator  # noqa: E402
except ImportError:  # Airflow 2.x
    from airflow.operators.python import PythonOperator  # noqa: E402

from dags.pipeline_spec import STAGE_INDEX, STAGES, run_stage, seed_everything  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def run_stage_task(stage_name: str) -> dict:
    """Airflow callable: load the config (``$RECSYS_CONFIG`` aware) and run one stage."""
    cfg = load_config()
    seed_everything(int(cfg["project"]["seed"]))
    return run_stage(STAGE_INDEX[stage_name], cfg)


DEFAULT_ARGS = {"owner": "dm4ml", "depends_on_past": False}

with DAG(
    dag_id="recommendation_pipeline",
    description="MovieLens-100K end-to-end recommendation pipeline (Airflow alternative)",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["recsys", "dm4ml"],
    doc_md=__doc__,
) as dag:
    operators = {
        stage.name: PythonOperator(
            task_id=stage.task_name,
            python_callable=run_stage_task,
            op_kwargs={"stage_name": stage.name},
            retries=stage.retries,
            retry_delay=timedelta(seconds=stage.retry_delay_sec or 60),
            retry_exponential_backoff=stage.retries > 0,
            execution_timeout=timedelta(seconds=stage.timeout_sec),
            doc_md=stage.description,
        )
        for stage in STAGES
    }
    for stage in STAGES:
        for upstream in stage.upstream:
            operators[upstream] >> operators[stage.name]
