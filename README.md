# End-to-End Data Management Pipeline for a Recommendation System

A reproducible, modular recommendation pipeline on the **MovieLens-100K** dataset, built for
*Data Management for Machine Learning – Assignment 2*. It covers the full data-management
lifecycle — **ingestion → validation → preparation → transformation → feature management →
training → evaluation → orchestration** — with experiment tracking in MLflow and a schedulable
Prefect flow.

> Each stage is a self-contained module exposing `run(cfg) -> dict`. Stages communicate **only
> through files** (data contracts in [`docs/data_contracts.md`](docs/data_contracts.md)), so any
> stage can be re-run, tested or debugged in isolation. The orchestration layer adds scheduling,
> dependency ordering, retries and a run record — but no business logic.

## 1. Pipeline at a glance

```
MovieLens-100K
     │
     ▼
1 Ingestion → 2 Validation → 3 Preparation → 4 Transformation
     → 5 Feature management → 6 Training → 7 Evaluation      (8 Orchestration wraps all of it)
```

Full architecture and the per-stage DAG (with timeouts/retries) are in
[`docs/architecture.md`](docs/architecture.md); the rendered diagram is
[`reports/figures/architecture.png`](reports/figures/architecture.png).

| # | Stage | Module | Key outputs |
|---|-------|--------|-------------|
| 1 | Ingestion | `src/ingestion/ingest.py` | `data/raw/*.csv`, `ingestion_manifest.json`, seeded `injected_anomalies.json` |
| 2 | Validation | `src/validation/validate.py` | `data/interim/*.parquet`, `data/errors/*_errors.csv`, `reports/validation_report.{json,md}` |
| 3 | Preparation | `src/preparation/prepare.py` | `*_clean.parquet`, time-based `train.parquet`/`test.parquet`, `split_metadata.json` |
| 4 | Transformation | `src/transformation/transform.py` | `user_activity_summary`, `item_popularity`, `user_item_matrix.npz` |
| 5 | Feature management | `src/features/` | offline feature store `data/features/*.parquet`, `reports/feature_catalog.md` |
| 6 | Training | `src/training/` | `models/{model}/model.pkl` + MLflow runs (`mlflow.db`, `mlartifacts/`) |
| 7 | Evaluation | `src/evaluation/` | `reports/model_comparison.{csv,md}`, `evaluation_results.json`, figures |
| 8 | Orchestration | `dags/`, `run_pipeline.py` | `reports/pipeline_run_summary.json` |

## 2. Dataset

[MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) (GroupLens): 943 users, 1,682
movies, 100,000 ratings (Sep 1997 – Apr 1998). The archive is **downloaded automatically** by the
ingestion stage (and cached under `data/raw/source/`); its licence forbids redistribution, so the
data itself is not committed — a clean clone regenerates everything with `make pipeline`.

To demonstrate that the validation layer actually catches problems, ingestion injects a small,
**seeded and fully documented** set of synthetic data-quality anomalies (duplicates, out-of-range
ratings, wrong types, missing keys, invalid timestamps, orphan foreign keys, bad ages/genders).
Validation detects **100%** of them (see the validation report).

## 3. Quick start

Requires **Python 3.11**. Everything is driven by the `Makefile` (`make help` lists all targets).

```bash
# 1. create the virtualenv and install pinned dependencies
make setup

# 2. run the whole pipeline end-to-end (downloads MovieLens on first run, ~25 s)
make pipeline            # dependency-free runner (run_pipeline.py)
#   or, as a Prefect flow:
make flow

# 3. run the tests
make test
```

Outputs land in `data/`, `models/`, `reports/` and MLflow (`mlflow.db` + `mlartifacts/`).

### Run individual stages

```bash
make ingest validate prepare transform features train evaluate   # any subset, in order
# or with the runner directly:
.venv/bin/python run_pipeline.py --from-stage features            # a stage + everything downstream
.venv/bin/python run_pipeline.py --stages training evaluation     # an explicit subset (DAG order kept)
.venv/bin/python run_pipeline.py --list                           # print the execution plan only
```

### Inspect runs in the UIs

```bash
make mlflow-ui     # MLflow tracking UI → http://127.0.0.1:5000
make prefect-ui    # Prefect server + UI → http://127.0.0.1:4200
```

Both UIs bind to `127.0.0.1` only; view them from another machine via an SSH port-forward rather
than binding to a public interface.

### Schedule it

```bash
.venv/bin/python dags/recommendation_flow.py --serve   # Prefect deployment, cron 0 2 * * * (02:00 UTC)
```

An equivalent Airflow DAG is provided in `dags/airflow_recommendation_dag.py` for teams
standardised on Airflow (Airflow is not a project dependency).

### Get recommendations

```bash
.venv/bin/python -m src.training.predict --model hybrid --user-id 1 --k 10
```

## 4. Models

Two required families, four models in total (all tracked in MLflow):

| Model | Type | Notes |
|-------|------|-------|
| `popularity` | **Baseline** | popularity-ranked; Bayesian-damped item mean for rating prediction |
| `item_knn` | Personalized — collaborative (memory-based) | item–item cosine kNN with shrinkage |
| `mf_svd` | Personalized — matrix factorization | biased SGD latent-factor model |
| `hybrid` | Personalized — hybrid | MF collaborative score + genre-affinity content score |

Cold-start users (no training history) fall back to the popularity recommender.

## 5. Results (time-based test split, K = 10)

Primary metric **NDCG@10** on warm users — best model: **hybrid**. Full table (Precision/Recall/
NDCG/HitRate/MAP @5 & @10, Coverage, Diversity, Novelty, RMSE, MAE) with a cold-start slice is in
[`reports/model_comparison.md`](reports/model_comparison.md).

| model | precision@10 | recall@10 | ndcg@10 | coverage | diversity | rmse | mae |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.108 | 0.083 | 0.134 | 0.047 | 0.747 | 1.031 | 0.833 |
| item_knn | 0.106 | 0.057 | 0.119 | 0.113 | 0.708 | 1.074 | 0.827 |
| mf_svd | 0.078 | 0.086 | 0.114 | 0.102 | 0.732 | 0.971 | 0.764 |
| **hybrid** | 0.094 | **0.101** | **0.153** | 0.113 | 0.458 | **0.971** | **0.764** |

## 6. Deliverables map

| Assignment deliverable | Location |
|---|---|
| Source code, modular | `src/`, `dags/`, `run_pipeline.py` |
| `requirements.txt` | [`requirements.txt`](requirements.txt) (pinned; `make lock` → `requirements.lock`) |
| Architecture diagram | [`reports/figures/architecture.png`](reports/figures/architecture.png), [`docs/architecture.md`](docs/architecture.md) |
| Data validation report / logs | [`reports/validation_report.md`](reports/validation_report.md), `data/errors/`, `reports/logs/` |
| Feature catalog (table) | [`reports/feature_catalog.md`](reports/feature_catalog.md) |
| Model evaluation comparing models | [`reports/model_comparison.md`](reports/model_comparison.md), `reports/evaluation_results.json`, `reports/figures/` |
| Orchestration DAG / workflow | `dags/recommendation_flow.py` (Prefect), `dags/airflow_recommendation_dag.py` (Airflow), `run_pipeline.py` |
| Successful-run evidence | `reports/pipeline_run_summary.json`, `reports/logs/prefect_flow_run.log`; Prefect UI screenshots [`pipeline_run.png`](reports/figures/pipeline_run.png), [`pipeline_tasks.png`](reports/figures/pipeline_tasks.png), [`prefect_runs_list.png`](reports/figures/prefect_runs_list.png); MLflow UI screenshots [`mlflow_runs.png`](reports/figures/mlflow_runs.png), [`mlflow_compare.png`](reports/figures/mlflow_compare.png) |
| Final report (2–3 pages) | [`reports/final_report.md`](reports/final_report.md) / [`reports/final_report.pdf`](reports/final_report.pdf) (3-page report + figure appendix; rebuild with `make report`) |

## 7. Project structure

```
recommendation-pipeline/
├── configs/config.yaml          # single source of configuration (paths, params, seeds)
├── dags/                        # orchestration: Prefect flow, Airflow DAG, shared pipeline spec
├── data/                        # data lake: raw / interim / processed / features / errors (gitignored except errors)
├── docs/                        # architecture.md, data_contracts.md
├── models/                      # trained model artifacts (gitignored; regenerated)
├── reports/                     # validation report, feature catalog, model comparison, figures, final report
├── scripts/                     # diagram + report builders
├── src/
│   ├── ingestion/ validation/ preparation/ transformation/
│   ├── features/ training/ evaluation/
│   └── utils/                   # config, io, logging
├── tests/                       # pytest suite (validation, preparation, transformation, features, metrics, models, orchestration)
├── Makefile  requirements.txt  run_pipeline.py  pytest.ini
```

## 8. Reproducibility

- **One config** (`configs/config.yaml`); no hard-coded paths. Its SHA-256 is recorded in every run summary.
- **Fixed seeds** for models, sampling and anomaly injection; global RNGs seeded before each run.
- **Deterministic data**: checksummed source archive, train-only aggregates/features, global time-based split (no leakage).
- **Pinned environment** (`requirements.txt`, Python 3.11); exact versions captured per run in `reports/pipeline_run_summary.json`.
- **Lineage** end to end: ingestion manifest → validation report → split metadata → feature metadata → MLflow run IDs → evaluation → run summary.
- `make clean` removes all generated data, models and reports but never the cached source archive.
