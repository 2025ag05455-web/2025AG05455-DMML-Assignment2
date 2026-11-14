# Reproducible entry points for the MovieLens-100K recommendation pipeline.
# Run `make help` for the list of targets. Override variables on the command line,
# e.g. `make pipeline CONFIG=configs/other.yaml` or `make test PYTHON=python3`.

SHELL          := /bin/bash
.DEFAULT_GOAL  := help

VENV           ?= .venv
PYTHON         ?= $(VENV)/bin/python
CONFIG         ?= configs/config.yaml
RUN            := RECSYS_CONFIG=$(CONFIG) $(PYTHON)

# Generated pipeline outputs removed by `make clean` (data/raw/source is never touched).
GENERATED_DATA    := data/interim data/processed data/features data/errors
GENERATED_REPORTS := validation_report.json validation_report.md feature_catalog.md \
                     model_comparison.csv model_comparison.md evaluation_results.json \
                     pipeline_run_summary.json

.PHONY: help setup lock ingest validate prepare transform features train evaluate \
        pipeline flow flow-serve test mlflow-ui prefect-ui diagram report clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

# ----------------------------------------------------------------------------- environment
setup: ## Create .venv with Python 3.11 and install requirements.txt
	python3.11 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

lock: ## Freeze the exact resolved environment to requirements.lock
	@{ $(PYTHON) -m pip freeze 2>/dev/null || uv pip freeze --python $(PYTHON); } > requirements.lock
	@echo "wrote requirements.lock ($$(wc -l < requirements.lock) packages)"

# ----------------------------------------------------------------------------- stages
ingest: ## Stage 1 - download MovieLens-100K and land raw CSVs
	$(RUN) -m src.ingestion.ingest

validate: ## Stage 2 - validate raw data, quarantine bad rows, validation report
	$(RUN) -m src.validation.validate

prepare: ## Stage 3 - clean, impute, de-duplicate, time-based train/test split
	$(RUN) -m src.preparation.prepare

transform: ## Stage 4 - train-only aggregates and user-item matrix
	$(RUN) -m src.transformation.transform

features: ## Stage 5 - build the offline feature store
	$(RUN) -m src.features.build_features

train: ## Stage 6 - train all recommenders (tracked in MLflow)
	$(RUN) -m src.training.train

evaluate: ## Stage 7 - evaluate models and write the comparison report
	$(RUN) -m src.evaluation.evaluate

# ----------------------------------------------------------------------------- orchestration
pipeline: ## Run the full DAG with the dependency-free runner (run_pipeline.py)
	$(PYTHON) run_pipeline.py --config $(CONFIG)

flow: ## Run the full DAG once as a Prefect flow
	$(PYTHON) dags/recommendation_flow.py --config $(CONFIG)

flow-serve: ## Serve the daily (02:00 UTC) Prefect deployment - long-running
	$(PYTHON) dags/recommendation_flow.py --config $(CONFIG) --serve

test: ## Run the unit tests
	$(PYTHON) -m pytest

# UIs bind to localhost only; use an SSH port-forward to view them from another machine.
mlflow-ui: ## MLflow tracking UI on http://127.0.0.1:5000
	$(VENV)/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

prefect-ui: ## Prefect server + UI on http://127.0.0.1:4200 (then export PREFECT_API_URL=http://127.0.0.1:4200/api)
	$(VENV)/bin/prefect server start --host 127.0.0.1

# ----------------------------------------------------------------------------- documentation
diagram: ## Render reports/figures/architecture.{png,svg}
	$(PYTHON) scripts/make_architecture_diagram.py

report: diagram ## Regenerate the architecture diagram and build reports/final_report.pdf
	$(RUN) scripts/build_report.py

# ----------------------------------------------------------------------------- housekeeping
clean: ## Remove generated data, models, MLflow store, reports and logs (keeps data/raw/source)
	rm -rf $(addsuffix /*,$(GENERATED_DATA)) models/*/ mlflow.db mlartifacts mlruns logs/*.log .pytest_cache
	rm -f data/raw/*.csv data/raw/*.json $(addprefix reports/,$(GENERATED_REPORTS))
	find reports/figures -type f ! -name 'architecture.*' -delete
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -exec rm -rf {} +
