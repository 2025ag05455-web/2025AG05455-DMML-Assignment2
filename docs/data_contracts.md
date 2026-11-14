# Data & Module Contracts

Every pipeline stage is a module exposing `run(cfg: dict | None = None) -> dict` (a JSON-serialisable
summary) and is runnable as `python -m src.<package>.<module>`. Stages communicate **only through files**
in the layered data lake below, which makes each stage independently re-runnable and testable.
All paths come from `configs/config.yaml` via `src.utils.config.load_config()` / `path(cfg, key, ...)`.
Shared helpers: `src.utils.io` (`write_parquet`, `read_parquet`, `write_json`, `read_json`), `src.utils.logger.get_logger`
(console + `logs/pipeline.log`; `$RECSYS_LOG_DIR` overrides the directory, which the test suite uses to keep the run log clean).

| # | Stage | Module | Reads | Writes |
|---|-------|--------|-------|--------|
| 1 | Ingestion | `src/ingestion/ingest.py` | MovieLens-100K zip (URL) | `data/raw/{users,items,interactions}.csv`, `ingestion_manifest.json`, `injected_anomalies.json` |
| 2 | Validation | `src/validation/validate.py` | `data/raw/*.csv` | `data/interim/{users,items,interactions}.parquet`, `data/errors/*_errors.csv`, `reports/validation_report.{json,md}` |
| 3 | Preparation | `src/preparation/prepare.py` | `data/interim/*.parquet` | `data/processed/{users,items,interactions}_clean.parquet`, `train.parquet`, `test.parquet`, `split_metadata.json`, `preparation_report.json` |
| 4 | Transformation | `src/transformation/transform.py` | `data/processed/{train,*_clean}.parquet` | `data/processed/user_activity_summary.parquet`, `item_popularity.parquet`, `genre_popularity.parquet`, `user_item_matrix.npz`, `matrix_index.json` |
| 5 | Feature mgmt | `src/features/build_features.py` (+ `registry.py`, `store.py`) | processed data | `data/features/{user,item}_features.parquet`, `user_genre_affinity.parquet`, `feature_metadata.json`, `reports/feature_catalog.md` |
| 6 | Training | `src/training/train.py` (+ `models.py`) | train + features | `models/<model>/model.pkl`, `models/<model>/metadata.json` (incl. MLflow `run_id`) |
| 7 | Evaluation | `src/evaluation/evaluate.py` (+ `metrics.py`, `plots.py`) | models + test | `reports/model_comparison.{csv,md}`, `reports/evaluation_results.json`, `reports/figures/model_comparison_{ranking,rating}.png`, `models/<model>/evaluation/{metrics.json,sample_recommendations.csv}`, metrics logged to MLflow |
| 8 | Orchestration | `dags/recommendation_flow.py` (Prefect), `run_pipeline.py` | — | `reports/pipeline_run_summary.json` |

## Layer schemas

### Raw landing (`data/raw/*.csv`) — every column is a string, exactly as received
* `users.csv`: `user_id, age, gender, occupation, zip_code`
* `items.csv`: `item_id, title, release_date (dd-Mon-YYYY), video_release_date, imdb_url, genre_unknown … genre_western` (19 `genre_*` 0/1 flags; names in `src.ingestion.ingest.GENRE_COLS`)
* `interactions.csv`: `user_id, item_id, rating, timestamp (unix seconds)`
* `injected_anomalies.json` = ground truth of the synthetic data-quality problems (used to measure validation recall).

### Validated (`data/interim/*.parquet`) — typed, only rows that passed hard checks
* users: `user_id:int64, age:Int64 (nullable), gender:string (nullable), occupation:string, zip_code:string` (occupation / zip_code would be nulled + warned if missing; never null in MovieLens - `'none'` is a valid occupation, not a null token)
* items: `item_id:int64, title:string (nullable), release_date:datetime64[ns] (nullable), imdb_url:string (nullable), genre_*:int8` (`video_release_date` dropped – 100% empty)
* interactions: `user_id:int64, item_id:int64, rating:int64 (1..5), timestamp:int64 (unix s)`
* Hard (row-rejecting) errors → `data/errors/<entity>_errors.csv` with the raw columns + `error_type`, `error_detail`, `severity="error"` (plus, for traceability, a leading `source_row` = 0-based data row of the raw CSV and an `error_column`; one record per failed check, so a row can appear more than once).
* Soft (attribute-level) problems — e.g. missing/invalid age, invalid gender, missing title, unparsable release date — keep the row, null the offending value, and are logged to the same error file with `severity="warning"`. Preparation imputes them.
* Exact duplicate rows are quarantined (`error_type="exact_duplicate"`), first occurrence kept.

### Processed (`data/processed/`)
* `users_clean`: `user_id, age:int64 (imputed), gender ('M'/'F'/'unknown'), occupation (lower-case), zip_code, age_group`
* `items_clean`: `item_id, title, year:Int64, release_date, imdb_url, genres (pipe-joined names), n_genres, genre_*`
* `interactions_clean`: `user_id, item_id, rating:int64, timestamp:int64, event_time:datetime64[ns, UTC]` — one row per (user_id,item_id) (latest kept)
* `train.parquet` / `test.parquet`: same columns as `interactions_clean`; **global time-based split**: rows with `timestamp <= cutoff` → train, later → test (`cutoff` = 80th percentile of timestamps). `split_metadata.json`: `cutoff_timestamp, cutoff_time_utc, n_train, n_test, n_test_users, n_warm_test_users, n_cold_test_users, n_cold_test_items`.
* Derived datasets are computed **on train only** (no test leakage):
  * `user_activity_summary`: `user_id, n_ratings, mean_rating, std_rating, first_ts, last_ts, active_days`
  * `item_popularity`: `item_id, n_ratings, mean_rating, bayesian_avg_rating, popularity_rank, popularity_pct, std_rating, first_ts, last_ts` (rated items only; rank 1 = most rated, ties broken by Bayesian average then `item_id`; damping = `features.min_item_ratings_for_avg`)
  * `genre_popularity`: `genre, n_items, n_rated_items, n_ratings, share_of_ratings, mean_rating` (one row per genre)
  * `user_item_matrix.npz` (scipy CSR float64, users × items, explicit ratings, 0 = not rated in train; rows/cols = **all** catalogue users/items sorted by id, so cold users/items are empty rows/cols) + `matrix_index.json` (`{"user_ids": [...], "item_ids": [...]}` row/col order)

### Features (`data/features/`) — point-in-time: computed with `as_of = split cutoff` from train only
* `user_features`: keyed by `user_id` (see `reports/feature_catalog.md`), one row per catalogue user
* `item_features`: keyed by `item_id`, one row per catalogue item
* `user_genre_affinity`: `user_id, genre_* (share of the user's genre-tagged ratings in each genre; rows sum to 1)`
* Every table also carries `feature_timestamp:datetime64[ns, UTC]` (= as_of) and `feature_set_version:string`; schemas are defined in `src/features/registry.py`.
* Entities without history before as_of get documented defaults (population priors); unknown ids get the same defaults from the store with `is_cold=True`.
* `feature_metadata.json`: version, as_of, row counts, feature list, defaults, created_at.
* Access via `src.features.store.FeatureStore(cfg)` – the **same** API is used by training and by prediction/serving.

### Models
Common interface in `src/training/models.py`:
`fit(train_df, feature_store=None)`, `recommend(user_ids, k, exclude_seen=True) -> dict[user_id, list[item_id]]`,
`predict(user_ids, item_ids) -> np.ndarray` (rating estimates, all models), `save(path)`, `load(path)`.
Models: `popularity` (baseline), `item_knn` (item-item CF), `mf_svd` (biased matrix factorisation, SGD), `hybrid` (MF + genre-affinity content score).
* Candidates are the items seen in training; `exclude_seen` drops the user's training items. Users without
  training history get the popularity (train count) ranking from every model; ties break by popularity, then `item_id`.
* `hybrid` = `alpha * minmax_u(MF score) + (1 - alpha) * user_item_genre_affinity`, the affinity read from the
  `FeatureStore` at request time (the store is not pickled; `BaseRecommender.load(path, feature_store=store)` re-attaches it).
* `models/<model>/metadata.json`: `model, model_class, run_id, experiment_id, tracking_uri, params, lineage, trained_at,
  train_metrics{fit_time_sec, train_rmse, train_mae}, epoch_train_rmse, n_users, n_items`.
* MLflow (`training.mlflow`, store/artifacts resolved under the project root): one run per model named after it;
  params = hyperparameters + `data.*` lineage (split cutoff, n_train, train sha256, feature version / as_of, raw archive
  sha256) + `config_sha256`; tags `model_type, model_name, stage, pipeline, pipeline_version`; metrics `fit_time_sec,
  train_rmse, train_mae, train_rmse_epoch` (per step); artifacts `model/`, `config/`. Evaluation re-opens the run and adds
  `test_<metric>_at_<k>`, `test_cold_*`, `test_rmse/mae` and the `evaluation/` artifacts.
* Serving: `python -m src.training.predict --user-id 42 --model hybrid --k 10` / `recommend_for_user(user_id, model_name, k, cfg)`.

### Evaluation
* Relevant = test rating >= `evaluation.relevance_threshold`. Ranking users = test users with training history and >= 1
  relevant test item; metrics at `evaluation.k` and `extra_k`; coverage (/ training catalogue), genre intra-list
  diversity and novelty (`-log2` share of training users) at `k`; RMSE / MAE on all test ratings of warm users.
  Cold test users are reported as a separate popularity-fallback slice.
* `model_comparison.csv` columns: `model, precision@10, recall@10, ndcg@10, hit_rate@10, map@10, coverage, diversity,
  novelty, rmse, mae, precision@5, recall@5, ndcg@5, hit_rate@5, map@5` (the `.md` bolds the best value per column).
* `evaluation_results.json`: `stage, created_at, protocol, data (counts), primary_metric, best_model,
  models{<name>: {run_id, warm{...}, cold_start{...}}}, artifacts`.
