# Data Validation Report - MovieLens-100K

Stage 2 of the recommendation pipeline (`src/validation/validate.py`). This file is generated from `reports/validation_report.json`; both are rewritten on every run.

| Item | Value |
|---|---|
| Run timestamp (UTC) | 2026-11-14T07:15:24+00:00 |
| Status | **PASSED** |
| Quality gate | interaction hard-error rate 0.0709% <= 5.00%: PASS |
| Post-condition gate | strict Pandera contract schema + exact dtypes on every output: PASS |
| Ground-truth detection recall | 100.0% (90/90 injected anomalies) |
| Config | `configs/config.yaml` (section `validation`) |
| Duration | 0.72 s |

## 1. Row counts

| Entity | Input rows | Exact duplicates | Rejected rows (hard errors, incl. duplicates) | Output rows | Rows with warnings | Hard-error rate |
|---|---:|---:|---:|---:|---:|---:|
| users | 946 | 3 | 3 | 943 | 10 | 0.317% |
| items | 1684 | 2 | 2 | 1682 | 7 | 0.119% |
| interactions | 100071 | 25 | 71 | 100000 | 0 | 0.071% |

Inputs (lineage):

| Entity | File | Rows | sha256 (prefix) |
|---|---|---:|---|
| users | `data/raw/users.csv` | 946 | `e6daf7243b10` |
| items | `data/raw/items.csv` | 1684 | `cbe37c6e7ed9` |
| interactions | `data/raw/interactions.csv` | 100071 | `fdc17110e952` |

## 2. Validation policy

| Severity | Problems | Action |
|---|---|---|
| error (hard) | missing / non-numeric / non-integer key, duplicate primary key, missing / non-numeric / non-integer / not-allowed rating, missing / non-numeric / out-of-window timestamp, orphan foreign key (not in validated users / items), genre flag missing or not 0/1 | row rejected -> `data/errors/<entity>_errors.csv` |
| error (exact duplicate) | row identical to an earlier row | row quarantined (first occurrence kept) |
| warning (soft) | missing / non-numeric / out-of-range age, missing or not-allowed gender, missing occupation or zip_code, missing title, missing or unparsable release_date, missing imdb_url | row kept, value set to null; logged in the error file; imputed in preparation |
| warning (pair) | duplicate (user_id, item_id) interaction | row kept, resolved in preparation |

* Null tokens (case-insensitive, stripped): '', 'n/a', 'na', 'nan', 'none', 'null'; exemptions (legitimate categories): `occupation`: ['none'].
* Quality gate: stage fails if interaction hard-error rate > 0.05. Post-condition gate: strict Pandera contract schema + exact dtype comparison on every output.
* Error files: source_row (0-based data row; CSV line = source_row + 2), raw columns, error_type, error_column, error_detail, severity. A row may have several records (one per failed check).
* `items.video_release_date` is dropped (100% empty, see findings).

Check order (per entity; users and items are validated before interactions so foreign keys are checked against the *validated* parents):

1. structure: all contract columns present (abort otherwise)
2. normalise: strip whitespace, null tokens -> null
3. exact duplicates quarantined (first occurrence kept); later checks run on unique rows
4. missing values per column
5. type coercion: non-numeric / non-integer numbers, unparsable dates (custom, vectorised)
6. Pandera diagnostic schema (lazy): ranges, allowed values, key uniqueness, foreign keys
7. policy: reject rows with hard errors, null soft-failed attributes
8. duplicate (user_id, item_id) pairs among kept rows (warning)
9. Pandera contract schema + exact dtype comparison on the clean output (post-condition gate)

## 3. Check results

Every automated check, including the ones that passed. `Checked` is the number of rows (values for collapsed `genre_*` rows, columns for `required_columns`) the check ran on.

| # | Entity | Check | Column | Severity | Method | Checked | Failed | % failed | Result | Action taken |
|---:|---|---|---|---|---|---:|---:|---:|---|---|
| 1 | users | `required_columns` | (all columns) | error | custom | 5 | 0 | 0.000% | pass | - |
| 2 | users | `exact_duplicate` | (all columns) | error | custom | 946 | 3 | 0.317% | FAIL | row quarantined (first occurrence kept) |
| 3 | users | `missing_value` | user_id | error | custom | 943 | 0 | 0.000% | pass | - |
| 4 | users | `missing_value` | age | warning | custom | 943 | 5 | 0.530% | WARN | row kept, value set to null |
| 5 | users | `missing_value` | gender | warning | custom | 943 | 1 | 0.106% | WARN | row kept, value set to null |
| 6 | users | `missing_value` | occupation | warning | custom | 943 | 0 | 0.000% | pass | - |
| 7 | users | `missing_value` | zip_code | warning | custom | 943 | 0 | 0.000% | pass | - |
| 8 | users | `invalid_type` | user_id | error | custom | 943 | 0 | 0.000% | pass | - |
| 9 | users | `non_integer_value` | user_id | error | custom | 943 | 0 | 0.000% | pass | - |
| 10 | users | `invalid_type` | age | warning | custom | 943 | 1 | 0.106% | WARN | row kept, value set to null |
| 11 | users | `non_integer_value` | age | warning | custom | 943 | 0 | 0.000% | pass | - |
| 12 | users | `duplicate_key` | user_id | error | pandera | 943 | 0 | 0.000% | pass | - |
| 13 | users | `age_out_of_range` | age | warning | pandera | 943 | 2 | 0.212% | WARN | row kept, value set to null |
| 14 | users | `gender_not_allowed` | gender | warning | pandera | 943 | 1 | 0.106% | WARN | row kept, value set to null |
| 15 | items | `required_columns` | (all columns) | error | custom | 24 | 0 | 0.000% | pass | - |
| 16 | items | `exact_duplicate` | (all columns) | error | custom | 1684 | 2 | 0.119% | FAIL | row quarantined (first occurrence kept) |
| 17 | items | `missing_value` | item_id | error | custom | 1682 | 0 | 0.000% | pass | - |
| 18 | items | `missing_value` | title | warning | custom | 1682 | 2 | 0.119% | WARN | row kept, value set to null |
| 19 | items | `missing_value` | release_date | warning | custom | 1682 | 1 | 0.059% | WARN | row kept, value set to null |
| 20 | items | `missing_value` | imdb_url | warning | custom | 1682 | 3 | 0.178% | WARN | row kept, value set to null |
| 21 | items | `missing_value` | genre_* (19 cols) | error | custom | 31958 | 0 | 0.000% | pass | - |
| 22 | items | `invalid_type` | item_id | error | custom | 1682 | 0 | 0.000% | pass | - |
| 23 | items | `non_integer_value` | item_id | error | custom | 1682 | 0 | 0.000% | pass | - |
| 24 | items | `invalid_type` | release_date | warning | custom | 1682 | 2 | 0.119% | WARN | row kept, value set to null |
| 25 | items | `invalid_type` | genre_* (19 cols) | error | custom | 31958 | 0 | 0.000% | pass | - |
| 26 | items | `non_integer_value` | genre_* (19 cols) | error | custom | 31958 | 0 | 0.000% | pass | - |
| 27 | items | `duplicate_key` | item_id | error | pandera | 1682 | 0 | 0.000% | pass | - |
| 28 | items | `genre_flag_not_binary` | genre_* (19 cols) | error | pandera | 31958 | 0 | 0.000% | pass | - |
| 29 | interactions | `required_columns` | (all columns) | error | custom | 4 | 0 | 0.000% | pass | - |
| 30 | interactions | `exact_duplicate` | (all columns) | error | custom | 100071 | 25 | 0.025% | FAIL | row quarantined (first occurrence kept) |
| 31 | interactions | `missing_value` | user_id | error | custom | 100046 | 5 | 0.005% | FAIL | row rejected |
| 32 | interactions | `missing_value` | item_id | error | custom | 100046 | 5 | 0.005% | FAIL | row rejected |
| 33 | interactions | `missing_value` | rating | error | custom | 100046 | 1 | 0.001% | FAIL | row rejected |
| 34 | interactions | `missing_value` | timestamp | error | custom | 100046 | 0 | 0.000% | pass | - |
| 35 | interactions | `invalid_type` | user_id | error | custom | 100046 | 0 | 0.000% | pass | - |
| 36 | interactions | `non_integer_value` | user_id | error | custom | 100046 | 0 | 0.000% | pass | - |
| 37 | interactions | `invalid_type` | item_id | error | custom | 100046 | 0 | 0.000% | pass | - |
| 38 | interactions | `non_integer_value` | item_id | error | custom | 100046 | 0 | 0.000% | pass | - |
| 39 | interactions | `invalid_type` | rating | error | custom | 100046 | 3 | 0.003% | FAIL | row rejected |
| 40 | interactions | `non_integer_value` | rating | error | custom | 100046 | 2 | 0.002% | FAIL | row rejected |
| 41 | interactions | `invalid_type` | timestamp | error | custom | 100046 | 4 | 0.004% | FAIL | row rejected |
| 42 | interactions | `non_integer_value` | timestamp | error | custom | 100046 | 0 | 0.000% | pass | - |
| 43 | interactions | `referential_integrity` | user_id | error | pandera | 100046 | 5 | 0.005% | FAIL | row rejected |
| 44 | interactions | `referential_integrity` | item_id | error | pandera | 100046 | 5 | 0.005% | FAIL | row rejected |
| 45 | interactions | `rating_not_allowed` | rating | error | pandera | 100046 | 10 | 0.010% | FAIL | row rejected |
| 46 | interactions | `timestamp_out_of_range` | timestamp | error | pandera | 100046 | 6 | 0.006% | FAIL | row rejected |
| 47 | interactions | `duplicate_interaction` | user_id+item_id | warning | custom | 100000 | 0 | 0.000% | pass | - |

## 4. Ground-truth comparison

The ingestion stage injects a seeded, documented set of anomalies (`data/raw/injected_anomalies.json`). Detected = distinct source rows flagged by the checks mapped to each category.

| Entity | Injected anomaly | Severity | Expected | Detected | Recall | Detected by |
|---|---|---|---:|---:|---:|---|
| interactions | exact_duplicate | error | 25 | 25 | 100.0% | `exact_duplicate[*]` (25) |
| interactions | rating_out_of_range | error | 10 | 10 | 100.0% | `rating_not_allowed[rating]` (10) |
| interactions | rating_wrong_type | error | 6 | 6 | 100.0% | `invalid_type[rating]` (3), `missing_value[rating]` (1), `non_integer_value[rating]` (2) |
| interactions | missing_key | error | 10 | 10 | 100.0% | `missing_value[item_id]` (5), `missing_value[user_id]` (5) |
| interactions | invalid_timestamp | error | 10 | 10 | 100.0% | `invalid_type[timestamp]` (4), `timestamp_out_of_range[timestamp]` (6) |
| interactions | orphan_user | error | 5 | 5 | 100.0% | `referential_integrity[user_id]` (5) |
| interactions | orphan_item | error | 5 | 5 | 100.0% | `referential_integrity[item_id]` (5) |
| users | missing_age | warning | 5 | 5 | 100.0% | `missing_value[age]` (5) |
| users | invalid_age | warning | 3 | 3 | 100.0% | `age_out_of_range[age]` (2), `invalid_type[age]` (1) |
| users | invalid_gender | warning | 2 | 2 | 100.0% | `gender_not_allowed[gender]` (1), `missing_value[gender]` (1) |
| users | exact_duplicate | error | 3 | 3 | 100.0% | `exact_duplicate[*]` (3) |
| items | missing_title | warning | 2 | 2 | 100.0% | `missing_value[title]` (2) |
| items | invalid_release_date | warning | 2 | 2 | 100.0% | `invalid_type[release_date]` (2) |
| items | exact_duplicate | error | 2 | 2 | 100.0% | `exact_duplicate[*]` (2) |

**Overall detection recall: 100.0%** (90 of 90 injected anomalies detected).

Reconciliation of rejected / warned rows with the injected anomalies:

| Entity | Rejected rows | Injected hard | Rows with warnings | Injected soft | Explanation |
|---|---:|---:|---:|---:|---|
| users | 3 | 3 | 10 | 10 | rejected rows match the injected anomalies exactly (3); warning rows match the injected anomalies exactly (10) |
| items | 2 | 2 | 7 | 4 | rejected rows match the injected anomalies exactly (2); warning: 7 rows = 4 injected + 3 pre-existing source issue(s) |
| interactions | 71 | 71 | 0 | 0 | rejected rows match the injected anomalies exactly (71); warning rows match the injected anomalies exactly (0) |

Detections not explained by injection (pre-existing issues in the source data):

| Entity | Check | Column | Severity | Rows | Examples |
|---|---|---|---|---:|---|
| items | `missing_value` | imdb_url | warning | 3 | item_id=267, item_id=1358, item_id=1359 |
| items | `missing_value` | release_date | warning | 1 | item_id=267 |

## 5. Data profile

Null share per column (after null-token normalisation; validated = after rejection and soft nulling). All columns not listed are 0% null.

| Entity | Column | Null % raw | Null % validated |
|---|---|---:|---:|
| users | age | 0.634% | 0.848% |
| users | gender | 0.106% | 0.212% |
| items | title | 0.119% | 0.119% |
| items | release_date | 0.059% | 0.178% |
| items | video_release_date | 100.000% | dropped |
| items | imdb_url | 0.178% | 0.178% |
| interactions | user_id | 0.005% | 0.000% |
| interactions | item_id | 0.005% | 0.000% |
| interactions | rating | 0.001% | 0.000% |

Validated data at a glance:

| Metric | Value |
|---|---|
| Interactions / users / items | 100,000 / 943 / 1,682 |
| Density | 6.305% |
| Rating mean (std) | 3.530 (1.126) |
| Timestamp range (UTC) | 1997-09-20T03:05:10+00:00 .. 1998-04-22T23:10:38+00:00 (214.8 days) |
| Ratings per user min / median / max | 20 / 65 / 737 |
| Ratings per item min / median / max | 1 / 27 / 583 |
| Users: age min / median / max | 7 / 31 / 73 |
| Users: gender | M=669, F=272, null=2 |
| Users: occupations | 21 |
| Items: release dates | 1922-01-01 .. 1998-10-23 |
| Items: multi-genre share | 50.48% |

Rating distribution (validated interactions):

| Rating | Count | % |
|---:|---:|---:|
| 1 | 6,110 | 6.11% |
| 2 | 11,370 | 11.37% |
| 3 | 27,145 | 27.14% |
| 4 | 34,174 | 34.17% |
| 5 | 21,201 | 21.20% |

## 6. Profiling findings

* `users.occupation`: 9 value(s) 'none' kept as a legitimate category value (it is in the MovieLens reference list), not treated as a null token.
* `items.video_release_date` is 100% empty in the raw file - it carries no information and is dropped from the validated layer.
* `items.title`: 3 value(s) had leading/trailing whitespace (stripped).
* `items.imdb_url`: 3 row(s) flagged by `missing_value` (warning) were not injected - pre-existing in the MovieLens source (item_id=267, item_id=1358, item_id=1359).
* `items.release_date`: 1 row(s) flagged by `missing_value` (warning) were not injected - pre-existing in the MovieLens source (item_id=267).
* 18 titles are shared by more than one item_id (known MovieLens quirk); kept as distinct items because item_id is the key.
* Ratings are skewed positive: mean 3.53, 55.4% of ratings are >= 4.
* User-item matrix density is 6.30% (sparsity 93.70%); every user has >= 20 ratings, items have 1..583 ratings (long tail).
* Interactions span 1997-09-20 .. 1998-04-22 (214.8 days), inside the configured validity window.
* 0 duplicate (user_id, item_id) pair(s) remain after the hard checks (warning only; preparation keeps one row per pair).

## 7. Outputs

| Artifact | Rows | Details |
|---|---:|---|
| `data/interim/users.parquet` | 943 | user_id:int64, age:Int64, gender:string, occupation:string, zip_code:string |
| `data/interim/items.parquet` | 1682 | item_id:int64, title:string, release_date:datetime64[ns], imdb_url:string, genre_*:int8 |
| `data/interim/interactions.parquet` | 100000 | user_id:int64, item_id:int64, rating:int64, timestamp:int64 |
| `data/errors/users_errors.csv` | 13 | 3 error / 10 warning records |
| `data/errors/items_errors.csv` | 10 | 2 error / 8 warning records |
| `data/errors/interactions_errors.csv` | 71 | 71 error / 0 warning records |
