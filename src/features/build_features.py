"""Stage 5 - Feature engineering: point-in-time user and item features for the offline feature store.

All behavioural features are computed **as of the split cutoff** (``as_of``) from train
interactions with ``timestamp <= as_of`` only; static profile/catalogue attributes come from
``users_clean`` / ``items_clean``. The per-user and per-item rating aggregates and the global
genre distribution are **read from the transformation stage** (``user_activity_summary``,
``item_popularity``, ``genre_popularity`` in ``processed_dir``; train-only, hence as of the
cutoff) rather than recomputed; :func:`check_aggregates` verifies that they describe exactly the
point-in-time history (no event after ``as_of``, same entities and rating counts) and the stage
fails if they are missing or stale. Every catalogue user and item gets a row: entities without
history before ``as_of`` (cold start) receive documented default values - population-level
priors such as the global mean rating or the global genre distribution - which are stored in
``feature_metadata.json`` and reused by :class:`src.features.store.FeatureStore` for ids that
are unknown altogether.

Inputs:
    processed_dir/train.parquet, users_clean.parquet, items_clean.parquet, split_metadata.json (preparation)
    processed_dir/user_activity_summary.parquet, item_popularity.parquet, genre_popularity.parquet (transformation)

Outputs:
    features_dir/user_features.parquet, item_features.parquet, user_genre_affinity.parquet,
    features_dir/feature_metadata.json, reports_dir/feature_catalog.md
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.registry import (FEATURE_REGISTRY, ON_DEMAND, TABLE_KEYS, display_name, expected_schema,
                                   feature_columns, markdown_table, render_catalog, validate_feature_frame)
from src.features.store import FeatureStore
from src.ingestion.ingest import GENRE_COLS
from src.preparation.prepare import age_group
from src.transformation.transform import SECONDS_PER_DAY
from src.utils.config import load_config, path, rel_path
from src.utils.io import read_json, read_parquet, write_json, write_parquet
from src.utils.logger import get_logger

log = get_logger("features")

GENRE_NAMES = np.array([col.removeprefix("genre_") for col in GENRE_COLS])
USER_STATIC = ["user_id", "age", "gender", "occupation", "age_group"]
INTERACTION_SAMPLE_SIZE = 20_000
AGGREGATE_TABLES = ("user_activity_summary", "item_popularity", "genre_popularity")


@dataclass(frozen=True)
class TrainAggregates:
    """Train-only aggregates materialised by the transformation stage (``processed_dir``).

    Attributes:
        user_activity: ``user_activity_summary`` - one row per user with train history.
        item_popularity: ``item_popularity`` - one row per rated item (Bayesian average, rank, pct).
        genre_popularity: ``genre_popularity`` - one row per genre (rating counts and shares).
    """

    user_activity: pd.DataFrame
    item_popularity: pd.DataFrame
    genre_popularity: pd.DataFrame


# --------------------------------------------------------------------------- generic helpers
def point_in_time(interactions: pd.DataFrame, as_of_ts: int) -> pd.DataFrame:
    """Keep only events observable at ``as_of_ts`` (``timestamp <= as_of_ts``)."""
    return interactions[interactions["timestamp"] <= as_of_ts]


def _days(seconds: pd.Series | float) -> pd.Series | float:
    """Convert a duration in seconds to fractional days."""
    return seconds / SECONDS_PER_DAY


def as_of_utc(as_of_ts: int) -> pd.Timestamp:
    """The feature timestamp: ``as_of_ts`` as a nanosecond-precision UTC timestamp."""
    return pd.Timestamp(int(as_of_ts), unit="s", tz="UTC").as_unit("ns")


def history_stats(history: pd.DataFrame, as_of_ts: int) -> dict[str, float]:
    """Population statistics of the history used as priors / defaults for cold entities."""
    return {
        "mean_rating": float(history["rating"].mean()),
        "std_rating": float(history["rating"].std(ddof=0)),
        "observation_window_days": float(_days(as_of_ts - history["timestamp"].min())),
        "n_interactions": int(len(history)),
    }


def aggregate_paths(cfg: dict) -> dict[str, Path]:
    """Locations of the transformation outputs consumed by this stage."""
    return {name: path(cfg, "processed_dir", f"{name}.parquet") for name in AGGREGATE_TABLES}


def load_train_aggregates(cfg: dict) -> TrainAggregates:
    """Read the transformation stage's train-only aggregates.

    Raises:
        FileNotFoundError: if any of them is missing (the transformation stage has not run).
    """
    sources = aggregate_paths(cfg)
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Transformation outputs not found: {missing}. Run the transformation stage "
                                "first (python -m src.transformation.transform) before building features.")
    return TrainAggregates(*(read_parquet(p) for p in sources.values()))


def check_aggregates(aggregates: TrainAggregates, history: pd.DataFrame, as_of_ts: int) -> None:
    """Verify the aggregates describe exactly ``history`` (point-in-time and in sync with train).

    Checks, for the user and item aggregates: no event after ``as_of``, the same entity ids as
    the history and the same total number of ratings; and that every genre is present.

    Raises:
        ValueError: listing every inconsistency (stale or leaking transformation outputs).
    """
    problems = []
    for name, frame, key in (("user_activity_summary", aggregates.user_activity, "user_id"),
                             ("item_popularity", aggregates.item_popularity, "item_id")):
        if int(frame["last_ts"].max()) > as_of_ts:
            problems.append(f"{name} contains events after as_of={as_of_ts}")
        if int(frame["n_ratings"].sum()) != len(history):
            problems.append(f"{name} counts {int(frame['n_ratings'].sum())} ratings, the history has {len(history)}")
        if not np.array_equal(np.sort(frame[key].to_numpy()), np.unique(history[key].to_numpy())):
            problems.append(f"{name} {key}s differ from the history's")
    missing_genres = set(GENRE_NAMES) - set(aggregates.genre_popularity["genre"])
    if missing_genres:
        problems.append(f"genre_popularity lacks genres {sorted(missing_genres)}")
    if problems:
        raise ValueError("Transformation outputs are inconsistent with train.parquet as of the cutoff: "
                         + "; ".join(problems) + ". Re-run the transformation stage.")


def finalise_table(frame: pd.DataFrame, table: str, as_of_ts: int, version: str) -> pd.DataFrame:
    """Add metadata columns, order/cast columns to the registry schema and sort by key."""
    schema = expected_schema(table)
    out = frame.assign(feature_timestamp=as_of_utc(as_of_ts), feature_set_version=version)
    return out[list(schema)].astype(schema).sort_values(TABLE_KEYS[table]).reset_index(drop=True)


# --------------------------------------------------------------------------- genre affinity
def normalised_entropy(shares: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy divided by ``ln(n_columns)`` (0 = single genre, 1 = uniform)."""
    log_shares = np.log(shares, out=np.zeros_like(shares), where=shares > 0)
    entropy = (shares * log_shares).sum(axis=1) / -np.log(shares.shape[1])
    return entropy + 0.0  # turn -0.0 (single-genre users) into 0.0


def genre_prior(genre_popularity: pd.DataFrame) -> pd.Series:
    """Global genre distribution of the history (``GENRE_COLS`` index, sums to 1; uniform if empty).

    Normalises the per-genre train rating counts of the transformation's ``genre_popularity``.
    """
    counts = (genre_popularity.set_index("genre")["n_ratings"].reindex(GENRE_NAMES)
              .set_axis(GENRE_COLS).astype("float64"))
    total = counts.sum()
    return counts / total if total > 0 else pd.Series(1.0 / len(GENRE_COLS), index=GENRE_COLS)


def user_genre_shares(history: pd.DataFrame, items: pd.DataFrame, user_ids: pd.Series,
                      prior: pd.Series) -> pd.DataFrame:
    """Share of each user's genre-tagged ratings per genre.

    Returns a frame indexed by ``user_ids`` with ``GENRE_COLS`` columns (rows sum to 1); users
    without genre-tagged history get ``prior`` (the global genre distribution).
    """
    flags = items.set_index("item_id")[GENRE_COLS].reindex(history["item_id"]).fillna(0).astype("float64")
    counts = flags.set_axis(history["user_id"].to_numpy()).groupby(level=0).sum()
    values = counts.reindex(user_ids.to_numpy(), fill_value=0.0).to_numpy()
    row_sum = values.sum(axis=1, keepdims=True)
    # Rows without any genre-tagged rating keep the prior that pre-fills the output buffer.
    shares = np.divide(values, row_sum, out=np.tile(prior.to_numpy(), (len(values), 1)), where=row_sum > 0)
    index = pd.Index(user_ids.to_numpy(), name="user_id")
    return pd.DataFrame(shares, index=index, columns=GENRE_COLS)


def genre_profile(shares: np.ndarray, top_n: int) -> pd.DataFrame:
    """Preferred genre, pipe-joined top-N genres (share > 0) and taste entropy for each share row."""
    order = np.argsort(-shares, axis=1, kind="stable")[:, :top_n]
    top_shares = np.take_along_axis(shares, order, axis=1)
    top_names = GENRE_NAMES[order]
    return pd.DataFrame({
        "user_preferred_genre": top_names[:, 0],
        "user_top_genres": ["|".join(names[vals > 0]) for names, vals in zip(top_names, top_shares)],
        "user_genre_entropy": normalised_entropy(shares),
    })


# --------------------------------------------------------------------------- user features
def user_behaviour(act: pd.DataFrame, as_of_ts: int) -> pd.DataFrame:
    """Activity, rating-bias and recency features from the ``user_activity_summary`` aggregate."""
    return pd.DataFrame({
        "user_id": act["user_id"],
        "user_n_ratings": act["n_ratings"],
        "user_avg_rating": act["mean_rating"],
        "user_rating_std": act["std_rating"],
        "user_days_since_last_interaction": _days(as_of_ts - act["last_ts"]),
        "user_tenure_days": _days(as_of_ts - act["first_ts"]),
        "user_ratings_per_active_day": act["n_ratings"] / act["active_days"],
    })


def user_defaults(users: pd.DataFrame, stats: dict[str, float], prior: pd.Series, top_n: int,
                  unknown_token: str) -> dict:
    """Default user feature values for cold (no history) and unseen user ids."""
    median_age = int(round(float(users["age"].median())))
    prior_profile = genre_profile(prior.to_numpy()[None, :], top_n).iloc[0]
    return {
        "user_n_ratings": 0,
        "user_avg_rating": stats["mean_rating"],
        "user_rating_std": stats["std_rating"],
        "user_days_since_last_interaction": stats["observation_window_days"],
        "user_tenure_days": 0.0,
        "user_ratings_per_active_day": 0.0,
        "user_preferred_genre": str(prior_profile["user_preferred_genre"]),
        "user_top_genres": str(prior_profile["user_top_genres"]),
        "user_genre_entropy": float(prior_profile["user_genre_entropy"]),
        "age": median_age,
        "gender": unknown_token,
        "occupation": unknown_token,
        "age_group": str(age_group(pd.Series([median_age])).iloc[0]),
    }


def build_user_features(activity: pd.DataFrame, users: pd.DataFrame, shares: pd.DataFrame,
                        as_of_ts: int, top_n: int, defaults: dict) -> pd.DataFrame:
    """Join static profile, behavioural and genre-profile features for every catalogue user."""
    profile = genre_profile(shares[GENRE_COLS].to_numpy(), top_n).assign(user_id=shares.index.to_numpy())
    frame = (users[USER_STATIC]
             .merge(user_behaviour(activity, as_of_ts), on="user_id", how="left", validate="one_to_one")
             .merge(profile, on="user_id", how="left", validate="one_to_one"))
    return frame.fillna(defaults)


# --------------------------------------------------------------------------- item features
def item_behaviour(pop: pd.DataFrame, as_of_ts: int) -> pd.DataFrame:
    """Popularity, rating and recency features from the ``item_popularity`` aggregate."""
    return pd.DataFrame({
        "item_id": pop["item_id"],
        "item_n_ratings": pop["n_ratings"],
        "item_avg_rating": pop["mean_rating"],
        "item_bayesian_avg_rating": pop["bayesian_avg_rating"],
        "item_rating_std": pop["std_rating"],
        "item_popularity_rank": pop["popularity_rank"],
        "item_popularity_pct": pop["popularity_pct"],
        "item_days_since_last_interaction": _days(as_of_ts - pop["last_ts"]),
    })


def item_defaults(items: pd.DataFrame, stats: dict[str, float], n_rated_items: int, as_of_ts: int) -> dict:
    """Default item feature values for cold (never rated) and unseen item ids."""
    median_year = int(round(float(items["year"].median())))
    return {
        "item_n_ratings": 0,
        "item_avg_rating": stats["mean_rating"],
        "item_bayesian_avg_rating": stats["mean_rating"],
        "item_rating_std": stats["std_rating"],
        "item_popularity_rank": int(n_rated_items + 1),
        "item_popularity_pct": 0.0,
        "item_days_since_last_interaction": stats["observation_window_days"],
        "item_year": median_year,
        "item_age_years": float(max(as_of_utc(as_of_ts).year - median_year, 0)),
        "item_n_genres": 0,
        **{col: 0 for col in GENRE_COLS},
    }


def build_item_features(popularity: pd.DataFrame, items: pd.DataFrame, as_of_ts: int,
                        defaults: dict) -> pd.DataFrame:
    """Join catalogue attributes and popularity features for every catalogue item."""
    frame = (items[["item_id", "year", "n_genres", *GENRE_COLS]]
             .rename(columns={"year": "item_year", "n_genres": "item_n_genres"})
             .merge(item_behaviour(popularity, as_of_ts), on="item_id", how="left", validate="one_to_one"))
    frame["item_year"] = frame["item_year"].fillna(defaults["item_year"]).astype("int64")
    frame["item_age_years"] = (as_of_utc(as_of_ts).year - frame["item_year"]).clip(lower=0).astype("float64")
    return frame.fillna(defaults)


# --------------------------------------------------------------------------- orchestration
def build_feature_tables(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame,
                         aggregates: TrainAggregates, as_of_ts: int, version: str, top_n: int,
                         unknown_token: str) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    """Build all materialised feature tables as of ``as_of_ts``.

    ``aggregates`` are the transformation stage's train-only aggregates; they must describe
    exactly the point-in-time history of ``train`` (checked by :func:`check_aggregates`).

    Returns ``(tables, defaults, stats)``: the registry-conformant tables keyed by table name,
    the default values per table and the population statistics of the history.
    """
    history = point_in_time(train, as_of_ts)
    if history.empty:
        raise ValueError(f"No interactions at or before as_of={as_of_ts}")
    check_aggregates(aggregates, history, as_of_ts)
    stats = history_stats(history, as_of_ts)
    prior = genre_prior(aggregates.genre_popularity)
    shares = user_genre_shares(history, items, users["user_id"], prior)
    defaults = {
        "user_features": user_defaults(users, stats, prior, top_n, unknown_token),
        "item_features": item_defaults(items, stats, len(aggregates.item_popularity), as_of_ts),
        "user_genre_affinity": prior.to_dict(),
    }
    raw = {
        "user_features": build_user_features(aggregates.user_activity, users, shares, as_of_ts, top_n,
                                             defaults["user_features"]),
        "item_features": build_item_features(aggregates.item_popularity, items, as_of_ts,
                                             defaults["item_features"]),
        "user_genre_affinity": shares.reset_index(),
    }
    tables = {name: finalise_table(df, name, as_of_ts, version) for name, df in raw.items()}
    return tables, defaults, stats


def feature_metadata(tables: dict[str, pd.DataFrame], defaults: dict, stats: dict, as_of_ts: int,
                     version: str, sources: dict[str, str]) -> dict:
    """Assemble ``feature_metadata.json``: version, as_of, row counts, feature names, defaults."""
    return {
        "version": version,
        "as_of": as_of_utc(as_of_ts).isoformat(),
        "as_of_timestamp": int(as_of_ts),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "history": stats,
        "tables": {name: {"file": f"{name}.parquet", "key": TABLE_KEYS[name], "rows": int(len(df)),
                          "columns": list(df.columns)} for name, df in tables.items()},
        "n_cold_users": int(tables["user_features"]["user_n_ratings"].eq(0).sum()),
        "n_cold_items": int(tables["item_features"]["item_n_ratings"].eq(0).sum()),
        "feature_names": {name: feature_columns(name) for name in (*tables, ON_DEMAND)},
        "defaults": defaults,
    }


# --------------------------------------------------------------------------- catalogue
def _fmt(value: object) -> str:
    """Compact number formatting for the statistics table."""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    return str(value)


def column_stats(values: pd.DataFrame) -> dict[str, str]:
    """Null %, min / mean / max for numeric columns, most frequent value for text columns.

    For a column family (several columns) the numeric statistics span all cells and the top
    value is the column with the highest mean.
    """
    null_pct = f"{100.0 * values.isna().to_numpy().mean():.2f}"
    if all(pd.api.types.is_numeric_dtype(t) for t in values.dtypes):
        cells = values.to_numpy(dtype=np.float64)
        top = ""
        if values.shape[1] > 1:
            means = values.mean()
            top = f"{means.idxmax()} (mean {means.max():.3f})"
        return {"null_pct": null_pct, "min": _fmt(np.nanmin(cells)), "mean": _fmt(np.nanmean(cells)),
                "max": _fmt(np.nanmax(cells)), "top": top}
    counts = values.iloc[:, 0].value_counts(normalize=True)
    return {"null_pct": null_pct, "min": "-", "mean": "-", "max": "-",
            "top": f"{counts.index[0]} ({100 * counts.iloc[0]:.1f}%)"}


def statistics_table(tables: dict[str, pd.DataFrame], on_demand: pd.DataFrame) -> str:
    """Markdown table of statistics of every registry feature computed from the built data."""
    rows = []
    for fd in FEATURE_REGISTRY:
        frame = on_demand if fd.table == ON_DEMAND else tables[fd.table]
        s = column_stats(frame[list(fd.stored_columns)])
        rows.append([f"`{display_name(fd)}`", fd.entity, s["null_pct"], s["min"], s["mean"], s["max"], s["top"]])
    return markdown_table(["Feature", "Entity", "Null %", "Min", "Mean", "Max", "Top value"], rows)


def defaults_table(defaults: dict) -> str:
    """Markdown table of the default values served for cold / unseen entities."""
    rows = [[f"`{col}`", table, _fmt(value)] for table, values in defaults.items()
            for col, value in values.items() if not col.startswith("genre_")]
    rows.append(["`genre_*` (item_features)", "item_features", "0 for every genre"])
    rows.append(["`genre_*` (user_genre_affinity)", "user_genre_affinity",
                 "global genre distribution of the history (see feature_metadata.json)"])
    return markdown_table(["Feature", "Table", "Default"], rows)


def render_feature_catalog(metadata: dict, tables: dict[str, pd.DataFrame], on_demand: pd.DataFrame) -> str:
    """Full markdown feature catalogue: definitions, statistics, defaults and PIT guarantees."""
    table_lines = "\n".join(f"- `{name}.parquet`: {info['rows']} rows x {len(info['columns'])} columns "
                            f"(key `{info['key']}`)" for name, info in metadata["tables"].items())
    return "\n\n".join([
        "# Feature Catalog",
        "_Generated by `src/features/build_features.py` from the registry in `src/features/registry.py`; "
        "do not edit by hand._",
        f"- **Feature set version:** `{metadata['version']}`\n"
        f"- **as_of (point-in-time cutoff):** {metadata['as_of']} (= train/test split cutoff)\n"
        f"- **History used:** {metadata['history']['n_interactions']} train interactions with "
        f"`timestamp <= as_of`; user/item/genre aggregates from the transformation stage "
        f"(`user_activity_summary`, `item_popularity`, `genre_popularity`); static attributes from "
        f"`users_clean` / `items_clean`\n"
        f"- **Cold entities (no history before as_of):** {metadata['n_cold_users']} users, "
        f"{metadata['n_cold_items']} items\n"
        f"- **Access:** `src.features.store.FeatureStore(cfg)` - the same API for training and serving\n"
        f"{table_lines}",
        "## 1. Feature definitions",
        render_catalog(),
        "## 2. Feature statistics (built feature set)",
        f"User/item statistics cover every row of the materialised tables. On-demand user x item features "
        f"are computed by the store for {len(on_demand)} random (user, item) pairs (fixed seed).",
        statistics_table(tables, on_demand),
        "## 3. Defaults for cold / unseen entities",
        "Applied to catalogue entities without history before as_of and by the store to unknown ids "
        "(`is_cold = True`). Behavioural defaults are population priors of the history.",
        defaults_table(metadata["defaults"]),
        "## 4. Point-in-time guarantees",
        "- Behavioural features use only interactions with `timestamp <= as_of`; adding or changing any "
        "later (test-period) interaction cannot change a feature value (covered by `tests/test_features.py`).\n"
        "- Recency/tenure features are measured relative to `as_of`, never to the wall clock.\n"
        "- Every table carries `feature_timestamp` (= as_of) and `feature_set_version`; the store refuses "
        "to serve a feature set whose version differs from `features.feature_set_version` in the config.\n"
        "- Tables are validated against the registry (columns, dtypes, unique keys, no nulls) before "
        "being written and again when the store loads them.",
    ]) + "\n"


def sample_pairs(user_ids: np.ndarray, item_ids: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` random (user, item) pairs with a fixed seed (for on-demand feature statistics)."""
    rng = np.random.default_rng(seed)
    return rng.choice(user_ids, size=n), rng.choice(item_ids, size=n)


# --------------------------------------------------------------------------- entrypoint
def run(cfg: dict | None = None) -> dict:
    """Build, validate and publish the feature set; return a JSON-serialisable summary."""
    cfg = cfg or load_config()
    fcfg = cfg["features"]
    version = str(fcfg["feature_set_version"])
    inputs = {name: path(cfg, "processed_dir", f"{name}.parquet") for name in ("train", "users_clean", "items_clean")}
    as_of_ts = int(read_json(path(cfg, "processed_dir", "split_metadata.json"))["cutoff_timestamp"])
    aggregates = load_train_aggregates(cfg)
    train, users, items = (read_parquet(p) for p in inputs.values())
    sources = {**inputs, **aggregate_paths(cfg)}

    tables, defaults, stats = build_feature_tables(
        train, users, items, aggregates, as_of_ts, version, int(fcfg["top_n_genres"]),
        cfg["preparation"]["unknown_token"])
    for name, df in tables.items():
        validate_feature_frame(df, name)
        write_parquet(df, path(cfg, "features_dir", f"{name}.parquet"))
    metadata = feature_metadata(tables, defaults, stats, as_of_ts, version,
                                {name: rel_path(p, cfg["project"]["root"]) for name, p in sources.items()})
    write_json(metadata, path(cfg, "features_dir", "feature_metadata.json"))

    store = FeatureStore(cfg)  # read-back through the serving API (also validates the written tables)
    users_s, items_s = sample_pairs(tables["user_features"]["user_id"].to_numpy(),
                                    tables["item_features"]["item_id"].to_numpy(),
                                    INTERACTION_SAMPLE_SIZE, int(cfg["project"]["seed"]))
    on_demand = store.get_interaction_features(users_s, items_s)
    catalog_path = path(cfg, "reports_dir", "feature_catalog.md")
    catalog_path.write_text(render_feature_catalog(metadata, tables, on_demand), encoding="utf-8")

    log.info("Features built as of %s: %s", metadata["as_of"],
             {name: info["rows"] for name, info in metadata["tables"].items()})
    return {"stage": "features", "version": version, "as_of": metadata["as_of"],
            "rows": {name: info["rows"] for name, info in metadata["tables"].items()},
            "n_features": {name: len(cols) for name, cols in metadata["feature_names"].items()},
            "n_cold_users": metadata["n_cold_users"], "n_cold_items": metadata["n_cold_items"],
            "catalog": rel_path(catalog_path, cfg["project"]["root"])}


if __name__ == "__main__":
    run()
