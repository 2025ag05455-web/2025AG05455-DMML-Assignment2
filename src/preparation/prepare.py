"""Stage 3 - Data preparation: cleaning, imputation, de-duplication and the time-based split.

Reads the validated, typed tables produced by the validation stage (``interim_dir``) and
turns them into analysis-ready tables plus a leakage-free global temporal train/test split.

Cleaning rules
    users        : missing age imputed (median, per config), invalid/missing gender -> 'unknown',
                   occupation lower-cased, zip_code upper-cased, ``age_group`` bucket added.
    items        : missing title -> 'unknown title', ``year`` from release_date or from the
                   "(YYYY)" suffix of the title, pipe-joined ``genres`` and ``n_genres`` added.
    interactions : rows with unknown user/item dropped (referential integrity), duplicate
                   (user_id, item_id) pairs resolved per ``duplicate_policy``, UTC ``event_time`` added.

Split
    Global time-based split: ``cutoff = quantile(1 - test_fraction)`` of all interaction
    timestamps; ``timestamp <= cutoff`` -> train, later -> test. Every train event therefore
    precedes every test event, which mirrors how a recommender is trained on the past and
    evaluated on the future.

Outputs (processed_dir):
    users_clean.parquet, items_clean.parquet, interactions_clean.parquet,
    train.parquet, test.parquet, split_metadata.json, preparation_report.json
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.ingestion.ingest import GENRE_COLS
from src.utils.config import load_config, path
from src.utils.io import read_parquet, write_json, write_parquet
from src.utils.logger import get_logger

log = get_logger("preparation")

AGE_BINS = [-np.inf, 17, 24, 34, 44, 54, np.inf]
AGE_LABELS = ["under_18", "18_24", "25_34", "35_44", "45_54", "55_plus"]
UNKNOWN_TITLE = "unknown title"
TITLE_YEAR_PATTERN = r"\((\d{4})\)$"
GENRE_NAMES = [col.removeprefix("genre_") for col in GENRE_COLS]

USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code", "age_group"]
ITEM_COLUMNS = ["item_id", "title", "year", "release_date", "imdb_url", "genres", "n_genres", *GENRE_COLS]
INTERACTION_COLUMNS = ["user_id", "item_id", "rating", "timestamp", "event_time"]
INTERACTION_DTYPES = {"user_id": "int64", "item_id": "int64", "rating": "int64", "timestamp": "int64"}

_AGE_IMPUTERS = {"median": pd.Series.median, "mean": pd.Series.mean}
_DUPLICATE_KEEP = {"keep_latest": "last", "keep_earliest": "first"}


# --------------------------------------------------------------------------- helpers
def _step(name: str, before: int, after: int) -> dict:
    """Describe one cleaning step as a row-count record for the preparation report."""
    return {"step": name, "rows_before": int(before), "rows_after": int(after)}


def _utc_iso(ts: int) -> str:
    """Format a unix timestamp (seconds) as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def age_group(age: pd.Series) -> pd.Series:
    """Bucket integer ages into the fixed groups ``AGE_LABELS`` (string dtype)."""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS).astype("string")


def _age_fill_value(age: pd.Series, strategy: str) -> int:
    """Return the integer value used to impute missing ages (``median`` or ``mean``)."""
    if strategy not in _AGE_IMPUTERS:
        raise ValueError(f"Unsupported impute_age strategy {strategy!r}; use one of {sorted(_AGE_IMPUTERS)}")
    return int(round(float(_AGE_IMPUTERS[strategy](age.dropna()))))


def _clean_text(values: pd.Series, fill: str) -> pd.Series:
    """Strip whitespace, turn empty strings into missing values and fill them with ``fill``."""
    text = values.astype("string").str.strip()
    return text.mask(text.fillna("").eq(""), fill)


# --------------------------------------------------------------------------- users
def clean_users(users: pd.DataFrame, impute_age: str, allowed_genders: list[str],
                unknown_token: str) -> tuple[pd.DataFrame, dict]:
    """Clean the validated users table and return ``(users_clean, stats)``."""
    out = users.drop_duplicates("user_id", keep="first").copy()
    fill_age = _age_fill_value(out["age"], impute_age)
    n_age_imputed = int(out["age"].isna().sum())
    out["age"] = out["age"].fillna(fill_age).astype("int64")

    gender = out["gender"].astype("string").str.strip().str.upper()
    valid_gender = gender.isin(allowed_genders).to_numpy(dtype=bool)
    out["gender"] = gender.where(valid_gender, unknown_token).astype("string")

    out["occupation"] = _clean_text(out["occupation"], unknown_token).str.lower()
    out["zip_code"] = _clean_text(out["zip_code"], unknown_token).str.upper()
    out["age_group"] = age_group(out["age"])

    out = out[USER_COLUMNS].sort_values("user_id").reset_index(drop=True)
    stats = {
        "steps": [_step("drop_duplicate_user_id", len(users), len(out))],
        "age_imputed": n_age_imputed,
        "age_fill_value": fill_age,
        "age_impute_strategy": impute_age,
        "gender_set_unknown": int((~valid_gender).sum()),
        "age_group_counts": out["age_group"].value_counts().reindex(AGE_LABELS, fill_value=0).to_dict(),
    }
    return out, stats


# --------------------------------------------------------------------------- items
def _genre_string(flags: pd.DataFrame) -> pd.Series:
    """Pipe-join the names of the active genre flags of every item (e.g. ``action|comedy``)."""
    tokens = pd.Series([f"{name}|" for name in GENRE_NAMES], index=GENRE_COLS, dtype=object)
    # int flag * "name|" repeats the token 0 or 1 times; the dot product concatenates them.
    return flags.astype(int).dot(tokens).str.rstrip("|").astype("string")


def clean_items(items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the validated items table and return ``(items_clean, stats)``."""
    out = items.drop_duplicates("item_id", keep="first").copy()
    n_title_missing = int(_clean_text(out["title"], "").eq("").sum())
    out["title"] = _clean_text(out["title"], UNKNOWN_TITLE)

    year_from_date = out["release_date"].dt.year.astype("Int64")
    year_from_title = out["title"].str.extract(TITLE_YEAR_PATTERN, expand=False).astype("Int64")
    out["year"] = year_from_date.fillna(year_from_title)

    flags = out[GENRE_COLS].astype("int8")
    out[GENRE_COLS] = flags
    out["genres"] = _genre_string(flags)
    out["n_genres"] = flags.sum(axis=1).astype("int64")
    out["imdb_url"] = out["imdb_url"].astype("string")

    out = out[ITEM_COLUMNS].sort_values("item_id").reset_index(drop=True)
    stats = {
        "steps": [_step("drop_duplicate_item_id", len(items), len(out))],
        "title_imputed": n_title_missing,
        "year_from_release_date": int(year_from_date.notna().sum()),
        "year_from_title": int((year_from_date.isna() & year_from_title.notna()).sum()),
        "year_missing": int(out["year"].isna().sum()),
        "items_without_genre": int(out["n_genres"].eq(0).sum()),
    }
    return out, stats


# --------------------------------------------------------------------------- interactions
def resolve_duplicates(interactions: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Keep one row per (user_id, item_id): the latest (``keep_latest``) or earliest event."""
    if policy not in _DUPLICATE_KEEP:
        raise ValueError(f"Unsupported duplicate_policy {policy!r}; use one of {sorted(_DUPLICATE_KEEP)}")
    ordered = interactions.sort_values("timestamp", kind="mergesort")  # stable: ties keep file order
    return ordered.drop_duplicates(["user_id", "item_id"], keep=_DUPLICATE_KEEP[policy])


def clean_interactions(interactions: pd.DataFrame, user_ids: pd.Series, item_ids: pd.Series,
                       duplicate_policy: str) -> tuple[pd.DataFrame, dict]:
    """Enforce dtypes and referential integrity, resolve duplicates and add ``event_time``."""
    typed = interactions[list(INTERACTION_DTYPES)].astype(INTERACTION_DTYPES)
    known = typed[typed["user_id"].isin(user_ids) & typed["item_id"].isin(item_ids)]
    deduped = resolve_duplicates(known, duplicate_policy).copy()
    deduped["event_time"] = pd.to_datetime(deduped["timestamp"], unit="s", utc=True).astype("datetime64[ns, UTC]")
    out = (deduped[INTERACTION_COLUMNS]
           .sort_values(["timestamp", "user_id", "item_id"], kind="mergesort")
           .reset_index(drop=True))
    stats = {
        "steps": [_step("drop_unknown_user_or_item", len(typed), len(known)),
                  _step(f"resolve_duplicate_pairs ({duplicate_policy})", len(known), len(out))],
        "orphans_dropped": int(len(typed) - len(known)),
        "duplicates_resolved": int(len(known) - len(out)),
    }
    return out, stats


# --------------------------------------------------------------------------- split
def check_no_temporal_overlap(train: pd.DataFrame, test: pd.DataFrame, cutoff: int) -> None:
    """Raise ``AssertionError`` unless ``max(train.ts) <= cutoff < min(test.ts)``."""
    if not (train["timestamp"].max() <= cutoff < test["timestamp"].min()):
        raise AssertionError(
            f"Temporal leakage: max train ts={train['timestamp'].max()}, cutoff={cutoff}, "
            f"min test ts={test['timestamp'].min()}")


def time_based_split(interactions: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split globally by time: ``timestamp <= quantile(1 - test_fraction)`` -> train, rest -> test."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    # Timestamps are integers, so flooring the (interpolated) quantile keeps the same partition.
    cutoff = int(np.floor(interactions["timestamp"].quantile(1.0 - test_fraction)))
    is_train = interactions["timestamp"] <= cutoff
    train = interactions[is_train].reset_index(drop=True)
    test = interactions[~is_train].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"Degenerate split at cutoff {cutoff}: n_train={len(train)}, n_test={len(test)}")
    check_no_temporal_overlap(train, test, cutoff)
    return train, test, cutoff


def split_metadata(train: pd.DataFrame, test: pd.DataFrame, cutoff: int, test_fraction: float,
                   min_user_train_interactions: int) -> dict:
    """Summarise the split: sizes, time ranges and warm/cold test users and items.

    A test user is *warm* when it has at least ``min_user_train_interactions`` train
    interactions; a test item is *cold* when it never occurs in train.
    """
    train_counts = train["user_id"].value_counts()
    warm_users = train_counts.index[train_counts >= min_user_train_interactions]
    test_users = pd.Series(test["user_id"].unique())
    test_items = pd.Series(test["item_id"].unique())
    is_warm_user = test_users.isin(warm_users)
    is_cold_item = ~test_items.isin(train["item_id"].unique())
    return {
        "method": "time_based",
        "cutoff_timestamp": int(cutoff),
        "cutoff_time_utc": _utc_iso(cutoff),
        "test_fraction_configured": float(test_fraction),
        "test_fraction_actual": round(len(test) / (len(train) + len(test)), 6),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_train_users": int(train["user_id"].nunique()),
        "n_train_items": int(train["item_id"].nunique()),
        "n_test_users": int(len(test_users)),
        "n_warm_test_users": int(is_warm_user.sum()),
        "n_cold_test_users": int((~is_warm_user).sum()),
        "n_test_items": int(len(test_items)),
        "n_cold_test_items": int(is_cold_item.sum()),
        "n_test_interactions_cold_user": int((~test["user_id"].isin(warm_users)).sum()),
        "min_user_train_interactions": int(min_user_train_interactions),
        "train_time_range_utc": [_utc_iso(train["timestamp"].min()), _utc_iso(train["timestamp"].max())],
        "test_time_range_utc": [_utc_iso(test["timestamp"].min()), _utc_iso(test["timestamp"].max())],
    }


# --------------------------------------------------------------------------- entrypoint
def run(cfg: dict | None = None) -> dict:
    """Run the preparation stage end to end and return a JSON-serialisable summary."""
    cfg = cfg or load_config()
    pcfg, scfg = cfg["preparation"], cfg["split"]
    if scfg["method"] != "time_based":
        raise ValueError(f"Unsupported split method {scfg['method']!r}; only 'time_based' is implemented")

    users_raw, items_raw, inter_raw = (read_parquet(path(cfg, "interim_dir", f"{name}.parquet"))
                                       for name in ("users", "items", "interactions"))
    log.info("Loaded validated data: users=%d items=%d interactions=%d", len(users_raw), len(items_raw), len(inter_raw))

    users, user_stats = clean_users(users_raw, pcfg["impute_age"], cfg["validation"]["allowed_genders"],
                                    pcfg["unknown_token"])
    items, item_stats = clean_items(items_raw)
    interactions, inter_stats = clean_interactions(inter_raw, users["user_id"], items["item_id"],
                                                   pcfg["duplicate_policy"])
    train, test, cutoff = time_based_split(interactions, float(scfg["test_fraction"]))
    split_meta = split_metadata(train, test, cutoff, float(scfg["test_fraction"]),
                                int(scfg["min_user_train_interactions"]))

    outputs = {"users_clean": users, "items_clean": items, "interactions_clean": interactions,
               "train": train, "test": test}
    for name, df in outputs.items():
        write_parquet(df, path(cfg, "processed_dir", f"{name}.parquet"))
    write_json(split_meta, path(cfg, "processed_dir", "split_metadata.json"))

    report = {
        "stage": "preparation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": {name: int(len(df)) for name, df in outputs.items()},
        "users": user_stats,
        "items": item_stats,
        "interactions": inter_stats,
        "split": split_meta,
    }
    write_json(report, path(cfg, "processed_dir", "preparation_report.json"))
    log.info("Preparation complete: cutoff=%s train=%d test=%d (warm test users=%d, cold=%d)",
             split_meta["cutoff_time_utc"], len(train), len(test),
             split_meta["n_warm_test_users"], split_meta["n_cold_test_users"])
    return {"stage": "preparation", "rows": report["rows"],
            "age_imputed": user_stats["age_imputed"], "duplicates_resolved": inter_stats["duplicates_resolved"],
            "split": {k: split_meta[k] for k in ("cutoff_timestamp", "cutoff_time_utc", "n_train", "n_test",
                                                 "n_warm_test_users", "n_cold_test_users", "n_cold_test_items")}}


if __name__ == "__main__":
    run()
