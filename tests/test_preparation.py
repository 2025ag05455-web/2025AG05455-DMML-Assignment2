"""Tests for stage 3 (src/preparation/prepare.py) on small synthetic frames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.ingest import GENRE_COLS
from src.preparation import prepare
from src.utils.config import load_config
from src.utils.io import read_json, read_parquet, write_parquet


def interim_users() -> pd.DataFrame:
    """Validated-layer users with a missing age, a missing and an invalid gender."""
    return pd.DataFrame({
        "user_id": pd.Series([1, 2, 3, 4, 5], dtype="int64"),
        "age": pd.Series([17, None, 40, 30, 60], dtype="Int64"),
        "gender": pd.Series(["M", None, "X", "f", "F"], dtype="string"),
        "occupation": pd.Series([" Student ", "writer", "OTHER", "doctor", "none"], dtype="string"),
        "zip_code": pd.Series([" ab12 ", "94043", "1", "2", "3"], dtype="string"),
    })


def interim_items() -> pd.DataFrame:
    """Validated-layer items: one missing title, one missing release date (year from title)."""
    items = pd.DataFrame({
        "item_id": pd.Series([10, 11, 12], dtype="int64"),
        "title": pd.Series(["Alpha (1995)", None, " Gamma (1987) "], dtype="string"),
        "release_date": pd.to_datetime(["1995-01-01", "1990-05-01", None]),
        "imdb_url": pd.Series(["u1", None, "u3"], dtype="string"),
    })
    for col in GENRE_COLS:
        items[col] = np.int8(0)
    items.loc[0, ["genre_action", "genre_comedy"]] = 1
    items.loc[2, "genre_drama"] = 1
    return items


def interim_interactions() -> pd.DataFrame:
    """Validated-layer interactions with timestamps 1..10 for users 1-5 and items 10-12."""
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "item_id": [10, 11, 10, 12, 11, 12, 10, 11, 12, 10],
        "rating": [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
        "timestamp": list(range(1, 11)),
    }).astype("int64")


# --------------------------------------------------------------------------- users / items
def test_clean_users_imputes_age_and_normalises_text() -> None:
    users, stats = prepare.clean_users(interim_users(), "median", ["M", "F"], "unknown")
    assert users["age"].dtype == "int64" and users.loc[1, "age"] == 35  # median of 17, 40, 30, 60
    assert stats["age_imputed"] == 1 and stats["gender_set_unknown"] == 2
    assert users["gender"].tolist() == ["M", "unknown", "unknown", "F", "F"]
    assert users.loc[0, "occupation"] == "student" and users.loc[0, "zip_code"] == "AB12"
    assert users["age_group"].tolist() == ["under_18", "35_44", "35_44", "25_34", "55_plus"]
    assert list(users.columns) == prepare.USER_COLUMNS


def test_clean_users_rejects_unknown_impute_strategy() -> None:
    with pytest.raises(ValueError):
        prepare.clean_users(interim_users(), "mode", ["M", "F"], "unknown")


def test_age_group_boundaries() -> None:
    ages = pd.Series([5, 17, 18, 24, 25, 34, 35, 44, 45, 54, 55, 99])
    expected = ["under_18", "under_18", "18_24", "18_24", "25_34", "25_34", "35_44", "35_44",
                "45_54", "45_54", "55_plus", "55_plus"]
    assert prepare.age_group(ages).tolist() == expected


def test_clean_items_title_year_and_genres() -> None:
    items, stats = prepare.clean_items(interim_items())
    assert items["title"].tolist() == ["Alpha (1995)", prepare.UNKNOWN_TITLE, "Gamma (1987)"]
    assert items["year"].tolist() == [1995, 1990, 1987]  # last one parsed from the title
    assert items["genres"].tolist() == ["action|comedy", "", "drama"]
    assert items["n_genres"].tolist() == [2, 0, 1]
    assert stats["title_imputed"] == 1 and stats["year_from_title"] == 1
    assert all(items[col].dtype == "int8" for col in GENRE_COLS)
    assert list(items.columns) == prepare.ITEM_COLUMNS


# --------------------------------------------------------------------------- interactions
def test_resolve_duplicates_keeps_latest_event() -> None:
    df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 10, 10],
                       "rating": [2, 5, 3], "timestamp": [200, 100, 150]})
    latest = prepare.resolve_duplicates(df, "keep_latest")
    assert len(latest) == 2
    assert latest.query("user_id == 1")["rating"].item() == 2  # ts 200 is the latest
    earliest = prepare.resolve_duplicates(df, "keep_earliest")
    assert earliest.query("user_id == 1")["rating"].item() == 5
    with pytest.raises(ValueError):
        prepare.resolve_duplicates(df, "average")


def test_clean_interactions_drops_orphans_and_adds_utc_time() -> None:
    raw = pd.concat([interim_interactions(),
                     pd.DataFrame({"user_id": [99, 1], "item_id": [10, 10], "rating": [3, 1],
                                   "timestamp": [11, 12]})], ignore_index=True)
    out, stats = prepare.clean_interactions(raw, pd.Series([1, 2, 3, 4, 5]), pd.Series([10, 11, 12]),
                                            "keep_latest")
    assert stats["orphans_dropped"] == 1 and stats["duplicates_resolved"] == 1
    assert out.query("user_id == 1 and item_id == 10")["rating"].item() == 1
    assert str(out["event_time"].dtype) == "datetime64[ns, UTC]"
    assert out["event_time"].eq(pd.to_datetime(out["timestamp"], unit="s", utc=True)).all()
    assert not out.duplicated(["user_id", "item_id"]).any()
    assert list(out.columns) == prepare.INTERACTION_COLUMNS


# --------------------------------------------------------------------------- split
def test_time_based_split_has_no_temporal_leakage() -> None:
    df = pd.DataFrame({"user_id": 1, "item_id": np.arange(100), "rating": 3,
                       "timestamp": np.arange(1, 101)}).astype("int64")
    train, test, cutoff = prepare.time_based_split(df, 0.2)
    assert cutoff == 80 and len(train) == 80 and len(test) == 20
    assert train["timestamp"].max() <= cutoff < test["timestamp"].min()


def test_time_based_split_keeps_ties_on_the_train_side() -> None:
    df = pd.DataFrame({"user_id": 1, "item_id": np.arange(10), "rating": 3,
                       "timestamp": [1, 2, 3, 4, 5, 8, 8, 8, 8, 9]}).astype("int64")
    train, test, cutoff = prepare.time_based_split(df, 0.2)
    assert cutoff == 8 and set(test["timestamp"]) == {9} and (train["timestamp"] <= 8).all()


def test_split_rejects_bad_fraction_and_detects_overlap() -> None:
    df = interim_interactions()
    with pytest.raises(ValueError):
        prepare.time_based_split(df, 1.0)
    with pytest.raises(AssertionError):
        prepare.check_no_temporal_overlap(df.iloc[:5], df.iloc[3:], cutoff=5)


def test_split_metadata_counts_warm_and_cold_entities() -> None:
    train = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 11, 10], "timestamp": [1, 2, 3]})
    test = pd.DataFrame({"user_id": [1, 3, 3], "item_id": [12, 10, 11], "timestamp": [4, 5, 6]})
    meta = prepare.split_metadata(train, test, cutoff=3, test_fraction=0.5, min_user_train_interactions=1)
    assert (meta["n_test_users"], meta["n_warm_test_users"], meta["n_cold_test_users"]) == (2, 1, 1)
    assert meta["n_cold_test_items"] == 1 and meta["n_test_interactions_cold_user"] == 2
    strict = prepare.split_metadata(train, test, 3, 0.5, min_user_train_interactions=3)
    assert strict["n_warm_test_users"] == 0


# --------------------------------------------------------------------------- end to end
def test_run_writes_all_contract_outputs(tmp_path) -> None:
    cfg = load_config(overrides={"paths": {"interim_dir": str(tmp_path / "interim"),
                                           "processed_dir": str(tmp_path / "processed")}})
    for name, df in (("users", interim_users()), ("items", interim_items()),
                     ("interactions", interim_interactions())):
        write_parquet(df, tmp_path / "interim" / f"{name}.parquet")

    summary = prepare.run(cfg)

    processed = tmp_path / "processed"
    train, test = read_parquet(processed / "train.parquet"), read_parquet(processed / "test.parquet")
    meta = read_json(processed / "split_metadata.json")
    assert summary["rows"]["train"] + summary["rows"]["test"] == 10
    assert train["timestamp"].max() <= meta["cutoff_timestamp"] < test["timestamp"].min()
    required = {"cutoff_timestamp", "cutoff_time_utc", "n_train", "n_test", "n_test_users",
                "n_warm_test_users", "n_cold_test_users", "n_cold_test_items"}
    assert required <= set(meta)
    report = read_json(processed / "preparation_report.json")
    assert report["users"]["age_imputed"] == 1 and report["rows"]["users_clean"] == 5
    assert read_parquet(processed / "users_clean.parquet")["age"].notna().all()
