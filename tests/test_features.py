"""Tests for stage 5 (src/features: registry, build_features, store) on small synthetic data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf
from src.features.registry import (FEATURE_REGISTRY, ON_DEMAND, TABLE_KEYS, FeatureValidationError,
                                   expected_schema, feature_columns, validate_feature_frame)
from src.features.store import FeatureStore, cosine_rows
from src.ingestion.ingest import GENRE_COLS
from src.preparation.prepare import clean_items, clean_users
from src.transformation import transform
from src.utils.config import load_config
from src.utils.io import write_json, write_parquet

AS_OF = 300
TOP_N = 3
DAMPING = 2.0


def users_clean() -> pd.DataFrame:
    """Four users; user 4 never rates anything before as_of (cold)."""
    raw = pd.DataFrame({
        "user_id": pd.Series([1, 2, 3, 4], dtype="int64"),
        "age": pd.Series([25, 40, None, 19], dtype="Int64"),
        "gender": pd.Series(["M", "F", "F", None], dtype="string"),
        "occupation": pd.Series(["student"] * 4, dtype="string"),
        "zip_code": pd.Series(["1", "2", "3", "4"], dtype="string"),
    })
    return clean_users(raw, "median", ["M", "F"], "unknown")[0]


def items_clean() -> pd.DataFrame:
    """Items 10 (action+comedy), 11 (drama), 12 (drama+romance), 13 (sci_fi, never rated)."""
    raw = pd.DataFrame({
        "item_id": pd.Series([10, 11, 12, 13], dtype="int64"),
        "title": pd.Series(["A (1990)", "B (1995)", "C (1997)", "D (1998)"], dtype="string"),
        "release_date": pd.to_datetime([None] * 4),
        "imdb_url": pd.Series([None] * 4, dtype="string"),
    })
    for col in GENRE_COLS:
        raw[col] = np.int8(0)
    for row, genres in enumerate([["action", "comedy"], ["drama"], ["drama", "romance"], ["sci_fi"]]):
        raw.loc[row, [f"genre_{g}" for g in genres]] = 1
    return clean_items(raw)[0]


def train_frame() -> pd.DataFrame:
    """Interactions up to as_of=300 for users 1-3."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3],
        "item_id": [10, 11, 12, 10, 11, 11],
        "rating": [5, 3, 4, 2, 4, 5],
        "timestamp": [100, 200, 300, 150, 250, 50],
    }).astype("int64")


def aggregates_of(train: pd.DataFrame) -> bf.TrainAggregates:
    """The transformation stage's aggregates of ``train`` (which never extends past the cutoff)."""
    return bf.TrainAggregates(transform.user_activity_summary(train), transform.item_popularity(train, DAMPING),
                              transform.genre_popularity(train, items_clean()))


def build(train: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    """Build the feature tables of the synthetic data as of ``AS_OF`` from its point-in-time history."""
    aggregates = aggregates_of(bf.point_in_time(train, AS_OF))
    return bf.build_feature_tables(train, users_clean(), items_clean(), aggregates, AS_OF, "v1", TOP_N, "unknown")


@pytest.fixture(scope="module")
def store_cfg(tmp_path_factory) -> dict:
    """Run the full feature stage into temporary directories and return the config."""
    root = tmp_path_factory.mktemp("features")
    cfg = load_config(overrides={"paths": {"processed_dir": str(root / "processed"),
                                           "features_dir": str(root / "features"),
                                           "reports_dir": str(root / "reports")},
                                 "features": {"min_item_ratings_for_avg": DAMPING}})
    write_parquet(train_frame(), root / "processed" / "train.parquet")
    write_parquet(users_clean(), root / "processed" / "users_clean.parquet")
    write_parquet(items_clean(), root / "processed" / "items_clean.parquet")
    write_json({"cutoff_timestamp": AS_OF}, root / "processed" / "split_metadata.json")
    transform.run(cfg)
    bf.run(cfg)
    return cfg


# --------------------------------------------------------------------------- registry
def test_registry_names_are_unique_per_table() -> None:
    for table in (*TABLE_KEYS, ON_DEMAND):
        cols = feature_columns(table)
        assert len(cols) == len(set(cols)), table
    assert {fd.name for fd in FEATURE_REGISTRY if fd.table == ON_DEMAND} == {
        "user_item_genre_affinity", "user_item_avg_rating_gap", "item_seen_by_user"}


def test_built_tables_match_registry_exactly() -> None:
    tables, _, _ = build(train_frame())
    for name, df in tables.items():
        validate_feature_frame(df, name)
        assert list(df.columns) == list(expected_schema(name))


@pytest.mark.parametrize("corrupt", [
    lambda df: df.drop(columns="user_avg_rating"),
    lambda df: df.astype({"user_n_ratings": "float64"}),
    lambda df: df.assign(user_avg_rating=np.nan),
    lambda df: pd.concat([df, df.iloc[:1]]),
])
def test_validation_rejects_corrupted_tables(corrupt) -> None:
    tables, _, _ = build(train_frame())
    with pytest.raises(FeatureValidationError):
        validate_feature_frame(corrupt(tables["user_features"]), "user_features")


# --------------------------------------------------------------------------- point-in-time
def test_future_interactions_do_not_change_features() -> None:
    later = pd.DataFrame({"user_id": [1, 4, 2], "item_id": [13, 10, 12], "rating": [1, 5, 1],
                          "timestamp": [AS_OF + 1, AS_OF + 10, AS_OF + 999]}).astype("int64")
    base, _, _ = build(train_frame())
    leaked, _, _ = build(pd.concat([train_frame(), later], ignore_index=True))
    for name in base:
        pd.testing.assert_frame_equal(base[name], leaked[name])


def test_aggregates_must_match_the_point_in_time_history() -> None:
    later = pd.DataFrame({"user_id": [4], "item_id": [13], "rating": [5], "timestamp": [AS_OF + 1]}).astype("int64")
    leaked = aggregates_of(pd.concat([train_frame(), later], ignore_index=True))
    with pytest.raises(ValueError, match="after as_of"):
        bf.build_feature_tables(train_frame(), users_clean(), items_clean(), leaked, AS_OF, "v1", TOP_N, "unknown")
    stale = aggregates_of(train_frame().iloc[1:])
    with pytest.raises(ValueError, match="re-run the transformation|Re-run the transformation"):
        bf.build_feature_tables(train_frame(), users_clean(), items_clean(), stale, AS_OF, "v1", TOP_N, "unknown")


def test_genre_prior_follows_genre_popularity_counts() -> None:
    prior = bf.genre_prior(transform.genre_popularity(train_frame(), items_clean()))
    assert list(prior.index) == GENRE_COLS and prior.sum() == pytest.approx(1.0)
    # genre tags of the 6 ratings: action 2, comedy 2, drama 3 + 1, romance 1 -> drama 4 / 9
    assert prior["genre_drama"] == pytest.approx(4 / 9) and prior["genre_sci_fi"] == 0.0
    empty = transform.genre_popularity(train_frame().iloc[:0], items_clean())
    np.testing.assert_allclose(bf.genre_prior(empty), 1 / len(GENRE_COLS))


def test_run_requires_transformation_outputs(store_cfg, tmp_path: Path) -> None:
    cfg = {**store_cfg, "paths": {**store_cfg["paths"], "processed_dir": str(tmp_path)}}
    with pytest.raises(FileNotFoundError, match="transformation stage"):
        bf.load_train_aggregates(cfg)


def test_history_before_as_of_does_change_features() -> None:
    changed = train_frame().assign(rating=lambda d: d["rating"].where(d["user_id"] != 1, 1))
    base, _, _ = build(train_frame())
    other, _, _ = build(changed)
    assert base["user_features"].loc[0, "user_avg_rating"] != other["user_features"].loc[0, "user_avg_rating"]


# --------------------------------------------------------------------------- feature values
def test_user_feature_values_and_cold_defaults() -> None:
    tables, defaults, stats = build(train_frame())
    users = tables["user_features"].set_index("user_id")
    assert users.loc[1, "user_n_ratings"] == 3 and users.loc[1, "user_avg_rating"] == pytest.approx(4.0)
    assert users.loc[1, "user_days_since_last_interaction"] == 0.0
    assert users.loc[1, "user_tenure_days"] == pytest.approx(200 / 86_400)
    assert users.loc[1, "user_preferred_genre"] == "drama"
    assert users.loc[1, "user_top_genres"] == "drama|action|comedy"  # ties follow GENRE_COLS order
    assert users.loc[3, "user_genre_entropy"] == 0.0  # single genre
    assert users.loc[4, "user_n_ratings"] == 0
    assert users.loc[4, "user_avg_rating"] == pytest.approx(stats["mean_rating"])
    assert users.loc[4, "user_preferred_genre"] == defaults["user_features"]["user_preferred_genre"] == "drama"
    assert (tables["user_features"]["feature_timestamp"] == pd.Timestamp(AS_OF, unit="s", tz="UTC")).all()


def test_genre_affinity_rows_sum_to_one_and_cold_users_get_prior() -> None:
    tables, defaults, _ = build(train_frame())
    affinity = tables["user_genre_affinity"].set_index("user_id")[GENRE_COLS]
    np.testing.assert_allclose(affinity.sum(axis=1), 1.0)
    assert affinity.loc[1, "genre_drama"] == pytest.approx(2 / 5)
    np.testing.assert_allclose(affinity.loc[4].to_numpy(), pd.Series(defaults["user_genre_affinity"])[GENRE_COLS])


def test_item_feature_values_and_cold_item() -> None:
    tables, defaults, stats = build(train_frame())
    items = tables["item_features"].set_index("item_id")
    assert items.loc[11, "item_n_ratings"] == 3 and items.loc[11, "item_popularity_rank"] == 1
    assert items.loc[10, "item_bayesian_avg_rating"] == pytest.approx((2 * 3.5 + 2 * stats["mean_rating"]) / 4)
    assert items.loc[13, "item_n_ratings"] == 0 and items.loc[13, "item_popularity_rank"] == 4
    assert items.loc[13, "item_popularity_pct"] == 0.0 and items.loc[13, "genre_sci_fi"] == 1
    assert items.loc[12, "item_year"] == 1997 and items.loc[12, "item_age_years"] == 0.0  # 1970 epoch -> clipped
    assert defaults["item_features"]["item_popularity_rank"] == 4


def test_normalised_entropy_bounds() -> None:
    shares = np.vstack([np.eye(4)[0], np.full(4, 0.25)])
    np.testing.assert_allclose(bf.normalised_entropy(shares), [0.0, 1.0])


def test_cosine_rows_handles_zero_vectors() -> None:
    a = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([[1.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(cosine_rows(a, b), [1 / np.sqrt(2), 0.0])


# --------------------------------------------------------------------------- store
def test_store_returns_requested_order_and_fills_unseen_ids(store_cfg) -> None:
    store = FeatureStore(store_cfg)
    defaults = store.metadata()["defaults"]["user_features"]
    out = store.get_user_features([2, 999, 1, 2], ["user_n_ratings", "user_avg_rating", "gender"])
    assert out["user_id"].tolist() == [2, 999, 1, 2]
    assert out["is_cold"].tolist() == [False, True, False, False]
    assert out.loc[1, "user_avg_rating"] == pytest.approx(defaults["user_avg_rating"])
    assert out.loc[1, "gender"] == "unknown" and out["user_n_ratings"].dtype == "int64"
    cold = store.get_user_features([4])
    assert cold["is_cold"].item() and list(cold.columns) == ["user_id", *feature_columns("user_features"), "is_cold"]
    with pytest.raises(KeyError):
        store.get_item_features([10], ["no_such_feature"])


def test_store_genre_matrices(store_cfg) -> None:
    store = FeatureStore(store_cfg)
    affinity = store.get_user_genre_affinity([1, 999])
    assert affinity.shape == (2, len(GENRE_COLS))
    np.testing.assert_allclose(affinity.sum(axis=1), 1.0)
    genres = store.get_item_genre_matrix([10, 999])
    assert genres.shape == (2, len(GENRE_COLS)) and genres[1].sum() == 0
    assert genres[0, GENRE_COLS.index("genre_action")] == 1.0


def test_store_interaction_features(store_cfg) -> None:
    store = FeatureStore(store_cfg)
    pairs = store.get_interaction_features([1, 4, 3], [10, 10, 13])
    assert pairs["item_seen_by_user"].tolist() == [1, 0, 0]
    expected = cosine_rows(store.get_user_genre_affinity([3]), store.get_item_genre_matrix([13]))
    assert pairs.loc[2, "user_item_genre_affinity"] == pytest.approx(expected[0]) == 0.0
    user_avg = store.get_user_features([1], ["user_avg_rating"])["user_avg_rating"].item()
    item_avg = store.get_item_features([10], ["item_avg_rating"])["item_avg_rating"].item()
    assert pairs.loc[0, "user_item_avg_rating_gap"] == pytest.approx(user_avg - item_avg)
    with pytest.raises(ValueError):
        store.get_interaction_features([1, 2], [10])


def test_store_builds_training_frame(store_cfg) -> None:
    store = FeatureStore(store_cfg)
    frame = store.build_training_frame(train_frame(), user_columns=["user_avg_rating"],
                                       item_columns=["item_n_ratings"])
    assert len(frame) == len(train_frame())
    assert {"user_avg_rating", "user_is_cold", "item_n_ratings", "item_is_cold",
            "user_item_genre_affinity", "item_seen_by_user"} <= set(frame.columns)
    assert frame["item_seen_by_user"].eq(1).all()
    with pytest.raises(ValueError):
        store.build_training_frame(frame)


def test_store_rejects_version_mismatch(store_cfg) -> None:
    cfg = {**store_cfg, "features": {**store_cfg["features"], "feature_set_version": "v999"}}
    with pytest.raises(ValueError):
        FeatureStore(cfg).metadata()


def test_run_writes_metadata_and_catalog(store_cfg) -> None:
    meta = FeatureStore(store_cfg).metadata()
    assert meta["version"] == "v1" and meta["as_of_timestamp"] == AS_OF
    assert meta["tables"]["user_features"]["rows"] == 4 and meta["n_cold_users"] == 1
    assert set(meta["defaults"]) == set(TABLE_KEYS)
    text = (Path(store_cfg["paths"]["reports_dir"]) / "feature_catalog.md").read_text(encoding="utf-8")
    assert "| Feature | Entity | Type | Description | Logic/Source | Point-in-time | Used by |" in text
    assert "| Feature | Entity | Null % | Min | Mean | Max | Top value |" in text
