"""Tests for stage 4 (src/transformation/transform.py) on small synthetic frames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.ingestion.ingest import GENRE_COLS
from src.transformation import transform
from src.utils.config import load_config
from src.utils.io import read_json, read_parquet, write_parquet

DAY = transform.SECONDS_PER_DAY


def train_frame() -> pd.DataFrame:
    """User 1: three ratings over two days; user 2: one rating; items 10 and 11."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2],
        "item_id": [10, 11, 12, 10],
        "rating": [5, 3, 4, 2],
        "timestamp": [DAY, DAY + 60, 3 * DAY, 2 * DAY],
    }).astype("int64")


def items_frame() -> pd.DataFrame:
    """Items 10 (action+comedy), 11 (drama), 12 (drama) and 13 (drama, never rated)."""
    items = pd.DataFrame({"item_id": [10, 11, 12, 13]})
    for col in GENRE_COLS:
        items[col] = np.int8(0)
    items.loc[0, ["genre_action", "genre_comedy"]] = 1
    items.loc[1:, "genre_drama"] = 1
    return items


def test_user_activity_summary_statistics() -> None:
    summary = transform.user_activity_summary(train_frame()).set_index("user_id")
    assert summary.loc[1, "n_ratings"] == 3 and summary.loc[1, "active_days"] == 2
    assert summary.loc[1, "mean_rating"] == pytest.approx(4.0)
    assert summary.loc[1, "std_rating"] == pytest.approx(np.std([5, 3, 4]))  # population std
    assert summary.loc[2, "std_rating"] == 0.0  # single rating -> 0, not NaN
    assert (summary.loc[1, "first_ts"], summary.loc[1, "last_ts"]) == (DAY, 3 * DAY)
    assert list(summary.reset_index().columns) == ["user_id", "n_ratings", "mean_rating", "std_rating",
                                                   "first_ts", "last_ts", "active_days"]


def test_item_popularity_bayesian_average_rank_and_pct() -> None:
    pop = transform.item_popularity(train_frame(), damping=2).set_index("item_id")
    global_mean = 3.5
    assert pop.loc[10, "bayesian_avg_rating"] == pytest.approx((2 * 3.5 + 2 * global_mean) / 4)
    assert pop.loc[12, "bayesian_avg_rating"] == pytest.approx((4 + 2 * global_mean) / 3)
    # item 10 is most rated; 11 and 12 tie on count -> higher Bayesian average (12) ranks first
    assert pop["popularity_rank"].to_dict() == {10: 1, 11: 3, 12: 2}
    assert pop.loc[10, "popularity_pct"] == 1.0 and pop.loc[11, "popularity_pct"] == pytest.approx(2 / 3)
    assert list(pop.reset_index().columns) == transform.ITEM_POPULARITY_COLUMNS


def test_interaction_matrix_shape_and_values() -> None:
    user_ids, item_ids = np.array([1, 2, 3]), np.array([10, 11, 12, 13])
    matrix = transform.build_interaction_matrix(train_frame(), user_ids, item_ids)
    assert isinstance(matrix, sparse.csr_matrix) and matrix.shape == (3, 4) and matrix.nnz == 4
    dense = matrix.toarray()
    assert dense[0, 0] == 5 and dense[0, 1] == 3 and dense[0, 2] == 4 and dense[1, 0] == 2
    assert dense[2].sum() == 0 and dense[:, 3].sum() == 0  # cold user / item stay empty
    summary = transform.matrix_summary(matrix)
    assert summary["density"] == pytest.approx(4 / 12) and summary["empty_rows"] == 1
    assert summary["empty_cols"] == 1


def test_interaction_matrix_rejects_unknown_ids_and_duplicates() -> None:
    with pytest.raises(ValueError):
        transform.build_interaction_matrix(train_frame(), np.array([1]), np.array([10, 11, 12]))
    duplicated = pd.concat([train_frame(), train_frame().iloc[:1]])
    with pytest.raises(ValueError):
        transform.build_interaction_matrix(duplicated, np.array([1, 2]), np.array([10, 11, 12]))


def test_genre_popularity_counts_multi_genre_items() -> None:
    genres = transform.genre_popularity(train_frame(), items_frame()).set_index("genre")
    assert genres.loc["drama", "n_items"] == 3 and genres.loc["drama", "n_rated_items"] == 2
    assert genres.loc["drama", "n_ratings"] == 2 and genres.loc["drama", "mean_rating"] == pytest.approx(3.5)
    assert genres.loc["action", "n_ratings"] == 2 and genres.loc["action", "share_of_ratings"] == 0.5
    assert np.isnan(genres.loc["war", "mean_rating"]) and genres.loc["war", "n_ratings"] == 0


def test_run_uses_train_only_and_writes_outputs(tmp_path) -> None:
    cfg = load_config(overrides={"paths": {"processed_dir": str(tmp_path)}})
    write_parquet(train_frame(), tmp_path / "train.parquet")
    write_parquet(pd.DataFrame({"user_id": [1, 2, 3]}), tmp_path / "users_clean.parquet")
    write_parquet(items_frame(), tmp_path / "items_clean.parquet")

    summary = transform.run(cfg)

    assert summary["matrix"]["shape"] == [3, 4] and summary["matrix"]["nnz"] == 4
    index = read_json(tmp_path / "matrix_index.json")
    assert index["user_ids"] == [1, 2, 3] and index["item_ids"] == [10, 11, 12, 13]
    matrix = sparse.load_npz(tmp_path / "user_item_matrix.npz")
    assert matrix.sum() == train_frame()["rating"].sum()
    assert len(read_parquet(tmp_path / "item_popularity.parquet")) == 3
    assert len(read_parquet(tmp_path / "genre_popularity.parquet")) == len(GENRE_COLS)
