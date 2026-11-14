"""Stage 4 - Data transformation: aggregated datasets and the user-item matrix (TRAIN ONLY).

Every derived dataset is computed from ``train.parquet`` only, so nothing observed after the
split cutoff can leak into models or features. Downstream consumers: the feature stage
(``src.features.build_features``) reads the three aggregates and the training stage
(``src.training.train``) fits every model from the user-item matrix.

Outputs (processed_dir):
    user_activity_summary.parquet : user_id, n_ratings, mean_rating, std_rating, first_ts, last_ts, active_days
    item_popularity.parquet       : item_id, n_ratings, mean_rating, bayesian_avg_rating, popularity_rank,
                                    popularity_pct, std_rating, first_ts, last_ts
    genre_popularity.parquet      : genre, n_items, n_rated_items, n_ratings, share_of_ratings, mean_rating
    user_item_matrix.npz          : scipy CSR (users x items, explicit ratings; rows/cols = full catalogue)
    matrix_index.json             : {"user_ids": [...], "item_ids": [...]} row / column order
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.ingestion.ingest import GENRE_COLS
from src.utils.config import load_config, path
from src.utils.io import read_parquet, write_json, write_parquet
from src.utils.logger import get_logger

log = get_logger("transformation")

SECONDS_PER_DAY = 86_400
MATRIX_FILE = "user_item_matrix.npz"
MATRIX_INDEX_FILE = "matrix_index.json"
ITEM_POPULARITY_COLUMNS = ["item_id", "n_ratings", "mean_rating", "bayesian_avg_rating", "popularity_rank",
                           "popularity_pct", "std_rating", "first_ts", "last_ts"]


# --------------------------------------------------------------------------- aggregates
def rating_aggregates(interactions: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-``key`` rating statistics: count, mean, population std (0 for a single rating), first/last ts."""
    grouped = interactions.groupby(key, sort=True)
    return pd.DataFrame({
        "n_ratings": grouped["rating"].size().astype("int64"),
        "mean_rating": grouped["rating"].mean().astype("float64"),
        "std_rating": grouped["rating"].std(ddof=0).astype("float64"),
        "first_ts": grouped["timestamp"].min().astype("int64"),
        "last_ts": grouped["timestamp"].max().astype("int64"),
    }).reset_index()


def user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Per-user activity: rating statistics plus the number of distinct (UTC) active days."""
    summary = rating_aggregates(train, "user_id")
    active_days = (train["timestamp"] // SECONDS_PER_DAY).groupby(train["user_id"]).nunique()
    summary["active_days"] = summary["user_id"].map(active_days).astype("int64")
    return summary


def bayesian_average(n_ratings: pd.Series, mean_rating: pd.Series, prior_mean: float, damping: float) -> pd.Series:
    """Shrink each mean towards ``prior_mean``: ``(n * mean + m * prior) / (n + m)`` with ``m = damping``."""
    return (n_ratings * mean_rating + damping * prior_mean) / (n_ratings + damping)


def item_popularity(train: pd.DataFrame, damping: float) -> pd.DataFrame:
    """Per-item popularity with Bayesian-average rating, ordinal rank (1 = most rated) and percentile.

    Ties in ``n_ratings`` are broken by the Bayesian average (desc) and then ``item_id`` (asc)
    so the ranking is fully deterministic. ``popularity_pct`` is the share of rated items whose
    rating count is <= the item's (1.0 = most popular).
    """
    pop = rating_aggregates(train, "item_id")
    pop["bayesian_avg_rating"] = bayesian_average(pop["n_ratings"], pop["mean_rating"],
                                                  float(train["rating"].mean()), damping)
    order = pop.sort_values(["n_ratings", "bayesian_avg_rating", "item_id"],
                            ascending=[False, False, True], kind="mergesort").index
    pop["popularity_rank"] = pd.Series(np.arange(1, len(pop) + 1), index=order).astype("int64")
    pop["popularity_pct"] = pop["n_ratings"].rank(method="max", pct=True).astype("float64")
    return pop[ITEM_POPULARITY_COLUMNS]


def genre_popularity(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Per-genre catalogue size, number of train ratings, share of all ratings and mean rating.

    Items can carry several genres, so ``share_of_ratings`` does not sum to 1 across genres.
    """
    flags = items.set_index("item_id")[GENRE_COLS]
    rated = flags.reindex(train["item_id"]).fillna(0).to_numpy(dtype=np.float64)
    ratings = train["rating"].to_numpy(dtype=np.float64)
    n_ratings = rated.sum(axis=0)
    rating_sum = rated.T @ ratings
    mean_rating = np.divide(rating_sum, n_ratings, out=np.full_like(rating_sum, np.nan), where=n_ratings > 0)
    rated_items = flags.loc[flags.index.isin(train["item_id"].unique())]
    out = pd.DataFrame({
        "genre": pd.Series([c.removeprefix("genre_") for c in GENRE_COLS], dtype="string"),
        "n_items": flags.sum(axis=0).to_numpy(dtype=np.int64),
        "n_rated_items": rated_items.sum(axis=0).to_numpy(dtype=np.int64),
        "n_ratings": n_ratings.astype(np.int64),
        "share_of_ratings": n_ratings / max(len(train), 1),
        "mean_rating": mean_rating,
    })
    return out.sort_values(["n_ratings", "genre"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- matrix
def build_interaction_matrix(train: pd.DataFrame, user_ids: np.ndarray, item_ids: np.ndarray) -> sparse.csr_matrix:
    """Build a CSR users x items matrix of explicit ratings with the given row/column id order."""
    if train.duplicated(["user_id", "item_id"]).any():
        raise ValueError("train has duplicate (user_id, item_id) pairs; CSR construction would sum them")
    rows = pd.Index(user_ids).get_indexer(train["user_id"])
    cols = pd.Index(item_ids).get_indexer(train["item_id"])
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("train references user/item ids that are missing from the matrix index")
    values = train["rating"].to_numpy(dtype=np.float64)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))


def matrix_summary(matrix: sparse.csr_matrix) -> dict:
    """Shape, non-zeros, density/sparsity and empty rows/columns of a sparse matrix."""
    n_cells = matrix.shape[0] * matrix.shape[1]
    density = matrix.nnz / n_cells if n_cells else 0.0
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "density": round(density, 6),
        "sparsity": round(1.0 - density, 6),
        "empty_rows": int((matrix.getnnz(axis=1) == 0).sum()),
        "empty_cols": int((matrix.getnnz(axis=0) == 0).sum()),
    }


# --------------------------------------------------------------------------- entrypoint
def run(cfg: dict | None = None) -> dict:
    """Run the transformation stage on the train split and return a JSON-serialisable summary."""
    cfg = cfg or load_config()
    train = read_parquet(path(cfg, "processed_dir", "train.parquet"))
    users = read_parquet(path(cfg, "processed_dir", "users_clean.parquet"))
    items = read_parquet(path(cfg, "processed_dir", "items_clean.parquet"))
    log.info("Loaded train=%d users=%d items=%d", len(train), len(users), len(items))

    datasets = {
        "user_activity_summary": user_activity_summary(train),
        "item_popularity": item_popularity(train, float(cfg["features"]["min_item_ratings_for_avg"])),
        "genre_popularity": genre_popularity(train, items),
    }
    for name, df in datasets.items():
        write_parquet(df, path(cfg, "processed_dir", f"{name}.parquet"))

    user_ids = np.sort(users["user_id"].to_numpy(dtype=np.int64))
    item_ids = np.sort(items["item_id"].to_numpy(dtype=np.int64))
    matrix = build_interaction_matrix(train, user_ids, item_ids)
    sparse.save_npz(path(cfg, "processed_dir", MATRIX_FILE), matrix)
    write_json({"user_ids": user_ids.tolist(), "item_ids": item_ids.tolist(),
                "values": "explicit rating (1-5); 0 = not rated in train", "source": "train.parquet"},
               path(cfg, "processed_dir", MATRIX_INDEX_FILE))

    summary = {
        "stage": "transformation",
        "rows": {name: int(len(df)) for name, df in datasets.items()},
        "global_mean_rating": round(float(train["rating"].mean()), 6),
        "matrix": matrix_summary(matrix),
        "top_genres_by_ratings": datasets["genre_popularity"]["genre"].head(3).tolist(),
    }
    log.info("Transformation complete: matrix %s nnz=%d density=%.4f",
             summary["matrix"]["shape"], summary["matrix"]["nnz"], summary["matrix"]["density"])
    return summary


if __name__ == "__main__":
    run()
