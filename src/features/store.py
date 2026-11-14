"""Lightweight offline feature store (Feast-like) - the single feature access layer.

Training, evaluation and prediction/serving all read features through :class:`FeatureStore`,
so a model always sees features computed by exactly the same code, version and ``as_of``
(no training/serving skew). The store

* lazily loads and caches the materialised tables in ``features_dir``,
* validates them against the registry and checks the feature-set version against the config,
* returns rows in the requested id order, filling unknown ids with the documented defaults
  from ``feature_metadata.json`` and flagging them with ``is_cold``,
* computes user x item features on demand (they are never materialised).
"""
from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.features.registry import (METADATA_COLUMNS, TABLE_KEYS, feature_columns, validate_feature_frame)
from src.ingestion.ingest import GENRE_COLS
from src.utils.config import load_config, path
from src.utils.io import read_json, read_parquet

IdsLike = Sequence[int] | np.ndarray | pd.Series
COLD_INDICATOR = {"user_features": "user_n_ratings", "item_features": "item_n_ratings"}
PAIR_KEY_BASE = np.int64(2**31)  # item ids are < 2**31, so user * BASE + item is a unique int64


def _as_ids(ids: IdsLike) -> np.ndarray:
    """Coerce a sequence of ids to a 1-D int64 array."""
    return np.atleast_1d(np.asarray(ids, dtype=np.int64))


def _pair_keys(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    """Encode aligned (user_id, item_id) pairs as unique int64 keys."""
    if len(item_ids) and (item_ids.min() < 0 or item_ids.max() >= PAIR_KEY_BASE):
        raise ValueError("item ids must be in [0, 2**31) for pair encoding")
    return user_ids * PAIR_KEY_BASE + item_ids


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two equally shaped matrices (0 where a row is all zeros)."""
    dot = np.einsum("ij,ij->i", a, b)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(dot, norms, out=np.zeros_like(dot), where=norms > 0)


class FeatureStore:
    """Read-only access to the point-in-time feature set built by ``src.features.build_features``.

    Args:
        cfg: pipeline config (``load_config()`` when omitted); ``paths.features_dir`` locates the
            tables and ``paths.processed_dir`` the train history used for ``item_seen_by_user``.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()
        self._tables: dict[str, pd.DataFrame] = {}
        self._metadata: dict | None = None
        self._seen_keys: np.ndarray | None = None

    # ------------------------------------------------------------------ loading
    def _meta(self) -> dict:
        """Load (once) and version-check ``feature_metadata.json``."""
        if self._metadata is None:
            meta = read_json(path(self.cfg, "features_dir", "feature_metadata.json"))
            expected = str(self.cfg["features"]["feature_set_version"])
            if meta["version"] != expected:
                raise ValueError(f"Feature set version {meta['version']!r} does not match config {expected!r}; "
                                 "rebuild features with src.features.build_features")
            self._metadata = meta
        return self._metadata

    def _table(self, name: str) -> pd.DataFrame:
        """Load (once), validate and cache a feature table indexed by its key."""
        if name not in self._tables:
            df = read_parquet(path(self.cfg, "features_dir", f"{name}.parquet"))
            validate_feature_frame(df, name)
            if not df["feature_set_version"].eq(self._meta()["version"]).all():
                raise ValueError(f"{name}: rows with a feature_set_version other than {self._meta()['version']!r}")
            self._tables[name] = df.drop(columns=list(METADATA_COLUMNS)).set_index(TABLE_KEYS[name])
        return self._tables[name]

    def _seen_pair_keys(self) -> np.ndarray:
        """Sorted unique keys of the (user, item) pairs observed up to ``as_of`` (train history)."""
        if self._seen_keys is None:
            train = read_parquet(path(self.cfg, "processed_dir", "train.parquet"))
            history = train[train["timestamp"] <= self._meta()["as_of_timestamp"]]
            self._seen_keys = np.unique(_pair_keys(history["user_id"].to_numpy(np.int64),
                                                   history["item_id"].to_numpy(np.int64)))
        return self._seen_keys

    # ------------------------------------------------------------------ entity lookups
    def _lookup(self, table: str, ids: IdsLike, columns: Sequence[str] | None) -> pd.DataFrame:
        """Rows of ``table`` for ``ids`` (request order, duplicates allowed) with defaults and ``is_cold``."""
        frame = self._table(table)
        cols = list(columns) if columns is not None else feature_columns(table)
        unknown = [c for c in cols if c not in frame.columns]
        if unknown:
            raise KeyError(f"Unknown {table} columns: {unknown}")
        keys = _as_ids(ids)
        defaults = self._meta()["defaults"][table]
        out = (frame[cols].reindex(keys)
               .fillna({c: defaults[c] for c in cols})
               .astype(frame[cols].dtypes.to_dict())
               .reset_index(drop=True))
        out.insert(0, TABLE_KEYS[table], keys)
        out["is_cold"] = frame[COLD_INDICATOR[table]].reindex(keys).fillna(0).eq(0).to_numpy()
        return out

    def get_user_features(self, user_ids: IdsLike, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """User features for ``user_ids`` in request order: ``user_id``, the features, ``is_cold``.

        ``is_cold`` is True for users without history before ``as_of`` (including unknown ids,
        which receive the defaults from ``feature_metadata.json``).
        """
        return self._lookup("user_features", user_ids, columns)

    def get_item_features(self, item_ids: IdsLike, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Item features for ``item_ids`` in request order: ``item_id``, the features, ``is_cold``."""
        return self._lookup("item_features", item_ids, columns)

    def get_user_genre_affinity(self, user_ids: IdsLike) -> np.ndarray:
        """``[n_users, 19]`` float64 genre-share matrix (columns in ``GENRE_COLS`` order).

        Unknown users receive the global genre distribution (the cold-start prior).
        """
        prior = pd.Series(self._meta()["defaults"]["user_genre_affinity"])[GENRE_COLS]
        return (self._table("user_genre_affinity")[GENRE_COLS]
                .reindex(_as_ids(user_ids)).fillna(prior).to_numpy(dtype=np.float64))

    def get_item_genre_matrix(self, item_ids: IdsLike) -> np.ndarray:
        """``[n_items, 19]`` float64 genre one-hot matrix (columns in ``GENRE_COLS`` order; unknown = 0)."""
        return self.get_item_features(item_ids, GENRE_COLS)[GENRE_COLS].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ user x item features
    def get_interaction_features(self, user_ids: IdsLike, item_ids: IdsLike) -> pd.DataFrame:
        """On-demand features of aligned (user, item) pairs.

        Returns ``user_id, item_id, user_item_genre_affinity, user_item_avg_rating_gap,
        item_seen_by_user`` with one row per input pair (input order).
        """
        users, items = _as_ids(user_ids), _as_ids(item_ids)
        if len(users) != len(items):
            raise ValueError(f"user_ids ({len(users)}) and item_ids ({len(items)}) must be aligned")
        affinity = cosine_rows(self.get_user_genre_affinity(users), self.get_item_genre_matrix(items))
        gap = (self.get_user_features(users, ["user_avg_rating"])["user_avg_rating"].to_numpy()
               - self.get_item_features(items, ["item_avg_rating"])["item_avg_rating"].to_numpy())
        seen = np.isin(_pair_keys(users, items), self._seen_pair_keys())
        return pd.DataFrame({
            "user_id": users,
            "item_id": items,
            "user_item_genre_affinity": affinity,
            "user_item_avg_rating_gap": gap,
            "item_seen_by_user": seen.astype(np.int8),
        })

    def build_training_frame(self, interactions: pd.DataFrame, user_columns: Sequence[str] | None = None,
                             item_columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Join user, item and on-demand user x item features onto an interactions frame.

        ``interactions`` needs ``user_id`` and ``item_id``; all its columns are kept. The cold
        flags are returned as ``user_is_cold`` / ``item_is_cold``.

        Raises:
            ValueError: if ``interactions`` already contains a feature column.
        """
        base = interactions.reset_index(drop=True)
        parts = [
            self.get_user_features(base["user_id"], user_columns)
                .drop(columns="user_id").rename(columns={"is_cold": "user_is_cold"}),
            self.get_item_features(base["item_id"], item_columns)
                .drop(columns="item_id").rename(columns={"is_cold": "item_is_cold"}),
            self.get_interaction_features(base["user_id"], base["item_id"]).drop(columns=["user_id", "item_id"]),
        ]
        overlap = set(base.columns) & set().union(*(p.columns for p in parts))
        if overlap:
            raise ValueError(f"interactions already contain feature columns: {sorted(overlap)}")
        return pd.concat([base, *parts], axis=1)

    # ------------------------------------------------------------------ metadata
    def metadata(self) -> dict:
        """A copy of ``feature_metadata.json`` (version, as_of, row counts, feature names, defaults)."""
        return copy.deepcopy(self._meta())
