"""Recommender models behind one common interface (stage 6).

Every model implements :class:`BaseRecommender`:

* ``fit(train_df, feature_store=None, interactions=None)`` - learn from explicit
  ``(user_id, item_id, rating)`` rows, optionally from a prebuilt :class:`InteractionMatrix`
  (the transformation stage's ``user_item_matrix.npz``) that must hold exactly those ratings;
* ``recommend(user_ids, k, exclude_seen=True) -> {user_id: [item_id, ...]}`` - top-k lists;
* ``predict(user_ids, item_ids) -> np.ndarray`` - rating estimates clipped to ``[1, 5]``;
* ``get_params()``, ``save(path)`` / ``load(path)`` (pickle).

Models
    ``popularity``  - non-personalised baseline (train interaction count or Bayesian average).
    ``item_knn``    - item-item collaborative filtering (shrunk adjusted cosine, top-k neighbours).
    ``mf_svd``      - biased matrix factorisation trained with SGD (Funk-SVD / Koren 2009).
    ``hybrid``      - ``alpha * minmax(collaborative score) + (1 - alpha) * user_item_genre_affinity``,
                      the content score being read from the feature store at request time; it can
                      reuse an already fitted base model (:meth:`HybridRecommender.use_fitted_base`).

Conventions shared by all models
    * The candidate catalogue is the set of items seen in training (there is no collaborative
      signal for other items); ids are mapped to dense indices sorted by id.
    * Users without training history (cold start) receive the popularity ranking, so every
      model can serve every user. Unknown users / items get zero biases and factors in ``predict``.
    * Ranking ties are broken by training popularity, then by item id, so results are deterministic.
    * All randomness comes from ``numpy.random.default_rng(seed)``.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
from scipy import sparse

RATING_MIN, RATING_MAX = 1.0, 5.0
REQUIRED_COLUMNS = ("user_id", "item_id", "rating")
SCORE_BATCH_USERS = 512  # bounds the dense [users x items] score block held in memory

IdsLike = Sequence[int] | np.ndarray | pd.Series


class FeatureProvider(Protocol):
    """The part of :class:`src.features.store.FeatureStore` that models consume."""

    def get_interaction_features(self, user_ids: IdsLike, item_ids: IdsLike) -> pd.DataFrame:
        """On-demand user x item features with a ``user_item_genre_affinity`` column."""


@dataclass(frozen=True)
class InteractionMatrix:
    """A users x items explicit-rating matrix with its row / column id index.

    The transformation stage materialises one for the training split over the whole catalogue
    (``user_item_matrix.npz`` + ``matrix_index.json``), so cold users / items are empty rows /
    columns; :meth:`from_frame` builds the compact equivalent of a ratings frame.

    Attributes:
        matrix: CSR ``[len(user_ids), len(item_ids)]`` of ratings (0 = not rated).
        user_ids: strictly increasing row ids.
        item_ids: strictly increasing column ids.

    Raises:
        ValueError: if the index does not fit the matrix or is not strictly increasing.
    """

    matrix: sparse.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray

    def __post_init__(self) -> None:
        """Validate the shape against the index and the index ordering."""
        if self.matrix.shape != (len(self.user_ids), len(self.item_ids)):
            raise ValueError(f"matrix shape {self.matrix.shape} does not match the index "
                             f"({len(self.user_ids)} users x {len(self.item_ids)} items)")
        for name, ids in (("user_ids", self.user_ids), ("item_ids", self.item_ids)):
            if np.any(np.diff(ids) <= 0):
                raise ValueError(f"{name} of the matrix index must be strictly increasing")

    @classmethod
    def from_frame(cls, train_df: pd.DataFrame) -> InteractionMatrix:
        """Matrix of a ratings frame over its own users and items (sorted by id)."""
        user_ids = np.unique(train_df["user_id"].to_numpy(np.int64))
        item_ids = np.unique(train_df["item_id"].to_numpy(np.int64))
        rows = index_of(user_ids, train_df["user_id"].to_numpy(np.int64))
        cols = index_of(item_ids, train_df["item_id"].to_numpy(np.int64))
        matrix = sparse.csr_matrix((train_df["rating"].to_numpy(np.float64), (rows, cols)),
                                   shape=(len(user_ids), len(item_ids)))
        return cls(matrix, user_ids, item_ids)

    def check_matches(self, train_df: pd.DataFrame) -> None:
        """Require the matrix to hold exactly the ratings of ``train_df`` (ids, nnz and values).

        ``train_df`` has unique ``(user_id, item_id)`` pairs and ratings >= 1, so equal non-zero
        counts plus equal values at every training pair mean the matrix contains nothing else.

        Raises:
            ValueError: describing the first inconsistency found.
        """
        if self.matrix.nnz != len(train_df):
            raise ValueError(f"user-item matrix has {self.matrix.nnz} non-zeros but the training frame has "
                             f"{len(train_df)} ratings; re-run the transformation stage")
        rows = index_of(self.user_ids, train_df["user_id"].to_numpy(np.int64))
        cols = index_of(self.item_ids, train_df["item_id"].to_numpy(np.int64))
        if (rows < 0).any() or (cols < 0).any():
            raise ValueError("the training frame references user / item ids missing from the matrix index; "
                             "re-run the transformation stage")
        if not np.array_equal(np.asarray(self.matrix[rows, cols]).ravel(), train_df["rating"].to_numpy(np.float64)):
            raise ValueError("user-item matrix ratings differ from the training frame; re-run the transformation stage")

    def compact(self) -> InteractionMatrix:
        """The sub-matrix of rows and columns holding at least one rating."""
        rows = np.flatnonzero(self.matrix.getnnz(axis=1))
        cols = np.flatnonzero(self.matrix.getnnz(axis=0))
        return InteractionMatrix(self.matrix[rows][:, cols].tocsr(), self.user_ids[rows], self.item_ids[cols])


@dataclass(frozen=True)
class TrainingData:
    """Training interactions mapped to dense indices.

    Attributes:
        users: user index of every rating (``0..n_users-1``).
        items: item index of every rating (``0..n_items-1``).
        ratings: explicit ratings as float64.
        n_users: number of distinct training users.
        n_items: number of distinct training items.
    """

    users: np.ndarray
    items: np.ndarray
    ratings: np.ndarray
    n_users: int
    n_items: int


# --------------------------------------------------------------------------- helpers
def as_id_array(ids: IdsLike) -> np.ndarray:
    """Coerce a sequence of ids to a 1-D int64 array."""
    return np.atleast_1d(np.asarray(ids, dtype=np.int64))


def index_of(sorted_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Position of each id in ``sorted_ids`` (-1 when absent)."""
    pos = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
    return np.where(sorted_ids[pos] == ids, pos, -1)


def take_or_default(values: np.ndarray, idx: np.ndarray, default: float = 0.0) -> np.ndarray:
    """``values[idx]`` (rows for 2-D ``values``) with ``default`` where ``idx == -1``."""
    taken = values[np.maximum(idx, 0)]
    mask = (idx >= 0).reshape((-1,) + (1,) * (values.ndim - 1))
    return np.where(mask, taken, default)


def damped_mean_offsets(index: np.ndarray, residuals: np.ndarray, size: int, damping: float) -> np.ndarray:
    """Per-group mean residual shrunk towards 0: ``sum(residual) / (count + damping)``."""
    totals = np.bincount(index, weights=residuals, minlength=size)
    counts = np.bincount(index, minlength=size)
    return totals / (counts + damping)


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Scale every row to ``[0, 1]``; constant rows become 0."""
    low = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - low
    return np.divide(scores - low, span, out=np.zeros_like(scores), where=span > 0)


def shrunk_cosine_similarity(values: sparse.csr_matrix, support: sparse.csr_matrix,
                             shrinkage: float) -> np.ndarray:
    """Dense column-column cosine similarity with significance shrinkage.

    ``sim(i, j) = cos(values[:, i], values[:, j]) * n_ij / (n_ij + shrinkage)`` where ``n_ij``
    is the number of users who rated both items (from the 0/1 ``support`` matrix). The
    diagonal is zeroed so an item is never its own neighbour.
    """
    gram = (values.T @ values).toarray()
    norms = np.sqrt(np.clip(np.diag(gram), 0.0, None))
    denom = np.outer(norms, norms)
    sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    co_counts = (support.T @ support).toarray()
    sim *= co_counts / (co_counts + shrinkage)
    np.fill_diagonal(sim, 0.0)
    return sim


def top_k_neighbours(sim: np.ndarray, k: int) -> sparse.csr_matrix:
    """Keep, for each row, its ``k`` largest positive similarities (sparse CSR)."""
    n = sim.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return sparse.csr_matrix(sim.shape)
    cols = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(sim, cols, axis=1)
    keep = vals > 0
    rows = np.repeat(np.arange(n), k).reshape(n, k)
    return sparse.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=sim.shape)


def _row_sums(matrix: sparse.spmatrix) -> np.ndarray:
    """Row sums of a sparse matrix as a flat float array."""
    return np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()


# --------------------------------------------------------------------------- base class
class BaseRecommender(ABC):
    """Common fit / recommend / predict / persist logic; subclasses implement the scoring.

    Subclasses define ``name``, ``param_names`` (constructor hyperparameters reported by
    :meth:`get_params`), :meth:`_fit`, :meth:`_score` and :meth:`_predict`.
    """

    name: ClassVar[str]
    param_names: ClassVar[tuple[str, ...]] = ()
    requires_feature_store: ClassVar[bool] = False

    def __init__(self, seed: int = 42) -> None:
        """Store the seed; fitted state is created by :meth:`fit`."""
        self.seed = int(seed)
        self.user_ids_: np.ndarray | None = None
        self.item_ids_: np.ndarray | None = None
        self.ratings_: sparse.csr_matrix | None = None
        self.item_counts_: np.ndarray | None = None
        self.popularity_order_: np.ndarray | None = None
        self.epoch_rmse_: list[float] = []
        self._feature_store: FeatureProvider | None = None

    # ------------------------------------------------------------------ public API
    def fit(self, train_df: pd.DataFrame, feature_store: FeatureProvider | None = None,
            interactions: InteractionMatrix | None = None) -> BaseRecommender:
        """Learn the model from explicit ratings (one row per ``(user_id, item_id)``).

        The model state (id index, rating matrix, item counts) is taken from ``interactions``
        restricted to its non-empty rows / columns. The per-rating triplets used by SGD and the
        bias estimates keep ``train_df``'s row order (values read from the matrix): the seeded
        SGD visiting order and floating-point sums depend on it, so fitting from the prebuilt
        matrix gives exactly the same model as fitting from the frame alone.

        Args:
            train_df: frame with ``user_id``, ``item_id`` and ``rating`` columns.
            feature_store: feature access layer; required by models that consume features.
            interactions: prebuilt matrix holding exactly the ratings of ``train_df`` (e.g. the
                transformation stage's ``user_item_matrix.npz``); built from ``train_df`` if omitted.

        Returns:
            ``self`` (fitted).

        Raises:
            ValueError: on an invalid training frame or a matrix inconsistent with it.
        """
        self._validate_training_frame(train_df)
        self.attach_feature_store(feature_store)
        if interactions is None:
            interactions = InteractionMatrix.from_frame(train_df)
        interactions.check_matches(train_df)
        compact = interactions.compact()
        self.user_ids_, self.item_ids_, self.ratings_ = compact.user_ids, compact.item_ids, compact.matrix
        users = index_of(self.user_ids_, train_df["user_id"].to_numpy(np.int64))
        items = index_of(self.item_ids_, train_df["item_id"].to_numpy(np.int64))
        data = TrainingData(users=users, items=items, ratings=np.asarray(self.ratings_[users, items]).ravel(),
                            n_users=len(self.user_ids_), n_items=len(self.item_ids_))
        self.item_counts_ = np.bincount(data.items, minlength=data.n_items)
        self.popularity_order_ = np.lexsort((self.item_ids_, -self.item_counts_))
        self._fit(train_df, data)
        return self

    def recommend(self, user_ids: IdsLike, k: int, exclude_seen: bool = True) -> dict[int, list[int]]:
        """Top-``k`` item ids per user (best first); cold users get the popularity ranking.

        Args:
            user_ids: users to serve (duplicates collapse into one dict entry).
            k: list length (fewer items are returned only if the catalogue runs out).
            exclude_seen: drop items the user rated in training.
        """
        self._check_fitted()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        users = np.unique(as_id_array(user_ids))
        idx = index_of(self.user_ids_, users)
        cold_list = self._rank(self._fallback_scores()[None, :], k)[0]
        recs = {int(u): list(cold_list) for u in users[idx < 0]}
        warm_users, warm_idx = users[idx >= 0], idx[idx >= 0]
        for start in range(0, len(warm_idx), SCORE_BATCH_USERS):
            batch = warm_idx[start:start + SCORE_BATCH_USERS]
            scores = np.array(self._score(batch), dtype=np.float64)
            if exclude_seen:
                rows, cols = self.ratings_[batch].nonzero()
                scores[rows, cols] = -np.inf
            ranked = self._rank(scores, k)
            recs.update({int(u): items for u, items in zip(warm_users[start:start + len(batch)], ranked)})
        return recs

    def predict(self, user_ids: IdsLike, item_ids: IdsLike) -> np.ndarray:
        """Rating estimates for aligned ``(user, item)`` pairs, clipped to ``[1, 5]``."""
        self._check_fitted()
        users, items = as_id_array(user_ids), as_id_array(item_ids)
        if len(users) != len(items):
            raise ValueError(f"user_ids ({len(users)}) and item_ids ({len(items)}) must be aligned")
        estimates = self._predict(index_of(self.user_ids_, users), index_of(self.item_ids_, items))
        return np.clip(estimates, RATING_MIN, RATING_MAX)

    def is_known_user(self, user_ids: IdsLike) -> np.ndarray:
        """Boolean mask: True for users with training history (served by the model itself)."""
        self._check_fitted()
        return index_of(self.user_ids_, as_id_array(user_ids)) >= 0

    def get_params(self) -> dict[str, Any]:
        """Model name, seed and constructor hyperparameters (MLflow-loggable)."""
        return {"model": self.name, "seed": self.seed, **{p: getattr(self, p) for p in self.param_names}}

    def attach_feature_store(self, feature_store: FeatureProvider | None) -> None:
        """(Re-)attach the feature access layer; raises if this model needs one and none is given."""
        if self.requires_feature_store and feature_store is None:
            raise ValueError(f"{self.name} needs a feature store (src.features.store.FeatureStore)")
        self._feature_store = feature_store

    def save(self, path: str | Path) -> Path:
        """Pickle the fitted model (the feature store is not persisted; re-attach it on load)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: str | Path, feature_store: FeatureProvider | None = None) -> BaseRecommender:
        """Unpickle a model written by :meth:`save` (trusted pipeline artefacts only).

        Raises:
            TypeError: if the file does not hold an instance of ``cls``.
        """
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, cls):
            raise TypeError(f"{path} holds {type(model).__name__}, expected {cls.__name__}")
        model.attach_feature_store(feature_store)
        return model

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the (large, re-loadable) feature store from the pickle."""
        state = self.__dict__.copy()
        state["_feature_store"] = None
        return state

    # ------------------------------------------------------------------ subclass hooks
    @abstractmethod
    def _fit(self, train_df: pd.DataFrame, data: TrainingData) -> None:
        """Model-specific training on the indexed data."""

    @abstractmethod
    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        """Dense ``[len(user_idx), n_items]`` ranking scores for known users (higher = better)."""

    @abstractmethod
    def _predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Unclipped rating estimates; indices are -1 for unknown users / items."""

    def _fallback_scores(self) -> np.ndarray:
        """Item scores used for cold-start users (training interaction count)."""
        return self.item_counts_.astype(np.float64)

    # ------------------------------------------------------------------ internals
    def _rank(self, scores: np.ndarray, k: int) -> list[list[int]]:
        """Top-k item ids per row; ties broken by popularity then id; ``-inf`` entries dropped."""
        order = self.popularity_order_
        top = order[np.argsort(-scores[:, order], axis=1, kind="stable")[:, :k]]
        finite = np.isfinite(np.take_along_axis(scores, top, axis=1))
        return [self.item_ids_[row[ok]].tolist() for row, ok in zip(top, finite)]

    def _check_fitted(self) -> None:
        """Raise if :meth:`fit` has not been called."""
        if self.user_ids_ is None:
            raise RuntimeError(f"{self.name} is not fitted; call fit() first")

    @staticmethod
    def _validate_training_frame(train_df: pd.DataFrame) -> None:
        """Require the rating columns, at least one row and unique ``(user_id, item_id)`` pairs."""
        missing = [c for c in REQUIRED_COLUMNS if c not in train_df.columns]
        if missing:
            raise ValueError(f"training frame lacks columns {missing}")
        if train_df.empty:
            raise ValueError("training frame is empty")
        if train_df.duplicated(["user_id", "item_id"]).any():
            raise ValueError("training frame has duplicate (user_id, item_id) pairs")


# --------------------------------------------------------------------------- models
class PopularityRecommender(BaseRecommender):
    """Non-personalised baseline: the same most-popular unseen items for everybody.

    ``score='count'`` ranks by the number of training ratings; ``score='bayesian_avg'`` ranks by
    the damped mean rating ``(n * mean + m * mu) / (n + m)``, which equals the rating
    prediction ``mu + b_i`` with ``b_i = sum(r - mu) / (n_i + damping)``.
    """

    name = "popularity"
    param_names = ("score", "damping")
    SCORES = ("count", "bayesian_avg")

    def __init__(self, score: str = "count", damping: float = 10.0, seed: int = 42) -> None:
        """Configure the ranking criterion and the item-bias damping (pseudo-count)."""
        super().__init__(seed)
        if score not in self.SCORES:
            raise ValueError(f"score must be one of {self.SCORES}, got {score!r}")
        self.score = score
        self.damping = float(damping)

    def _fit(self, train_df: pd.DataFrame, data: TrainingData) -> None:
        """Global mean, damped item biases and the item ranking scores."""
        self.global_mean_ = float(data.ratings.mean())
        self.item_bias_ = damped_mean_offsets(data.items, data.ratings - self.global_mean_,
                                              data.n_items, self.damping)
        self.scores_ = (self.item_counts_.astype(np.float64) if self.score == "count"
                        else self.global_mean_ + self.item_bias_)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        """Identical item scores for every user."""
        return np.broadcast_to(self.scores_, (len(user_idx), len(self.scores_)))

    def _predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Global mean plus the damped item bias (0 for unknown items)."""
        return self.global_mean_ + take_or_default(self.item_bias_, item_idx)

    def _fallback_scores(self) -> np.ndarray:
        """Cold users get the same ranking as everybody else."""
        return self.scores_


class ItemKNNRecommender(BaseRecommender):
    """Item-item collaborative filtering on user-mean-centred ratings.

    Similarity: adjusted cosine (columns of the user-centred rating matrix) multiplied by the
    significance shrinkage ``n_ij / (n_ij + shrinkage)``; each item keeps its ``k_neighbors``
    most similar items with positive similarity (``N``).

    * Ranking: ``score(u, i) = sum_{j in I_u} N[i, j] * r_uj`` - similarity mass towards the
      user's rated items, weighted by how much they were liked (every rating is a signal of
      interest; higher ratings count more).
    * Rating: ``r_hat(u, i) = mean_u + sum_j N[i, j] (r_uj - mean_u) / sum_j N[i, j]`` over the
      neighbours rated by ``u``; the damped baseline ``mu + b_u + b_i`` when there are none.
    """

    name = "item_knn"
    param_names = ("k_neighbors", "similarity", "shrinkage", "baseline_damping")
    SIMILARITIES = ("cosine",)

    def __init__(self, k_neighbors: int = 50, similarity: str = "cosine", shrinkage: float = 10.0,
                 baseline_damping: float = 10.0, seed: int = 42) -> None:
        """Configure neighbourhood size, similarity, shrinkage and baseline damping."""
        super().__init__(seed)
        if similarity not in self.SIMILARITIES:
            raise ValueError(f"similarity must be one of {self.SIMILARITIES}, got {similarity!r}")
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        self.k_neighbors = int(k_neighbors)
        self.similarity = similarity
        self.shrinkage = float(shrinkage)
        self.baseline_damping = float(baseline_damping)

    def _fit(self, train_df: pd.DataFrame, data: TrainingData) -> None:
        """User means, centred ratings, pruned item-item similarities and baseline biases."""
        support = self.ratings_.copy()
        support.data[:] = 1.0
        self.support_ = support
        self.user_mean_ = _row_sums(self.ratings_) / _row_sums(support)
        centred = self.ratings_.copy()
        centred.data -= np.repeat(self.user_mean_, np.diff(centred.indptr))
        self.centred_ = centred
        sim = shrunk_cosine_similarity(centred.tocsc(), support.tocsc(), self.shrinkage)
        self.neighbours_ = top_k_neighbours(sim, self.k_neighbors)
        self.global_mean_ = float(data.ratings.mean())
        self.item_bias_ = damped_mean_offsets(data.items, data.ratings - self.global_mean_,
                                              data.n_items, self.baseline_damping)
        user_resid = data.ratings - self.global_mean_ - self.item_bias_[data.items]
        self.user_bias_ = damped_mean_offsets(data.users, user_resid, data.n_users, self.baseline_damping)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        """Rating-weighted similarity mass ``R_u @ N.T``."""
        return (self.ratings_[user_idx] @ self.neighbours_.T).toarray()

    def _predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Neighbourhood estimate where available, damped baseline otherwise."""
        estimate = (self.global_mean_ + take_or_default(self.user_bias_, user_idx)
                    + take_or_default(self.item_bias_, item_idx))
        known = np.flatnonzero((user_idx >= 0) & (item_idx >= 0))
        if len(known):
            u, i = user_idx[known], item_idx[known]
            weights = self.neighbours_[i]
            numerator = _row_sums(weights.multiply(self.centred_[u]))
            denominator = _row_sums(weights.multiply(self.support_[u]))
            has_neighbours = denominator > 0
            knn = self.user_mean_[u] + np.divide(numerator, denominator, out=np.zeros_like(numerator),
                                                 where=has_neighbours)
            estimate[known] = np.where(has_neighbours, knn, estimate[known])
        return estimate


class MatrixFactorizationRecommender(BaseRecommender):
    """Biased matrix factorisation ``r_hat = mu + b_u + b_i + p_u . q_i`` trained with SGD.

    Plain SGD over shuffled ratings (seeded) with L2 regularisation ``reg`` on biases and
    factors, learning rate ``lr`` and factors drawn from ``N(0, init_std)``. The training RMSE
    after every epoch is recorded in ``epoch_rmse_``. Ranking uses the predicted rating.
    """

    name = "mf_svd"
    param_names = ("n_factors", "n_epochs", "lr", "reg", "init_std")

    def __init__(self, n_factors: int = 50, n_epochs: int = 30, lr: float = 0.007, reg: float = 0.05,
                 init_std: float = 0.1, seed: int = 42) -> None:
        """Configure the latent dimension and the SGD schedule."""
        super().__init__(seed)
        self.n_factors = int(n_factors)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.reg = float(reg)
        self.init_std = float(init_std)

    def _fit(self, train_df: pd.DataFrame, data: TrainingData) -> None:
        """Run ``n_epochs`` of SGD and record the training RMSE per epoch."""
        rng = np.random.default_rng(self.seed)
        self.global_mean_ = float(data.ratings.mean())
        self.user_bias_ = np.zeros(data.n_users)
        self.item_bias_ = np.zeros(data.n_items)
        self.user_factors_ = rng.normal(0.0, self.init_std, (data.n_users, self.n_factors))
        self.item_factors_ = rng.normal(0.0, self.init_std, (data.n_items, self.n_factors))
        self.epoch_rmse_ = []
        for _ in range(self.n_epochs):
            self._sgd_epoch(data, rng.permutation(len(data.ratings)))
            residual = data.ratings - self._predict(data.users, data.items)
            self.epoch_rmse_.append(float(np.sqrt(np.mean(residual ** 2))))

    def _sgd_epoch(self, data: TrainingData, order: np.ndarray) -> None:
        """One pass of per-rating SGD updates (lean scalar loop over in-place row views)."""
        mu, lr, reg = self.global_mean_, self.lr, self.reg
        bu, bi, P, Q = self.user_bias_, self.item_bias_, self.user_factors_, self.item_factors_
        for u, i, r in zip(data.users[order].tolist(), data.items[order].tolist(), data.ratings[order].tolist()):
            pu, qi = P[u], Q[i]
            err = r - (mu + bu[u] + bi[i] + pu.dot(qi))
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            pu_grad = err * qi - reg * pu
            qi += lr * (err * pu - reg * qi)
            pu += lr * pu_grad

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        """Predicted ratings of every catalogue item."""
        return (self.global_mean_ + self.user_bias_[user_idx, None] + self.item_bias_[None, :]
                + self.user_factors_[user_idx] @ self.item_factors_.T)

    def _predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Biased dot product; unknown users / items contribute zero bias and factors."""
        p = take_or_default(self.user_factors_, user_idx)
        q = take_or_default(self.item_factors_, item_idx)
        return (self.global_mean_ + take_or_default(self.user_bias_, user_idx)
                + take_or_default(self.item_bias_, item_idx) + np.einsum("ij,ij->i", p, q))


class HybridRecommender(BaseRecommender):
    """Weighted hybrid of a collaborative model and the genre-affinity content feature.

    ``score(u, i) = alpha * minmax_u(base score) + (1 - alpha) * user_item_genre_affinity(u, i)``
    where the min-max scaling is per user over the catalogue and the affinity (cosine of the
    user's genre-share vector and the item's genre flags, in ``[0, 1]``) is computed by the
    feature store at request time - the same feature definition at training and serving time.
    Rating predictions are those of the base model.
    """

    name = "hybrid"
    param_names = ("alpha", "base_model")
    requires_feature_store = True

    def __init__(self, alpha: float = 0.7, base_model: str = "mf_svd", base_params: dict | None = None,
                 seed: int = 42) -> None:
        """Build the (unfitted) base model; ``alpha`` weights the collaborative component."""
        super().__init__(seed)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if base_model not in MODEL_REGISTRY or base_model == self.name:
            raise ValueError(f"base_model must be a non-hybrid model of {sorted(MODEL_REGISTRY)}")
        self.alpha = float(alpha)
        self.base_model = base_model
        self.base = MODEL_REGISTRY[base_model](seed=seed, **(base_params or {}))
        self.base_reused_ = False

    def use_fitted_base(self, model: BaseRecommender) -> None:
        """Adopt an already fitted, identically configured base model; :meth:`fit` then reuses it.

        The training stage passes the ``base_model`` it fitted earlier in the same run, so the
        (deterministic, seeded) base is not trained twice; :meth:`fit` checks that it was fitted
        on the same ratings.

        Raises:
            ValueError: if ``model`` is not a fitted ``base_model`` with the same hyperparameters.
        """
        if model.name != self.base_model or model.get_params() != self.base.get_params():
            raise ValueError(f"cannot reuse {model.get_params()} as base; expected {self.base.get_params()}")
        model._check_fitted()
        self.base = model
        self.base_reused_ = True

    def get_params(self) -> dict[str, Any]:
        """Hybrid parameters plus the base model's hyperparameters prefixed with ``base.``."""
        base = {f"base.{k}": v for k, v in self.base.get_params().items() if k not in ("model", "seed")}
        return {**super().get_params(), **base}

    def _fit(self, train_df: pd.DataFrame, data: TrainingData) -> None:
        """Fit the base model on the same ratings, or check that the reused base was fitted on them.

        Raises:
            ValueError: if a reused base was fitted on other ratings than this model.
        """
        if self.base_reused_:
            same_data = (np.array_equal(self.base.user_ids_, self.user_ids_)
                         and np.array_equal(self.base.item_ids_, self.item_ids_)
                         and (self.base.ratings_ != self.ratings_).nnz == 0)
            if not same_data:
                raise ValueError(f"the reused {self.base_model} base was fitted on different ratings")
        else:
            self.base.fit(train_df, interactions=InteractionMatrix(self.ratings_, self.user_ids_, self.item_ids_))
        self.epoch_rmse_ = list(self.base.epoch_rmse_)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        """Blend of the scaled base score and the store's user x item genre affinity."""
        n_items = len(self.item_ids_)
        users = self.user_ids_[user_idx]
        affinity = self._feature_store.get_interaction_features(
            np.repeat(users, n_items), np.tile(self.item_ids_, len(users)))["user_item_genre_affinity"]
        content = affinity.to_numpy(np.float64).reshape(len(users), n_items)
        return self.alpha * minmax_rows(self.base._score(user_idx)) + (1.0 - self.alpha) * content

    def _predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        """Rating estimates of the base model."""
        return self.base._predict(user_idx, item_idx)


MODEL_REGISTRY: dict[str, type[BaseRecommender]] = {
    cls.name: cls for cls in (PopularityRecommender, ItemKNNRecommender, MatrixFactorizationRecommender,
                              HybridRecommender)
}


def model_hyperparams(models_cfg: dict, name: str) -> dict[str, Any]:
    """Constructor keyword arguments of model ``name`` from ``cfg['training']['models']``.

    Drops the ``enabled`` switch; for the hybrid, injects the base model's hyperparameters.
    """
    params = {k: v for k, v in models_cfg[name].items() if k != "enabled"}
    if name == HybridRecommender.name:
        params["base_params"] = model_hyperparams(models_cfg, params.get("base_model", "mf_svd"))
    return params


def create_model(name: str, models_cfg: dict, seed: int) -> BaseRecommender:
    """Instantiate model ``name`` with its configured hyperparameters and ``seed``."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](seed=seed, **model_hyperparams(models_cfg, name))
