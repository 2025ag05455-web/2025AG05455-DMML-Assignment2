"""Pure offline evaluation metrics for top-k recommendation and rating prediction.

Ranking metrics use binary relevance: an item is relevant to a user when it is in the user's
``relevant`` collection (in this pipeline: test items rated >= ``relevance_threshold``).
Per-user functions take one ranked list (best first); :func:`ranking_metrics` and
:func:`map_at_k` average them over users. Beyond-accuracy metrics (coverage, diversity,
novelty) take the recommendation lists of all evaluated users.

Definitions (``rec@k`` = first ``k`` recommended items, ``R`` = relevant set, ``hits`` = ``|rec@k ∩ R|``):
    precision@k   = hits / k
    recall@k      = hits / |R|
    hit_rate@k    = 1 if hits > 0 else 0
    ndcg@k        = DCG@k / IDCG@k, ``DCG = sum_r rel_r / log2(r + 1)``, IDCG over ``min(k, |R|)`` hits
    AP@k          = sum_r P@r * rel_r / min(k, |R|);  MAP@k = mean AP@k over users
    coverage      = |distinct recommended items| / |training catalogue|
    diversity     = mean over lists of (1 - mean pairwise cosine similarity of item vectors)
    novelty       = mean over lists of mean(-log2 p_i), ``p_i`` = share of users who rated item i
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

import numpy as np


def _check_k(k: int) -> None:
    """Reject non-positive cut-offs."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _hits(recommended: Sequence[int], relevant: Collection[int], k: int) -> np.ndarray:
    """0/1 relevance of the first ``k`` recommended items."""
    _check_k(k)
    return np.array([item in relevant for item in recommended[:k]], dtype=np.float64)


# --------------------------------------------------------------------------- per-user ranking metrics
def precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Fraction of the top-``k`` slots holding a relevant item (short lists are not rewarded)."""
    return float(_hits(recommended, relevant, k).sum() / k)


def recall_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Fraction of the relevant items retrieved in the top ``k`` (0 when nothing is relevant)."""
    hits = _hits(recommended, relevant, k).sum()
    return float(hits / len(relevant)) if relevant else 0.0


def hit_rate_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """1.0 if at least one relevant item is in the top ``k``, else 0.0."""
    return float(_hits(recommended, relevant, k).any())


def ndcg_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Normalised DCG with binary gains and a ``log2(rank + 1)`` discount (0 when nothing is relevant)."""
    hits = _hits(recommended, relevant, k)
    n_ideal = min(k, len(relevant))
    if n_ideal == 0:
        return 0.0
    dcg = float(np.sum(hits / np.log2(np.arange(2, len(hits) + 2))))
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, n_ideal + 2))))
    return dcg / idcg


def average_precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Mean of precision@r at each relevant rank r <= k, normalised by ``min(k, |relevant|)``."""
    hits = _hits(recommended, relevant, k)
    n_ideal = min(k, len(relevant))
    if n_ideal == 0:
        return 0.0
    precision_at_rank = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(np.sum(precision_at_rank * hits) / n_ideal)


# --------------------------------------------------------------------------- aggregated ranking metrics
PER_USER_METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hit_rate": hit_rate_at_k,
    "map": average_precision_at_k,
}


def mean_over_users(metric: Callable[[Sequence[int], Collection[int], int], float],
                    recommendations: Mapping[int, Sequence[int]], relevant: Mapping[int, Collection[int]],
                    k: int) -> float:
    """Average a per-user metric over the users of ``relevant`` (missing lists count as empty)."""
    _check_k(k)
    if not relevant:
        return 0.0
    return float(np.mean([metric(recommendations.get(u, []), rel, k) for u, rel in relevant.items()]))


def map_at_k(recommendations: Mapping[int, Sequence[int]], relevant: Mapping[int, Collection[int]],
             k: int) -> float:
    """Mean average precision@k over the users of ``relevant``."""
    return mean_over_users(average_precision_at_k, recommendations, relevant, k)


def ranking_metrics(recommendations: Mapping[int, Sequence[int]], relevant: Mapping[int, Collection[int]],
                    k: int) -> dict[str, float]:
    """``{precision, recall, ndcg, hit_rate, map}@k`` averaged over the users of ``relevant``."""
    return {f"{name}@{k}": mean_over_users(fn, recommendations, relevant, k) for name, fn in PER_USER_METRICS.items()}


# --------------------------------------------------------------------------- beyond-accuracy metrics
def catalog_coverage(recommendation_lists: Iterable[Sequence[int]], n_catalog_items: int) -> float:
    """Share of the training catalogue that appears in at least one recommendation list."""
    if n_catalog_items <= 0:
        raise ValueError("n_catalog_items must be positive")
    distinct = set().union(*map(set, recommendation_lists))
    return len(distinct) / n_catalog_items


def intra_list_diversity(recommendation_lists: Iterable[Sequence[int]],
                         item_vectors: Mapping[int, np.ndarray]) -> float:
    """Mean over lists of ``1 - mean pairwise cosine similarity`` of the items' vectors.

    Lists with fewer than two items are skipped (0.0 if none qualifies); a zero vector has
    cosine 0 with every other item.
    """
    scores = []
    for items in recommendation_lists:
        n = len(items)
        if n < 2:
            continue
        vectors = np.vstack([np.asarray(item_vectors[i], dtype=np.float64) for i in items])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        sim = unit @ unit.T
        mean_pairwise = (sim.sum() - np.trace(sim)) / (n * (n - 1))
        scores.append(1.0 - mean_pairwise)
    return float(np.mean(scores)) if scores else 0.0


def novelty(recommendation_lists: Iterable[Sequence[int]], item_share: Mapping[int, float]) -> float:
    """Mean self-information ``-log2(p_i)`` of recommended items (per list, then over lists).

    ``item_share[i]`` is the fraction of training users who rated item ``i`` (must be in ``(0, 1]``).
    """
    per_list = [np.mean(-np.log2([item_share[i] for i in items])) for items in recommendation_lists if len(items)]
    return float(np.mean(per_list)) if per_list else 0.0


# --------------------------------------------------------------------------- rating metrics
def _residuals(y_true: Sequence[float], y_pred: Sequence[float]) -> np.ndarray:
    """``y_true - y_pred`` after checking both are non-empty and aligned."""
    true, pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError(f"y_true {true.shape} and y_pred {pred.shape} must be non-empty and aligned")
    return true - pred


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean(_residuals(y_true, y_pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(_residuals(y_true, y_pred))))
