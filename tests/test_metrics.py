"""Hand-computed checks of the evaluation metrics and report helpers (src.evaluation)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics as m
from src.evaluation.evaluate import (best_values, comparison_frame, markdown_table, primary_metric_key,
                                     relevant_by_user)

# One user: hits at ranks 2 and 5 of a 5-item list; 3 relevant items (9 is never recommended).
RECS = [1, 2, 3, 4, 5]
RELEVANT = {2, 5, 9}
LOG2_3, LOG2_6 = math.log2(3), math.log2(6)
IDCG_3 = 1 + 1 / LOG2_3 + 1 / 2  # three ideal hits at ranks 1..3


@pytest.mark.parametrize("k, expected", [(5, 2 / 5), (3, 1 / 3), (1, 0.0)])
def test_precision_at_k(k: int, expected: float) -> None:
    assert m.precision_at_k(RECS, RELEVANT, k) == pytest.approx(expected)


def test_precision_divides_by_k_for_short_lists() -> None:
    assert m.precision_at_k([2], RELEVANT, 5) == pytest.approx(1 / 5)


@pytest.mark.parametrize("k, expected", [(5, 2 / 3), (3, 1 / 3)])
def test_recall_at_k(k: int, expected: float) -> None:
    assert m.recall_at_k(RECS, RELEVANT, k) == pytest.approx(expected)


def test_hit_rate_at_k() -> None:
    assert m.hit_rate_at_k(RECS, RELEVANT, 2) == 1.0
    assert m.hit_rate_at_k(RECS, RELEVANT, 1) == 0.0


def test_ndcg_at_k_hand_computed() -> None:
    expected = (1 / LOG2_3 + 1 / LOG2_6) / IDCG_3
    assert m.ndcg_at_k(RECS, RELEVANT, 5) == pytest.approx(expected)
    assert m.ndcg_at_k(RECS, RELEVANT, 3) == pytest.approx((1 / LOG2_3) / IDCG_3)


def test_ndcg_ideal_dcg_uses_min_of_k_and_relevant() -> None:
    # one relevant item placed first -> perfect even though k = 5
    assert m.ndcg_at_k([7, 1, 2], {7}, 5) == pytest.approx(1.0)
    assert m.ndcg_at_k([9, 2, 5], RELEVANT, 3) == pytest.approx(1.0)


def test_average_precision_at_k() -> None:
    # precision at the hit ranks: 1/2 and 2/5, normalised by min(5, 3) = 3
    assert m.average_precision_at_k(RECS, RELEVANT, 5) == pytest.approx((1 / 2 + 2 / 5) / 3)
    assert m.average_precision_at_k([9, 2, 5], RELEVANT, 3) == pytest.approx(1.0)


def test_empty_relevant_set_scores_zero() -> None:
    for fn in (m.precision_at_k, m.recall_at_k, m.ndcg_at_k, m.hit_rate_at_k, m.average_precision_at_k):
        assert fn(RECS, set(), 3) == 0.0


def test_non_positive_k_rejected() -> None:
    with pytest.raises(ValueError):
        m.precision_at_k(RECS, RELEVANT, 0)


def test_map_and_ranking_metrics_average_over_relevant_users() -> None:
    recs = {1: RECS}                 # user 2 has no list -> scored as an empty list
    relevant = {1: RELEVANT, 2: {4}}
    assert m.map_at_k(recs, relevant, 5) == pytest.approx(0.3 / 2)
    out = m.ranking_metrics(recs, relevant, 5)
    assert set(out) == {"precision@5", "recall@5", "ndcg@5", "hit_rate@5", "map@5"}
    assert out["precision@5"] == pytest.approx(0.2)
    assert out["hit_rate@5"] == pytest.approx(0.5)


def test_catalog_coverage() -> None:
    assert m.catalog_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        m.catalog_coverage([[1]], 0)


def test_intra_list_diversity() -> None:
    vectors = {1: np.array([1, 0]), 2: np.array([0, 1]), 3: np.array([1, 1]), 4: np.array([0, 0])}
    cos_13 = 1 / math.sqrt(2)
    assert m.intra_list_diversity([[1, 2]], vectors) == pytest.approx(1.0)
    assert m.intra_list_diversity([[1, 2], [1, 3], [1]], vectors) == pytest.approx((1 + (1 - cos_13)) / 2)
    assert m.intra_list_diversity([[1, 2, 3]], vectors) == pytest.approx(1 - 2 * cos_13 / 3)
    assert m.intra_list_diversity([[1, 4]], vectors) == pytest.approx(1.0)  # zero vector -> similarity 0
    assert m.intra_list_diversity([[1]], vectors) == 0.0


def test_novelty() -> None:
    share = {1: 0.5, 2: 0.25, 3: 1.0}
    assert m.novelty([[1, 2], [3]], share) == pytest.approx((1.5 + 0.0) / 2)


def test_rmse_and_mae() -> None:
    assert m.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(math.sqrt(4 / 3))
    assert m.mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        m.rmse([1, 2], [1])


# --------------------------------------------------------------------------- evaluation report helpers
def test_relevant_by_user_applies_threshold() -> None:
    test = pd.DataFrame({"user_id": [1, 1, 2, 3], "item_id": [10, 11, 10, 12], "rating": [4, 3, 5, 1]})
    assert relevant_by_user(test, 4) == {1: {10}, 2: {10}}


def test_primary_metric_key() -> None:
    cfg = {"evaluation": {"primary_metric": "ndcg_at_k", "k": 10}}
    assert primary_metric_key(cfg) == "ndcg@10"
    assert primary_metric_key({"evaluation": {"primary_metric": "rmse", "k": 10}}) == "rmse"


def test_comparison_frame_and_bold_best() -> None:
    def result(ndcg: float, rmse: float) -> dict:
        warm = {f"{name}@{k}": ndcg for k in (10, 5) for name in ("precision", "recall", "ndcg", "hit_rate", "map")}
        warm.update(coverage=0.1, diversity=0.5, novelty=2.0, rmse=rmse, mae=rmse)
        return {"warm": warm}

    frame = comparison_frame({"a": result(0.2, 1.1), "b": result(0.1, 0.9)}, 10, [5])
    assert frame.columns[:11].tolist() == ["model", "precision@10", "recall@10", "ndcg@10", "hit_rate@10", "map@10",
                                           "coverage", "diversity", "novelty", "rmse", "mae"]
    assert best_values(frame)["ndcg@10"] == 0.2 and best_values(frame)["rmse"] == 0.9
    table = markdown_table(frame)
    assert "| a | **0.2000** |" in table and "| b | 0.1000 |" in table
    assert "**0.9000**" in table.splitlines()[3]
