"""Unit tests of the recommender models and training helpers on a tiny synthetic frame."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.training.models import (MODEL_REGISTRY, BaseRecommender, HybridRecommender, ItemKNNRecommender,
                                 MatrixFactorizationRecommender, PopularityRecommender, create_model,
                                 shrunk_cosine_similarity, top_k_neighbours)
from src.training.tracking import resolve_artifact_location, resolve_tracking_uri
from src.training.train import enabled_models

# 4 users x 6 items; train counts: item 10 -> 3, 11 -> 3, 12/13/14 -> 2, 15 -> 1
RATINGS = [(1, 10, 5), (1, 11, 4), (1, 12, 1),
           (2, 10, 4), (2, 11, 5), (2, 13, 2),
           (3, 11, 4), (3, 12, 2), (3, 14, 5),
           (4, 10, 5), (4, 13, 3), (4, 14, 4), (4, 15, 1)]
POPULARITY_ORDER = [10, 11, 12, 13, 14, 15]  # count desc, ties by item id
COLD_USER = 999


class StubFeatureStore:
    """Feature-store stand-in: genre affinity 1.0 for ``preferred_item``, 0.0 elsewhere."""

    def __init__(self, preferred_item: int = 15) -> None:
        self.preferred_item = preferred_item

    def get_interaction_features(self, user_ids, item_ids) -> pd.DataFrame:
        items = np.asarray(item_ids)
        return pd.DataFrame({"user_id": np.asarray(user_ids), "item_id": items,
                             "user_item_genre_affinity": (items == self.preferred_item).astype(float)})


MF_PARAMS = {"n_factors": 4, "n_epochs": 15, "lr": 0.02, "reg": 0.02, "init_std": 0.1}
FACTORIES: dict[str, Callable[[], BaseRecommender]] = {
    "popularity": lambda: PopularityRecommender(damping=1.0),
    "item_knn": lambda: ItemKNNRecommender(k_neighbors=2, shrinkage=1.0, baseline_damping=1.0),
    "mf_svd": lambda: MatrixFactorizationRecommender(**MF_PARAMS),
    "hybrid": lambda: HybridRecommender(alpha=0.5, base_params=MF_PARAMS),
}


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(RATINGS, columns=["user_id", "item_id", "rating"])


def fitted(name: str, train: pd.DataFrame) -> BaseRecommender:
    return FACTORIES[name]().fit(train, feature_store=StubFeatureStore())


def seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("user_id")["item_id"].apply(set).to_dict()


# --------------------------------------------------------------------------- shared interface
@pytest.mark.parametrize("name", list(FACTORIES))
def test_recommend_returns_k_unseen_catalogue_items(name: str, train: pd.DataFrame) -> None:
    recs = fitted(name, train).recommend([1, 2, 3, 4], k=2)
    seen = seen_items(train)
    assert set(recs) == {1, 2, 3, 4}
    for user, items in recs.items():
        assert len(items) == 2 and len(set(items)) == 2
        assert not set(items) & seen[user]
        assert set(items) <= set(POPULARITY_ORDER)


@pytest.mark.parametrize("name", list(FACTORIES))
def test_cold_user_gets_popularity_fallback(name: str, train: pd.DataFrame) -> None:
    model = fitted(name, train)
    assert model.recommend([COLD_USER], k=3)[COLD_USER] == POPULARITY_ORDER[:3]
    assert not model.is_known_user([COLD_USER])[0] and model.is_known_user([1])[0]


@pytest.mark.parametrize("name", list(FACTORIES))
def test_list_is_truncated_when_unseen_catalogue_runs_out(name: str, train: pd.DataFrame) -> None:
    assert sorted(fitted(name, train).recommend([4], k=10)[4]) == [11, 12]  # user 4 rated 4 of 6 items


@pytest.mark.parametrize("name", list(FACTORIES))
def test_predict_is_clipped_and_handles_unknown_ids(name: str, train: pd.DataFrame) -> None:
    preds = fitted(name, train).predict([1, 2, COLD_USER, 1], [10, 15, 10, 12345])
    assert preds.shape == (4,)
    assert np.all(np.isfinite(preds)) and np.all((preds >= 1.0) & (preds <= 5.0))


@pytest.mark.parametrize("name", list(FACTORIES))
def test_save_load_roundtrip(name: str, train: pd.DataFrame, tmp_path: Path) -> None:
    model = fitted(name, train)
    path = model.save(tmp_path / name / "model.pkl")
    loaded = BaseRecommender.load(path, feature_store=StubFeatureStore())
    assert type(loaded) is type(model) and loaded.get_params() == model.get_params()
    assert loaded.recommend([1, 2, COLD_USER], k=3) == model.recommend([1, 2, COLD_USER], k=3)
    np.testing.assert_allclose(loaded.predict([1, 3], [13, 10]), model.predict([1, 3], [13, 10]))


def test_load_rejects_other_model_class(train: pd.DataFrame, tmp_path: Path) -> None:
    path = fitted("popularity", train).save(tmp_path / "model.pkl")
    with pytest.raises(TypeError):
        ItemKNNRecommender.load(path)


def test_input_validation(train: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        PopularityRecommender().recommend([1], k=1)
    with pytest.raises(ValueError, match="duplicate"):
        PopularityRecommender().fit(pd.concat([train, train.head(1)]))
    with pytest.raises(ValueError, match="columns"):
        PopularityRecommender().fit(train.drop(columns="rating"))
    with pytest.raises(ValueError):
        fitted("popularity", train).recommend([1], k=0)
    with pytest.raises(ValueError):
        fitted("popularity", train).predict([1, 2], [10])


# --------------------------------------------------------------------------- model-specific behaviour
def test_popularity_ranking_and_damped_prediction(train: pd.DataFrame) -> None:
    model = fitted("popularity", train)
    assert model.recommend([1], k=3, exclude_seen=False)[1] == [10, 11, 12]
    mu = train["rating"].mean()
    # item 15: one rating of 1, damping 1 -> mu + (1 - mu) / 2; unknown item -> mu
    np.testing.assert_allclose(model.predict([1, 1], [15, 12345]), [mu + (1 - mu) / 2, mu])


def test_popularity_bayesian_average_score(train: pd.DataFrame) -> None:
    model = PopularityRecommender(score="bayesian_avg", damping=1.0).fit(train)
    mu = train["rating"].mean()
    stats = train.groupby("item_id")["rating"].agg(["sum", "size"])
    bayes = (stats["sum"] + mu) / (stats["size"] + 1)
    expected = bayes.sort_values(ascending=False, kind="stable").index.tolist()
    assert model.recommend([COLD_USER], k=6)[COLD_USER] == expected
    with pytest.raises(ValueError):
        PopularityRecommender(score="median")


def test_shrunk_cosine_and_top_k_neighbours() -> None:
    values = sparse.csc_matrix(np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    sim = shrunk_cosine_similarity(values, values, shrinkage=1.0)
    np.testing.assert_allclose(sim, [[0, 2 / 3, 0], [2 / 3, 0, 0], [0, 0, 0]])  # cos 1 * 2 / (2 + 1)
    neighbours = top_k_neighbours(sim, k=1).toarray()
    np.testing.assert_allclose(neighbours, [[0, 2 / 3, 0], [2 / 3, 0, 0], [0, 0, 0]])


def test_item_knn_prediction_neighbour_estimate_or_baseline(train: pd.DataFrame) -> None:
    model = fitted("item_knn", train)
    branches = set()
    for u, user in enumerate(model.user_ids_):
        user_row = model.ratings_[u].toarray().ravel()
        rated = user_row > 0
        mean_u = user_row[rated].mean()
        for i, item in enumerate(model.item_ids_):
            weights = model.neighbours_[i].toarray().ravel()[rated]
            if weights.sum() > 0:
                expected = mean_u + (weights * (user_row[rated] - mean_u)).sum() / weights.sum()
                branches.add("neighbours")
            else:
                expected = model.global_mean_ + model.user_bias_[u] + model.item_bias_[i]
                branches.add("baseline")
            assert model.predict([user], [item])[0] == pytest.approx(np.clip(expected, 1, 5))
    assert branches == {"neighbours", "baseline"}


def test_mf_is_deterministic_and_learns(train: pd.DataFrame) -> None:
    first, second = fitted("mf_svd", train), fitted("mf_svd", train)
    np.testing.assert_array_equal(first.predict([1, 2, 3], [13, 12, 10]), second.predict([1, 2, 3], [13, 12, 10]))
    assert len(first.epoch_rmse_) == MF_PARAMS["n_epochs"]
    assert first.epoch_rmse_[-1] < first.epoch_rmse_[0]
    other_seed = MatrixFactorizationRecommender(seed=7, **MF_PARAMS).fit(train)
    assert not np.allclose(other_seed.user_factors_, first.user_factors_)


def test_hybrid_blends_base_and_store_affinity(train: pd.DataFrame) -> None:
    store = StubFeatureStore(preferred_item=15)
    content_only = HybridRecommender(alpha=0.0, base_params=MF_PARAMS).fit(train, feature_store=store)
    assert content_only.recommend([1], k=1)[1] == [15]
    collaborative_only = HybridRecommender(alpha=1.0, base_params=MF_PARAMS).fit(train, feature_store=store)
    assert collaborative_only.recommend([1, 2, 3], k=3) == collaborative_only.base.recommend([1, 2, 3], k=3)
    np.testing.assert_allclose(collaborative_only.predict([1, 2], [13, 15]),
                               collaborative_only.base.predict([1, 2], [13, 15]))


def test_hybrid_requires_a_feature_store(train: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="feature store"):
        HybridRecommender(base_params=MF_PARAMS).fit(train)
    path = fitted("hybrid", train).save(tmp_path / "hybrid.pkl")
    with pytest.raises(ValueError, match="feature store"):
        BaseRecommender.load(path)
    with pytest.raises(ValueError):
        HybridRecommender(alpha=1.5)


# --------------------------------------------------------------------------- factory and training helpers
def test_create_model_from_config_section() -> None:
    models_cfg = {
        "popularity": {"enabled": True, "score": "count", "damping": 5},
        "mf_svd": {"enabled": False, **MF_PARAMS},
        "hybrid": {"enabled": True, "base_model": "mf_svd", "alpha": 0.6},
    }
    popularity = create_model("popularity", models_cfg, seed=3)
    assert popularity.get_params() == {"model": "popularity", "seed": 3, "score": "count", "damping": 5.0}
    hybrid = create_model("hybrid", models_cfg, seed=3)
    assert hybrid.alpha == 0.6 and hybrid.base.n_factors == MF_PARAMS["n_factors"] and hybrid.base.seed == 3
    assert hybrid.get_params()["base.n_epochs"] == MF_PARAMS["n_epochs"]
    assert enabled_models({"training": {"models": models_cfg}}) == ["popularity", "hybrid"]
    assert set(MODEL_REGISTRY) == {"popularity", "item_knn", "mf_svd", "hybrid"}
    with pytest.raises(KeyError):
        create_model("svd++", models_cfg, seed=3)


def test_mlflow_locations_are_anchored_at_project_root(tmp_path: Path) -> None:
    assert resolve_tracking_uri("sqlite:///mlflow.db", tmp_path) == f"sqlite:///{tmp_path.resolve()}/mlflow.db"
    assert resolve_tracking_uri("sqlite:////abs/x.db", tmp_path) == "sqlite:////abs/x.db"
    assert resolve_tracking_uri("http://127.0.0.1:5000", tmp_path) == "http://127.0.0.1:5000"
    assert resolve_artifact_location("mlartifacts", tmp_path) == (tmp_path / "mlartifacts").resolve().as_uri()
