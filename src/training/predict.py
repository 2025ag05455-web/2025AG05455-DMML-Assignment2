"""Prediction / serving entrypoint: top-k recommendations for one user.

Serving reuses the offline feature store (:class:`src.features.store.FeatureStore`) exactly as
training does: the hybrid model reads ``user_item_genre_affinity`` from it at request time, and
the explanation columns (user profile, item popularity, genre affinity) come from the same
point-in-time feature tables. Users without training history get the popularity fallback.

Usage:
    python -m src.training.predict --user-id 42 --model hybrid --k 10
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from src.features.store import FeatureStore
from src.training.models import MODEL_REGISTRY, BaseRecommender
from src.training.train import MODEL_FILE
from src.utils.config import load_config, path
from src.utils.io import read_parquet

USER_PROFILE_FEATURES = ("user_n_ratings", "user_avg_rating", "user_preferred_genre", "user_top_genres",
                         "age", "gender", "occupation")
ITEM_EXPLANATION_FEATURES = ("item_n_ratings", "item_bayesian_avg_rating")


def load_model(model_name: str, cfg: dict, store: FeatureStore) -> BaseRecommender:
    """Load ``models/<model_name>/model.pkl`` with the feature store attached."""
    model_path = path(cfg, "models_dir", model_name, MODEL_FILE)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found; run `python -m src.training.train` first")
    return BaseRecommender.load(model_path, feature_store=store)


def user_profile(user_id: int, store: FeatureStore) -> dict:
    """Key served features of one user (defaults and ``is_cold=True`` for unknown users)."""
    row = store.get_user_features([user_id], list(USER_PROFILE_FEATURES)).iloc[0]
    return {key: (value.item() if hasattr(value, "item") else value) for key, value in row.items()}


def recommend_for_user(user_id: int, model_name: str = "hybrid", k: int = 10, cfg: dict | None = None,
                       store: FeatureStore | None = None) -> pd.DataFrame:
    """Top-``k`` unseen items for ``user_id`` with titles, genres and explanation features.

    Args:
        user_id: user to serve (unknown / cold users receive the popularity fallback).
        model_name: one of ``MODEL_REGISTRY`` (must have been trained).
        k: number of recommendations.
        cfg: pipeline config (``load_config()`` when omitted).
        store: an already opened feature store (opened from ``cfg`` when omitted).

    Returns:
        Columns ``rank, item_id, title, genres, predicted_rating, user_item_genre_affinity,
        item_n_ratings, item_bayesian_avg_rating, source`` (``source`` = ``model`` or
        ``popularity_fallback``).
    """
    cfg = cfg or load_config()
    store = store or FeatureStore(cfg)
    model = load_model(model_name, cfg, store)
    item_ids = model.recommend([user_id], k=k, exclude_seen=True)[int(user_id)]
    users = [user_id] * len(item_ids)
    items = read_parquet(path(cfg, "processed_dir", "items_clean.parquet")).set_index("item_id")
    item_features = store.get_item_features(item_ids, list(ITEM_EXPLANATION_FEATURES))
    affinity = store.get_interaction_features(users, item_ids)["user_item_genre_affinity"]
    return pd.DataFrame({
        "rank": range(1, len(item_ids) + 1),
        "item_id": item_ids,
        "title": items["title"].reindex(item_ids).to_numpy(),
        "genres": items["genres"].reindex(item_ids).to_numpy(),
        "predicted_rating": model.predict(users, item_ids).round(3),
        "user_item_genre_affinity": affinity.round(3).to_numpy(),
        **{col: item_features[col].round(3).to_numpy() for col in ITEM_EXPLANATION_FEATURES},
        "source": "model" if model.is_known_user([user_id])[0] else "popularity_fallback",
    })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options of the prediction entrypoint."""
    parser = argparse.ArgumentParser(description="Top-k movie recommendations for one user.")
    parser.add_argument("--user-id", type=int, required=True, help="MovieLens user id")
    parser.add_argument("--model", default="hybrid", choices=sorted(MODEL_REGISTRY), help="trained model to use")
    parser.add_argument("--k", type=int, default=10, help="number of recommendations")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """Print the user's key features and the recommendation table; return the table."""
    args = parse_args(argv)
    cfg = load_config()
    store = FeatureStore(cfg)
    profile = user_profile(args.user_id, store)
    recs = recommend_for_user(args.user_id, args.model, args.k, cfg=cfg, store=store)
    status = "cold start - popularity fallback" if profile["is_cold"] else "warm"
    print(f"User {args.user_id} ({status}); features as of {store.metadata()['as_of']}")
    shown = {key: round(value, 3) if isinstance(value, float) else value for key, value in profile.items()
             if key not in ("user_id", "is_cold")}
    print("  " + ", ".join(f"{key}={value}" for key, value in shown.items()))
    print(f"\nTop-{args.k} recommendations from '{args.model}':")
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        print(recs.to_string(index=False))
    return recs


if __name__ == "__main__":
    main()
