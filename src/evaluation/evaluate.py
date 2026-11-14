"""Stage 7 - Offline evaluation of the trained recommenders on the temporal test split.

Protocol
    * Relevant items: test interactions rated >= ``evaluation.relevance_threshold``.
    * Ranking users ("warm"): test users with training history and >= 1 relevant test item.
      Items rated in training are excluded from their lists; candidates are training items.
    * Ranking metrics at ``evaluation.k`` and every ``evaluation.extra_k``; coverage, diversity
      (genre vectors) and novelty (training popularity) at ``evaluation.k``.
    * Rating metrics (RMSE, MAE): every test interaction of warm users.
    * Cold-start slice: test users without training history are served by each model's
      popularity fallback and reported separately (the global time split makes most test users cold).

Item genre vectors and training popularity come from :class:`src.features.store.FeatureStore`,
the same feature layer used by training and serving. Each model's MLflow training run is
re-opened by ``run_id`` and receives the test metrics plus per-model artifacts.

Outputs:
    reports/model_comparison.csv, reports/model_comparison.md, reports/evaluation_results.json,
    reports/figures/model_comparison_ranking.png, reports/figures/model_comparison_rating.png,
    models/<model>/evaluation/{metrics.json, sample_recommendations.csv} (also logged to MLflow)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.evaluation.metrics import (catalog_coverage, intra_list_diversity, mae, novelty, ranking_metrics,
                                    rmse)
from src.evaluation.plots import plot_rating_comparison, plot_ranking_comparison
from src.features.store import FeatureStore
from src.training.models import BaseRecommender
from src.training.tracking import configure_mlflow
from src.training.train import METADATA_FILE, MODEL_FILE, enabled_models
from src.utils.config import load_config, path, rel_path
from src.utils.io import read_json, read_parquet, write_json
from src.utils.logger import get_logger

log = get_logger("evaluation")

RANKING_NAMES = ("precision", "recall", "ndcg", "hit_rate", "map")
BEYOND_ACCURACY = ("coverage", "diversity", "novelty")
LOWER_IS_BETTER = ("rmse", "mae")
N_SAMPLE_USERS = 20


@dataclass(frozen=True)
class EvaluationData:
    """Everything the per-model evaluation needs, computed once.

    Attributes:
        relevant: warm ranking users -> relevant test item ids.
        cold_relevant: cold-start test users -> relevant test item ids.
        warm_test: all test interactions of users with training history.
        item_vectors: training item id -> genre one-hot vector (feature store).
        item_share: training item id -> share of training users who rated it (feature store).
        n_catalog_items: number of items in the training catalogue.
        stats: descriptive counts reported in ``evaluation_results.json``.
    """

    relevant: dict[int, set[int]]
    cold_relevant: dict[int, set[int]]
    warm_test: pd.DataFrame
    item_vectors: dict[int, np.ndarray]
    item_share: dict[int, float]
    n_catalog_items: int
    stats: dict


# --------------------------------------------------------------------------- data
def relevant_by_user(test: pd.DataFrame, threshold: float) -> dict[int, set[int]]:
    """``{user_id: {item_id, ...}}`` of test interactions rated >= ``threshold``."""
    liked = test.loc[test["rating"] >= threshold, ["user_id", "item_id"]]
    return {int(u): set(items.tolist()) for u, items in liked.groupby("user_id")["item_id"]}


def prepare_evaluation_data(cfg: dict, store: FeatureStore) -> EvaluationData:
    """Load train/test, split test users into warm / cold and fetch item features from the store."""
    train = read_parquet(path(cfg, "processed_dir", "train.parquet"))
    test = read_parquet(path(cfg, "processed_dir", "test.parquet"))
    relevant = relevant_by_user(test, float(cfg["evaluation"]["relevance_threshold"]))
    train_users = set(train["user_id"].unique().tolist())
    catalog = np.sort(train["item_id"].unique())
    item_counts = store.get_item_features(catalog, ["item_n_ratings"])["item_n_ratings"].to_numpy()
    warm_test = test[test["user_id"].isin(train_users)]
    warm_relevant = {u: items for u, items in relevant.items() if u in train_users}
    catalog_set = set(catalog.tolist())
    relevant_pairs = sum(len(items) for items in warm_relevant.values())
    on_catalog = sum(len(items & catalog_set) for items in warm_relevant.values())
    stats = {
        "n_test_interactions": int(len(test)),
        "n_test_users": int(test["user_id"].nunique()),
        "n_warm_test_users": int(warm_test["user_id"].nunique()),
        "n_cold_test_users": int(test["user_id"].nunique() - warm_test["user_id"].nunique()),
        "n_warm_test_interactions": int(len(warm_test)),
        "n_ranking_users": len(warm_relevant),
        "n_cold_ranking_users": len(relevant) - len(warm_relevant),
        "n_relevant_pairs_warm": int(relevant_pairs),
        "share_relevant_pairs_on_catalog": round(on_catalog / relevant_pairs, 6) if relevant_pairs else 0.0,
        "n_catalog_items": int(len(catalog)),
        "n_train_users": len(train_users),
    }
    return EvaluationData(
        relevant=warm_relevant,
        cold_relevant={u: items for u, items in relevant.items() if u not in train_users},
        warm_test=warm_test,
        item_vectors=dict(zip(catalog.tolist(), store.get_item_genre_matrix(catalog))),
        item_share=dict(zip(catalog.tolist(), (item_counts / len(train_users)).tolist())),
        n_catalog_items=len(catalog),
        stats=stats,
    )


def load_trained_model(cfg: dict, name: str, store: FeatureStore) -> tuple[BaseRecommender, dict]:
    """Load ``models/<name>/model.pkl`` (feature store attached) and its training metadata."""
    model_dir = path(cfg, "models_dir", name)
    if not (model_dir / MODEL_FILE).exists():
        raise FileNotFoundError(f"{model_dir / MODEL_FILE} not found; run src.training.train first")
    return BaseRecommender.load(model_dir / MODEL_FILE, feature_store=store), read_json(model_dir / METADATA_FILE)


# --------------------------------------------------------------------------- metrics
def ranking_report(recommendations: dict[int, list[int]], relevant: dict[int, set[int]], ks: list[int],
                   data: EvaluationData) -> dict[str, float]:
    """Accuracy metrics at every k plus coverage / diversity / novelty at ``ks[0]``."""
    metrics: dict[str, float] = {}
    for k in ks:
        metrics.update(ranking_metrics({u: recs[:k] for u, recs in recommendations.items()}, relevant, k))
    lists = [recommendations[u][:ks[0]] for u in relevant]
    metrics["coverage"] = catalog_coverage(lists, data.n_catalog_items)
    metrics["diversity"] = intra_list_diversity(lists, data.item_vectors)
    metrics["novelty"] = novelty(lists, data.item_share)
    return metrics


def evaluate_model(model: BaseRecommender, data: EvaluationData, ks: list[int]) -> tuple[dict, dict[int, list[int]]]:
    """Warm ranking + rating metrics, the cold-start slice and the warm recommendation lists."""
    recs = model.recommend(list(data.relevant), k=max(ks), exclude_seen=True)
    warm = ranking_report(recs, data.relevant, ks, data)
    predictions = model.predict(data.warm_test["user_id"], data.warm_test["item_id"])
    warm["rmse"] = rmse(data.warm_test["rating"], predictions)
    warm["mae"] = mae(data.warm_test["rating"], predictions)
    cold_recs = model.recommend(list(data.cold_relevant), k=max(ks), exclude_seen=True)
    cold = ranking_report(cold_recs, data.cold_relevant, ks, data) if data.cold_relevant else {}
    return {"warm": warm, "cold_start": cold}, recs


def primary_metric_key(cfg: dict) -> str:
    """Translate ``evaluation.primary_metric`` (e.g. ``ndcg_at_k``) into a result column (``ndcg@10``)."""
    name = str(cfg["evaluation"]["primary_metric"])
    return f"{name[:-len('_at_k')]}@{cfg['evaluation']['k']}" if name.endswith("_at_k") else name


# --------------------------------------------------------------------------- reporting
def comparison_frame(results: dict[str, dict], k: int, extra_k: list[int]) -> pd.DataFrame:
    """One row per model: metrics@k, beyond-accuracy, rmse, mae, then the extra-k metrics."""
    columns = ([f"{m}@{k}" for m in RANKING_NAMES] + list(BEYOND_ACCURACY) + list(LOWER_IS_BETTER)
               + [f"{m}@{x}" for x in extra_k for m in RANKING_NAMES])
    rows = [{"model": name, **{c: res["warm"][c] for c in columns}} for name, res in results.items()]
    return pd.DataFrame(rows, columns=["model", *columns])


def best_values(frame: pd.DataFrame) -> dict[str, float]:
    """Best value of every metric column (minimum for errors, maximum otherwise)."""
    return {c: (frame[c].min() if c in LOWER_IS_BETTER else frame[c].max()) for c in frame.columns if c != "model"}


def markdown_table(frame: pd.DataFrame, bold_best: bool = True) -> str:
    """Markdown table with 4-decimal values; the best value of each column in bold."""
    best = best_values(frame) if bold_best else {}
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "|".join(["---"] + ["---:"] * (frame.shape[1] - 1)) + "|"]
    for _, row in frame.iterrows():
        cells = [str(row["model"])]
        for column in frame.columns[1:]:
            text = f"{row[column]:.4f}"
            cells.append(f"**{text}**" if bold_best and np.isclose(row[column], best[column]) else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_report(frame: pd.DataFrame, cold: pd.DataFrame, summary: dict) -> str:
    """``reports/model_comparison.md``: protocol, comparison table and the cold-start slice."""
    stats, protocol = summary["data"], summary["protocol"]
    k = protocol["k"]
    return "\n".join([
        "# Model comparison",
        "",
        f"Generated {summary['created_at']} by `src/evaluation/evaluate.py`; best value per column in **bold** "
        "(lowest for RMSE/MAE, highest otherwise).",
        "",
        "## Protocol",
        f"* Temporal split at {protocol['split_cutoff_utc']}; relevant = test rating >= {protocol['relevance_threshold']}.",
        f"* Ranking: {stats['n_ranking_users']} warm test users (training history + >= 1 relevant test item); "
        f"items rated in training excluded; candidates = {stats['n_catalog_items']} training items.",
        f"* coverage / diversity / novelty at K={k}; RMSE / MAE on {stats['n_warm_test_interactions']:,} test "
        "ratings of warm users.",
        f"* {stats['share_relevant_pairs_on_catalog']:.1%} of the warm users' relevant test items exist in the "
        "training catalogue (upper bound on recall).",
        "",
        "## Warm users",
        markdown_table(frame),
        "",
        f"Primary metric: `{summary['primary_metric']}` - best model: **{summary['best_model']}**.",
        "",
        "## Cold-start slice (popularity fallback)",
        f"{stats['n_cold_test_users']} of {stats['n_test_users']} test users have no training history; "
        f"{stats['n_cold_ranking_users']} of them have a relevant test item. Every model serves them its "
        "popularity fallback:",
        "",
        markdown_table(cold, bold_best=False) if not cold.empty else "_no cold-start users with relevant items_",
        "",
    ])


def sample_recommendations(recs: dict[int, list[int]], relevant: dict[int, set[int]], items: pd.DataFrame,
                           n_users: int = N_SAMPLE_USERS) -> pd.DataFrame:
    """Long table of the lists of the first ``n_users`` ranking users, with titles and hit flags."""
    rows = [{"user_id": u, "rank": rank, "item_id": item, "relevant": item in relevant[u]}
            for u in sorted(recs)[:n_users] for rank, item in enumerate(recs[u], start=1)]
    return pd.DataFrame(rows).merge(items[["item_id", "title", "genres"]], on="item_id", how="left")


def log_to_mlflow(run_id: str, metrics: dict, artifacts: list[Path]) -> None:
    """Add test metrics (``test_*`` / ``test_cold_*``) and evaluation artifacts to a training run."""
    flat = {f"test_{key}": value for key, value in metrics["warm"].items()}
    flat.update({f"test_cold_{key}": value for key, value in metrics["cold_start"].items()})
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({key.replace("@", "_at_"): float(value) for key, value in flat.items()})
        mlflow.set_tag("evaluated", "true")
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path="evaluation")


# --------------------------------------------------------------------------- stage
def evaluate_and_log(name: str, cfg: dict, store: FeatureStore, data: EvaluationData, ks: list[int],
                     items: pd.DataFrame) -> dict:
    """Evaluate one trained model, write its artifacts and log everything to its MLflow run."""
    model, metadata = load_trained_model(cfg, name, store)
    metrics, recs = evaluate_model(model, data, ks)
    out_dir = path(cfg, "models_dir", name, "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "sample_recommendations.csv"
    sample_recommendations(recs, data.relevant, items).to_csv(sample_path, index=False)
    log_to_mlflow(metadata["run_id"], metrics, [write_json(metrics, out_dir / "metrics.json"), sample_path])
    k = ks[0]
    log.info("%-10s ndcg@%d=%.4f precision@%d=%.4f rmse=%.4f", name, k, metrics["warm"][f"ndcg@{k}"], k,
             metrics["warm"][f"precision@{k}"], metrics["warm"]["rmse"])
    return {"run_id": metadata["run_id"], **metrics}


def report_paths(cfg: dict) -> dict[str, Path]:
    """Locations of the evaluation reports and figures."""
    return {
        "comparison_csv": path(cfg, "reports_dir", "model_comparison.csv"),
        "comparison_md": path(cfg, "reports_dir", "model_comparison.md"),
        "results_json": path(cfg, "reports_dir", "evaluation_results.json"),
        "ranking_figure": path(cfg, "figures_dir", "model_comparison_ranking.png"),
        "rating_figure": path(cfg, "figures_dir", "model_comparison_rating.png"),
    }


def protocol_description(cfg: dict, ks: list[int]) -> dict:
    """Machine-readable description of the evaluation protocol."""
    split = read_json(path(cfg, "processed_dir", "split_metadata.json"))
    return {
        "split_method": split["method"], "split_cutoff_utc": split["cutoff_time_utc"], "k": ks[0], "extra_k": ks[1:],
        "relevance_threshold": cfg["evaluation"]["relevance_threshold"], "exclude_seen": True,
        "ranking_users": "test users with training history and >= 1 relevant test item",
        "candidate_items": "items in the training catalogue",
        "rating_metrics_on": "all test interactions of users with training history",
        "beyond_accuracy": f"coverage, diversity (genre ILD) and novelty (-log2 user share) at K={ks[0]}",
    }


def cold_start_frame(results: dict[str, dict], k: int) -> pd.DataFrame:
    """One row per model with its cold-start (popularity fallback) ranking metrics at ``k``."""
    return pd.DataFrame([{"model": name, **{f"{m}@{k}": res["cold_start"].get(f"{m}@{k}", 0.0) for m in RANKING_NAMES}}
                         for name, res in results.items()])


def write_reports(frame: pd.DataFrame, summary: dict, data: EvaluationData, reports: dict[str, Path]) -> None:
    """Comparison CSV / markdown, full JSON results and the two comparison figures."""
    k = summary["protocol"]["k"]
    frame.to_csv(reports["comparison_csv"], index=False)
    cold = cold_start_frame(summary["models"], k)
    reports["comparison_md"].write_text(markdown_report(frame, cold, summary), encoding="utf-8")
    write_json(summary, reports["results_json"])
    plot_ranking_comparison(frame, k, data.stats["n_ranking_users"], reports["ranking_figure"])
    plot_rating_comparison(frame, data.stats["n_warm_test_interactions"], reports["rating_figure"])


def run(cfg: dict | None = None) -> dict:
    """Evaluate every enabled, trained model; write reports and figures; log to MLflow."""
    cfg = cfg or load_config()
    ks = [int(cfg["evaluation"]["k"]), *[int(x) for x in cfg["evaluation"].get("extra_k", [])]]
    store = FeatureStore(cfg)
    data = prepare_evaluation_data(cfg, store)
    items = read_parquet(path(cfg, "processed_dir", "items_clean.parquet"))
    configure_mlflow(cfg)
    results = {name: evaluate_and_log(name, cfg, store, data, ks, items) for name in enabled_models(cfg)}

    frame = comparison_frame(results, ks[0], ks[1:])
    primary = primary_metric_key(cfg)
    best_row = frame[primary].idxmin() if primary in LOWER_IS_BETTER else frame[primary].idxmax()
    reports = report_paths(cfg)
    summary = {
        "stage": "evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol_description(cfg, ks),
        "data": data.stats,
        "primary_metric": primary,
        "best_model": str(frame.loc[best_row, "model"]),
        "models": results,
        "artifacts": {key: rel_path(p, cfg["project"]["root"]) for key, p in reports.items()},
    }
    write_reports(frame, summary, data, reports)
    log.info("Evaluation complete: %d ranking users, best %s by %s", data.stats["n_ranking_users"],
             summary["best_model"], primary)
    return {
        "stage": "evaluation",
        "n_ranking_users": data.stats["n_ranking_users"],
        "n_cold_test_users": data.stats["n_cold_test_users"],
        "primary_metric": primary,
        "best_model": summary["best_model"],
        "scores": {name: round(res["warm"][primary], 6) for name, res in results.items()},
        "artifacts": summary["artifacts"],
    }


if __name__ == "__main__":
    run()
