#!/usr/bin/env python
"""Render the pipeline architecture diagram (``reports/figures/architecture.{png,svg}``).

Left-to-right layered Graphviz diagram: the MovieLens source, the seven data stages
(each a coloured cluster with the files it writes), the experiment-tracking and
serving paths, and the orchestration layer whose Prefect tasks are rank-aligned
with the stage they run. Requires the ``graphviz`` Python package and the ``dot``
binary on PATH.

Usage::

    python scripts/make_architecture_diagram.py [--dpi 200]
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/make_architecture_diagram.py`
    sys.path.insert(0, str(ROOT))

import graphviz  # noqa: E402

from dags.pipeline_spec import STAGES  # noqa: E402
from src.utils.config import load_config, path  # noqa: E402

FONT = "Helvetica"
GREY = "#78909C"

#: Layer -> (strong colour for process nodes / borders, light tint for cluster backgrounds).
PALETTE: dict[str, tuple[str, str]] = {
    "source": ("#546E7A", "#F4F6F7"),
    "ingestion": ("#1E88E5", "#EEF6FD"),
    "validation": ("#F09300", "#FFF8EB"),
    "quarantine": ("#E53935", "#FFEBEE"),
    "preparation": ("#00897B", "#EAF6F5"),
    "transformation": ("#43A047", "#EEF7EE"),
    "features": ("#8E24AA", "#F6EDF8"),
    "training": ("#3949AB", "#EEF0F9"),
    "evaluation": ("#D81B60", "#FCEEF3"),
    "serving": ("#00838F", "#E9F6F7"),
    "orchestration": ("#37474F", "#F1F3F4"),
}


# --------------------------------------------------------------------------- node helpers
def _label(title: str, detail: str = "", size: int = 13) -> str:
    """HTML-like label: bold title plus an optional smaller detail block (lines split on '|')."""
    lines = "".join(f'<br/><font point-size="{size - 3}">{line}</font>' for line in detail.split("|") if line)
    return f"<<b>{title}</b>{lines}>"


def process(g: graphviz.Digraph, node_id: str, title: str, detail: str, layer: str) -> None:
    """A processing step (script / module): solid rounded box in the layer colour."""
    strong, _ = PALETTE[layer]
    g.node(node_id, _label(title, detail, 14), shape="box", style="rounded,filled",
           fillcolor=strong, color=strong, fontcolor="white", margin="0.18,0.10")


def dataset(g: graphviz.Digraph, node_id: str, title: str, detail: str, layer: str,
            shape: str = "note") -> None:
    """A dataset or store: white node with a layer-coloured border (note = file, cylinder = store)."""
    strong, _ = PALETTE[layer]
    g.node(node_id, _label(title, detail), shape=shape, style="filled", fillcolor="white",
           color=strong, penwidth="1.6", fontcolor="#263238", margin="0.12,0.06")


@contextmanager
def cluster(g: graphviz.Digraph, name: str, label: str, layer: str,
            labeljust: str = "r") -> Iterator[graphviz.Digraph]:
    """A rounded, tinted cluster for one layer (right-aligned label keeps it clear of task edges)."""
    strong, tint = PALETTE[layer]
    with g.subgraph(name=f"cluster_{name}") as c:
        c.attr(label=f"<<b>{label}</b>>", style="rounded,filled", fillcolor=tint, color=strong,
               penwidth="1.8", fontcolor=strong, fontsize="17", margin="14", labeljust=labeljust)
        yield c


def same_rank(g: graphviz.Digraph, *node_ids: str) -> None:
    """Force nodes into the same left-to-right column (requires ``newrank``)."""
    with g.subgraph() as s:
        s.attr(rank="same")
        for node_id in node_ids:
            s.node(node_id)


# --------------------------------------------------------------------------- layers
def add_data_stages(g: graphviz.Digraph) -> None:
    """Source plus stages 1-5: ingestion, validation, preparation, transformation, features."""
    with cluster(g, "source", "Source", "source") as c:
        dataset(c, "src_zip", "MovieLens-100K", "ml-100k.zip (GroupLens)|943 users · 1,682 items|100k ratings",
                "source", shape="folder")
    with cluster(g, "ingestion", "1 · Ingestion", "ingestion") as c:
        process(c, "p_ingestion", "ingest.py", "download / cache · checksum|seeded anomaly injection", "ingestion")
        dataset(c, "raw_csv", "data/raw/*.csv", "users · items · interactions|string-typed landing zone", "ingestion")
        dataset(c, "raw_manifest", "ingestion_manifest.json", "lineage · sha256 · row counts|injected_anomalies.json",
                "ingestion")
    with cluster(g, "validation", "2 · Validation", "validation") as c:
        process(c, "p_validation", "validate.py", "pandera schemas|referential + range rules", "validation")
        dataset(c, "interim", "data/interim/*.parquet", "typed rows that passed|hard checks", "validation")
        dataset(c, "errors", "data/errors/*_errors.csv", "quarantine: rejected rows|+ soft warnings", "quarantine")
        dataset(c, "val_report", "validation_report.{json,md}", "checks · error rates · recall", "validation")
    with cluster(g, "preparation", "3 · Preparation", "preparation") as c:
        process(c, "p_preparation", "prepare.py", "impute · normalise · dedupe|time-based 80/20 split",
                "preparation")
        dataset(c, "clean", "*_clean.parquet", "users · items · interactions", "preparation")
        dataset(c, "split", "train / test.parquet", "split_metadata.json|cutoff = 80th pct timestamp",
                "preparation")
    with cluster(g, "transformation", "4 · Transformation", "transformation") as c:
        process(c, "p_transformation", "transform.py", "train-only aggregates", "transformation")
        dataset(c, "derived", "derived tables", "user_activity_summary|item_popularity", "transformation")
        dataset(c, "matrix", "user_item_matrix.npz", "CSR users × items|matrix_index.json", "transformation")
    with cluster(g, "features", "5 · Feature management", "features") as c:
        process(c, "p_features", "build_features.py", "registry.py · point-in-time|as_of = split cutoff",
                "features")
        dataset(c, "feat_tables", "offline feature store",
                "data/features/: user_features|item_features · user_genre_affinity", "features", shape="cylinder")
        dataset(c, "feat_meta", "feature_metadata.json", "version · as_of · feature list|feature_catalog.md",
                "features")
        process(c, "fs_api", "FeatureStore API", "store.py · same API for|training and serving", "features")

    for src, dst in [("src_zip", "p_ingestion"), ("p_ingestion", "raw_csv"), ("p_ingestion", "raw_manifest"),
                     ("raw_csv", "p_validation"), ("p_validation", "interim"), ("p_validation", "errors"),
                     ("p_validation", "val_report"), ("interim", "p_preparation"),
                     ("p_preparation", "clean"), ("p_preparation", "split"),
                     ("clean", "p_transformation"), ("split", "p_transformation"),
                     ("p_transformation", "derived"), ("p_transformation", "matrix"),
                     ("clean", "p_features"), ("derived", "p_features"),
                     ("p_features", "feat_tables"), ("p_features", "feat_meta"), ("feat_tables", "fs_api")]:
        g.edge(src, dst)


def add_model_stages(g: graphviz.Digraph) -> None:
    """Stages 6-7 (training, evaluation), MLflow tracking and the prediction path."""
    with cluster(g, "training", "6 · Training", "training") as c:
        process(c, "p_training", "train.py", "models.py: popularity · item-kNN|MF (SGD) · hybrid", "training")
        dataset(c, "models", "models/&lt;model&gt;/", "model.pkl · metadata.json|(incl. MLflow run_id)", "training")
        dataset(c, "mlflow", "MLflow tracking", "mlflow.db (sqlite) · mlartifacts/|params · metrics · artifacts",
                "training", shape="cylinder")
    with cluster(g, "evaluation", "7 · Evaluation", "evaluation") as c:
        process(c, "p_evaluation", "evaluate.py", "metrics.py: P/R/NDCG/MAP@K|coverage · RMSE", "evaluation")
        dataset(c, "eval_reports", "reports/", "model_comparison.{csv,md}|evaluation_results.json", "evaluation")
        dataset(c, "figures", "reports/figures/*.png", "metric comparison plots", "evaluation")
    with cluster(g, "serving", "Prediction / serving", "serving") as c:
        process(c, "p_predict", "predict.py", "load model + FeatureStore|recommend(user_ids, k)", "serving")
        dataset(c, "topk", "top-K recommendations", "per user · seen items excluded", "serving")

    for src, dst in [("fs_api", "p_training"), ("matrix", "p_training"), ("p_training", "models"),
                     ("p_training", "mlflow"), ("models", "p_evaluation"), ("p_evaluation", "eval_reports"),
                     ("p_evaluation", "figures"), ("models", "p_predict"), ("p_predict", "topk")]:
        g.edge(src, dst)
    g.edge("split", "p_training", label=" train", color=GREY, fontcolor=GREY, fontsize="10")
    g.edge("split", "p_evaluation", label=" test", color=GREY, fontcolor=GREY, fontsize="10")
    g.edge("p_evaluation", "mlflow", label=" metrics", color=GREY, fontcolor=GREY, fontsize="10",
           constraint="false")
    g.edge("fs_api", "p_predict", color=PALETTE["serving"][0], style="bold", label=" same features",
           fontcolor=PALETTE["serving"][0], fontsize="10")


def add_orchestration(g: graphviz.Digraph) -> None:
    """Prefect flow + scheduler lane; each task is aligned with (and runs) its stage."""
    strong, _ = PALETTE["orchestration"]
    with cluster(g, "orchestration", "8 · Orchestration  (dags/recommendation_flow.py · Prefect 3)",
                 "orchestration", labeljust="l") as c:
        process(c, "scheduler", "Prefect deployment", "daily-recsys-pipeline|cron 0 2 * * * (UTC)", "orchestration")
        for stage in STAGES:
            dataset(c, f"t_{stage.name}", f"@task {stage.task_name}",
                    f"timeout {stage.timeout_sec // 60} min"
                    + (f" · {stage.retries} retr{'y' if stage.retries == 1 else 'ies'}" if stage.retries else ""),
                    "orchestration", shape="box")
        dataset(c, "run_summary", "pipeline_run_summary.json", "status · timings · versions|+ Prefect artifacts",
                "orchestration")
        dataset(c, "alt_runners", "alternatives", "run_pipeline.py (plain Python)|airflow_recommendation_dag.py",
                "orchestration", shape="box")
    chain = ["scheduler", *[f"t_{s.name}" for s in STAGES], "run_summary"]
    for src, dst in zip(chain, chain[1:]):
        g.edge(src, dst, color=strong, penwidth="1.6")
    for stage in STAGES:
        g.edge(f"t_{stage.name}", f"p_{stage.name}", style="dashed", color=GREY, arrowhead="vee",
               constraint="false")
        same_rank(g, f"t_{stage.name}", f"p_{stage.name}")
    same_rank(g, "scheduler", "alt_runners", "src_zip")
    same_rank(g, "run_summary", "eval_reports")


def title_label() -> str:
    """Graph title with a one-line legend (shapes, edge styles and layer colours)."""
    swatches = "".join(
        f'<td bgcolor="{PALETTE[layer][0]}" width="14"> </td>'
        f'<td align="left"><font point-size="13">{text}</font></td>'
        for layer, text in (("ingestion", "ingest"), ("validation", "validate"), ("preparation", "prepare"),
                            ("transformation", "transform"), ("features", "features"),
                            ("training", "train"), ("evaluation", "evaluate"), ("serving", "serve"),
                            ("orchestration", "orchestrate")))
    legend = "■ process (script / API)   □ file   ⛁ store   → data flow   ⇢ task runs stage      layers:"
    return ("<<table border='0' cellspacing='4'>"
            "<tr><td><font point-size='28'><b>End-to-end data management pipeline · "
            "MovieLens-100K recommender</b></font></td></tr>"
            "<tr><td><table border='0' cellspacing='3'><tr>"
            f"<td><font point-size='13' color='#37474F'>{legend}</font></td>{swatches}"
            "</tr></table></td></tr></table>>")


def build_graph(dpi: int) -> graphviz.Digraph:
    """Assemble the complete architecture graph."""
    g = graphviz.Digraph("recsys_architecture")
    g.attr(rankdir="LR", newrank="true", splines="spline", nodesep="0.25",
           ranksep="0.4", pad="0.4", dpi=str(dpi), bgcolor="white", fontname=FONT, labelloc="t",
           fontsize="13", fontcolor="#263238", label=title_label())
    g.attr("node", fontname=FONT, fontsize="13")
    g.attr("edge", fontname=FONT, color="#455A64", penwidth="1.3", arrowsize="0.8")
    add_data_stages(g)
    add_model_stages(g)
    add_orchestration(g)
    return g


def render(out_dir: Path, dpi: int = 200) -> dict[str, str]:
    """Render PNG and SVG into ``out_dir``; return the written paths."""
    graph = build_graph(dpi)
    written = {}
    for fmt in ("png", "svg"):
        target = out_dir / f"architecture.{fmt}"
        target.write_bytes(graph.pipe(format=fmt))
        written[fmt] = str(target)
    return written


def main(argv: list[str] | None = None) -> dict[str, str]:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Render the pipeline architecture diagram.")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default 200)")
    args = parser.parse_args(argv)
    written = render(path(load_config(), "figures_dir"), dpi=args.dpi)
    print("\n".join(f"wrote {p}" for p in written.values()))
    return written


if __name__ == "__main__":
    main()
