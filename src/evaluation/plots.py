"""Static model-comparison figures for the evaluation report (matplotlib, PNG).

Design (reference data-viz method): every metric gets its own small-multiple panel with one
horizontal bar per model, so metrics on different scales never share an axis. The job is
"which model wins on each metric", so the form is *emphasis*: the best model of each panel is
drawn in the accent blue and the others in a recessive gray; models keep the same row in every
panel, values are printed at the bar tips, bars are thin with a rounded data end and a square
baseline, and grid/axes are solid hairlines. ``reports/model_comparison.md`` is the table view.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"          # categorical slot 1 (validated on the light surface)
DEEMPHASIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DPI = 150
BAR_HEIGHT = 0.42           # with PANEL_HEIGHT_IN below: ~22 px per bar for four models (cap 24 px)
PANEL_WIDTH_IN, PANEL_HEIGHT_IN = 3.0, 1.75
LEFT_MARGIN_IN, HEADER_IN = 0.85, 0.9    # room for model names / title + subtitle
CORNER_PX = 4.0

RANKING_PANELS: tuple[tuple[str, str], ...] = (
    ("precision@{k}", "Precision@{k}"), ("recall@{k}", "Recall@{k}"), ("ndcg@{k}", "NDCG@{k}"),
    ("hit_rate@{k}", "Hit rate@{k}"), ("map@{k}", "MAP@{k}"), ("coverage", "Catalogue coverage"),
    ("diversity", "Intra-list diversity"), ("novelty", "Novelty (bits)"),
)
RATING_PANELS: tuple[tuple[str, str], ...] = (("rmse", "RMSE"), ("mae", "MAE"))


def _style_axes(ax: Axes) -> None:
    """Recessive chrome: hairline vertical grid, baseline on the left, no box."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["left"].set_linewidth(1.0)
    ax.grid(axis="x", color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=8, colors=INK_SECONDARY)
    ax.tick_params(axis="x", labelcolor=INK_MUTED, labelsize=7)


def _bar_path(value: float, bottom: float, height: float, rx: float, ry: float) -> MplPath:
    """Outline of a bar from x=0 to ``value``: square at the baseline, rounded corners at the data end."""
    top = bottom + height
    vertices = [(0, bottom), (value - rx, bottom), (value, bottom), (value, bottom + ry), (value, top - ry),
                (value, top), (value - rx, top), (0, top), (0, bottom)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    return MplPath(vertices, codes)


def _draw_bar(ax: Axes, y: float, value: float, color: str) -> None:
    """Horizontal bar centred on row ``y`` with a 4 px rounded data end (axis limits must be set)."""
    bbox = ax.get_window_extent()
    x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / bbox.width
    y_per_px = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) / bbox.height
    rx = min(CORNER_PX * x_per_px, value / 2)
    ry = min(CORNER_PX * y_per_px, BAR_HEIGHT / 2)
    ax.add_patch(PathPatch(_bar_path(value, y - BAR_HEIGHT / 2, BAR_HEIGHT, rx, ry), facecolor=color,
                           edgecolor="none"))


def _bar_panel(ax: Axes, models: Sequence[str], values: Sequence[float], title: str, lower_is_better: bool,
               value_format: str) -> None:
    """One metric: a bar per model, the best one in the accent colour, values at the tips."""
    best = min(values) if lower_is_better else max(values)
    ax.set_xlim(0, max(values) * 1.32 if max(values) > 0 else 1.0)
    ax.set_ylim(len(models) - 0.5, -0.5)  # first model on top
    ax.set_yticks(range(len(models)), models)
    for row, value in enumerate(values):
        _draw_bar(ax, row, value, ACCENT if value == best else DEEMPHASIS)
        ax.text(value + ax.get_xlim()[1] * 0.02, row, format(value, value_format), va="center", ha="left",
                fontsize=7.5, color=INK_PRIMARY if value == best else INK_SECONDARY)
    ax.set_title(title, loc="left", fontsize=9, color=INK_PRIMARY, pad=6)


def _panel_figure(frame: pd.DataFrame, panels: Sequence[tuple[str, str]], n_cols: int, title: str,
                  subtitle: str, lower_is_better: bool, out_path: Path) -> Path:
    """Grid of single-metric panels sharing the model order; returns the written PNG path."""
    n_rows = -(-len(panels) // n_cols)
    width, height = PANEL_WIDTH_IN * n_cols + LEFT_MARGIN_IN, PANEL_HEIGHT_IN * n_rows + HEADER_IN
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), dpi=DPI, squeeze=False, facecolor=SURFACE)
    fig.subplots_adjust(left=LEFT_MARGIN_IN / width, right=0.98, top=1 - (HEADER_IN - 0.1) / height,
                        bottom=0.25 / height, wspace=0.3, hspace=0.6)
    models = frame["model"].tolist()
    for index, (ax, (column, label)) in enumerate(zip(axes.flat, panels)):
        _style_axes(ax)
        values = frame[column].astype(float).tolist()
        _bar_panel(ax, models, values, label, lower_is_better, ".2f" if column == "novelty" else ".3f")
        if index % n_cols:
            ax.set_yticklabels([])
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.text(0.01, 1 - 0.18 / height, title, fontsize=11.5, color=INK_PRIMARY, weight="bold",
             va="top")
    fig.text(0.01, 1 - 0.42 / height, subtitle, fontsize=8.5, color=INK_SECONDARY, va="top")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_ranking_comparison(frame: pd.DataFrame, k: int, n_users: int, out_path: Path) -> Path:
    """Ranking-accuracy and beyond-accuracy metrics at ``k``, one panel per metric."""
    panels = [(column.format(k=k), label.format(k=k)) for column, label in RANKING_PANELS]
    return _panel_figure(
        frame, panels, n_cols=4,
        title=f"Top-{k} recommendation quality on the temporal test split",
        subtitle=f"{n_users} warm test users with at least one relevant item - higher is better; "
                 "blue marks the best model for each metric",
        lower_is_better=False, out_path=out_path)


def plot_rating_comparison(frame: pd.DataFrame, n_ratings: int, out_path: Path) -> Path:
    """Rating-prediction error (RMSE, MAE), one panel per metric."""
    return _panel_figure(
        frame, RATING_PANELS, n_cols=2,
        title="Rating-prediction error on the temporal test split",
        subtitle=f"{n_ratings:,} test ratings of warm users - lower is better; blue marks the best model",
        lower_is_better=True, out_path=out_path)
