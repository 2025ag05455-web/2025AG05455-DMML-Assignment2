"""Feature registry: the single declarative definition of every feature in the feature store.

Each :class:`FeatureDefinition` records the feature's entity, storage dtype, meaning,
derivation, point-in-time semantics and consuming models. The registry drives three things:

* schema validation of the built feature tables before they are written (:func:`validate_feature_frame`),
* the human-readable catalogue in ``reports/feature_catalog.md`` (:func:`render_catalog`),
* the list of columns served by :class:`src.features.store.FeatureStore`.

Storage tables
    ``user_features`` (key ``user_id``), ``item_features`` (key ``item_id``),
    ``user_genre_affinity`` (key ``user_id``) and ``on_demand`` (user x item features that the
    store computes at request time and never materialises).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from src.ingestion.ingest import GENRE_COLS

ON_DEMAND = "on_demand"
TABLE_KEYS = {"user_features": "user_id", "item_features": "item_id", "user_genre_affinity": "user_id"}
METADATA_COLUMNS = {"feature_timestamp": "datetime64[ns, UTC]", "feature_set_version": "string"}
ENTITIES = ("user", "item", "user_item")


class FeatureValidationError(ValueError):
    """Raised when a built feature table does not match the registry."""


@dataclass(frozen=True)
class FeatureDefinition:
    """Declarative description of one feature (or of a fixed family of columns such as ``genre_*``).

    Attributes:
        name: feature name; equals the stored column unless ``columns`` is given.
        entity: ``'user'``, ``'item'`` or ``'user_item'``.
        dtype: pandas dtype of the stored / served column(s).
        description: what the feature means.
        logic: source table(s) and derivation.
        point_in_time: True if the value depends on interaction history up to ``as_of``
            (False for static profile / catalogue attributes).
        used_by: consumers of the feature: model names, ``evaluation`` (beyond-accuracy metrics),
            ``serving`` (``src.training.predict`` explanations) or ``eda`` (analysis / catalogue only).
        table: storage table name, or ``'on_demand'`` for request-time features.
        columns: stored column names for a feature family; defaults to ``(name,)``.
    """

    name: str
    entity: str
    dtype: str
    description: str
    logic: str
    point_in_time: bool
    used_by: tuple[str, ...]
    table: str
    columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject definitions with an unknown entity or table."""
        if self.entity not in ENTITIES:
            raise ValueError(f"{self.name}: unknown entity {self.entity!r}")
        if self.table not in TABLE_KEYS and self.table != ON_DEMAND:
            raise ValueError(f"{self.name}: unknown table {self.table!r}")

    @property
    def stored_columns(self) -> tuple[str, ...]:
        """Column names this definition occupies in its table."""
        return self.columns or (self.name,)


def _user(name: str, dtype: str, description: str, logic: str, pit: bool = True,
          used_by: tuple[str, ...] = ("eda",)) -> FeatureDefinition:
    """Shorthand for a scalar feature stored in ``user_features``."""
    return FeatureDefinition(name, "user", dtype, description, logic, pit, used_by, "user_features")


def _item(name: str, dtype: str, description: str, logic: str, pit: bool = True,
          used_by: tuple[str, ...] = ("eda",)) -> FeatureDefinition:
    """Shorthand for a scalar feature stored in ``item_features``."""
    return FeatureDefinition(name, "item", dtype, description, logic, pit, used_by, "item_features")


FEATURE_REGISTRY: list[FeatureDefinition] = [
    # ------------------------------------------------------------------ user (behavioural, as of cutoff)
    _user("user_n_ratings", "int64", "Activity: number of ratings given before as_of",
          "user_activity_summary.n_ratings = count(train) per user_id", used_by=("serving", "eda")),
    _user("user_avg_rating", "float64", "Mean rating given by the user (rating bias)",
          "user_activity_summary.mean_rating = mean(train.rating) per user_id", used_by=("serving", "eda")),
    _user("user_rating_std", "float64", "Population std of the user's ratings (0 for one rating)",
          "user_activity_summary.std_rating = std(train.rating, ddof=0) per user_id"),
    _user("user_days_since_last_interaction", "float64", "Recency: days between the last rating and as_of",
          "(as_of - user_activity_summary.last_ts) / 86400"),
    _user("user_tenure_days", "float64", "Days between the user's first rating and as_of",
          "(as_of - user_activity_summary.first_ts) / 86400"),
    _user("user_ratings_per_active_day", "float64", "Activity intensity: ratings per distinct active UTC day",
          "user_n_ratings / user_activity_summary.active_days (distinct train.timestamp // 86400)"),
    _user("user_preferred_genre", "string", "Genre with the largest share of the user's ratings",
          "argmax(user_genre_affinity); ties -> GENRE_COLS order", used_by=("serving", "eda")),
    _user("user_top_genres", "string", "Top-N genres by share (pipe-joined, N = features.top_n_genres)",
          "top-N of user_genre_affinity with share > 0", used_by=("serving", "eda")),
    _user("user_genre_entropy", "float64", "Taste diversity: normalised Shannon entropy of genre shares (0..1)",
          "-sum(p * ln p) / ln(19) over user_genre_affinity"),
    # ------------------------------------------------------------------ user (static profile)
    _user("age", "int64", "User age in years (missing ages imputed)", "users_clean.age", pit=False,
          used_by=("serving", "eda")),
    _user("gender", "string", "Gender: M, F or unknown", "users_clean.gender", pit=False,
          used_by=("serving", "eda")),
    _user("occupation", "string", "Occupation (lower-case)", "users_clean.occupation", pit=False,
          used_by=("serving", "eda")),
    _user("age_group", "string", "Age bucket (under_18, 18_24, ..., 55_plus)", "users_clean.age_group", pit=False),
    # ------------------------------------------------------------------ user genre-affinity vector
    FeatureDefinition("user_genre_affinity", "user", "float64",
                      "Share of the user's genre-tagged ratings falling in each genre (row sums to 1)",
                      "sum of items_clean.genre_* over the user's train ratings, row-normalised; "
                      "no history -> genre_popularity.n_ratings normalised (global prior)",
                      True, ("hybrid",), "user_genre_affinity", tuple(GENRE_COLS)),
    # ------------------------------------------------------------------ item (behavioural, as of cutoff)
    _item("item_n_ratings", "int64", "Popularity: number of ratings received before as_of",
          "item_popularity.n_ratings", used_by=("evaluation", "serving", "eda")),
    _item("item_avg_rating", "float64", "Raw mean rating of the item", "item_popularity.mean_rating"),
    _item("item_bayesian_avg_rating", "float64",
          "Mean rating shrunk to the global mean (damping = features.min_item_ratings_for_avg)",
          "item_popularity.bayesian_avg_rating = (n * mean + m * global_mean) / (n + m)",
          used_by=("serving", "eda")),
    _item("item_rating_std", "float64", "Population std of the item's ratings (0 for one rating)",
          "item_popularity.std_rating"),
    _item("item_popularity_rank", "int64", "Ordinal popularity rank (1 = most rated; cold items rank last)",
          "item_popularity.popularity_rank"),
    _item("item_popularity_pct", "float64", "Popularity percentile among rated items (1 = most rated; cold = 0)",
          "item_popularity.popularity_pct"),
    _item("item_days_since_last_interaction", "float64", "Recency: days between the item's last rating and as_of",
          "(as_of - item_popularity.last_ts) / 86400"),
    # ------------------------------------------------------------------ item (static catalogue)
    _item("item_year", "int64", "Release year (missing years imputed with the catalogue median)",
          "items_clean.year", pit=False),
    _item("item_age_years", "float64", "Item age at as_of in years (clipped at 0)",
          "max(year(as_of) - item_year, 0)"),
    _item("item_n_genres", "int64", "Number of genres the item belongs to", "items_clean.n_genres", pit=False),
    FeatureDefinition("genre_flags", "item", "int8", "One-hot genre membership flags (19 MovieLens genres)",
                      "items_clean.genre_*", False, ("hybrid", "evaluation", "eda"), "item_features",
                      tuple(GENRE_COLS)),
    # ------------------------------------------------------------------ user x item (computed on demand)
    FeatureDefinition("user_item_genre_affinity", "user_item", "float64",
                      "Content match: cosine similarity of user genre affinity and item genre flags (0..1)",
                      "cos(user_genre_affinity[u], genre_flags[i]); 0 if either vector is zero",
                      True, ("hybrid", "serving"), ON_DEMAND),
    FeatureDefinition("user_item_avg_rating_gap", "user_item", "float64",
                      "User rating bias relative to the item's average rating",
                      "user_avg_rating[u] - item_avg_rating[i]", True, ("eda",), ON_DEMAND),
    FeatureDefinition("item_seen_by_user", "user_item", "int8",
                      "1 if the user rated the item before as_of (models apply the same seen-item exclusion internally)",
                      "(u, i) in train", True, ("eda",), ON_DEMAND),
]


# --------------------------------------------------------------------------- lookups
def definitions_for(table: str) -> list[FeatureDefinition]:
    """Registry entries stored in ``table`` (or computed ``on_demand``), in registry order."""
    return [fd for fd in FEATURE_REGISTRY if fd.table == table]


def feature_columns(table: str) -> list[str]:
    """Stored feature column names of ``table`` (excluding the key and metadata columns)."""
    return [col for fd in definitions_for(table) for col in fd.stored_columns]


def expected_schema(table: str) -> dict[str, str]:
    """Ordered ``{column: dtype}`` of a materialised table: key, registry features, metadata."""
    schema = {TABLE_KEYS[table]: "int64"}
    schema.update({col: fd.dtype for fd in definitions_for(table) for col in fd.stored_columns})
    schema.update(METADATA_COLUMNS)
    return schema


def validate_feature_frame(df: pd.DataFrame, table: str) -> None:
    """Check a built table against the registry: column order, dtypes, unique key and no nulls.

    Raises:
        FeatureValidationError: listing every problem found.
    """
    expected = expected_schema(table)
    problems = []
    if list(df.columns) != list(expected):
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        problems.append(f"column mismatch (missing={missing}, unexpected={extra}, order must match registry)")
    problems += [f"{col}: dtype {df[col].dtype} != {dtype}" for col, dtype in expected.items()
                 if col in df.columns and str(df[col].dtype) != dtype]
    key = TABLE_KEYS[table]
    if key in df.columns and df[key].duplicated().any():
        problems.append(f"duplicate {key} values")
    null_cols = [c for c in df.columns if df[c].isna().any()]
    if null_cols:
        problems.append(f"null values in {null_cols}")
    if problems:
        raise FeatureValidationError(f"{table}: " + "; ".join(problems))


# --------------------------------------------------------------------------- rendering
def _escape(value: object) -> str:
    """Make a value safe for a markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_escape(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def display_name(fd: FeatureDefinition) -> str:
    """Catalogue label of a definition: the name, or ``name (genre_*, 19 cols)`` for families."""
    return f"{fd.name} (genre_*, {len(fd.columns)} cols)" if fd.columns else fd.name


def render_catalog(definitions: Iterable[FeatureDefinition] = FEATURE_REGISTRY) -> str:
    """Render the registry as the catalogue table (Feature | Entity | Type | ... | Used by)."""
    headers = ["Feature", "Entity", "Type", "Description", "Logic/Source", "Point-in-time", "Used by"]
    rows = [[f"`{display_name(fd)}`", fd.entity, fd.dtype, fd.description, f"`{fd.logic}`",
             "yes" if fd.point_in_time else "no (static)", ", ".join(fd.used_by)] for fd in definitions]
    return markdown_table(headers, rows)
