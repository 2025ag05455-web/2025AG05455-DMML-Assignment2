"""Declarative validation rules: entity specs, Pandera schemas and the check catalogue.

* ``EntitySpec`` describes one raw entity: its raw columns, the typed output contract
  (``docs/data_contracts.md``), the primary key and which columns are *required*
  (a missing / unparsable value is a hard error) versus *attributes* (soft: warning + null).
* ``build_schema`` is the single Pandera ``DataFrameSchema`` factory, used in two modes:
    - diagnostic (``strict=False``): nullable dtypes and every column nullable, so lazy
      validation pinpoints the offending rows of each range / allowed-value / uniqueness /
      foreign-key rule without re-reporting values already flagged as missing or un-parsable;
    - contract (``strict=True``): exact dtypes, nullability, column set and order - the
      post-condition gate on the clean output.
* ``CHECKS`` is the catalogue of check ids used in error files, logs and reports.

Pandera labels each failure case with the ``error`` string of the check; we set it to our
stable check id (``name=`` cannot be used on built-in checks: it selects the implementation).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd
import pandera as pa

from src.ingestion.ingest import GENRE_COLS, INTERACTION_COLS, ITEM_COLS, USER_COLS

ERROR = "error"
WARNING = "warning"
ALL_COLUMNS = "*"  # error_column value for row-level checks (e.g. exact duplicates)
RELEASE_DATE_FORMAT = "%d-%b-%Y"  # MovieLens dd-Mon-YYYY, e.g. 01-Jan-1995

# Compared after stripping whitespace and lower-casing.
NULL_TOKENS: frozenset[str] = frozenset({"", "na", "n/a", "null", "none", "nan"})
# Values that look like a null token but are legitimate categories of a column:
# 'none' is one of the 21 official MovieLens occupations (ml-100k/u.occupation).
NULL_TOKEN_EXEMPTIONS: Mapping[str, frozenset[str]] = {"occupation": frozenset({"none"})}

PANDERA_UNIQUE_LABEL = "field_uniqueness"  # Pandera's label for Column(unique=True) failures
FOREIGN_KEYS: Mapping[str, str] = {"user_id": "users", "item_id": "items"}  # interactions FK -> parent

Rules = dict[str, list[pa.Check]]

ACTION_REJECT = "row rejected"
ACTION_QUARANTINE = "row quarantined (first occurrence kept)"
ACTION_NULLIFY = "row kept, value set to null"
ACTION_FLAG = "row kept, resolved in preparation"
ACTION_ABORT = "stage aborted"


@dataclass(frozen=True)
class EntitySpec:
    """Static description of one entity of the raw landing zone."""

    name: str
    raw_columns: tuple[str, ...]
    output_dtypes: dict[str, str]  # typed contract of data/interim/<name>.parquet (ordered)
    required: frozenset[str]  # missing / invalid value -> hard error (row rejected)
    rules: Callable[[dict], Rules]  # vcfg -> {column: [pandera checks]}
    key: str | None = None  # primary key (uniqueness enforced)
    pair_key: tuple[str, ...] | None = None  # duplicate combination reported as a warning
    date_formats: dict[str, str] = field(default_factory=dict)

    @property
    def output_columns(self) -> list[str]:
        """Columns of the validated output, in contract order."""
        return list(self.output_dtypes)

    @property
    def integer_columns(self) -> list[str]:
        """Output columns with an integer contract dtype."""
        return [c for c, t in self.output_dtypes.items() if t.lower().startswith("int")]

    @property
    def dropped_columns(self) -> list[str]:
        """Raw columns that are not carried into the validated layer."""
        return [c for c in self.raw_columns if c not in self.output_dtypes]


@dataclass(frozen=True)
class CheckInfo:
    """Catalogue entry of a check: description template, severity and engine."""

    description: str  # may reference cfg["validation"] keys, e.g. {age_min}
    severity: str | None = None  # None -> ERROR for required columns, WARNING for attributes
    method: str = "custom"  # "custom" (vectorised pandas) | "pandera"


CHECKS: dict[str, CheckInfo] = {
    "required_columns": CheckInfo("all contract columns are present in the raw file", ERROR),
    "exact_duplicate": CheckInfo("row is an exact copy of an earlier row", ERROR),
    "missing_value": CheckInfo("value is empty or a null token (NA, N/A, null, None, nan)"),
    "invalid_type": CheckInfo("value cannot be parsed to the contract type"),
    "non_integer_value": CheckInfo("numeric value has a fractional part where an integer is required"),
    "duplicate_key": CheckInfo("primary key occurs more than once (first occurrence kept)", ERROR, "pandera"),
    "rating_not_allowed": CheckInfo("rating not in allowed_ratings {allowed_ratings}", ERROR, "pandera"),
    "timestamp_out_of_range": CheckInfo("unix timestamp outside [{min_timestamp}, {max_timestamp}]",
                                        ERROR, "pandera"),
    "referential_integrity": CheckInfo("foreign key not found in the validated parent table", ERROR, "pandera"),
    "genre_flag_not_binary": CheckInfo("genre flag is not 0/1", ERROR, "pandera"),
    "age_out_of_range": CheckInfo("age outside [{age_min}, {age_max}]", WARNING, "pandera"),
    "gender_not_allowed": CheckInfo("gender not in allowed_genders {allowed_genders}", WARNING, "pandera"),
    "duplicate_interaction": CheckInfo("(user_id, item_id) pair occurs more than once", WARNING),
}


# --------------------------------------------------------------------------- rules (config driven)
def timestamp_bounds(vcfg: dict) -> tuple[int, int]:
    """Return the configured ``[min_timestamp, max_timestamp]`` window as unix seconds."""
    return (int(pd.Timestamp(vcfg["min_timestamp"]).timestamp()),
            int(pd.Timestamp(vcfg["max_timestamp"]).timestamp()))


def _user_rules(vcfg: dict) -> Rules:
    return {
        "age": [pa.Check.in_range(vcfg["age_min"], vcfg["age_max"], error="age_out_of_range")],
        "gender": [pa.Check.isin(vcfg["allowed_genders"], error="gender_not_allowed")],
    }


def _item_rules(vcfg: dict) -> Rules:  # noqa: ARG001 - uniform signature
    return {col: [pa.Check.isin([0, 1], error="genre_flag_not_binary")] for col in GENRE_COLS}


def _interaction_rules(vcfg: dict) -> Rules:
    ts_min, ts_max = timestamp_bounds(vcfg)
    return {
        "rating": [pa.Check.isin(vcfg["allowed_ratings"], error="rating_not_allowed")],
        "timestamp": [pa.Check.in_range(ts_min, ts_max, error="timestamp_out_of_range")],
    }


USERS = EntitySpec(
    name="users",
    raw_columns=tuple(USER_COLS),
    output_dtypes={"user_id": "int64", "age": "Int64", "gender": "string",
                   "occupation": "string", "zip_code": "string"},
    required=frozenset({"user_id"}),
    rules=_user_rules,
    key="user_id",
)
ITEMS = EntitySpec(
    name="items",
    raw_columns=tuple(ITEM_COLS),
    output_dtypes={"item_id": "int64", "title": "string", "release_date": "datetime64[ns]",
                   "imdb_url": "string", **dict.fromkeys(GENRE_COLS, "int8")},
    required=frozenset({"item_id", *GENRE_COLS}),
    rules=_item_rules,
    key="item_id",
    date_formats={"release_date": RELEASE_DATE_FORMAT},
)
INTERACTIONS = EntitySpec(
    name="interactions",
    raw_columns=tuple(INTERACTION_COLS),
    output_dtypes=dict.fromkeys(INTERACTION_COLS, "int64"),
    required=frozenset(INTERACTION_COLS),
    rules=_interaction_rules,
    pair_key=("user_id", "item_id"),
)
# Validation order matters: parents first so foreign keys are checked against *validated* rows.
ENTITIES: dict[str, EntitySpec] = {spec.name: spec for spec in (USERS, ITEMS, INTERACTIONS)}


# --------------------------------------------------------------------------- catalogue helpers
def severity_of(check: str, column: str, spec: EntitySpec) -> str:
    """Severity of a failed check: fixed per check, or derived from the column's role."""
    fixed = CHECKS[check].severity
    return fixed or (ERROR if column in spec.required else WARNING)


def action_of(check: str, severity: str) -> str:
    """Remediation applied to rows failing ``check`` (documented in the report)."""
    if check == "required_columns":
        return ACTION_ABORT
    if check == "exact_duplicate":
        return ACTION_QUARANTINE
    if check == "duplicate_interaction":
        return ACTION_FLAG
    return ACTION_REJECT if severity == ERROR else ACTION_NULLIFY


def describe(check: str, vcfg: dict) -> str:
    """Human-readable rule text of ``check`` with configuration values filled in."""
    return CHECKS[check].description.format(**vcfg)


# --------------------------------------------------------------------------- pandera schemas
def diagnostic_dtype(dtype: str) -> str:
    """Nullable working dtype used before rejection (all integers -> pandas ``Int64``)."""
    return "Int64" if dtype.lower().startswith("int") else dtype


def build_schema(spec: EntitySpec, vcfg: dict, *, strict: bool,
                 references: Mapping[str, Iterable[int]] | None = None) -> pa.DataFrameSchema:
    """Build the Pandera schema of ``spec`` in diagnostic (``strict=False``) or contract mode.

    ``references`` maps foreign-key columns to the ids of the validated parent table and
    adds a referential-integrity ``isin`` check per column.
    """
    rules = spec.rules(vcfg)
    for column, ids in (references or {}).items():
        rules.setdefault(column, []).append(
            pa.Check.isin(sorted(set(ids)), error="referential_integrity"))
    columns = {
        col: pa.Column(
            dtype if strict else diagnostic_dtype(dtype),
            checks=rules.get(col, []),
            nullable=(col not in spec.required) if strict else True,
            unique=col == spec.key,
            report_duplicates="exclude_first",
        )
        for col, dtype in spec.output_dtypes.items()
    }
    return pa.DataFrameSchema(
        columns,
        strict=strict,
        ordered=strict,
        # without a primary key, the contract still forbids exact duplicate rows
        unique=spec.output_columns if strict and spec.key is None else None,
        report_duplicates="exclude_first",
        name=f"{spec.name}_{'contract' if strict else 'diagnostic'}",
    )


def schema_rules(schema: pa.DataFrameSchema) -> list[tuple[str, str, str]]:
    """List ``(column, pandera failure label, check id)`` for every column-level rule."""
    rules: list[tuple[str, str, str]] = []
    for name, column in schema.columns.items():
        if column.unique:
            rules.append((name, PANDERA_UNIQUE_LABEL, "duplicate_key"))
        rules.extend((name, check.error, check.error) for check in column.checks)
    return rules


def dtype_mismatches(df: pd.DataFrame, spec: EntitySpec) -> dict[str, str]:
    """Exact dtype comparison with the contract.

    Pandera treats ``Int64``/``int64`` and datetime units as compatible, so the contract
    gate complements it with this string-exact check.
    """
    actual = {col: str(dtype) for col, dtype in df.dtypes.items()}
    expected = spec.output_dtypes
    return {col: f"expected {expected.get(col, '<absent>')}, got {actual.get(col, '<absent>')}"
            for col in sorted(expected.keys() | actual.keys()) if expected.get(col) != actual.get(col)}
