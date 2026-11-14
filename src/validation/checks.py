"""Vectorised custom checks and the per-entity ``IssueLog``.

Pandera (``schemas.py``) covers the declarative single-table rules. The helpers here cover
what it is awkward at: null-token normalisation, *type coercion failures* (a value that
cannot become an integer or a date never reaches a typed schema), exact duplicate rows and
duplicate key combinations - plus mapping Pandera's lazy ``failure_cases`` back to rows.

Every outcome is recorded in an ``IssueLog`` as one boolean mask per (check, column), so the
error files, the reject / nullify policy and the report are all derived from one record.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd
import pandera as pa

from src.validation.exceptions import DataValidationError
from src.validation.schemas import (
    CHECKS, NULL_TOKEN_EXEMPTIONS, NULL_TOKENS, EntitySpec, action_of, describe, schema_rules, severity_of,
)

ISSUE_COLUMNS = ["source_row", "error_type", "error_column", "error_detail", "severity", "action"]
_FAILURE_CASE_COLUMNS = ["schema_context", "column", "check", "check_number", "failure_case", "index"]


class IssueLog:
    """Collects the outcome of every check run on one entity.

    ``results`` holds one summary dict per executed check (also for checks that passed);
    ``issues()`` returns one row per failing (source row, check, column).
    """

    def __init__(self, spec: EntitySpec, vcfg: dict, logger: logging.Logger) -> None:
        self.spec = spec
        self.vcfg = vcfg
        self.results: list[dict] = []
        self._logger = logger
        self._issues: list[pd.DataFrame] = []

    def record(self, check: str, column: str, failed: pd.Series, detail: pd.Series | str) -> None:
        """Record a check outcome.

        Args:
            check: check id from ``schemas.CHECKS``.
            column: checked column (``ALL_COLUMNS`` for row-level checks).
            failed: boolean mask indexed by source row (``True`` = row fails the check).
            detail: per-row message (Series aligned with ``failed``) or one message for all rows.
        """
        failed = failed.fillna(False).astype(bool)
        severity = severity_of(check, column, self.spec)
        action = action_of(check, severity)
        rows = failed.index[failed.to_numpy()]
        self._add_result(check, column, severity, action, n_checked=len(failed), n_failed=len(rows))
        if len(rows):
            text = detail.loc[rows].to_numpy() if isinstance(detail, pd.Series) else detail
            self._issues.append(pd.DataFrame({
                "source_row": rows.astype("int64"), "error_type": check, "error_column": column,
                "error_detail": text, "severity": severity, "action": action,
            }))

    def issues(self) -> pd.DataFrame:
        """All recorded failures, one row per (source row, check, column)."""
        if not self._issues:
            return pd.DataFrame(columns=ISSUE_COLUMNS).astype({"source_row": "int64"})
        return pd.concat(self._issues, ignore_index=True)

    def rows(self, severity: str) -> pd.Index:
        """Distinct source rows with at least one failure of the given severity."""
        issues = self.issues()
        return pd.Index(issues.loc[issues["severity"] == severity, "source_row"].unique())

    def _add_result(self, check: str, column: str, severity: str, action: str,
                    n_checked: int, n_failed: int) -> None:
        pct = round(100.0 * n_failed / n_checked, 4) if n_checked else 0.0
        self.results.append({
            "entity": self.spec.name, "check": check, "column": column, "severity": severity,
            "method": CHECKS[check].method, "n_checked": n_checked, "n_failed": n_failed,
            "pct_failed": pct, "action": action, "rule": describe(check, self.vcfg),
        })
        self._logger.log(
            logging.WARNING if n_failed else logging.INFO,
            "%-12s | %-22s | %-14s | %-7s | failed %6d / %-6d (%7.3f%%) | %s",
            self.spec.name, check, column, severity, n_failed, n_checked, pct,
            action if n_failed else "passed",
        )


# --------------------------------------------------------------------------- normalisation
def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and turn null tokens into ``<NA>`` (``string`` dtype frame).

    Tokens are matched case-insensitively; ``NULL_TOKEN_EXEMPTIONS`` keeps legitimate
    category values such as the MovieLens occupation ``'none'``.
    """
    stripped = raw.astype("string").apply(lambda s: s.str.strip())
    is_token = stripped.apply(
        lambda s: s.str.lower().isin(NULL_TOKENS - NULL_TOKEN_EXEMPTIONS.get(s.name, frozenset())))
    return stripped.mask(is_token.fillna(False).astype(bool))


def missing_detail(raw_values: pd.Series, column: str) -> pd.Series:
    """Per-row message for a missing value, quoting the raw token (e.g. ``'N/A'``)."""
    return f"{column} is missing (raw value '" + raw_values.astype("string").fillna("") + "')"


# --------------------------------------------------------------------------- duplicates
def find_duplicates(df: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.Series, pd.Series]:
    """Flag rows repeating an earlier row on ``columns`` (first occurrence is not flagged).

    Returns the boolean mask and, for every row, the source row of its first occurrence.
    """
    group = df.groupby(list(columns), dropna=False, sort=False).ngroup().to_numpy()
    first = pd.Series(df.index, index=df.index).groupby(group).transform("first")
    return first != df.index, first


# --------------------------------------------------------------------------- type coercion
def to_integer(values: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Parse strings to ``Int64``; return (parsed, non-numeric mask, non-integer mask).

    Integral numbers written as decimals (``"4.0"``) are accepted; failures become ``<NA>``.
    """
    number = pd.to_numeric(values, errors="coerce").astype("float64")
    non_numeric = values.notna() & number.isna()
    non_integer = number.notna() & (number % 1 != 0)  # also catches +/-inf
    return number.mask(non_integer).astype("Int64"), non_numeric, non_integer


def to_datetime(values: pd.Series, fmt: str) -> tuple[pd.Series, pd.Series]:
    """Parse strings with ``fmt``; return (``datetime64[ns]`` values, unparsable mask)."""
    parsed = pd.to_datetime(values, format=fmt, errors="coerce").astype("datetime64[ns]")
    return parsed, values.notna() & parsed.isna()


def coerce(norm: pd.DataFrame, spec: EntitySpec, log: IssueLog) -> pd.DataFrame:
    """Cast the normalised string frame to diagnostic dtypes, recording type failures.

    Un-parsable values become null here; whether that rejects the row (required column)
    or only nulls the attribute is decided by the severity policy.
    """
    return pd.DataFrame({col: _coerce_column(norm[col], col, spec, log) for col in spec.output_columns},
                        index=norm.index)


def _coerce_column(values: pd.Series, column: str, spec: EntitySpec, log: IssueLog) -> pd.Series:
    quoted = f"{column}='" + values + "'"
    if column in spec.date_formats:
        fmt = spec.date_formats[column]
        parsed, unparsable = to_datetime(values, fmt)
        log.record("invalid_type", column, unparsable, quoted + f" does not match format {fmt}")
        return parsed
    if column in spec.integer_columns:
        parsed, non_numeric, non_integer = to_integer(values)
        log.record("invalid_type", column, non_numeric, quoted + " is not numeric")
        log.record("non_integer_value", column, non_integer, quoted + " is not an integer")
        return parsed
    return values


# --------------------------------------------------------------------------- pandera bridge
def lazy_failures(schema: pa.DataFrameSchema, df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``df`` lazily and return Pandera's failure cases (empty frame when valid)."""
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return exc.failure_cases
    return pd.DataFrame(columns=_FAILURE_CASE_COLUMNS)


def record_schema_failures(log: IssueLog, schema: pa.DataFrameSchema, typed: pd.DataFrame) -> None:
    """Run the diagnostic schema and record each rule's offending rows in ``log``.

    Null failure cases of uniqueness rules (several missing keys) are ignored: missing keys
    are already reported by ``missing_value``. Any failure case not produced by a known rule
    (e.g. a dtype or column mismatch) is a programming error and aborts the stage.
    """
    failures = lazy_failures(schema, typed)
    matched = pd.Series(False, index=failures.index)
    for column, label, check in schema_rules(schema):
        hit = (failures["column"] == column) & (failures["check"] == label)
        matched |= hit
        cases = failures.loc[hit & failures["failure_case"].notna() & failures["index"].notna()]
        values = pd.Series(cases["failure_case"].to_numpy(), index=cases["index"].astype("int64").to_numpy())
        values = values[~values.index.duplicated()]
        detail = f"{column}=" + values.astype(str) + f": {describe(check, log.vcfg)}"
        log.record(check, column, pd.Series(typed.index.isin(values.index), index=typed.index),
                   detail.reindex(typed.index))
    if not matched.all():
        unexpected = failures.loc[~matched, ["column", "check", "failure_case"]].head(10)
        raise DataValidationError(f"{log.spec.name}: unexpected schema failures",
                                  {"failure_cases": unexpected.astype(str).to_dict("records")})
