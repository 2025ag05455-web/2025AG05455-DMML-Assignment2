"""Stage 2 - Data validation.

Reads the raw landing CSVs (every column a string, exactly as received) and validates the
entities in dependency order - users, items, then interactions - so that foreign keys are
checked against the *validated* parent tables. Per entity:

1. structure        all contract columns present, otherwise abort                    (custom)
2. normalise        strip whitespace; "", NA, N/A, null, None, nan -> null           (custom)
3. exact duplicates repeated rows quarantined, first occurrence kept                 (custom)
4. missing values   every output column                                              (custom)
5. type coercion    non-numeric / non-integer numbers, unparsable dates              (custom)
6. rules            ranges, allowed values, key uniqueness, foreign keys             (Pandera, lazy)
7. policy           hard errors reject the row; soft attribute problems null the value
8. duplicate pairs  repeated (user_id, item_id) -> warning only (preparation)        (custom)
9. contract gate    strict Pandera schema + exact dtypes on the clean output         (Pandera)

Quality gate: the stage raises ``DataValidationError`` when the interaction hard-error rate
exceeds ``validation.max_error_rate``. Error files and reports are still written for triage,
and stale interim outputs are removed so downstream stages cannot consume them.

Outputs: data/interim/{users,items,interactions}.parquet, data/errors/<entity>_errors.csv,
reports/validation_report.{json,md}.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, path
from src.utils.io import read_json, read_parquet, sha256_file, write_parquet
from src.utils.logger import get_logger
from src.validation.checks import (
    IssueLog, coerce, find_duplicates, lazy_failures, missing_detail, normalise, record_schema_failures,
)
from src.validation.exceptions import DataValidationError
from src.validation.report import build_report, relative, write_reports
from src.validation.schemas import (
    ACTION_NULLIFY, ALL_COLUMNS, ENTITIES, ERROR, FOREIGN_KEYS, INTERACTIONS, ITEMS, USERS, WARNING,
    EntitySpec, build_schema, dtype_mismatches,
)

log = get_logger("validation")

ERROR_FILE_TAIL = ["error_type", "error_column", "error_detail", "severity"]


@dataclass
class EntityResult:
    """Outcome of validating one entity."""

    spec: EntitySpec
    raw: pd.DataFrame  # as received (strings); index = source_row (0-based data row of the CSV)
    clean: pd.DataFrame  # typed rows that passed every hard check (contract dtypes)
    issues: pd.DataFrame  # one row per failing (source_row, check, column)
    checks: list[dict]  # one summary per executed check, including passed ones

    @property
    def n_input(self) -> int:
        """Rows in the raw file."""
        return len(self.raw)

    @property
    def rejected_rows(self) -> set[int]:
        """Source rows rejected by at least one hard error (incl. exact duplicates)."""
        return set(self.issues.loc[self.issues["severity"] == ERROR, "source_row"])

    @property
    def warning_rows(self) -> set[int]:
        """Kept source rows carrying at least one warning."""
        return set(self.issues.loc[self.issues["severity"] == WARNING, "source_row"]) - self.rejected_rows

    @property
    def n_rejected(self) -> int:
        """Number of rejected source rows."""
        return len(self.rejected_rows)

    def error_records(self) -> pd.DataFrame:
        """Rows of the error file: source_row, raw columns, error_type/column/detail, severity."""
        merged = self.issues.merge(self.raw, left_on="source_row", right_index=True, how="left")
        columns = ["source_row", *self.raw.columns, *ERROR_FILE_TAIL]
        return merged[columns].sort_values(["source_row", "severity"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- core (pure, no I/O)
def validate_entity(spec: EntitySpec, raw: pd.DataFrame, vcfg: dict,
                    references: Mapping[str, pd.Series] | None = None) -> EntityResult:
    """Run every check on one raw entity; return the clean typed rows and the issue log.

    Args:
        spec: entity description (``schemas.USERS`` / ``ITEMS`` / ``INTERACTIONS``).
        raw: raw frame with string columns as read from the landing CSV.
        vcfg: ``cfg["validation"]``.
        references: foreign-key column -> ids of the validated parent table.
    """
    raw = raw.reset_index(drop=True)
    issues = IssueLog(spec, vcfg, log)
    _check_structure(raw, spec, issues)
    norm = normalise(raw[list(spec.raw_columns)])

    is_dup, first = find_duplicates(norm, norm.columns)
    issues.record("exact_duplicate", ALL_COLUMNS, is_dup, "exact duplicate of source_row " + first.astype(str))
    unique_rows = norm.loc[~is_dup]
    for column in spec.output_columns:
        issues.record("missing_value", column, unique_rows[column].isna(), missing_detail(raw[column], column))

    typed = coerce(unique_rows, spec, issues)
    record_schema_failures(issues, build_schema(spec, vcfg, strict=False, references=references), typed)

    kept = typed.loc[~typed.index.isin(issues.rows(ERROR))]
    if spec.pair_key:
        _check_pairs(kept, spec.pair_key, issues)
    clean = apply_policy(kept, issues.issues(), spec)
    enforce_contract(clean, spec, vcfg, references)
    return EntityResult(spec, raw, clean, issues.issues(), issues.results)


def _check_structure(raw: pd.DataFrame, spec: EntitySpec, issues: IssueLog) -> None:
    """Abort when contract columns are missing; unexpected extra columns are ignored."""
    missing = [c for c in spec.raw_columns if c not in raw.columns]
    if missing:
        raise DataValidationError(f"{spec.name}: raw file lacks contract columns {missing}",
                                  {"entity": spec.name, "missing_columns": missing})
    extra = [c for c in raw.columns if c not in spec.raw_columns]
    if extra:
        log.warning("%s: ignoring unexpected raw columns %s", spec.name, extra)
    issues.record("required_columns", ALL_COLUMNS, pd.Series(False, index=list(spec.raw_columns)), "")


def _check_pairs(kept: pd.DataFrame, pair: tuple[str, ...], issues: IssueLog) -> None:
    """Warn about repeated key combinations among kept rows (resolved by preparation)."""
    is_dup, first = find_duplicates(kept, pair)
    label = f"({', '.join(pair)})"
    issues.record("duplicate_interaction", "+".join(pair), is_dup,
                  f"{label} also present at source_row " + first.astype(str))


def apply_policy(kept: pd.DataFrame, issues: pd.DataFrame, spec: EntitySpec) -> pd.DataFrame:
    """Null soft-failed attribute values and cast the surviving rows to the contract dtypes."""
    soft = issues[(issues["action"] == ACTION_NULLIFY) & issues["source_row"].isin(kept.index)]
    clean = kept.copy()
    for column, rows in soft.groupby("error_column")["source_row"]:
        clean[column] = clean[column].mask(clean.index.isin(rows))
    return clean.astype(spec.output_dtypes).reset_index(drop=True)


def enforce_contract(clean: pd.DataFrame, spec: EntitySpec, vcfg: dict,
                     references: Mapping[str, pd.Series] | None = None) -> None:
    """Post-condition gate: the clean output must satisfy the strict contract schema exactly."""
    mismatches = dtype_mismatches(clean, spec)
    failures = lazy_failures(build_schema(spec, vcfg, strict=True, references=references), clean)
    if mismatches or len(failures):
        raise DataValidationError(
            f"{spec.name}: clean output violates the data contract",
            {"dtype_mismatches": mismatches,
             "schema_failures": failures.head(20).astype(str).to_dict("records")})
    log.info("%-12s | contract gate passed (%d rows x %d columns)", spec.name, *clean.shape)


def validate_all(raw: Mapping[str, pd.DataFrame], vcfg: dict) -> dict[str, EntityResult]:
    """Validate users and items first, then interactions against the validated parents."""
    results = {spec.name: validate_entity(spec, raw[spec.name], vcfg) for spec in (USERS, ITEMS)}
    references = {col: results[parent].clean[col] for col, parent in FOREIGN_KEYS.items()}
    results[INTERACTIONS.name] = validate_entity(INTERACTIONS, raw[INTERACTIONS.name], vcfg, references)
    return results


def quality_gate(result: EntityResult, max_error_rate: float) -> dict:
    """Fail-fast gate on the share of rejected (hard-error) rows of one entity."""
    rate = result.n_rejected / result.n_input if result.n_input else 0.0
    return {"entity": result.spec.name, "metric": "hard_error_rate", "value": round(rate, 6),
            "threshold": float(max_error_rate), "n_rejected": result.n_rejected,
            "n_input": result.n_input, "passed": bool(rate <= max_error_rate)}


# --------------------------------------------------------------------------- I/O
def read_raw(csv_path: Path) -> pd.DataFrame:
    """Read a landing CSV keeping every value as the exact string received."""
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False)


def describe_inputs(inputs: Mapping[str, Path], results: Mapping[str, EntityResult], cfg: dict) -> dict:
    """Lineage of the raw inputs: file, row count and checksum."""
    return {name: {"file": relative(p, cfg), "rows": results[name].n_input, "sha256": sha256_file(p)}
            for name, p in inputs.items()}


def write_errors(result: EntityResult, cfg: dict) -> dict:
    """Write ``data/errors/<entity>_errors.csv`` (header-only when the entity is clean)."""
    records = result.error_records()
    target = path(cfg, "errors_dir", f"{result.spec.name}_errors.csv")
    records.to_csv(target, index=False)
    counts = records["severity"].value_counts()
    return {"file": relative(target, cfg), "records": len(records),
            "errors": int(counts.get(ERROR, 0)), "warnings": int(counts.get(WARNING, 0))}


def write_interim(results: Mapping[str, EntityResult], cfg: dict) -> dict:
    """Write the clean frames to parquet and verify the dtypes survive the round trip."""
    outputs = {}
    for name, result in results.items():
        target = write_parquet(result.clean, path(cfg, "interim_dir", f"{name}.parquet"))
        reloaded = read_parquet(target)
        mismatches = dtype_mismatches(reloaded, result.spec)
        if mismatches:
            raise DataValidationError(f"{name}: parquet round trip changed dtypes", mismatches)
        outputs[name] = {"file": relative(target, cfg), "rows": len(reloaded),
                         "dtypes": {col: str(dtype) for col, dtype in reloaded.dtypes.items()}}
    return outputs


def invalidate_interim(cfg: dict) -> dict:
    """Remove interim outputs of a previous run so a failed gate cannot leak stale data."""
    for name in ENTITIES:
        path(cfg, "interim_dir", f"{name}.parquet").unlink(missing_ok=True)
    return {}


def load_ground_truth(cfg: dict) -> dict | None:
    """Injected-anomaly counts written by ingestion (``None`` when nothing was injected)."""
    truth_path = path(cfg, "raw_dir", "injected_anomalies.json")
    return read_json(truth_path).get("anomalies") if truth_path.exists() else None


# --------------------------------------------------------------------------- entrypoint
def run(cfg: dict | None = None) -> dict:
    """Validate the raw landing files and write interim data, error files and reports.

    Raises:
        DataValidationError: missing contract columns, failed contract gate, or interaction
            hard-error rate above ``validation.max_error_rate``.
    """
    cfg = cfg or load_config()
    vcfg = cfg["validation"]
    started = datetime.now(timezone.utc)
    inputs = {name: path(cfg, "raw_dir", f"{name}.csv") for name in ENTITIES}
    log.info("Validating %s", {name: relative(p, cfg) for name, p in inputs.items()})

    results = validate_all({name: read_raw(p) for name, p in inputs.items()}, vcfg)
    gate = quality_gate(results[INTERACTIONS.name], vcfg["max_error_rate"])
    error_files = {name: write_errors(result, cfg) for name, result in results.items()}
    outputs = write_interim(results, cfg) if gate["passed"] else invalidate_interim(cfg)
    report = build_report(results=results, gate=gate, cfg=cfg, started=started,
                          inputs=describe_inputs(inputs, results, cfg), outputs=outputs,
                          error_files=error_files, truth=load_ground_truth(cfg))
    report_files = write_reports(report, cfg)
    report_files = {fmt: relative(p, cfg) for fmt, p in report_files.items()}

    if not gate["passed"]:
        log.error("Quality gate FAILED: interaction hard-error rate %.4f > %.4f",
                  gate["value"], gate["threshold"])
        raise DataValidationError("Interaction hard-error rate exceeds validation.max_error_rate",
                                  {"quality_gate": gate, "reports": report_files})

    summary = {
        "stage": "validation",
        "status": report["status"],
        "rows_in": {name: r.n_input for name, r in results.items()},
        "rows_out": {name: len(r.clean) for name, r in results.items()},
        "rejected_rows": {name: r.n_rejected for name, r in results.items()},
        "warning_rows": {name: len(r.warning_rows) for name, r in results.items()},
        "interaction_hard_error_rate": gate["value"],
        "ground_truth_recall": report["ground_truth"].get("overall", {}).get("recall"),
        "outputs": {name: o["file"] for name, o in outputs.items()},
        "reports": report_files,
    }
    log.info("Validation complete: %s", summary)
    return summary


if __name__ == "__main__":
    run()
