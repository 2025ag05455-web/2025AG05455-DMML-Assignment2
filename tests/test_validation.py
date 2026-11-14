"""Unit tests for the validation stage (tiny synthetic frames, no real data)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.ingest import GENRE_COLS, INTERACTION_COLS, ITEM_COLS, USER_COLS
from src.utils.config import load_config
from src.utils.io import read_json, write_json
from src.validation.checks import find_duplicates, normalise, to_integer
from src.validation.exceptions import DataValidationError
from src.validation.report import compare_ground_truth, render_markdown
from src.validation.schemas import ERROR, INTERACTIONS, ITEMS, USERS, WARNING
from src.validation.validate import enforce_contract, quality_gate, run, validate_all, validate_entity

# Hermetic copy of the validation config (tests must not depend on configs/config.yaml edits).
VCFG = {
    "rating_min": 1, "rating_max": 5, "allowed_ratings": [1, 2, 3, 4, 5],
    "min_timestamp": "1995-01-01T00:00:00Z", "max_timestamp": "2000-12-31T23:59:59Z",
    "age_min": 5, "age_max": 100, "allowed_genders": ["M", "F"], "max_error_rate": 0.05,
}
TS = "880000000"  # 1997-11-20, inside the validity window
NO_GENRES = ["0"] * len(GENRE_COLS)


def users_raw(*rows: list[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=USER_COLS, dtype=str)


def items_raw(*rows: list[str]) -> pd.DataFrame:
    return pd.DataFrame([r + NO_GENRES if len(r) == 5 else r for r in rows], columns=ITEM_COLS, dtype=str)


def interactions_raw(*rows: list[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=INTERACTION_COLS, dtype=str)


def validate_interactions(*rows: list[str], user_ids=(1, 2), item_ids=(10, 20)):
    refs = {"user_id": pd.Series(user_ids), "item_id": pd.Series(item_ids)}
    return validate_entity(INTERACTIONS, interactions_raw(*rows), VCFG, refs)


def checks_of(result, severity: str | None = None) -> set[tuple[str, str]]:
    issues = result.issues if severity is None else result.issues[result.issues["severity"] == severity]
    return set(zip(issues["error_type"], issues["error_column"]))


# --------------------------------------------------------------------------- building blocks
def test_normalise_maps_null_tokens_and_keeps_exempt_occupation():
    raw = pd.DataFrame({"age": [" 25 ", "NA", "n/a", "NULL", "None", "nan", ""],
                        "occupation": ["none", "None", "student", "x", "y", "z", "w"]})
    norm = normalise(raw)
    assert norm["age"].iloc[0] == "25"
    assert norm["age"].iloc[1:].isna().all()
    assert norm["occupation"].iloc[:2].tolist() == ["none", "None"]  # legitimate category, not null


def test_to_integer_classifies_failures():
    parsed, non_numeric, non_integer = to_integer(pd.Series(["5", "4.0", "4.5", "five", None], dtype="string"))
    assert parsed.iloc[:2].tolist() == [5, 4] and parsed.iloc[2:].isna().all()
    assert non_numeric.tolist() == [False, False, False, True, False]
    assert non_integer.tolist() == [False, False, True, False, False]


def test_find_duplicates_flags_later_occurrences_only():
    df = pd.DataFrame({"a": ["1", "2", "1", None, None]})
    is_dup, first = find_duplicates(df, ["a"])
    assert is_dup.tolist() == [False, False, True, False, True]
    assert first.tolist() == [0, 1, 0, 3, 3]


# --------------------------------------------------------------------------- interactions
@pytest.mark.parametrize("rating, check", [
    ("6", "rating_not_allowed"), ("0", "rating_not_allowed"), ("4.5", "non_integer_value"),
    ("five", "invalid_type"), ("N/A", "missing_value"),
])
def test_invalid_rating_rejected(rating, check):
    result = validate_interactions(["1", "10", "4", TS], ["2", "20", rating, TS])
    assert result.clean["user_id"].tolist() == [1]
    assert checks_of(result) == {(check, "rating")}
    assert result.issues["severity"].eq(ERROR).all()


@pytest.mark.parametrize("timestamp, check", [
    ("-1", "timestamp_out_of_range"), ("4102444800", "timestamp_out_of_range"),
    ("abc", "invalid_type"), ("1999-13-45", "invalid_type"),
])
def test_invalid_timestamp_rejected(timestamp, check):
    result = validate_interactions(["1", "10", "4", TS], ["2", "20", "3", timestamp])
    assert len(result.clean) == 1
    assert checks_of(result) == {(check, "timestamp")}


def test_missing_key_rejected():
    result = validate_interactions(["1", "10", "4", TS], ["", "20", "3", TS], ["2", "null", "3", TS])
    assert len(result.clean) == 1
    assert checks_of(result) == {("missing_value", "user_id"), ("missing_value", "item_id")}


def test_orphan_foreign_keys_rejected():
    result = validate_interactions(["1", "10", "4", TS], ["99", "10", "4", TS], ["1", "999", "4", TS])
    assert result.clean.to_dict("list") == {"user_id": [1], "item_id": [10], "rating": [4], "timestamp": [880000000]}
    assert checks_of(result) == {("referential_integrity", "user_id"), ("referential_integrity", "item_id")}


def test_exact_duplicate_quarantined_first_occurrence_kept():
    result = validate_interactions(["1", "10", "4", TS], ["2", "20", "3", TS], ["1", "10", "4", TS])
    assert len(result.clean) == 2
    errors = result.error_records()
    assert errors["error_type"].tolist() == ["exact_duplicate"]
    assert errors["source_row"].tolist() == [2]
    assert errors["error_detail"].iloc[0] == "exact duplicate of source_row 0"


def test_duplicate_pair_is_warning_only():
    result = validate_interactions(["1", "10", "4", TS], ["1", "10", "5", "880000500"])
    assert len(result.clean) == 2  # preparation resolves it
    assert checks_of(result, WARNING) == {("duplicate_interaction", "user_id+item_id")}
    assert result.n_rejected == 0


def test_interaction_output_matches_contract_dtypes():
    result = validate_interactions(["1", "10", "4", TS], ["2", "20", " 5 ", TS])
    assert result.clean.dtypes.astype(str).to_dict() == INTERACTIONS.output_dtypes


# --------------------------------------------------------------------------- users / items
def test_user_soft_problems_null_the_value_and_keep_the_row():
    result = validate_entity(USERS, users_raw(
        ["1", "abc", "M", "student", "11111"], ["2", "150", "F", "none", "22222"],
        ["3", "", "X", "writer", "33333"], ["4", "30", "M", "other", "44444"]), VCFG)
    assert result.clean["user_id"].tolist() == [1, 2, 3, 4]
    assert result.clean["age"].isna().tolist() == [True, True, True, False]
    assert result.clean["gender"].isna().tolist() == [False, False, True, False]
    assert result.clean["occupation"].iloc[1] == "none"
    assert checks_of(result, WARNING) == {("invalid_type", "age"), ("age_out_of_range", "age"),
                                          ("missing_value", "age"), ("gender_not_allowed", "gender")}
    assert result.n_rejected == 0
    assert result.clean.dtypes.astype(str).to_dict() == USERS.output_dtypes


def test_conflicting_user_key_rejects_later_row():
    result = validate_entity(USERS, users_raw(["1", "20", "M", "student", "11111"],
                                              ["1", "30", "F", "writer", "22222"]), VCFG)
    assert result.clean["age"].tolist() == [20]
    assert checks_of(result) == {("duplicate_key", "user_id")}


def test_item_attribute_warnings_and_dropped_column():
    result = validate_entity(ITEMS, items_raw(
        ["1", "A (1995)", "01-Jan-1995", "", "http://a"], ["2", "", "32-Foo-1995", "", ""]), VCFG)
    clean = result.clean
    assert "video_release_date" not in clean.columns
    assert clean.dtypes.astype(str).to_dict() == ITEMS.output_dtypes
    assert clean["release_date"].iloc[0] == pd.Timestamp("1995-01-01") and pd.isna(clean["release_date"].iloc[1])
    assert checks_of(result, WARNING) == {("missing_value", "title"), ("invalid_type", "release_date"),
                                          ("missing_value", "imdb_url")}


def test_invalid_genre_flag_rejects_item():
    bad = ["2", "B (1996)", "01-Jan-1996", "", "http://b", "2", *NO_GENRES[1:]]
    result = validate_entity(ITEMS, items_raw(["1", "A (1995)", "01-Jan-1995", "", "http://a"], bad), VCFG)
    assert result.clean["item_id"].tolist() == [1]
    assert checks_of(result) == {("genre_flag_not_binary", "genre_unknown")}


def test_interactions_are_checked_against_validated_parents():
    raw = {"users": users_raw(["1", "20", "M", "student", "1"], ["", "20", "F", "writer", "2"]),
           "items": items_raw(["10", "A (1995)", "01-Jan-1995", "", "http://a"]),
           "interactions": interactions_raw(["1", "10", "4", TS], ["1", "11", "4", TS])}
    results = validate_all(raw, VCFG)
    assert len(results["users"].clean) == 1 and len(results["interactions"].clean) == 1
    assert checks_of(results["interactions"]) == {("referential_integrity", "item_id")}


# --------------------------------------------------------------------------- gates
def test_quality_gate_fails_on_high_error_rate():
    result = validate_interactions(["1", "10", "4", TS], ["9", "10", "4", TS])
    gate = quality_gate(result, max_error_rate=0.05)
    assert gate["value"] == 0.5 and gate["passed"] is False
    assert quality_gate(result, max_error_rate=0.5)["passed"] is True


def test_contract_gate_raises_on_dtype_drift():
    result = validate_interactions(["1", "10", "4", TS])
    with pytest.raises(DataValidationError, match="violates the data contract"):
        enforce_contract(result.clean.astype({"rating": "float64"}), INTERACTIONS, VCFG)


def test_contract_gate_raises_on_rule_violation():
    clean = pd.DataFrame({"user_id": [1], "item_id": [10], "rating": [9], "timestamp": [880000000]})
    with pytest.raises(DataValidationError) as exc:
        enforce_contract(clean, INTERACTIONS, VCFG)
    assert "rating_not_allowed" in str(exc.value.details["schema_failures"])


def test_missing_contract_column_aborts():
    with pytest.raises(DataValidationError, match="lacks contract columns"):
        validate_entity(INTERACTIONS, interactions_raw(["1", "10", "4", TS]).drop(columns="rating"), VCFG)


# --------------------------------------------------------------------------- report
def test_ground_truth_comparison_and_reconciliation():
    raw = {"users": users_raw(["1", "", "M", "student", "1"]),
           "items": items_raw(["10", "A (1995)", "01-Jan-1995", "", "http://a"]),
           "interactions": interactions_raw(["1", "10", "4", TS], ["1", "10", "4", TS], ["5", "10", "4", TS])}
    results = validate_all(raw, VCFG)
    truth = {"interactions": {"exact_duplicate": 1, "orphan_user": 1}, "users": {"missing_age": 2}}
    gt = compare_ground_truth(results, truth)
    by_cat = {(c["entity"], c["category"]): c for c in gt["categories"]}
    assert by_cat[("interactions", "orphan_user")]["detected"] == 1
    assert by_cat[("users", "missing_age")]["recall"] == 0.5
    assert gt["overall"] == {"expected": 4, "detected": 3, "matched": 3, "recall": 0.75}
    recon = {r["entity"]: r for r in gt["reconciliation"]}
    assert recon["interactions"]["rejected_rows"] == recon["interactions"]["injected_hard"] == 2


# --------------------------------------------------------------------------- end to end
def _write_raw(raw_dir: Path, interactions: list[list[str]]) -> None:
    raw_dir.mkdir(parents=True)
    users_raw(["1", "25", "M", "student", "11111"], ["2", "30", "F", "none", "22222"]).to_csv(
        raw_dir / "users.csv", index=False)
    items_raw(["10", "A (1995)", "01-Jan-1995", "", "http://a"],
              ["20", "B (1996)", "01-Feb-1996", "", "http://b"]).to_csv(raw_dir / "items.csv", index=False)
    interactions_raw(*interactions).to_csv(raw_dir / "interactions.csv", index=False)
    write_json({"seed": 0, "anomalies": {"interactions": {"orphan_user": 1}}}, raw_dir / "injected_anomalies.json")


def _tmp_config(tmp_path: Path, **validation) -> dict:
    paths = {key: str(tmp_path / key) for key in ("raw_dir", "interim_dir", "errors_dir", "reports_dir")}
    return load_config(overrides={"paths": paths, "validation": validation})


GOOD_ROWS = [["1", "10", "4", TS], ["2", "20", "5", TS], ["1", "20", "3", TS]]


def test_run_writes_typed_outputs_errors_and_reports(tmp_path):
    cfg = _tmp_config(tmp_path, max_error_rate=0.5)
    _write_raw(Path(cfg["paths"]["raw_dir"]), GOOD_ROWS + [["7", "10", "4", TS]])
    summary = run(cfg)
    assert summary["status"] == "passed" and summary["rows_out"]["interactions"] == 3
    assert summary["ground_truth_recall"] == 1.0
    inter = pd.read_parquet(tmp_path / "interim_dir" / "interactions.parquet")
    assert inter.dtypes.astype(str).to_dict() == INTERACTIONS.output_dtypes
    errors = pd.read_csv(tmp_path / "errors_dir" / "interactions_errors.csv")
    assert errors["error_type"].tolist() == ["referential_integrity"]
    report = read_json(tmp_path / "reports_dir" / "validation_report.json")
    assert {c["check"] for c in report["checks"]} >= {"exact_duplicate", "missing_value", "referential_integrity"}
    assert "Ground-truth comparison" in (tmp_path / "reports_dir" / "validation_report.md").read_text()
    assert "## 7. Outputs" in render_markdown(report)


def test_run_fails_fast_when_error_rate_exceeds_threshold(tmp_path):
    cfg = _tmp_config(tmp_path, max_error_rate=0.05)
    _write_raw(Path(cfg["paths"]["raw_dir"]), GOOD_ROWS + [["7", "10", "4", TS]])
    with pytest.raises(DataValidationError, match="max_error_rate"):
        run(cfg)
    report = read_json(tmp_path / "reports_dir" / "validation_report.json")
    assert report["status"] == "failed" and report["quality_gate"]["passed"] is False
    assert not (tmp_path / "interim_dir" / "interactions.parquet").exists()
    assert (tmp_path / "errors_dir" / "interactions_errors.csv").exists()
