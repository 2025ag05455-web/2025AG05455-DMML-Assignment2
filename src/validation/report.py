"""Validation report: data profile, ground-truth comparison and JSON / Markdown rendering.

``build_report`` assembles one JSON-serialisable dict; ``render_markdown`` renders that same
dict, so the machine-readable and the human-readable report always agree.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.utils.config import path
from src.utils.io import write_json
from src.validation.checks import normalise
from src.validation.schemas import (
    ACTION_FLAG, ACTION_NULLIFY, ACTION_QUARANTINE, ACTION_REJECT, ALL_COLUMNS, ERROR,
    NULL_TOKEN_EXEMPTIONS, NULL_TOKENS, WARNING, severity_of,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type hints (avoids a cycle)
    from src.validation.validate import EntityResult

Pairs = list[tuple[str, str]]

# Injected anomaly category (data/raw/injected_anomalies.json) -> (check, column) detecting it.
TRUTH_TO_CHECKS: dict[str, dict[str, Pairs]] = {
    "interactions": {
        "exact_duplicate": [("exact_duplicate", ALL_COLUMNS)],
        "rating_out_of_range": [("rating_not_allowed", "rating")],
        # 'five' -> not numeric, '4.5' -> non-integer, 'N/A' -> a null token, i.e. missing
        "rating_wrong_type": [("invalid_type", "rating"), ("non_integer_value", "rating"),
                              ("missing_value", "rating")],
        "missing_key": [("missing_value", "user_id"), ("missing_value", "item_id")],
        # 'abc' / '1999-13-45' -> not numeric; '-1', '0', 2100-01-01 -> outside the window
        "invalid_timestamp": [("invalid_type", "timestamp"), ("non_integer_value", "timestamp"),
                              ("timestamp_out_of_range", "timestamp"), ("missing_value", "timestamp")],
        "orphan_user": [("referential_integrity", "user_id")],
        "orphan_item": [("referential_integrity", "item_id")],
    },
    "users": {
        "exact_duplicate": [("exact_duplicate", ALL_COLUMNS)],
        "missing_age": [("missing_value", "age")],
        "invalid_age": [("invalid_type", "age"), ("non_integer_value", "age"), ("age_out_of_range", "age")],
        "invalid_gender": [("gender_not_allowed", "gender"), ("missing_value", "gender")],
    },
    "items": {
        "exact_duplicate": [("exact_duplicate", ALL_COLUMNS)],
        "missing_title": [("missing_value", "title")],
        "invalid_release_date": [("invalid_type", "release_date")],
    },
}

CHECK_ORDER = [
    "structure: all contract columns present (abort otherwise)",
    "normalise: strip whitespace, null tokens -> null",
    "exact duplicates quarantined (first occurrence kept); later checks run on unique rows",
    "missing values per column",
    "type coercion: non-numeric / non-integer numbers, unparsable dates (custom, vectorised)",
    "Pandera diagnostic schema (lazy): ranges, allowed values, key uniqueness, foreign keys",
    "policy: reject rows with hard errors, null soft-failed attributes",
    "duplicate (user_id, item_id) pairs among kept rows (warning)",
    "Pandera contract schema + exact dtype comparison on the clean output (post-condition gate)",
]
HARD_PROBLEMS = ("missing / non-numeric / non-integer key, duplicate primary key, missing / non-numeric / "
                 "non-integer / not-allowed rating, missing / non-numeric / out-of-window timestamp, "
                 "orphan foreign key (not in validated users / items), genre flag missing or not 0/1")
SOFT_PROBLEMS = ("missing / non-numeric / out-of-range age, missing or not-allowed gender, missing occupation "
                 "or zip_code, missing title, missing or unparsable release_date, missing imdb_url")
GROUPED_PREFIX = "genre_"  # one-hot columns collapsed into one row in the Markdown check table


# --------------------------------------------------------------------------- assembly
def relative(p: str | Path, cfg: dict) -> str:
    """Path relative to the project root when possible (portable report paths)."""
    try:
        return str(Path(p).relative_to(cfg["project"]["root"]))
    except ValueError:
        return str(p)


def build_report(*, results: Mapping[str, EntityResult], gate: dict, cfg: dict, started: datetime,
                 inputs: dict, outputs: dict, error_files: dict, truth: dict | None) -> dict:
    """Assemble the full, JSON-serialisable validation report."""
    vcfg = cfg["validation"]
    ground_truth = compare_ground_truth(results, truth)
    profile = build_profile(results)
    report = {
        "stage": "validation",
        "status": "passed" if gate["passed"] else "failed",
        "run_timestamp_utc": started.isoformat(timespec="seconds"),
        "duration_sec": round((datetime.now(timezone.utc) - started).total_seconds(), 2),
        "config": {"file": relative(cfg["_config_path"], cfg), "validation": vcfg},
        "inputs": inputs,
        "outputs": {"interim": outputs, "errors": error_files},
        "summary": {name: entity_summary(result) for name, result in results.items()},
        "quality_gate": gate,
        "post_condition_gate": {name: {"schema": f"{name}_contract", "rows": len(r.clean), "passed": True}
                                for name, r in results.items()},
        "policy": policy(vcfg),
        "checks": [check for result in results.values() for check in result.checks],
        "profile": profile,
        "ground_truth": ground_truth,
    }
    report["findings"] = findings(report)
    return report


def entity_summary(result: EntityResult) -> dict:
    """Row-count funnel of one entity."""
    issues = result.issues
    severity_counts = issues["severity"].value_counts()
    return {
        "input_rows": result.n_input,
        "exact_duplicates": int((issues["error_type"] == "exact_duplicate").sum()),
        "rejected_rows": result.n_rejected,
        "output_rows": len(result.clean),
        "rows_with_warnings": len(result.warning_rows),
        "error_records": int(severity_counts.get(ERROR, 0)),
        "warning_records": int(severity_counts.get(WARNING, 0)),
        "hard_error_rate_pct": round(100.0 * result.n_rejected / result.n_input, 4) if result.n_input else 0.0,
        "dropped_columns": result.spec.dropped_columns,
    }


def policy(vcfg: dict) -> dict:
    """The data-quality policy applied by this stage (documented in the report)."""
    return {
        "hard_errors": {"severity": ERROR, "action": ACTION_REJECT, "problems": HARD_PROBLEMS},
        "exact_duplicates": {"severity": ERROR, "action": ACTION_QUARANTINE},
        "soft_warnings": {"severity": WARNING, "action": ACTION_NULLIFY, "problems": SOFT_PROBLEMS,
                          "note": "nulled attributes are imputed by the preparation stage"},
        "duplicate_interactions": {"severity": WARNING, "action": ACTION_FLAG,
                                   "note": "preparation keeps one row per (user_id, item_id)"},
        "null_tokens": sorted(NULL_TOKENS),
        "null_token_exemptions": {col: sorted(vals) for col, vals in NULL_TOKEN_EXEMPTIONS.items()},
        "quality_gate": f"stage fails if interaction hard-error rate > {vcfg['max_error_rate']}",
        "post_condition_gate": "strict Pandera contract schema + exact dtype comparison on every output",
        "error_file_columns": "source_row (0-based data row; CSV line = source_row + 2), raw columns, "
                              "error_type, error_column, error_detail, severity",
        "check_order": CHECK_ORDER,
    }


# --------------------------------------------------------------------------- ground truth
def _flagged(issues: pd.DataFrame, pairs: Pairs) -> pd.DataFrame:
    """Issues raised by any of the (check, column) pairs."""
    if not pairs or issues.empty:
        return issues.iloc[0:0]
    keys = pd.MultiIndex.from_frame(issues[["error_type", "error_column"]])
    return issues[keys.isin(pairs)]


def compare_ground_truth(results: Mapping[str, EntityResult], truth: dict | None) -> dict:
    """Compare detected issues with the injected anomalies (expected vs detected, recall)."""
    if not truth:
        return {"available": False, "note": "no injected_anomalies.json - ingestion ran without injection"}
    categories = [_category_row(results[entity], category, int(expected))
                  for entity, cats in truth.items() for category, expected in cats.items()]
    expected = sum(c["expected"] for c in categories)
    matched = sum(min(c["detected"], c["expected"]) for c in categories)
    unattributed = _unattributed(results)
    return {
        "available": True,
        "source": "data/raw/injected_anomalies.json",
        "categories": categories,
        "overall": {"expected": expected, "detected": sum(c["detected"] for c in categories),
                    "matched": matched, "recall": round(matched / expected, 4) if expected else 1.0},
        "reconciliation": [_reconcile(results[name], categories) for name in results],
        "unattributed_detections": unattributed,
    }


def _category_row(result: EntityResult, category: str, expected: int) -> dict:
    pairs = TRUTH_TO_CHECKS.get(result.spec.name, {}).get(category, [])
    flagged = _flagged(result.issues, pairs)
    detected = int(flagged["source_row"].nunique())
    by_check = flagged.groupby(["error_type", "error_column"])["source_row"].nunique()
    hard = bool(pairs) and all(severity_of(check, col, result.spec) == ERROR for check, col in pairs)
    return {
        "entity": result.spec.name, "category": category,
        "severity": ERROR if hard else WARNING,
        "expected": expected, "detected": detected,
        "recall": round(min(detected, expected) / expected, 4) if expected else 1.0,
        "detected_by": {f"{check}[{col}]": int(n) for (check, col), n in by_check.items()},
        "mapped_checks": [f"{check}[{col}]" for check, col in pairs],
    }


def _entity_pairs(entity: str) -> Pairs:
    """All (check, column) pairs mapped to any injected category of ``entity``."""
    return [pair for pairs in TRUTH_TO_CHECKS.get(entity, {}).values() for pair in pairs]


def _attributed_rows(result: EntityResult) -> set[int]:
    return set(_flagged(result.issues, _entity_pairs(result.spec.name))["source_row"])


def _unattributed(results: Mapping[str, EntityResult]) -> list[dict]:
    """Detections not explained by any injected anomaly (i.e. issues pre-existing in the source)."""
    rows = []
    for name, result in results.items():
        attributed = _flagged(result.issues, _entity_pairs(name)).index
        flagged = result.issues.drop(index=attributed)
        for (check, column, severity), grp in flagged.groupby(["error_type", "error_column", "severity"]):
            source_rows = sorted(grp["source_row"].unique().tolist())
            rows.append({"entity": name, "check": check, "column": column, "severity": severity,
                         "rows": len(source_rows), "examples": _row_labels(result, source_rows[:10])})
    return rows


def _row_labels(result: EntityResult, source_rows: list[int]) -> list[str]:
    """Human-friendly labels of source rows (primary key when the entity has one)."""
    key = result.spec.key
    if key:
        return [f"{key}={v}" for v in result.raw.loc[source_rows, key]]
    return [f"source_row={r}" for r in source_rows]


def _reconcile(result: EntityResult, categories: list[dict]) -> dict:
    """Rejected / warned row counts versus the injected hard / soft anomalies of one entity."""
    name = result.spec.name
    cats = [c for c in categories if c["entity"] == name]
    expected_hard = sum(c["expected"] for c in cats if c["severity"] == ERROR)
    expected_soft = sum(c["expected"] for c in cats if c["severity"] == WARNING)
    attributed = _attributed_rows(result)
    rejected, warned = result.rejected_rows, result.warning_rows
    out = {
        "entity": name,
        "rejected_rows": len(rejected), "injected_hard": expected_hard,
        "rows_with_warnings": len(warned), "injected_soft": expected_soft,
        "unattributed_rejected": len(rejected - attributed),
        "unattributed_warning_rows": len(warned - attributed),
    }
    out["explanation"] = _explain(out)
    return out


def _explain(rec: dict) -> str:
    parts = []
    for kind, observed, expected, extra in (
            ("rejected", rec["rejected_rows"], rec["injected_hard"], rec["unattributed_rejected"]),
            ("warning", rec["rows_with_warnings"], rec["injected_soft"], rec["unattributed_warning_rows"])):
        gap = observed - expected
        if gap == 0 and extra == 0:
            parts.append(f"{kind} rows match the injected anomalies exactly ({observed})")
        elif gap == extra:
            parts.append(f"{kind}: {observed} rows = {expected} injected + {extra} pre-existing source issue(s)")
        else:
            parts.append(f"{kind}: {observed} rows vs {expected} injected (gap {gap:+d}, "
                         f"{extra} not attributable to injection)")
    return "; ".join(parts)


# --------------------------------------------------------------------------- profile
def _null_pct(df: pd.DataFrame) -> dict[str, float]:
    return {col: round(100.0 * float(v), 4) for col, v in df.isna().mean().items()}


def _utc_iso(unix_seconds: int) -> str:
    return pd.Timestamp(unix_seconds, unit="s", tz="UTC").isoformat()


def _stats(s: pd.Series) -> dict:
    if s.empty:
        return {}
    return {"min": float(s.min()), "median": float(s.median()), "mean": round(float(s.mean()), 4),
            "max": float(s.max())}


def build_profile(results: Mapping[str, EntityResult]) -> dict:
    """Basic data profile: null % (raw vs validated) plus entity-specific statistics."""
    profile = {name: _base_profile(result) for name, result in results.items()}
    profile["users"].update(_user_profile(results["users"].clean))
    profile["items"].update(_item_profile(results["items"].clean))
    profile["interactions"].update(_interaction_profile(results["interactions"].clean))
    return profile


def _base_profile(result: EntityResult) -> dict:
    raw = result.raw[list(result.spec.raw_columns)].astype("string")
    whitespace = (raw != raw.apply(lambda s: s.str.strip())).sum()
    exemptions = {col: {tok: int((raw[col].str.strip().str.lower() == tok).sum()) for tok in sorted(toks)}
                  for col, toks in NULL_TOKEN_EXEMPTIONS.items() if col in raw}
    return {
        "rows_raw": result.n_input,
        "rows_validated": len(result.clean),
        "null_pct_raw": _null_pct(normalise(raw)),
        "null_pct_validated": _null_pct(result.clean),
        "values_with_surrounding_whitespace": {c: int(n) for c, n in whitespace.items() if n},
        "null_token_exemptions": exemptions,
    }


def _user_profile(users: pd.DataFrame) -> dict:
    return {
        "n_users": int(users["user_id"].nunique()),
        "age": _stats(users["age"].dropna().astype("float64")),
        "gender_distribution": {str(k): int(v) for k, v in users["gender"].fillna("null").value_counts().items()},
        "n_occupations": int(users["occupation"].nunique()),
    }


def _item_profile(items: pd.DataFrame) -> dict:
    genres = [c for c in items.columns if c.startswith(GROUPED_PREFIX)]
    dates = items["release_date"].dropna()
    titles = items["title"].dropna()
    return {
        "n_items": int(items["item_id"].nunique()),
        "release_date_range": ({"min": dates.min().date().isoformat(), "max": dates.max().date().isoformat()}
                               if not dates.empty else {}),
        "duplicate_titles": int(titles[titles.duplicated(keep=False)].nunique()),
        "items_per_genre": {g.removeprefix(GROUPED_PREFIX): int(items[g].sum()) for g in genres},
        "multi_genre_pct": round(100.0 * float((items[genres].sum(axis=1) > 1).mean()), 2) if len(items) else 0.0,
    }


def _interaction_profile(inter: pd.DataFrame) -> dict:
    n = len(inter)
    if n == 0:
        return {"n_interactions": 0}
    n_users, n_items = int(inter["user_id"].nunique()), int(inter["item_id"].nunique())
    counts = inter["rating"].value_counts().sort_index()
    ts_min, ts_max = int(inter["timestamp"].min()), int(inter["timestamp"].max())
    return {
        "n_interactions": n, "n_users": n_users, "n_items": n_items,
        "density_pct": round(100.0 * n / (n_users * n_items), 4),
        "rating_distribution": {str(k): {"count": int(v), "pct": round(100.0 * v / n, 2)} for k, v in counts.items()},
        "rating_mean": round(float(inter["rating"].mean()), 4),
        "rating_std": round(float(inter["rating"].std(ddof=0)), 4),
        "timestamp_range": {"min": ts_min, "max": ts_max, "min_utc": _utc_iso(ts_min), "max_utc": _utc_iso(ts_max),
                            "span_days": round((ts_max - ts_min) / 86400, 1)},
        "ratings_per_user": _stats(inter.groupby("user_id").size()),
        "ratings_per_item": _stats(inter.groupby("item_id").size()),
    }


# --------------------------------------------------------------------------- findings
def findings(report: dict) -> list[str]:
    """Auto-generated, data-driven observations for the report."""
    profile, notes = report["profile"], []
    for entity, prof in profile.items():
        notes += [f"`{entity}.{col}` is {pct:.0f}% empty in the raw file - it carries no information and "
                  "is dropped from the validated layer." for col, pct in prof["null_pct_raw"].items() if pct == 100]
        for col, toks in prof["null_token_exemptions"].items():
            notes += [f"`{entity}.{col}`: {n} value(s) '{tok}' kept as a legitimate category value "
                      "(it is in the MovieLens reference list), not treated as a null token." for tok, n in toks.items() if n]
        notes += [f"`{entity}.{col}`: {n} value(s) had leading/trailing whitespace (stripped)."
                  for col, n in prof["values_with_surrounding_whitespace"].items()]
    for u in report["ground_truth"].get("unattributed_detections", []):
        notes.append(f"`{u['entity']}.{u['column']}`: {u['rows']} row(s) flagged by `{u['check']}` ({u['severity']}) "
                     f"were not injected - pre-existing in the MovieLens source ({', '.join(u['examples'])}).")
    items, inter = profile["items"], profile["interactions"]
    if items.get("duplicate_titles"):
        notes.append(f"{items['duplicate_titles']} titles are shared by more than one item_id (known MovieLens "
                     "quirk); kept as distinct items because item_id is the key.")
    if inter.get("n_interactions"):
        high = sum(v["pct"] for k, v in inter["rating_distribution"].items() if int(k) >= 4)
        ts = inter["timestamp_range"]
        notes += [
            f"Ratings are skewed positive: mean {inter['rating_mean']:.2f}, {high:.1f}% of ratings are >= 4.",
            f"User-item matrix density is {inter['density_pct']:.2f}% (sparsity {100 - inter['density_pct']:.2f}%); "
            f"every user has >= {inter['ratings_per_user']['min']:.0f} ratings, items have "
            f"{inter['ratings_per_item']['min']:.0f}..{inter['ratings_per_item']['max']:.0f} ratings (long tail).",
            f"Interactions span {ts['min_utc'][:10]} .. {ts['max_utc'][:10]} ({ts['span_days']} days), inside the "
            "configured validity window.",
        ]
    pairs = sum(c["n_failed"] for c in report["checks"] if c["check"] == "duplicate_interaction")
    notes.append(f"{pairs} duplicate (user_id, item_id) pair(s) remain after the hard checks "
                 "(warning only; preparation keeps one row per pair).")
    return notes


# --------------------------------------------------------------------------- markdown
def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    """Markdown table; ``align`` is one 'l'/'r' per column (default: all left)."""
    align = align or "l" * len(headers)
    rule = "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"
    lines = ["| " + " | ".join(headers) + " |", rule]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def _collapse_grouped(checks: list[dict]) -> list[dict]:
    """Merge per-column results of one-hot columns (genre_*) into one row per check."""
    out: list[dict] = []
    groups: dict[tuple[str, str], dict] = {}
    for check in checks:
        if not check["column"].startswith(GROUPED_PREFIX):
            out.append(check)
            continue
        key = (check["entity"], check["check"])
        if key not in groups:
            groups[key] = dict(check, n_checked=0, n_failed=0, n_columns=0)
            out.append(groups[key])
        group = groups[key]
        group["n_checked"] += check["n_checked"]
        group["n_failed"] += check["n_failed"]
        group["n_columns"] += 1
    for group in groups.values():
        group["column"] = f"{GROUPED_PREFIX}* ({group['n_columns']} cols)"
        group["pct_failed"] = round(100.0 * group["n_failed"] / group["n_checked"], 4) if group["n_checked"] else 0.0
    return out


def _md_header(r: dict) -> str:
    gate, gt = r["quality_gate"], r["ground_truth"]
    recall = (f"{100 * gt['overall']['recall']:.1f}% ({gt['overall']['matched']}/{gt['overall']['expected']} "
              "injected anomalies)" if gt.get("available") else "n/a (no injected anomalies)")
    rows = [
        ["Run timestamp (UTC)", r["run_timestamp_utc"]],
        ["Status", f"**{r['status'].upper()}**"],
        ["Quality gate", f"interaction hard-error rate {_pct(100 * gate['value'], 4)} "
                         f"<= {_pct(100 * gate['threshold'])}: {'PASS' if gate['passed'] else 'FAIL'}"],
        ["Post-condition gate", "strict Pandera contract schema + exact dtypes on every output: "
                                + ("PASS" if all(g["passed"] for g in r["post_condition_gate"].values()) else "FAIL")],
        ["Ground-truth detection recall", recall],
        ["Config", f"`{r['config']['file']}` (section `validation`)"],
        ["Duration", f"{r['duration_sec']} s"],
    ]
    intro = ("Stage 2 of the recommendation pipeline (`src/validation/validate.py`). This file is generated "
             "from `reports/validation_report.json`; both are rewritten on every run.")
    return "# Data Validation Report - MovieLens-100K\n\n" + intro + "\n\n" + _table(["Item", "Value"], rows)


def _md_counts(r: dict) -> str:
    rows = [[name, s["input_rows"], s["exact_duplicates"], s["rejected_rows"], s["output_rows"],
             s["rows_with_warnings"], _pct(s["hard_error_rate_pct"], 3)] for name, s in r["summary"].items()]
    inputs = [[name, f"`{i['file']}`", i["rows"], f"`{i['sha256'][:12]}`"] for name, i in r["inputs"].items()]
    return ("## 1. Row counts\n\n"
            + _table(["Entity", "Input rows", "Exact duplicates", "Rejected rows (hard errors, incl. duplicates)",
                      "Output rows", "Rows with warnings", "Hard-error rate"], rows, "lrrrrrr")
            + "\n\nInputs (lineage):\n\n" + _table(["Entity", "File", "Rows", "sha256 (prefix)"], inputs, "llrl"))


def _md_policy(r: dict) -> str:
    p = r["policy"]
    rows = [
        ["error (hard)", p["hard_errors"]["problems"], p["hard_errors"]["action"] + " -> `data/errors/<entity>_errors.csv`"],
        ["error (exact duplicate)", "row identical to an earlier row", p["exact_duplicates"]["action"]],
        ["warning (soft)", p["soft_warnings"]["problems"], p["soft_warnings"]["action"] + "; logged in the error "
                                                                                          "file; imputed in preparation"],
        ["warning (pair)", "duplicate (user_id, item_id) interaction", p["duplicate_interactions"]["action"]],
    ]
    exemptions = ", ".join(f"`{c}`: {v}" for c, v in p["null_token_exemptions"].items())
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(p["check_order"], start=1))
    return ("## 2. Validation policy\n\n" + _table(["Severity", "Problems", "Action"], rows)
            + f"\n\n* Null tokens (case-insensitive, stripped): {', '.join(repr(t) for t in p['null_tokens'])}; "
              f"exemptions (legitimate categories): {exemptions}."
            + f"\n* Quality gate: {p['quality_gate']}. Post-condition gate: {p['post_condition_gate']}."
            + f"\n* Error files: {p['error_file_columns']}. A row may have several records (one per failed check)."
            + "\n* `items.video_release_date` is dropped (100% empty, see findings)."
            + "\n\nCheck order (per entity; users and items are validated before interactions so foreign keys are "
              "checked against the *validated* parents):\n\n" + steps)


def _md_checks(r: dict) -> str:
    rows = []
    for i, c in enumerate(_collapse_grouped(r["checks"]), start=1):
        result = "pass" if not c["n_failed"] else ("FAIL" if c["severity"] == ERROR else "WARN")
        column = "(all columns)" if c["column"] == ALL_COLUMNS else c["column"]
        rows.append([i, c["entity"], f"`{c['check']}`", column, c["severity"], c["method"], c["n_checked"],
                     c["n_failed"], _pct(c["pct_failed"], 3), result, c["action"] if c["n_failed"] else "-"])
    return ("## 3. Check results\n\nEvery automated check, including the ones that passed. `Checked` is the number "
            "of rows (values for collapsed `genre_*` rows, columns for `required_columns`) the check ran on.\n\n"
            + _table(["#", "Entity", "Check", "Column", "Severity", "Method", "Checked", "Failed", "% failed",
                      "Result", "Action taken"], rows, "rlllllrrrll"))


def _md_ground_truth(r: dict) -> str:
    gt = r["ground_truth"]
    if not gt.get("available"):
        return "## 4. Ground-truth comparison\n\n" + gt["note"]
    rows = [[c["entity"], c["category"], c["severity"], c["expected"], c["detected"], _pct(100 * c["recall"], 1),
             ", ".join(f"`{k}` ({v})" for k, v in c["detected_by"].items()) or "-"] for c in gt["categories"]]
    rec = [[x["entity"], x["rejected_rows"], x["injected_hard"], x["rows_with_warnings"], x["injected_soft"],
            x["explanation"]] for x in gt["reconciliation"]]
    extra = [[u["entity"], f"`{u['check']}`", u["column"], u["severity"], u["rows"], ", ".join(u["examples"])]
             for u in gt["unattributed_detections"]]
    o = gt["overall"]
    text = (f"## 4. Ground-truth comparison\n\nThe ingestion stage injects a seeded, documented set of anomalies "
            f"(`{gt['source']}`). Detected = distinct source rows flagged by the checks mapped to each category.\n\n"
            + _table(["Entity", "Injected anomaly", "Severity", "Expected", "Detected", "Recall", "Detected by"],
                     rows, "lllrrrl")
            + f"\n\n**Overall detection recall: {_pct(100 * o['recall'], 1)}** ({o['matched']} of {o['expected']} "
              f"injected anomalies detected).\n\nReconciliation of rejected / warned rows with the injected anomalies:\n\n"
            + _table(["Entity", "Rejected rows", "Injected hard", "Rows with warnings", "Injected soft", "Explanation"],
                     rec, "lrrrrl"))
    if extra:
        text += ("\n\nDetections not explained by injection (pre-existing issues in the source data):\n\n"
                 + _table(["Entity", "Check", "Column", "Severity", "Rows", "Examples"], extra, "llllrl"))
    return text


def _md_profile(r: dict) -> str:
    prof = r["profile"]
    nulls = []
    for name, p in prof.items():
        validated = p["null_pct_validated"]
        nulls += [[name, col, _pct(pct, 3), _pct(validated[col], 3) if col in validated else "dropped"]
                  for col, pct in p["null_pct_raw"].items() if pct or validated.get(col)]
    inter, users, items = prof["interactions"], prof["users"], prof["items"]
    parts = ["## 5. Data profile\n\nNull share per column (after null-token normalisation; validated = after "
             "rejection and soft nulling). All columns not listed are 0% null.\n\n"
             + _table(["Entity", "Column", "Null % raw", "Null % validated"], nulls, "llrr")]
    if inter.get("n_interactions"):
        ts, pu, pi = inter["timestamp_range"], inter["ratings_per_user"], inter["ratings_per_item"]
        metrics = [
            ["Interactions / users / items", f"{inter['n_interactions']:,} / {inter['n_users']:,} / {inter['n_items']:,}"],
            ["Density", _pct(inter["density_pct"], 3)],
            ["Rating mean (std)", f"{inter['rating_mean']:.3f} ({inter['rating_std']:.3f})"],
            ["Timestamp range (UTC)", f"{ts['min_utc']} .. {ts['max_utc']} ({ts['span_days']} days)"],
            ["Ratings per user min / median / max", f"{pu['min']:.0f} / {pu['median']:.0f} / {pu['max']:.0f}"],
            ["Ratings per item min / median / max", f"{pi['min']:.0f} / {pi['median']:.0f} / {pi['max']:.0f}"],
            ["Users: age min / median / max", " / ".join(f"{users['age'][k]:.0f}" for k in ("min", "median", "max"))
             if users["age"] else "-"],
            ["Users: gender", ", ".join(f"{k}={v}" for k, v in users["gender_distribution"].items())],
            ["Users: occupations", users["n_occupations"]],
            ["Items: release dates", " .. ".join(items["release_date_range"].values()) or "-"],
            ["Items: multi-genre share", _pct(items["multi_genre_pct"])],
        ]
        dist = [[k, f"{v['count']:,}", _pct(v["pct"])] for k, v in inter["rating_distribution"].items()]
        parts += ["Validated data at a glance:\n\n" + _table(["Metric", "Value"], metrics),
                  "Rating distribution (validated interactions):\n\n" + _table(["Rating", "Count", "%"], dist, "rrr")]
    return "\n\n".join(parts)


def _md_findings(r: dict) -> str:
    return "## 6. Profiling findings\n\n" + "\n".join(f"* {note}" for note in r["findings"])


def _md_outputs(r: dict) -> str:
    rows = [[f"`{o['file']}`", o["rows"], ", ".join(f"{c}:{t}" for c, t in o["dtypes"].items()
                                                    if not c.startswith(GROUPED_PREFIX))
             + (", genre_*:int8" if any(c.startswith(GROUPED_PREFIX) for c in o["dtypes"]) else "")]
            for o in r["outputs"]["interim"].values()]
    rows += [[f"`{e['file']}`", e["records"], f"{e['errors']} error / {e['warnings']} warning records"]
             for e in r["outputs"]["errors"].values()]
    return "## 7. Outputs\n\n" + _table(["Artifact", "Rows", "Details"], rows, "lrl")


def render_markdown(report: dict) -> str:
    """Render the report dict as Markdown."""
    sections = [_md_header(report), _md_counts(report), _md_policy(report), _md_checks(report),
                _md_ground_truth(report), _md_profile(report), _md_findings(report), _md_outputs(report)]
    return "\n\n".join(sections) + "\n"


def write_reports(report: dict, cfg: dict) -> dict[str, Path]:
    """Write ``reports/validation_report.{json,md}`` and return their paths."""
    json_path = write_json(report, path(cfg, "reports_dir", "validation_report.json"))
    md_path = path(cfg, "reports_dir", "validation_report.md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "md": md_path}
