"""Stage 1 - Data ingestion.

Downloads (or reuses) the public MovieLens-100K dataset, converts the three source
files into standardized CSV landing files (all columns kept as *strings*, exactly as
received, so that validation can detect type problems), optionally injects a small,
seeded set of synthetic data-quality anomalies, and writes an ingestion manifest
(source URL, checksum, row counts, timestamp) for lineage.

Outputs (data/raw/):
    users.csv, items.csv, interactions.csv   - landing files (string columns)
    source/ml-100k/...                       - extracted original files
    ingestion_manifest.json                  - lineage / row counts / checksums
    injected_anomalies.json                  - ground truth of injected issues
"""
from __future__ import annotations

import io
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils.config import load_config, path, rel_path
from src.utils.io import sha256_file, write_json
from src.utils.logger import get_logger

log = get_logger("ingestion")

GENRES = [
    "unknown", "action", "adventure", "animation", "childrens", "comedy", "crime",
    "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery",
    "romance", "sci_fi", "thriller", "war", "western",
]
GENRE_COLS = [f"genre_{g}" for g in GENRES]
USER_COLS = ["user_id", "age", "gender", "occupation", "zip_code"]
ITEM_COLS = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS
INTERACTION_COLS = ["user_id", "item_id", "rating", "timestamp"]


# --------------------------------------------------------------------------- download
def _download_zip(url: str, dest: Path, timeout: int) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Reusing cached archive %s", rel_path(dest))
        return dest
    log.info("Downloading %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _extract(zip_path: Path, out_dir: Path) -> Path:
    target = out_dir / "ml-100k"
    if target.exists():
        shutil.rmtree(target)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():  # guard against path traversal
            if member.startswith("/") or ".." in Path(member).parts:
                raise ValueError(f"Unsafe path in archive: {member}")
        zf.extractall(out_dir)
    return target


# --------------------------------------------------------------------------- parsing
def _read_source(src: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse MovieLens source files into string-typed frames with standard headers."""
    users = pd.read_csv(src / "u.user", sep="|", header=None, names=USER_COLS,
                        dtype=str, keep_default_na=False, encoding="latin-1")
    items = pd.read_csv(src / "u.item", sep="|", header=None, names=ITEM_COLS,
                        dtype=str, keep_default_na=False, encoding="latin-1")
    inter = pd.read_csv(src / "u.data", sep="\t", header=None, names=INTERACTION_COLS,
                        dtype=str, keep_default_na=False)
    return users, items, inter


# --------------------------------------------------------------------------- anomaly injection
def inject_anomalies(users: pd.DataFrame, items: pd.DataFrame, inter: pd.DataFrame,
                     seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Inject a documented, reproducible set of data-quality problems.

    Returns modified copies plus a ground-truth dictionary {entity: {anomaly: count}}.
    Injected rows are appended and then the frame is shuffled deterministically so the
    anomalies are not trivially located at the end of the file.
    """
    rng = np.random.default_rng(seed)
    truth: dict[str, dict[str, int]] = {"interactions": {}, "users": {}, "items": {}}

    def sample_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[rng.choice(len(df), size=n, replace=False)].copy()

    # ---- interactions
    new = []
    dup = sample_rows(inter, 25)
    new.append(dup); truth["interactions"]["exact_duplicate"] = 25

    bad = sample_rows(inter, 10); bad["rating"] = rng.choice(["0", "6", "7", "-1", "10"], 10)
    new.append(bad); truth["interactions"]["rating_out_of_range"] = 10

    bad = sample_rows(inter, 6); bad["rating"] = rng.choice(["five", "4.5", "N/A"], 6)
    new.append(bad); truth["interactions"]["rating_wrong_type"] = 6

    bad = sample_rows(inter, 10)
    bad.iloc[:5, bad.columns.get_loc("user_id")] = ""
    bad.iloc[5:, bad.columns.get_loc("item_id")] = ""
    new.append(bad); truth["interactions"]["missing_key"] = 10

    bad = sample_rows(inter, 10)
    bad["timestamp"] = rng.choice(["-1", "abc", "4102444800", "0", "1999-13-45"], 10)
    new.append(bad); truth["interactions"]["invalid_timestamp"] = 10

    bad = sample_rows(inter, 5); bad["user_id"] = "9999"
    new.append(bad); truth["interactions"]["orphan_user"] = 5
    bad = sample_rows(inter, 5); bad["item_id"] = "99999"
    new.append(bad); truth["interactions"]["orphan_item"] = 5

    inter2 = pd.concat([inter] + new, ignore_index=True)
    inter2 = inter2.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # ---- users (attribute problems are applied in place, duplicates appended)
    users2 = users.copy()
    idx = rng.choice(len(users2), size=10, replace=False)
    users2.loc[idx[:5], "age"] = ""; truth["users"]["missing_age"] = 5
    users2.loc[idx[5:8], "age"] = ["abc", "-5", "150"]; truth["users"]["invalid_age"] = 3
    users2.loc[idx[8:10], "gender"] = ["X", ""]; truth["users"]["invalid_gender"] = 2
    users2 = pd.concat([users2, sample_rows(users2, 3)], ignore_index=True)
    truth["users"]["exact_duplicate"] = 3

    # ---- items
    items2 = items.copy()
    idx = rng.choice(len(items2), size=4, replace=False)
    items2.loc[idx[:2], "title"] = ""; truth["items"]["missing_title"] = 2
    items2.loc[idx[2:4], "release_date"] = ["32-Foo-1995", "not a date"]
    truth["items"]["invalid_release_date"] = 2
    items2 = pd.concat([items2, sample_rows(items2, 2)], ignore_index=True)
    truth["items"]["exact_duplicate"] = 2

    return users2, items2, inter2, truth


# --------------------------------------------------------------------------- entrypoint
def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    icfg = cfg["ingestion"]
    raw_dir = path(cfg, "raw_dir")
    src_dir = path(cfg, "raw_dir", "source")
    src_dir.mkdir(parents=True, exist_ok=True)

    zip_path = _download_zip(icfg["url"], src_dir / "ml-100k.zip", icfg.get("download_timeout_sec", 60))
    extracted = _extract(zip_path, src_dir)
    users, items, inter = _read_source(extracted)
    log.info("Parsed source: users=%d items=%d interactions=%d", len(users), len(items), len(inter))

    truth: dict = {}
    if icfg.get("inject_anomalies", False):
        users, items, inter, truth = inject_anomalies(users, items, inter, int(icfg.get("anomaly_seed", 0)))
        write_json({"seed": icfg.get("anomaly_seed"), "anomalies": truth}, raw_dir / "injected_anomalies.json")
        log.info("Injected synthetic anomalies: %s", truth)

    outputs = {}
    for name, df in (("users", users), ("items", items), ("interactions", inter)):
        out = raw_dir / f"{name}.csv"
        df.to_csv(out, index=False)
        outputs[name] = {"file": str(out.relative_to(cfg["project"]["root"])), "rows": int(len(df)),
                         "columns": list(df.columns), "sha256": sha256_file(out)}

    manifest = {
        "stage": "ingestion",
        "source": icfg["source"],
        "url": icfg["url"],
        "archive_sha256": sha256_file(zip_path),
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "anomalies_injected": bool(truth),
        "outputs": outputs,
    }
    write_json(manifest, raw_dir / "ingestion_manifest.json")
    log.info("Ingestion complete: %s", {k: v["rows"] for k, v in outputs.items()})
    return {"stage": "ingestion", "rows": {k: v["rows"] for k, v in outputs.items()},
            "anomalies_injected": truth}


if __name__ == "__main__":
    run()
