from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from .paths import CORPUS_PDF_DIR, EXTRACTED_DATA_DIR, PROJECT_ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_inventory() -> dict[str, dict[str, str]]:
    return {row["pdf"]: row for row in read_csv_dicts(EXTRACTED_DATA_DIR / "paper_inventory.csv")}


def _load_scope() -> dict[str, dict[str, str]]:
    return {row["pdf"]: row for row in read_csv_dicts(EXTRACTED_DATA_DIR / "corpus_scope_audit.csv")}


def source_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    token = stem.split("_", 1)[0]
    if token.isdigit():
        return f"SRC{int(token):03d}"
    return token.upper()


def build_sources_rows() -> list[dict[str, str]]:
    scope = _load_scope()
    rows: list[dict[str, str]] = []
    for pdf in sorted(CORPUS_PDF_DIR.glob("*.pdf")):
        sc = scope.get(pdf.name, {})
        recommended = sc.get("recommended_model_use", "unknown")
        rows.append(
            {
                "source_id": source_id_from_filename(pdf.name),
                "title": pdf.stem,
                "authors": "",
                "year": "",
                "doi": "",
                "url": "",
                "publisher": "",
                "source_type": "pdf",
                "peer_review_status": "",
                "license_or_access_status": "",
                "am_scope": sc.get("am_scope", "unknown"),
                "recommended_model_use": recommended,
                "exclusion_reason": "" if recommended in {"yes", "partial"} else sc.get("notes", ""),
            }
        )
    return rows


def build_source_file_rows() -> list[dict[str, str]]:
    inventory = _load_inventory()
    rows: list[dict[str, str]] = []
    for pdf in sorted(CORPUS_PDF_DIR.glob("*.pdf")):
        inv = inventory.get(pdf.name, {})
        rows.append(
            {
                "file_id": f"FILE_{source_id_from_filename(pdf.name)}",
                "source_id": source_id_from_filename(pdf.name),
                "local_path": str(pdf.relative_to(PROJECT_ROOT)),
                "filename": pdf.name,
                "bytes": str(pdf.stat().st_size),
                "sha256": sha256_file(pdf),
                "page_count": inv.get("pages", ""),
                "text_chars": inv.get("text_chars", ""),
                "table_count": inv.get("table_count", ""),
                "acquisition_route": "local_corpus",
                "acquisition_date": datetime.now(timezone.utc).date().isoformat(),
                "download_status": "available",
            }
        )
    return rows


def build_extraction_run_row(command: str) -> dict[str, str]:
    script = PROJECT_ROOT / "ml_project" / "extract_literature_data.py"
    outputs = sorted(p.name for p in EXTRACTED_DATA_DIR.glob("*.csv"))
    output_counts = {}
    for p in EXTRACTED_DATA_DIR.glob("*.csv"):
        with p.open(encoding="utf-8", errors="ignore") as f:
            output_counts[p.name] = max(0, sum(1 for _ in f) - 1)
    return {
        "run_id": datetime.now(timezone.utc).strftime("RUN_%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script_path": str(script.relative_to(PROJECT_ROOT)) if script.exists() else "",
        "script_sha256": sha256_file(script) if script.exists() else "",
        "command": command,
        "python_version": platform.python_version(),
        "tool_versions": json.dumps({"platform": platform.platform()}),
        "input_file_hashes": json.dumps({p.name: sha256_file(p) for p in sorted(CORPUS_PDF_DIR.glob("*.pdf"))}),
        "output_files": json.dumps(outputs),
        "output_row_counts": json.dumps(output_counts),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

