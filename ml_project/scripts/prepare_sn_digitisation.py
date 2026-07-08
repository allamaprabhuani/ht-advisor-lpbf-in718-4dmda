#!/usr/bin/env python3
"""Prepare the S-N digitisation review queue.

This script does not digitise figures. It checks corpus/source traceability,
scans extracted text for fatigue-relevant pages, and writes audit artifacts that
define the outstanding checks before marker-level digitisation can begin.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ML = PROJECT / "ml_project"
CURATED = ML / "curated_data"
DATA = ML / "data"
EXTRACTED = ML / "extracted_data"
CORPUS = ML / "corpus_pdfs"
REPORTS = ML / "reports"

SN_RE = re.compile(r"S\s*[-–]\s*N|stress[- ]life|W[oö]hler|cycles? to failure|fatigue life", re.I)
FATIGUE_RE = re.compile(r"fatigue|run[- ]?out|stress amplitude|stress ratio|R[- ]?ratio|Basquin|HCF|LCF|Nf", re.I)
FIGURE_RE = re.compile(r"\b(fig(?:ure)?\.?\s*\d+|S\s*[-–]\s*N)\b", re.I)
TABLE_RE = re.compile(r"\b(table\s*\d+|fatigue test|fatigue data)\b", re.I)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extracted_pages(pdf_name: str) -> dict[int, str]:
    text_path = EXTRACTED / "text" / f"{Path(pdf_name).stem}.txt"
    if not text_path.exists():
        return {}
    pages: dict[int, list[str]] = defaultdict(list)
    current_page: int | None = None
    for line in text_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        marker = re.match(r"===== .+ PAGE (\d+) =====", line.strip())
        if marker:
            current_page = int(marker.group(1))
            continue
        if current_page is not None:
            pages[current_page].append(line)
    return {page: "\n".join(lines) for page, lines in pages.items()}


def page_list(values: list[int], limit: int = 18) -> str:
    ordered = sorted(set(values))
    if not ordered:
        return ""
    rendered = ";".join(str(v) for v in ordered[:limit])
    if len(ordered) > limit:
        rendered += f";+{len(ordered) - limit}_more"
    return rendered


def target_counts_by_source(targets: list[dict[str, str]]) -> tuple[Counter[str], Counter[str]]:
    target_counts: Counter[str] = Counter()
    saved_counts: Counter[str] = Counter()
    for row in targets:
        source_id = row["source_id"]
        target_counts[source_id] += 1
        fig_path = row.get("figure_image_path", "")
        if fig_path and (PROJECT / fig_path).exists():
            saved_counts[source_id] += 1
    return target_counts, saved_counts


def review_action(
    source_id: str,
    target_count: int,
    saved_figure_count: int,
    fatigue_pages: list[int],
    sn_pages: list[int],
    recommended_use: str,
) -> tuple[str, str]:
    if recommended_use == "background_only":
        return ("background_check_only", "low")
    if target_count and saved_figure_count < target_count:
        return ("complete_registered_figure_snips_and_metadata", "high")
    if target_count:
        return ("verify_registered_figures_before_digitisation", "high")
    if sn_pages:
        return ("inspect_unregistered_sn_pages", "high")
    if fatigue_pages:
        return ("inspect_fatigue_pages_for_tables_or_figures", "medium")
    return ("no_fatigue_digitisation_action", "low")


def outstanding_checks(targets: list[dict[str, str]], reviewed_points: int) -> list[str]:
    checks: list[str] = []
    if reviewed_points == 0:
        checks.append("no_reviewed_sn_points")
    if any(row.get("stress_metric_type", "unknown") == "unknown" for row in targets):
        checks.append("stress_metric_unknown")
    if any(row.get("figure_id", "") in {"", "not_yet_verified"} for row in targets):
        checks.append("figure_identity_unverified")
    if any(not row.get("source_page", "") for row in targets):
        checks.append("source_page_missing")
    if any(row.get("figure_image_path", "") == "" for row in targets):
        checks.append("figure_image_missing")
    if any(row.get("axis_scale_x", "unknown") == "unknown" or row.get("axis_scale_y", "unknown") == "unknown" for row in targets):
        checks.append("axis_scale_unknown")
    return checks


def main() -> None:
    sources = read_csv(CURATED / "sources.csv")
    source_files = {row["source_id"]: row for row in read_csv(CURATED / "source_files.csv")}
    scope = {row["pdf"]: row for row in read_csv(EXTRACTED / "corpus_scope_audit.csv")}
    targets = read_csv(DATA / "sn_digitisation_targets.csv")
    points = read_csv(DATA / "sn_curve_points.csv")
    target_counts, saved_counts = target_counts_by_source(targets)

    queue_rows: list[dict[str, object]] = []
    total_fatigue_pages = 0
    total_sn_pages = 0
    for source in sources:
        source_id = source["source_id"]
        file_row = source_files.get(source_id, {})
        pdf = file_row.get("filename", "")
        pages = extracted_pages(pdf)
        fatigue_pages = [page for page, text in pages.items() if FATIGUE_RE.search(text)]
        sn_pages = [page for page, text in pages.items() if SN_RE.search(text)]
        figure_pages = [page for page, text in pages.items() if FIGURE_RE.search(text)]
        table_pages = [page for page, text in pages.items() if TABLE_RE.search(text)]
        total_fatigue_pages += len(set(fatigue_pages))
        total_sn_pages += len(set(sn_pages))

        recommended_use = source.get("recommended_model_use", "")
        action, priority = review_action(
            source_id,
            target_counts[source_id],
            saved_counts[source_id],
            fatigue_pages,
            sn_pages,
            recommended_use,
        )
        queue_rows.append(
            {
                "source_id": source_id,
                "pdf": pdf,
                "title": source.get("title", ""),
                "am_scope": source.get("am_scope", ""),
                "recommended_model_use": recommended_use,
                "registered_target_count": target_counts[source_id],
                "saved_figure_count": saved_counts[source_id],
                "fatigue_page_count": len(set(fatigue_pages)),
                "sn_page_count": len(set(sn_pages)),
                "figure_page_count": len(set(figure_pages)),
                "table_page_count": len(set(table_pages)),
                "candidate_pages": page_list(fatigue_pages + sn_pages + figure_pages + table_pages),
                "review_action": action,
                "priority": priority,
                "notes": scope.get(pdf, {}).get("notes", ""),
            }
        )

    queue_rows.sort(key=lambda row: ({"high": 0, "medium": 1, "low": 2}[str(row["priority"])], str(row["source_id"])))
    queue_fields = [
        "source_id",
        "pdf",
        "title",
        "am_scope",
        "recommended_model_use",
        "registered_target_count",
        "saved_figure_count",
        "fatigue_page_count",
        "sn_page_count",
        "figure_page_count",
        "table_page_count",
        "candidate_pages",
        "review_action",
        "priority",
        "notes",
    ]
    write_csv(REPORTS / "sn_pdf_review_queue.csv", queue_rows, queue_fields)

    reviewed_points = sum(1 for row in points if row.get("review_status") == "reviewed")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf_count": len(list(CORPUS.glob("*.pdf"))),
        "source_count": len(sources),
        "registered_sn_targets": len(targets),
        "saved_sn_figure_images": sum(1 for row in targets if row.get("figure_image_path") and (PROJECT / row["figure_image_path"]).exists()),
        "reviewed_sn_points": reviewed_points,
        "fatigue_candidate_pages": total_fatigue_pages,
        "sn_candidate_pages": total_sn_pages,
        "high_priority_sources": sum(1 for row in queue_rows if row["priority"] == "high"),
        "outstanding_checks": outstanding_checks(targets, reviewed_points),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "sn_digitisation_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plan = [
        "# S-N Digitisation Preparation Plan",
        "",
        "This report is generated from the local corpus and source metadata. It does not digitise figure points.",
        "",
        "## Current Counts",
        "",
        f"- PDFs in local corpus: {summary['pdf_count']}",
        f"- Registered S-N targets: {summary['registered_sn_targets']}",
        f"- Saved S-N figure images: {summary['saved_sn_figure_images']}",
        f"- Reviewed S-N points: {summary['reviewed_sn_points']}",
        f"- High-priority review sources: {summary['high_priority_sources']}",
        "",
        "## Outstanding Checks Before Fatigue Model Use",
        "",
    ]
    plan.extend(f"- {check}" for check in summary["outstanding_checks"])
    plan.extend(
        [
            "",
            "## Required Workflow",
            "",
            "1. Confirm figure identity, panel, caption, axis scale, stress metric, units, and runout symbols.",
            "2. Save or update missing S-N figure snapshots.",
            "3. Digitise one curve at a time and keep calibration JSON beside the local figure.",
            "4. Replot digitised experimental marker points and compare against the source snapshot.",
            "5. Mark points as reviewed only after source metadata and visual QA are complete.",
            "6. Use reviewed points for Basquin or ML fatigue modelling only after sufficient conditions and source groups are available.",
        ]
    )
    (REPORTS / "sn_digitisation_phase_plan.md").write_text("\n".join(plan) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
