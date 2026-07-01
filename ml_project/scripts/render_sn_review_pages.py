#!/usr/bin/env python3
"""Render local PDF pages needed for S-N figure review.

The rendered PNGs are local-only review artifacts and are ignored by Git. The
CSV manifest is committed so the review set is reproducible without publishing
publisher images.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
ML = PROJECT / "ml_project"
CORPUS = ML / "corpus_pdfs"
DATA = ML / "data"
REPORTS = ML / "reports"
OUT = ML / "figures" / "sn_review_pages"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def candidate_pages(value: str, limit: int) -> list[int]:
    pages: list[int] = []
    for token in (value or "").split(";"):
        if not token or token.startswith("+"):
            continue
        try:
            pages.append(int(float(token)))
        except ValueError:
            continue
        if len(pages) >= limit:
            break
    return pages


def add_task(tasks: dict[tuple[str, int], dict[str, object]], pdf: str, page: int, source_id: str, reason: str, target_id: str = "") -> None:
    if not pdf or page <= 0:
        return
    key = (pdf, page)
    if key not in tasks:
        tasks[key] = {"source_id": source_id, "pdf": pdf, "source_page": page, "target_ids": set(), "review_reason": set()}
    if target_id:
        tasks[key]["target_ids"].add(target_id)
    tasks[key]["review_reason"].add(reason)


def render_page(pdf: Path, page: int, output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    for stale in output.parent.glob(f"{prefix.name}-*.png"):
        stale.unlink()
    cmd = ["pdftoppm", "-f", str(page), "-l", str(page), "-r", "180", "-png", str(pdf), str(prefix)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rendered = sorted(output.parent.glob(f"{prefix.name}-*.png"))
    if not rendered:
        return False
    rendered[0].replace(output)
    return output.exists()


def build_tasks(max_pages_per_source: int) -> dict[tuple[str, int], dict[str, object]]:
    tasks: dict[tuple[str, int], dict[str, object]] = {}
    targets = read_csv(DATA / "sn_digitisation_targets.csv")
    for row in targets:
        source_page = row.get("source_page", "")
        if not source_page:
            continue
        add_task(
            tasks,
            row["source_pdf"],
            int(float(source_page)),
            row["source_id"],
            "registered_sn_target",
            row["target_id"],
        )

    queue_path = REPORTS / "sn_pdf_review_queue.csv"
    if queue_path.exists():
        for row in read_csv(queue_path):
            if row.get("priority") != "high":
                continue
            for page in candidate_pages(row.get("candidate_pages", ""), max_pages_per_source):
                add_task(tasks, row["pdf"], page, row["source_id"], row["review_action"])
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages-per-source", type=int, default=2)
    args = parser.parse_args()

    tasks = build_tasks(args.max_pages_per_source)
    rows: list[dict[str, object]] = []
    for (pdf_name, page), task in sorted(tasks.items()):
        pdf = CORPUS / pdf_name
        target_ids = sorted(task["target_ids"])
        reasons = sorted(task["review_reason"])
        output = OUT / Path(pdf_name).stem / f"p{int(page):03d}.png"
        status = "missing_pdf"
        if pdf.exists():
            try:
                status = "rendered" if render_page(pdf, int(page), output) else "render_failed"
            except Exception as exc:
                status = f"render_failed: {exc}"
        rows.append(
            {
                "source_id": task["source_id"],
                "pdf": pdf_name,
                "source_page": int(page),
                "target_ids": ";".join(target_ids),
                "review_reason": ";".join(reasons),
                "local_review_image": str(output.relative_to(PROJECT)),
                "render_status": status,
            }
        )

    write_csv(
        REPORTS / "sn_rendered_review_pages.csv",
        rows,
        ["source_id", "pdf", "source_page", "target_ids", "review_reason", "local_review_image", "render_status"],
    )
    print(f"rendered manifest rows: {len(rows)}")


if __name__ == "__main__":
    main()
