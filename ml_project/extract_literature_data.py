#!/usr/bin/env python3
"""Extract audit-ready data candidates from the LPBF IN718 literature corpus.

This script deliberately produces candidate rows with source excerpts instead
of pretending that arbitrary PDF prose is already a clean ML table.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


BASE = Path(
    "/Users/allamaprabhuani/Library/CloudStorage/"
    "OneDrive-CityStGeorge's,UniversityofLondon (02-06-2026 13:31)/"
    "4DMDA Conference - Nic"
)
CORPUS = BASE / "ml_project" / "corpus_pdfs"
OUT = BASE / "ml_project" / "extracted_data"
TEXT_OUT = OUT / "text"
TABLE_OUT = OUT / "tables"


HT_RE = re.compile(
    r"\b("
    r"heat[- ]?treat(?:ment|ed)?|solution(?:ized|ing)?|homogen(?:e?i[sz](?:ation|ed|ing))?|"
    r"age(?:d|ing)?|anneal(?:ed|ing)?|HIP|hot isostatic|stress[- ]?relief|"
    r"furnace cool|air cool|water quench|quench(?:ed|ing)?"
    r")\b",
    re.I,
)
MECH_RE = re.compile(
    r"\b("
    r"UTS|ultimate tensile|tensile strength|yield(?: strength)?|YS|elongation|"
    r"hardness|HV|Vickers|Young'?s modulus|elastic modulus"
    r")\b|(?:\d{2,4}(?:\.\d+)?)\s*(?:MPa|GPa|HV|%)",
    re.I,
)
FATIGUE_RE = re.compile(
    r"\b("
    r"S\s*[-–]\s*N|fatigue life|fatigue strength|cycles? to failure|run[- ]?out|"
    r"stress amplitude|stress ratio|R[- ]?ratio|Basquin|W[oö]hler|Nf|HCF|LCF"
    r")\b",
    re.I,
)
TEMP_RE = re.compile(r"(?<!\d)(\d{3,4})(?:\s*)(?:°\s*)?C\b", re.I)
TIME_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", re.I)
MPA_RE = re.compile(r"(?<!\d)(\d{2,4}(?:\.\d+)?)\s*MPa\b", re.I)
GPA_RE = re.compile(r"(?<!\d)(\d{2,3}(?:\.\d+)?)\s*GPa\b", re.I)
HV_RE = re.compile(r"(?<!\d)(\d{2,4}(?:\.\d+)?)\s*HV\b", re.I)
PCT_RE = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*%")
CYCLES_RE = re.compile(r"(?<!\d)(?:10\^?\s*)?(\d+(?:\.\d+)?)\s*(?:cycles|cycle|Nf)\b", re.I)


@dataclass
class PageRecord:
    pdf: str
    page: int
    text: str


def clean(s: str, limit: int = 900) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:limit]


def split_sentences(text: str) -> list[str]:
    # Keep table-like lines too. Some PDF extraction collapses captions badly.
    parts = []
    for line in text.splitlines():
        line = clean(line, 2000)
        if not line:
            continue
        if len(line) > 350:
            chunks = re.split(r"(?<=[.;])\s+(?=[A-Z0-9])", line)
            parts.extend(c for c in chunks if c)
        else:
            parts.append(line)
    return parts


def unique_numbers(rx: re.Pattern[str], text: str) -> str:
    vals = []
    for m in rx.finditer(text):
        v = m.group(1)
        if v not in vals:
            vals.append(v)
    return ";".join(vals)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def extract_pdf(pdf: Path) -> tuple[dict, list[PageRecord]]:
    pages: list[PageRecord] = []
    table_count = 0
    text_chars = 0
    table_dir = TABLE_OUT / pdf.stem
    table_dir.mkdir(parents=True, exist_ok=True)

    try:
        with pdfplumber.open(pdf) as doc:
            for i, page in enumerate(doc.pages, start=1):
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                pages.append(PageRecord(pdf.name, i, text))
                text_chars += len(text)

                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for j, table in enumerate(tables or [], start=1):
                    if not table or sum(1 for row in table if any(cell for cell in row)) < 2:
                        continue
                    table_count += 1
                    out_csv = table_dir / f"p{i:03d}_table{j:02d}.csv"
                    with out_csv.open("w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        for row in table:
                            w.writerow([(cell or "").strip() if isinstance(cell, str) else cell for cell in row])

        text_path = TEXT_OUT / f"{pdf.stem}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        with text_path.open("w", encoding="utf-8") as f:
            for p in pages:
                f.write(f"\n\n===== {p.pdf} PAGE {p.page} =====\n")
                f.write(p.text)

        all_text = "\n".join(p.text for p in pages)
        return (
            {
                "pdf": pdf.name,
                "pages": len(pages),
                "text_chars": text_chars,
                "table_count": table_count,
                "has_heat_treatment_terms": bool(HT_RE.search(all_text)),
                "has_mechanical_terms": bool(MECH_RE.search(all_text)),
                "has_fatigue_terms": bool(FATIGUE_RE.search(all_text)),
                "text_file": str(text_path.relative_to(BASE)),
                "tables_dir": str(table_dir.relative_to(BASE)) if table_count else "",
            },
            pages,
        )
    except Exception as exc:
        return (
            {
                "pdf": pdf.name,
                "pages": 0,
                "text_chars": 0,
                "table_count": 0,
                "has_heat_treatment_terms": False,
                "has_mechanical_terms": False,
                "has_fatigue_terms": False,
                "text_file": "",
                "tables_dir": "",
                "error": str(exc),
            },
            [],
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TEXT_OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)

    inventory: list[dict] = []
    ht_rows: list[dict] = []
    mech_rows: list[dict] = []
    fatigue_rows: list[dict] = []

    pdfs = sorted(CORPUS.glob("*.pdf"))
    for n, pdf in enumerate(pdfs, start=1):
        print(f"[{n:02d}/{len(pdfs):02d}] {pdf.name}")
        inv, pages = extract_pdf(pdf)
        inventory.append(inv)

        for page in pages:
            for sent in split_sentences(page.text):
                if HT_RE.search(sent) and (TEMP_RE.search(sent) or TIME_RE.search(sent)):
                    ht_rows.append(
                        {
                            "pdf": page.pdf,
                            "page": page.page,
                            "temperatures_C": unique_numbers(TEMP_RE, sent),
                            "times_h": unique_numbers(TIME_RE, sent),
                            "excerpt": clean(sent),
                        }
                    )
                if MECH_RE.search(sent):
                    mech_rows.append(
                        {
                            "pdf": page.pdf,
                            "page": page.page,
                            "mpa_values": unique_numbers(MPA_RE, sent),
                            "gpa_values": unique_numbers(GPA_RE, sent),
                            "hv_values": unique_numbers(HV_RE, sent),
                            "percent_values": unique_numbers(PCT_RE, sent),
                            "excerpt": clean(sent),
                        }
                    )
                if FATIGUE_RE.search(sent):
                    fatigue_rows.append(
                        {
                            "pdf": page.pdf,
                            "page": page.page,
                            "mpa_values": unique_numbers(MPA_RE, sent),
                            "cycle_values": unique_numbers(CYCLES_RE, sent),
                            "excerpt": clean(sent),
                        }
                    )

    write_csv(
        OUT / "paper_inventory.csv",
        inventory,
        [
            "pdf",
            "pages",
            "text_chars",
            "table_count",
            "has_heat_treatment_terms",
            "has_mechanical_terms",
            "has_fatigue_terms",
            "text_file",
            "tables_dir",
        ],
    )
    write_csv(OUT / "candidate_heat_treatments.csv", ht_rows, ["pdf", "page", "temperatures_C", "times_h", "excerpt"])
    write_csv(
        OUT / "candidate_mechanical_properties.csv",
        mech_rows,
        ["pdf", "page", "mpa_values", "gpa_values", "hv_values", "percent_values", "excerpt"],
    )
    write_csv(OUT / "candidate_fatigue_data.csv", fatigue_rows, ["pdf", "page", "mpa_values", "cycle_values", "excerpt"])

    print("\nSummary")
    print(f"  PDFs: {len(inventory)}")
    print(f"  Heat-treatment candidate rows: {len(ht_rows)}")
    print(f"  Mechanical-property candidate rows: {len(mech_rows)}")
    print(f"  Fatigue candidate rows: {len(fatigue_rows)}")
    print(f"  Tables extracted: {sum(int(r['table_count']) for r in inventory)}")


if __name__ == "__main__":
    main()
