#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_project.ht_advisor.paths import CURATED_DATA_DIR
from ml_project.ht_advisor.provenance import build_extraction_run_row, build_source_file_rows, build_sources_rows, write_csv


def main() -> None:
    write_csv(CURATED_DATA_DIR / "sources.csv", build_sources_rows())
    write_csv(CURATED_DATA_DIR / "source_files.csv", build_source_file_rows())
    write_csv(CURATED_DATA_DIR / "extraction_runs.csv", [build_extraction_run_row("python3 ml_project/extract_literature_data.py")])
    print(f"Wrote provenance tables to {CURATED_DATA_DIR}")


if __name__ == "__main__":
    main()
