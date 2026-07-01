import hashlib

from ml_project.ht_advisor.provenance import build_source_file_rows, sha256_file


def test_sha256_file(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("abc", encoding="utf-8")
    assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_build_source_file_rows_has_required_columns():
    rows = build_source_file_rows()
    assert rows
    row = rows[0]
    for col in ["file_id", "source_id", "local_path", "filename", "bytes", "sha256", "page_count", "table_count"]:
        assert col in row
    assert row["sha256"]

