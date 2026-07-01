from ml_project.ht_advisor.paths import CORPUS_PDF_DIR, CURATED_DATA_DIR, EXTRACTED_DATA_DIR, PROJECT_ROOT


def test_project_paths_exist():
    assert PROJECT_ROOT.exists()
    assert EXTRACTED_DATA_DIR.exists()
    assert CORPUS_PDF_DIR.exists()


def test_curated_dir_is_under_ml_project():
    assert CURATED_DATA_DIR.parent.name == "ml_project"
    assert CURATED_DATA_DIR.name == "curated_data"

