from pathlib import Path

from .conftest import run_generate_test


def test_generate_cot(tmp_path: Path):
    run_generate_test(tmp_path, "cot_config.yaml")
