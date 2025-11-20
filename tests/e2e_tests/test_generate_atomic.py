from pathlib import Path

from .conftest import run_generate_test


def test_generate_atomic(tmp_path: Path):
    run_generate_test(tmp_path, "atomic_config.yaml")
