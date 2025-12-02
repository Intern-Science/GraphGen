from pathlib import Path

from .conftest import run_generate_test


def test_generate_aggregated(tmp_path: Path):
    run_generate_test(tmp_path, "aggregated_config.yaml")
