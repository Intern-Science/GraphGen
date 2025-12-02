from pathlib import Path

from .conftest import run_generate_test


def test_generate_multi_hop(tmp_path: Path):
    run_generate_test(tmp_path, "multi_hop_config.yaml")
