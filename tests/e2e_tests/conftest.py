import json
import os
import subprocess
from pathlib import Path


def run_generate_test(tmp_path: Path, config_name: str):
    """
    Run the generate test with the given configuration file and temporary path.

    Args:
        tmp_path: pytest temporary path
        config_name: configuration file name (e.g. "atomic_config.yaml")

    Returns:
        tuple: (run_folder, json_files[0])
    """
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    config_path = repo_root / "graphgen" / "configs" / config_name
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "python",
            "-m",
            "graphgen.generate",
            "--config_file",
            str(config_path),
            "--output_dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    data_root = output_dir / "data" / "graphgen"
    assert data_root.exists(), f"{data_root} does not exist"
    run_folders = sorted(data_root.iterdir(), key=lambda p: p.name, reverse=True)
    assert run_folders, f"No run folders found in {data_root}"
    run_folder = run_folders[0]

    config_saved = run_folder / "config.yaml"
    assert config_saved.exists(), f"{config_saved} not found"

    json_files = list(run_folder.glob("*.json"))
    assert json_files, f"No JSON output found in {run_folder}"

    log_files = list(run_folder.glob("*.log"))
    assert log_files, "No log file generated"

    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert (
        isinstance(data, list) and len(data) > 0
    ), "JSON output is empty or not a list"

    return run_folder, json_files[0]

