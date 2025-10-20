import shutil
import subprocess
import sys


def pick_device() -> str:
    """Return the best available device string for MinerU."""
    # 1. NVIDIA GPU
    if shutil.which("nvidia-smi") is not None:
        try:
            # check if there's any free GPU memory
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            if any(int(line) > 0 for line in out.strip().splitlines()):
                return "cuda:0"
        except Exception:  # pylint: disable=broad-except
            pass

    # 2. Apple Silicon
    if sys.platform == "darwin" and shutil.which("sysctl"):
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            )
            if "Apple" in brand:
                return "mps"
        except Exception:  # pylint: disable=broad-except
            pass

    # 3. Ascend NPU
    if shutil.which("npu-smi") is not None:
        try:
            subprocess.check_call(["npu-smi", "info"], stdout=subprocess.DEVNULL)
            return "npu"
        except Exception:  # pylint: disable=broad-except
            pass

    return "cpu"
