from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from graphgen.models import (
    CSVReader,
    JSONLReader,
    JSONReader,
    ParquetReader,
    PDFReader,
    PickleReader,
    RDFReader,
    TXTReader,
)
from graphgen.utils import logger

from .parallel_file_scanner import ParallelFileScanner

_MAPPING = {
    "jsonl": JSONLReader,
    "json": JSONReader,
    "txt": TXTReader,
    "csv": CSVReader,
    "md": TXTReader,
    "pdf": PDFReader,
    "parquet": ParquetReader,
    "pickle": PickleReader,
    "rdf": RDFReader,
    "owl": RDFReader,
    "ttl": RDFReader,
}


def _build_reader(suffix: str, cache_dir: str | None):
    suffix = suffix.lower()
    if suffix == "pdf" and cache_dir is not None:
        return _MAPPING[suffix](output_dir=cache_dir)
    return _MAPPING[suffix]()


def read_files(
    input_file: str,
    allowed_suffix: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    max_workers: int = 4,
    rescan: bool = False,
) -> Iterator[Dict[str, Any]]:
    """
    Read files from a path using parallel scanning and appropriate readers.

    Args:
        input_file: Path to a file or directory
        allowed_suffix: List of file suffixes to read. If None, uses all supported types
        cache_dir: Directory for caching PDF extraction and scan results
        max_workers: Number of workers for parallel scanning
        rescan: Whether to force rescan even if cached results exist
    """

    path = Path(input_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"input_path not found: {input_file}")

    if allowed_suffix is None:
        support_suffix = set(_MAPPING.keys())
    else:
        support_suffix = {s.lower().lstrip(".") for s in allowed_suffix}

    with ParallelFileScanner(
        cache_dir=cache_dir or "cache",
        allowed_suffix=support_suffix,
        rescan=rescan,
        max_workers=max_workers,
    ) as scanner:
        scan_results = scanner.scan(str(path), recursive=True)

    # Extract files from scan results
    files_to_read = []
    for path_result in scan_results.values():
        if "error" in path_result:
            logger.warning("Error scanning %s: %s", path_result.path, path_result.error)
            continue
        files_to_read.extend(path_result.get("files", []))

    logger.info(
        "Found %d eligible file(s) under folder %s (allowed_suffix=%s)",
        len(files_to_read),
        input_file,
        support_suffix,
    )

    for file_info in files_to_read:
        try:
            file_path = file_info["path"]
            suffix = Path(file_path).suffix.lstrip(".").lower()
            reader = _build_reader(suffix, cache_dir)

            yield from reader.read(file_path)

        except Exception as e:  # pylint: disable=broad-except
            logger.exception("Error reading %s: %s", file_info.get("path"), e)
