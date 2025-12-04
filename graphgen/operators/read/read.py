from pathlib import Path
from typing import Any, List, Optional, Union

import ray

from graphgen.models import (
    CSVReader,
    JSONReader,
    ParquetReader,
    PDFReader,
    PickleReader,
    RDFReader,
    TXTReader,
)
from graphgen.utils import compute_mm_hash, logger

from .parallel_file_scanner import ParallelFileScanner

_MAPPING = {
    "jsonl": JSONReader,
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


def _build_reader(suffix: str, cache_dir: str | None, **reader_kwargs):
    """Factory function to build appropriate reader instance"""
    suffix = suffix.lower()
    reader_cls = _MAPPING.get(suffix)
    if not reader_cls:
        raise ValueError(f"Unsupported file suffix: {suffix}")

    # Special handling for PDFReader which needs output_dir
    if suffix == "pdf":
        if cache_dir is None:
            raise ValueError("cache_dir must be provided for PDFReader")
        return reader_cls(output_dir=cache_dir, **reader_kwargs)

    return reader_cls(**reader_kwargs)


def read(
    input_path: Union[str, List[str]],
    allowed_suffix: Optional[List[str]] = None,
    cache_dir: Optional[str] = "cache",
    parallelism: int = 4,
    recursive: bool = True,
    **reader_kwargs: Any,
) -> ray.data.Dataset:
    """
    Unified entry point to read files of multiple types using Ray Data.

    :param input_path: File or directory path(s) to read from
    :param allowed_suffix: List of allowed file suffixes (e.g., ['pdf', 'txt'])
    :param cache_dir: Directory to cache intermediate files (PDF processing)
    :param parallelism: Number of parallel workers
    :param recursive: Whether to scan directories recursively
    :param reader_kwargs: Additional kwargs passed to readers
    :return: Ray Dataset containing all documents
    """
    try:
        # 1. Scan all paths to discover files
        logger.info("[READ] Scanning paths: %s", input_path)
        scanner = ParallelFileScanner(
            cache_dir=cache_dir,
            allowed_suffix=allowed_suffix,
            rescan=False,
            max_workers=parallelism if parallelism > 0 else 1,
        )

        all_files = []
        scan_results = scanner.scan(input_path, recursive=recursive)

        for result in scan_results.values():
            all_files.extend(result.get("files", []))

        logger.info("[READ] Found %d files to process", len(all_files))

        if not all_files:
            return ray.data.from_items([])

        # 2. Group files by suffix to use appropriate reader
        files_by_suffix = {}
        for file_info in all_files:
            suffix = Path(file_info["path"]).suffix.lower().lstrip(".")
            if allowed_suffix and suffix not in [
                s.lower().lstrip(".") for s in allowed_suffix
            ]:
                continue
            files_by_suffix.setdefault(suffix, []).append(file_info["path"])

        # 3. Create read tasks
        read_tasks = []
        for suffix, file_paths in files_by_suffix.items():
            reader = _build_reader(suffix, cache_dir, **reader_kwargs)
            ds = reader.read(file_paths)
            read_tasks.append(ds)

        # 4. Combine all datasets
        if not read_tasks:
            logger.warning("[READ] No datasets created")
            return ray.data.from_items([])

        if len(read_tasks) == 1:
            combined_ds = read_tasks[0]
        else:
            combined_ds = read_tasks[0].union(*read_tasks[1:])

        combined_ds = combined_ds.map(
            lambda record: {
                **record,
                "_doc_id": compute_mm_hash(record),
            }
        )

        logger.info("[READ] Successfully read files from %s", input_path)
        return combined_ds

    except Exception as e:
        logger.error("[READ] Failed to read files from %s: %s", input_path, e)
        raise
