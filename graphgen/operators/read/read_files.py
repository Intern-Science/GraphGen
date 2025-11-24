from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import pyarrow as pa
import ray
from ray.data.block import Block, BlockMetadata
from ray.data.datasource import Datasource, ReadTask

from graphgen.models import (
    CSVReader,
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


# pylint: disable=abstract-method
class UnifiedFileDatasource(Datasource):
    """
    A unified Ray DataSource that can read multiple file types
    and automatically route to the appropriate reader.
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        allowed_suffix: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        recursive: bool = True,
        **reader_kwargs,
    ):
        """
        Initialize the datasource.

        :param paths: File or directory paths to read
        :param allowed_suffix: List of allowed file suffixes (e.g., ['pdf', 'txt'])
        :param cache_dir: Directory to cache intermediate files (used for PDF processing)
        :param recursive: Whether to scan directories recursively
        :param reader_kwargs: Additional kwargs passed to readers
        """
        self.paths = [paths] if isinstance(paths, str) else paths
        self.allowed_suffix = (
            [s.lower().lstrip(".") for s in allowed_suffix]
            if allowed_suffix
            else list(_MAPPING.keys())
        )
        self.cache_dir = cache_dir
        self.recursive = recursive
        self.reader_kwargs = reader_kwargs

        # Validate allowed suffixes
        unsupported = set(self.allowed_suffix) - set(_MAPPING.keys())
        if unsupported:
            raise ValueError(f"Unsupported file suffixes: {unsupported}")

    def get_read_tasks(
        self, parallelism: int, per_task_row_limit: Optional[int] = None
    ) -> List[ReadTask]:
        """
        Create read tasks for all discovered files.

        :param parallelism: Number of parallel workers
        :param per_task_row_limit: Optional limit on rows per task
        :return: List of ReadTask objects
        """
        # 1. Scan all paths to discover files
        logger.info("[READ] Scanning paths: %s", self.paths)
        scanner = ParallelFileScanner(
            cache_dir=self.cache_dir,
            allowed_suffix=self.allowed_suffix,
            rescan=False,
            max_workers=parallelism if parallelism > 0 else 1,
        )

        all_files = []
        scan_results = scanner.scan(self.paths, recursive=self.recursive)

        for result in scan_results.values():
            all_files.extend(result.get("files", []))

        logger.info("[READ] Found %d files to process", len(all_files))
        if not all_files:
            return []

        # 2. Group files by suffix to use appropriate reader
        files_by_suffix = {}
        for file_info in all_files:
            suffix = Path(file_info["path"]).suffix.lower().lstrip(".")
            if suffix not in self.allowed_suffix:
                continue
            files_by_suffix.setdefault(suffix, []).append(file_info["path"])

        # 3. Create read tasks
        read_tasks = []

        for suffix, file_paths in files_by_suffix.items():
            # Split files into chunks for parallel processing
            num_chunks = min(parallelism, len(file_paths))
            if num_chunks == 0:
                continue

            chunks = [[] for _ in range(num_chunks)]
            for i, path in enumerate(file_paths):
                chunks[i % num_chunks].append(path)

            # Create a task for each chunk
            for chunk in chunks:
                if not chunk:
                    continue

                # Use factory function to avoid mutable default argument issue
                def make_read_fn(
                    file_paths_chunk, suffix_val, reader_kwargs_val, cache_dir_val
                ):
                    def _read_fn() -> Iterable[Block]:
                        """
                        Read a chunk of files and return blocks.
                        This function runs in a Ray worker.
                        """
                        all_records = []

                        for file_path in file_paths_chunk:
                            try:
                                # Build reader for this file
                                reader = _build_reader(
                                    suffix_val, cache_dir_val, **reader_kwargs_val
                                )

                                # Read the file - readers return Dataset
                                ds = reader.read(file_path, parallelism=parallelism)

                                # Convert Dataset to list of dicts
                                records = ds.take_all()
                                all_records.extend(records)

                            except Exception as e:
                                logger.error(
                                    "[READ] Error reading file %s: %s", file_path, e
                                )
                                continue

                        # Convert list of dicts to PyArrow Table (Block)
                        if all_records:
                            # Create PyArrow Table from records
                            # pylint: disable=no-value-for-parameter
                            table = pa.Table.from_pylist(mapping=all_records)
                            yield table

                    return _read_fn

                # Create closure with current loop variables
                read_fn = make_read_fn(
                    chunk, suffix, self.reader_kwargs, self.cache_dir
                )

                # Calculate metadata for this task
                total_bytes = sum(
                    Path(fp).stat().st_size for fp in chunk if Path(fp).exists()
                )

                # input_files must be Optional[str], not List[str]
                # Use first file as representative or None if empty
                first_file = chunk[0] if chunk else None

                metadata = BlockMetadata(
                    num_rows=None,  # Unknown until read
                    size_bytes=total_bytes,
                    input_files=first_file,
                    exec_stats=None,
                )

                read_tasks.append(
                    ReadTask(
                        read_fn=read_fn,
                        metadata=metadata,
                        schema=None,  # Will be inferred
                        per_task_row_limit=per_task_row_limit,
                    )
                )

        logger.info("[READ] Created %d read tasks", len(read_tasks))
        return read_tasks

    def estimate_inmemory_data_size(self) -> Optional[int]:
        """
        Estimate the total size of data in memory.
        This helps Ray optimize task scheduling.
        """
        try:
            total_size = 0
            for path in self.paths:
                scan_results = ParallelFileScanner(
                    cache_dir=self.cache_dir,
                    allowed_suffix=self.allowed_suffix,
                    rescan=False,
                    max_workers=1,
                ).scan(path, recursive=self.recursive)

                for result in scan_results.values():
                    total_size += result.get("stats", {}).get("total_size", 0)
            return total_size
        except Exception:
            # Return None if estimation fails
            return None


def read_files(
    input_path: Union[str, List[str]],
    allowed_suffix: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
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

    if not ray.is_initialized():
        ray.init()

    try:
        return ray.data.read_datasource(
            UnifiedFileDatasource(
                paths=input_path,
                allowed_suffix=allowed_suffix,
                cache_dir=cache_dir,
                recursive=recursive,
                **reader_kwargs,
            ),
            parallelism=parallelism,
        )
    except Exception as e:
        logger.error("[READ] Failed to read files from %s: %s", input_path, e)
        raise
