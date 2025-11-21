from ray.data.datasource import (
    Datasource, ReadTask
)
import pyarrow as pa
from typing import List, Dict, Any, Optional, Union

from typing import Iterator, List, Optional

import ray

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

class UnifiedFileDatasource(Datasource):
    pass


def read_files(
    input_path: str,
    allowed_suffix: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    parallelism: int = 4,
    **ray_kwargs,
) -> ray.data.Dataset:
    """
    Reads files from the specified input path, filtering by allowed suffixes,
    and returns a Ray Dataset containing the read documents.
    :param input_path: input file or directory path
    :param allowed_suffix: list of allowed file suffixes (e.g., ['pdf', 'txt'])
    :param cache_dir: directory to cache intermediate files (used for PDF reading)
    :param parallelism: number of parallel workers for reading files
    :param ray_kwargs: additional keyword arguments for Ray Dataset reading
    :return: Ray Dataset containing the read documents
    """

    if not ray.is_initialized():
        ray.init()


    return ray.data.read_datasource(
        UnifiedFileDatasource(
            paths=[input_path],
            allowed_suffix=allowed_suffix,
            cache_dir=cache_dir,
            **ray_kwargs,  # Pass additional Ray kwargs here
        ),
        parallelism=parallelism,
    )


    # path = Path(input_file).expanduser()
    # if not path.exists():
    #     raise FileNotFoundError(f"[Read] input_path not found: {input_file}")
    #
    # if allowed_suffix is None:
    #     support_suffix = set(_MAPPING.keys())
    # else:
    #     support_suffix = {s.lower().lstrip(".") for s in allowed_suffix}
    #
    # # single file
    # if path.is_file():
    #     suffix = path.suffix.lstrip(".").lower()
    #     if suffix not in support_suffix:
    #         logger.warning(
    #             "[Read] Skip file %s (suffix '%s' not in allowed_suffix %s)",
    #             path,
    #             suffix,
    #             support_suffix,
    #         )
    #         return
    #     reader = _build_reader(suffix, cache_dir)
    #     logger.info("[Read] Reading file %s", path)
    #     yield reader.read(str(path))
    #     return
    #
    # # folder
    # logger.info("[Read] Streaming directory %s", path)
    # for p in path.rglob("*"):
    #     if p.is_file() and p.suffix.lstrip(".").lower() in support_suffix:
    #         try:
    #             suffix = p.suffix.lstrip(".").lower()
    #             reader = _build_reader(suffix, cache_dir)
    #             logger.info("[Reader] Reading file %s", p)
    #             docs = reader.read(str(p))
    #             if docs:
    #                 yield docs
    #         except Exception:  # pylint: disable=broad-except
    #             logger.exception("[Reader] Error reading %s", p)
