import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from graphgen.models import RocksDBCache
from graphgen.utils import logger


class ParallelFileScanner:
    def __init__(
        self, cache_dir: str, allowed_suffix, rescan: bool = False, max_workers: int = 4
    ):
        self.cache = RocksDBCache(os.path.join(cache_dir, "file_paths_cache"))
        self.allowed_suffix = set(allowed_suffix) if allowed_suffix else None
        self.rescan = rescan
        self.max_workers = max_workers

    def scan(
        self, paths: Union[str, List[str]], recursive: bool = True
    ) -> Dict[str, Any]:
        if isinstance(paths, str):
            paths = [paths]

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {}
            for p in paths:
                if os.path.exists(p):
                    future = executor.submit(
                        self._scan_files, Path(p).resolve(), recursive, set()
                    )
                    future_to_path[future] = p
                else:
                    logger.warning("[READ] Path does not exist: %s", p)

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    logger.error("[READ] Error scanning path %s: %s", path, e)
                    results[path] = {
                        "error": str(e),
                        "files": [],
                        "dirs": [],
                        "stats": {},
                    }
        return results

    def _scan_files(
        self, path: Path, recursive: bool, visited: Set[str]
    ) -> Dict[str, Any]:
        path_str = str(path)

        # Avoid cycles due to symlinks
        if path_str in visited:
            logger.warning("[READ] Skipping already visited path: %s", path_str)
            return self._empty_result(path_str)

        # cache check
        cache_key = f"scan::{path_str}::recursive::{recursive}"
        cached = self.cache.get(cache_key)
        if cached and not self.rescan:
            logger.info("[READ] Using cached scan result for path: %s", path_str)
            return cached["data"]

        logger.info("[READ] Scanning path: %s", path_str)
        files, dirs = [], []
        stats = {"total_size": 0, "file_count": 0, "dir_count": 0, "errors": 0}

        try:
            path_stat = path.stat()
            if path.is_file():
                return self._scan_single_file(path, path_str, path_stat)
            if path.is_dir():
                with os.scandir(path_str) as entries:
                    for entry in entries:
                        try:
                            entry_stat = entry.stat(follow_symlinks=False)

                            if entry.is_dir():
                                dirs.append(
                                    {
                                        "path": entry.path,
                                        "name": entry.name,
                                        "mtime": entry_stat.st_mtime,
                                    }
                                )
                                stats["dir_count"] += 1
                            else:
                                # allowed suffix filter
                                if not self._is_allowed_file(Path(entry.path)):
                                    continue
                                files.append(
                                    {
                                        "path": entry.path,
                                        "name": entry.name,
                                        "size": entry_stat.st_size,
                                        "mtime": entry_stat.st_mtime,
                                    }
                                )
                                stats["total_size"] += entry_stat.st_size
                                stats["file_count"] += 1

                        except OSError:
                            stats["errors"] += 1

        except (PermissionError, FileNotFoundError, OSError) as e:
            logger.error("[READ] Failed to scan path %s: %s", path_str, e)
            return {"error": str(e), "files": [], "dirs": [], "stats": stats}

        if recursive:
            sub_visited = visited | {path_str}
            sub_results = self._scan_subdirs(dirs, sub_visited)

            for sub_data in sub_results.values():
                files.extend(sub_data.get("files", []))
                stats["total_size"] += sub_data["stats"].get("total_size", 0)
                stats["file_count"] += sub_data["stats"].get("file_count", 0)

        result = {"path": path_str, "files": files, "dirs": dirs, "stats": stats}
        self._cache_result(cache_key, result, path)
        return result

    def _scan_single_file(
        self, path: Path, path_str: str, stat: os.stat_result
    ) -> Dict[str, Any]:
        """Scan a single file and return its metadata"""
        if not self._is_allowed_file(path):
            return self._empty_result(path_str)

        return {
            "path": path_str,
            "files": [
                {
                    "path": path_str,
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            ],
            "dirs": [],
            "stats": {
                "total_size": stat.st_size,
                "file_count": 1,
                "dir_count": 0,
                "errors": 0,
            },
        }

    def _scan_subdirs(self, dir_list: List[Dict], visited: Set[str]) -> Dict[str, Any]:
        """
        Parallel scan subdirectories
        :param dir_list
        :param visited
        :return:
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._scan_files, Path(d["path"]), True, visited): d[
                    "path"
                ]
                for d in dir_list
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    logger.error("[READ] Error scanning subdirectory %s: %s", path, e)
                    results[path] = {
                        "error": str(e),
                        "files": [],
                        "dirs": [],
                        "stats": {},
                    }

        return results

    def _cache_result(self, key: str, result: Dict, path: Path):
        """Cache the scan result"""
        try:
            self.cache.set(
                key,
                {
                    "data": result,
                    "dir_mtime": path.stat().st_mtime,
                    "cached_at": time.time(),
                },
            )
            logger.info("[READ] Cached scan result for path: %s", path)
        except OSError as e:
            logger.error("[READ] Failed to cache scan result for path %s: %s", path, e)

    def _is_allowed_file(self, path: Path) -> bool:
        """Check if the file has an allowed suffix"""
        if self.allowed_suffix is None:
            return True
        suffix = path.suffix.lower().lstrip(".")
        return suffix in self.allowed_suffix

    def invalidate(self, path: str):
        """Invalidate cache for a specific path"""
        path = Path(path).resolve()
        keys = [k for k in self.cache if k.startswith(f"scan::{path}")]
        for k in keys:
            self.cache.delete(k)
        logger.info("[READ] Invalidated cache for path: %s", path)

    def close(self):
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _empty_result(path: str) -> Dict[str, Any]:
        return {
            "path": path,
            "files": [],
            "dirs": [],
            "stats": {"total_size": 0, "file_count": 0, "dir_count": 0, "errors": 0},
        }
