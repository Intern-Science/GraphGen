import os
import time
from typing import List, Dict, Any, Set, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from diskcache import Cache
from graphgen.utils import logger

class ParallelDirScanner:
    def __init__(self,
                 cache_dir: str,
                 allowed_suffix,
                 rescan: bool = False,
                 max_workers: int = 4
                 ):
        self.cache = Cache(cache_dir)
        self.allowed_suffix = set(allowed_suffix) if allowed_suffix else None
        self.rescan = rescan
        self.max_workers = max_workers

    def scan(self, paths: Union[str, List[str]], recursive: bool = True) -> Dict[str, Any]:
        if isinstance(paths, str):
            paths = [paths]

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._scan_dir, Path(p).resolve(), recursive, set()): p
                for p in paths if os.path.exists(p)
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    logger.error("Error scanning path %s: %s", path, e)
                    results[path] = {'error': str(e), 'files': [], 'dirs': [], 'stats': {}}

        return results

    def _scan_dir(self, path: Path, recursive: bool, visited: Set[str]) -> Dict[str, Any]:
        path_str = str(path)

        # Avoid cycles due to symlinks
        if path_str in visited:
            logger.warning("Skipping already visited path: %s", path_str)
            return self._empty_result(path_str)

        # cache check
        cache_key = f"scan::{path_str}::recursive::{recursive}"
        cached = self.cache.get(cache_key)
        if cached and not self.rescan:
            logger.info("Using cached scan result for path: %s", path_str)
            return cached['data']

        logger.info("Scanning path: %s", path_str)
        files, dirs = [], []
        stats = {'total_size': 0, 'file_count': 0, 'dir_count': 0, 'errors': 0}

        try:
            with os.scandir(path_str) as entries:
                for entry in entries:
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)

                        if entry.is_dir():
                            dirs.append({
                                'path': entry.path,
                                'name': entry.name,
                                'mtime': entry_stat.st_mtime
                            })
                            stats['dir_count'] += 1
                        else:
                            # allowed suffix filter
                            if self.allowed_suffix:
                                suffix = Path(entry.name).suffix.lower()
                                if suffix not in self.allowed_suffix:
                                    continue

                            files.append({
                                'path': entry.path,
                                'name': entry.name,
                                'size': entry_stat.st_size,
                                'mtime': entry_stat.st_mtime
                            })
                            stats['total_size'] += entry_stat.st_size
                            stats['file_count'] += 1

                    except OSError:
                        stats['errors'] += 1

        except (PermissionError, FileNotFoundError, OSError) as e:
            logger.error("Failed to scan directory %s: %s", path_str, e)
            return {'error': str(e), 'files': [], 'dirs': [], 'stats': stats}

        if recursive:
            sub_visited = visited | {path_str}
            sub_results = self._scan_subdirs(dirs, sub_visited)

            for sub_data in sub_results.values():
                files.extend(sub_data.get('files', []))
                stats['total_size'] += sub_data['stats'].get('total_size', 0)
                stats['file_count'] += sub_data['stats'].get('file_count', 0)

        result = {'path': path_str, 'files': files, 'dirs': dirs, 'stats': stats}
        self._cache_result(cache_key, result, path)
        return result

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
                executor.submit(self._scan_dir, Path(d['path']), True, visited): d['path']
                for d in dir_list
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    logger.error("Error scanning subdirectory %s: %s", path, e)
                    results[path] = {'error': str(e), 'files': [], 'dirs': [], 'stats': {}}

        return results

    def _cache_result(self, key: str, result: Dict, path: Path):
        """Cache the scan result"""
        try:
            self.cache.set(key, {
                'data': result,
                'dir_mtime': path.stat().st_mtime,
                'cached_at': time.time()
            })
            logger.info(f"Cached scan result for: {path}")
        except OSError:
            pass

    def invalidate(self, path: str):
        """Invalidate cache for a specific path"""
        path = Path(path).resolve()
        keys = [k for k in self.cache if k.startswith(f"scan:{path}")]
        for k in keys:
            self.cache.delete(k)
        logger.info(f"Invalidated cache for path: {path}")

    def close(self):
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _empty_result(path: str) -> Dict[str, Any]:
        return {
            'path': path,
            'files': [],
            'dirs': [],
            'stats': {'total_size': 0, 'file_count': 0, 'dir_count': 0, 'errors': 0}
        }
