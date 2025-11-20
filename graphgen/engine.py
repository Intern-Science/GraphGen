"""
orchestration engine for GraphGen
"""

import threading
import traceback
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, List, Optional


class AggMode(Enum):
    MAP = auto()  # whenever upstream produces a result, run
    ALL_REDUCE = auto()  # wait for all upstream results, then run


class Context(dict):
    _lock = threading.Lock()

    def set(self, k, v):
        with self._lock:
            self[k] = v

    def get(self, k, default=None):
        with self._lock:
            return super().get(k, default)


class OpNode:
    def __init__(
        self,
        name: str,
        deps: List[str],
        compute_func: Callable[["OpNode", Context], Any],
        callback_func: Optional[Callable[["OpNode", Context, List[Any]], None]] = None,
        agg_mode: AggMode = AggMode.ALL_REDUCE,
    ):
        self.name = name
        self.deps = deps
        self.compute_func = compute_func
        self.callback_func = callback_func or (lambda self, ctx, results: None)
        self.agg_mode = agg_mode


def op(name: str, deps=None):
    deps = deps or []

    def decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _wrapper.op_node = OpNode(name, deps, lambda self, ctx: func(self, **ctx))
        return _wrapper

    return decorator


class Engine:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def run(self, ops: List[OpNode], ctx: Context):
        name2op = {operation.name: operation for operation in ops}

        # topological sort
        graph = {n: set(name2op[n].deps) for n in name2op}
        topo = []
        q = [n for n, d in graph.items() if not d]
        while q:
            cur = q.pop(0)
            topo.append(cur)
            for child in [c for c, d in graph.items() if cur in d]:
                graph[child].remove(cur)
                if not graph[child]:
                    q.append(child)

        if len(topo) != len(ops):
            raise ValueError(
                "Cyclic dependencies detected among operations."
                "Please check your configuration."
            )

        # semaphore for max_workers
        sem = threading.Semaphore(self.max_workers)
        done = {n: threading.Event() for n in name2op}
        exc = {}

        def _exec(n: str):
            with sem:
                for d in name2op[n].deps:
                    done[d].wait()
                if any(d in exc for d in name2op[n].deps):
                    exc[n] = Exception("Skipped due to failed dependencies")
                    done[n].set()
                    return
                try:
                    name2op[n].func(name2op[n], ctx)
                except Exception:  # pylint: disable=broad-except
                    exc[n] = traceback.format_exc()
                done[n].set()

        ts = [threading.Thread(target=_exec, args=(n,), daemon=True) for n in topo]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if exc:
            raise RuntimeError(
                "Some operations failed:\n"
                + "\n".join(f"---- {op} ----\n{tb}" for op, tb in exc.items())
            )


class Bucket:
    """
    Bucket for a single operation, collecting computation results and triggering downstream ops
    """

    def __init__(
        self, name: str, size: int, mode: AggMode, callback: Callable[[List[Any]], None]
    ):
        self.name = name
        self.size = size
        self.mode = mode
        self.callback = callback
        self._lock = threading.Lock()
        self._results: List[Any] = []
        self._done = False

    def put(self, result: Any):
        with self._lock:
            if self._done:
                return
            self._results.append(result)

            if self.mode == AggMode.MAP or len(self._results) == self.size:
                self._fire()

    def _fire(self):
        self._done = True
        threading.Thread(target=self._callback_wrapper, daemon=True).start()

    def _callback_wrapper(self):
        try:
            self.callback(self._results)
        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()


class BucketManager:
    def __init__(self):
        self._buckets: dict[str, Bucket] = {}
        self._lock = threading.Lock()

    def register(
        self,
        node_name: str,
        bucket_size: int,
        mode: AggMode,
        callback: Callable[[List[Any]], None],
    ):
        with self._lock:
            if node_name in self._buckets:
                raise RuntimeError(f"Bucket {node_name} already registered")
            self._buckets[node_name] = Bucket(
                name=node_name, size=bucket_size, mode=mode, callback=callback
            )
            return self._buckets[node_name]

    def get(self, node_name: str) -> Optional[Bucket]:
        with self._lock:
            return self._buckets.get(node_name)


def collect_ops(config: dict, graph_gen) -> List[OpNode]:
    """
    build operation nodes from yaml config
    :param config
    :param graph_gen
    """
    ops: List[OpNode] = []
    for stage in config["pipeline"]:
        name = stage["name"]
        method = getattr(graph_gen, name)
        op_node = method.op_node

        # if there are runtime dependencies, override them
        runtime_deps = stage.get("deps", op_node.deps)
        op_node.deps = runtime_deps

        if "params" in stage:
            op_node.func = lambda self, ctx, m=method, sc=stage: m(sc.get("params", {}))
        else:
            op_node.func = lambda self, ctx, m=method: m()
        ops.append(op_node)
    return ops
