"""
orchestration engine for GraphGen
"""
import queue
import threading
import traceback
from enum import Enum, auto
from functools import wraps
from typing import Callable, Dict, List


class OpType(Enum):
    STREAMING = auto()  # once data from upstream arrives, process it immediately
    BARRIER = auto()  # wait for all upstream data to arrive before processing

    # TODO: implement batch processing
    # BATCH = auto()  # process data in batches


# signals the end of a data stream
class EndOfStream:
    pass


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
        func: Callable,
        op_type: OpType = OpType.BARRIER,  # use barrier by default
    ):
        self.name = name
        self.deps = deps
        self.func = func
        self.op_type = op_type


def op(name: str, deps=None, op_type: OpType = OpType.BARRIER):
    deps = deps or []

    def decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _wrapper.op_node = OpNode(
            name,
            deps,
            func,
            op_type=op_type,
        )
        return _wrapper

    return decorator


class Engine:
    def __init__(self, queue_size: int = 100):
        self.queue_size = queue_size

    @staticmethod
    def _topo_sort(name2op: Dict[str, OpNode]) -> List[str]:
        adj = {n: [] for n in name2op}
        in_degree = {n: 0 for n in name2op}

        for name, operation in name2op.items():
            for dep_name in operation.deps:
                if dep_name not in name2op:
                    raise ValueError(f"Dependency {dep_name} of {name} not found")
                adj[dep_name].append(name)
                in_degree[name] += 1

        # Kahn's algorithm for topological sorting
        queue_nodes = [n for n in name2op if in_degree[n] == 0]
        topo_order = []

        while queue_nodes:
            u = queue_nodes.pop(0)
            topo_order.append(u)

            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue_nodes.append(v)

        # cycle detection
        if len(topo_order) != len(name2op):
            cycle_nodes = set(name2op.keys()) - set(topo_order)
            raise ValueError(f"Cyclic dependency detected among: {cycle_nodes}")
        return topo_order

    def _build_channels(self, name2op):
        """Return channels / consumers_of / producer_counts"""
        channels, consumers_of, producer_counts = {}, {}, {n: 0 for n in name2op}
        for name, operator in name2op.items():
            consumers_of[name] = []
            for dep in operator.deps:
                if dep not in name2op:
                    raise ValueError(f"Dependency {dep} of {name} not found")
                channels[(dep, name)] = queue.Queue(maxsize=self.queue_size)
                consumers_of[dep].append(name)
                producer_counts[name] += 1
        return channels, consumers_of, producer_counts

    def _run_workers(self, ordered_ops, channels, consumers_of, producer_counts, ctx):
        """Run worker threads for each operation node."""
        exceptions, threads = {}, []
        for node in ordered_ops:
            t = threading.Thread(
                target=self._worker_loop,
                args=(node, channels, consumers_of, producer_counts, ctx, exceptions),
                daemon=True,
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return exceptions

    def _worker_loop(
        self, node, channels, consumers_of, producer_counts, ctx, exceptions
    ):
        op_name = node.name

        def input_generator():
            # if no dependencies, yield None once
            if not node.deps:
                yield None
                return

            active_producers = producer_counts[op_name]
            # collect all queues
            input_queues = [channels[(dep_name, op_name)] for dep_name in node.deps]

            # loop until all producers are done
            while active_producers > 0:
                got_data = False
                for q in input_queues:
                    try:
                        item = q.get(timeout=0.1)
                        if isinstance(item, EndOfStream):
                            active_producers -= 1
                        else:
                            yield item
                        got_data = True
                    except queue.Empty:
                        continue

                if not got_data and active_producers > 0:
                    # barrier wait on the first active queue
                    item = input_queues[0].get()
                    if isinstance(item, EndOfStream):
                        active_producers -= 1
                    else:
                        yield item

        in_stream = input_generator()

        try:
            # execute the operation
            result_iter = []
            if node.op_type == OpType.BARRIER:
                # consume all input
                buffered_inputs = list(in_stream)
                res = node.func(self, ctx, inputs=buffered_inputs)
                if res is not None:
                    result_iter = res if isinstance(res, (list, tuple)) else [res]

            elif node.op_type == OpType.STREAMING:
                # process input one by one
                res = node.func(self, ctx, input_stream=in_stream)
                if res is not None:
                    result_iter = res
            else:
                raise ValueError(f"Unknown OpType {node.op_type} for {op_name}")

            # output dispatch, send results to downstream consumers
            if result_iter:
                for item in result_iter:
                    for consumer_name in consumers_of[op_name]:
                        channels[(op_name, consumer_name)].put(item)

        except Exception:  # pylint: disable=broad-except
            traceback.print_exc()
            exceptions[op_name] = traceback.format_exc()

        finally:
            # signal end of stream to downstream consumers
            for consumer_name in consumers_of[op_name]:
                channels[(op_name, consumer_name)].put(EndOfStream())

    def run(self, ops: List[OpNode], ctx: Context):
        name2op = {op.name: op for op in ops}

        # Step 1: topo sort and validate
        sorted_op_names = self._topo_sort(name2op)

        # Step 2: build channels and tracking structures
        channels, consumers_of, producer_counts = self._build_channels(name2op)

        # Step3: start worker threads using topo order
        ordered_ops = [name2op[name] for name in sorted_op_names]
        exceptions = self._run_workers(
            ordered_ops, channels, consumers_of, producer_counts, ctx
        )

        if exceptions:
            raise RuntimeError(f"Engine encountered exceptions: {exceptions}")


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
        deps = stage.get("deps", op_node.deps)
        op_type = op_node.op_type

        if op_type == OpType.BARRIER:
            if "params" in stage:

                def func(self, ctx, inputs, m=method, sc=stage):
                    return m(sc.get("params", {}), inputs=inputs)

            else:

                def func(self, ctx, inputs, m=method):
                    return m(inputs=inputs)

        elif op_type == OpType.STREAMING:
            if "params" in stage:

                def func(self, ctx, input_stream, m=method, sc=stage):
                    return m(sc.get("params", {}), input_stream=input_stream)

            else:

                def func(self, ctx, input_stream, m=method):
                    return m(input_stream=input_stream)

        else:
            raise ValueError(f"Unknown OpType {op_type} for operation {name}")

        new_node = OpNode(
            name=name,
            deps=deps,
            func=func,
            op_type=op_type,
        )
        ops.append(new_node)
    return ops
