import inspect
import logging
from collections import defaultdict, deque
from functools import wraps
from typing import Any, Callable, Dict, List, Set

import ray
import ray.data

from graphgen.bases import Config, Node


class Engine:
    def __init__(
        self, config: Dict[str, Any], functions: Dict[str, Callable], **ray_init_kwargs
    ):
        self.config = Config(**config)
        self.global_params = self.config.global_params
        self.functions = functions
        self.datasets: Dict[str, ray.data.Dataset] = {}

        if not ray.is_initialized():
            context = ray.init(
                ignore_reinit_error=True,
                logging_level=logging.ERROR,
                log_to_driver=True,
                **ray_init_kwargs,
            )
            print(f"Ray Dashboard URL: {context.dashboard_url}")

    @staticmethod
    def _topo_sort(nodes: List[Node]) -> List[Node]:
        id_to_node: Dict[str, Node] = {}
        for n in nodes:
            id_to_node[n.id] = n

        indeg: Dict[str, int] = {nid: 0 for nid in id_to_node}
        adj: Dict[str, List[str]] = defaultdict(list)

        for n in nodes:
            nid = n.id
            deps: List[str] = n.dependencies
            uniq_deps: Set[str] = set(deps)
            for d in uniq_deps:
                if d not in id_to_node:
                    raise ValueError(
                        f"The dependency node id {d} of node {nid} is not defined in the configuration."
                    )
                indeg[nid] += 1
                adj[d].append(nid)

        zero_deg: deque = deque(
            [id_to_node[nid] for nid, deg in indeg.items() if deg == 0]
        )
        sorted_nodes: List[Node] = []

        while zero_deg:
            cur = zero_deg.popleft()
            sorted_nodes.append(cur)
            cur_id = cur.id
            for nb_id in adj.get(cur_id, []):
                indeg[nb_id] -= 1
                if indeg[nb_id] == 0:
                    zero_deg.append(id_to_node[nb_id])

        if len(sorted_nodes) != len(nodes):
            remaining = [nid for nid, deg in indeg.items() if deg > 0]
            raise ValueError(
                f"The configuration contains cycles, unable to execute. Remaining nodes with indegree > 0: {remaining}"
            )

        return sorted_nodes

    def _get_input_dataset(
        self, node: Node, initial_ds: ray.data.Dataset
    ) -> ray.data.Dataset:
        deps = node.dependencies

        if not deps:
            return initial_ds

        if len(deps) == 1:
            return self.datasets[deps[0]]

        main_ds = self.datasets[deps[0]]
        other_dss = [self.datasets[d] for d in deps[1:]]
        if not all(ds.schema() == main_ds.schema() for ds in other_dss):
            raise ValueError(
                f"Union requires all datasets to have the same schema for node {node.id}"
            )
        return main_ds.union(*other_dss)

    def _execute_node(self, node: Node, initial_ds: ray.data.Dataset):
        def _filter_kwargs(
            func_or_class: Callable,
            global_params: Dict[str, Any],
            func_params: Dict[str, Any],
        ) -> Dict[str, Any]:
            """
            1. global_params: only when specified in function signature, will be passed
            2. func_params: pass specified params first, then **kwargs if exists
            """
            try:
                sig = inspect.signature(func_or_class)
            except ValueError:
                return {}

            params = sig.parameters
            final_kwargs = {}

            has_var_keywords = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            valid_keys = set(params.keys())
            for k, v in global_params.items():
                if k in valid_keys:
                    final_kwargs[k] = v

            for k, v in func_params.items():
                if k in valid_keys or has_var_keywords:
                    final_kwargs[k] = v
            return final_kwargs

        if node.op_name not in self.functions:
            raise ValueError(f"Operator {node.op_name} not found for node {node.id}")

        op_handler = self.functions[node.op_name]
        node_params = _filter_kwargs(op_handler, self.global_params, node.params or {})

        if node.type == "source":
            self.datasets[node.id] = op_handler(**node_params)
            return

        input_ds = self._get_input_dataset(node, initial_ds)

        if inspect.isclass(op_handler):
            execution_params = node.execution_params or {}
            replicas = execution_params.get("replicas", 1)
            batch_size = (
                int(execution_params.get("batch_size"))
                if "batch_size" in execution_params
                else "default"
            )
            compute_resources = execution_params.get("compute_resources", {})

            if node.type == "aggregate":
                self.datasets[node.id] = input_ds.repartition(1).map_batches(
                    op_handler,
                    compute=ray.data.ActorPoolStrategy(min_size=1, max_size=1),
                    batch_size=None,  # aggregate processes the whole dataset at once
                    num_gpus=compute_resources.get("num_gpus", 0)
                    if compute_resources
                    else 0,
                    fn_constructor_kwargs=node_params,
                    batch_format="pandas",
                )
            else:
                # others like map, filter, flatmap, map_batch let actors process data inside batches
                self.datasets[node.id] = input_ds.map_batches(
                    op_handler,
                    compute=ray.data.ActorPoolStrategy(min_size=1, max_size=replicas),
                    batch_size=batch_size,
                    num_gpus=compute_resources.get("num_gpus", 0)
                    if compute_resources
                    else 0,
                    fn_constructor_kwargs=node_params,
                    batch_format="pandas",
                )

        else:

            @wraps(op_handler)
            def func_wrapper(row_or_batch: Dict[str, Any]) -> Dict[str, Any]:
                return op_handler(row_or_batch, **node_params)

            if node.type == "map":
                self.datasets[node.id] = input_ds.map(func_wrapper)
            elif node.type == "filter":
                self.datasets[node.id] = input_ds.filter(func_wrapper)
            elif node.type == "flatmap":
                self.datasets[node.id] = input_ds.flat_map(func_wrapper)
            elif node.type == "aggregate":
                self.datasets[node.id] = input_ds.repartition(1).map_batches(
                    func_wrapper, batch_format="default"
                )
            elif node.type == "map_batch":
                self.datasets[node.id] = input_ds.map_batches(func_wrapper)
            else:
                raise ValueError(
                    f"Unsupported node type {node.type} for node {node.id}"
                )

    @staticmethod
    def _find_leaf_nodes(nodes: List[Node]) -> Set[str]:
        all_ids = {n.id for n in nodes}
        deps_set = set()
        for n in nodes:
            deps_set.update(n.dependencies)
        return all_ids - deps_set

    def execute(self, initial_ds: ray.data.Dataset) -> Dict[str, List[Any]]:
        sorted_nodes = self._topo_sort(self.config.nodes)

        for node in sorted_nodes:
            self._execute_node(node, initial_ds)

        leaf_nodes = self._find_leaf_nodes(sorted_nodes)

        @ray.remote
        def _fetch_result(ds: ray.data.Dataset) -> List[Any]:
            return ds.take_all()

        results = ray.get(
            [_fetch_result.remote(self.datasets[node_id]) for node_id in leaf_nodes]
        )
        return dict(zip(leaf_nodes, results))
