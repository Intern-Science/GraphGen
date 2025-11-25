import asyncio
from typing import List, Tuple

import gradio as gr

from graphgen.bases import BaseGraphStorage, BaseTokenizer
from graphgen.utils import run_concurrent


async def pre_tokenize(
    graph_storage: BaseGraphStorage,
    tokenizer: BaseTokenizer,
    edges: List[Tuple],
    nodes: List[Tuple],
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
) -> Tuple[List, List]:
    """为 edges/nodes 补 token-length 并回写存储，并发 1000，带进度条。"""
    sem = asyncio.Semaphore(max_concurrent)

    async def _patch_and_write(obj: Tuple, *, is_node: bool) -> Tuple:
        async with sem:
            data = obj[1] if is_node else obj[2]
            if "length" not in data:
                loop = asyncio.get_event_loop()
                data["length"] = len(
                    await loop.run_in_executor(
                        None, tokenizer.encode, data["description"]
                    )
                )
            if is_node:
                graph_storage.update_node(obj[0], obj[1])
            else:
                graph_storage.update_edge(obj[0], obj[1], obj[2])
            return obj

    new_edges, new_nodes = await asyncio.gather(
        run_concurrent(
            lambda e: _patch_and_write(e, is_node=False),
            edges,
            desc="Pre-tokenizing edges",
            unit="edge",
            progress_bar=progress_bar,
        ),
        run_concurrent(
            lambda n: _patch_and_write(n, is_node=True),
            nodes,
            desc="Pre-tokenizing nodes",
            unit="node",
            progress_bar=progress_bar,
        ),
    )

    graph_storage.index_done_callback()
    return new_edges, new_nodes
