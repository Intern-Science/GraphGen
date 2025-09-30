import asyncio
from typing import Awaitable, Callable, List, Optional, TypeVar

import gradio as gr
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.utils.log import logger

T = TypeVar("T")
R = TypeVar("R")


async def run_concurrent(
    coro_fn: Callable[[T], Awaitable[R]],
    items: List[T],
    *,
    desc: str = "processing",
    unit: str = "item",
    progress_bar: Optional[gr.Progress] = None,
) -> List[R]:
    tasks = [asyncio.create_task(coro_fn(it)) for it in items]

    results = await tqdm_async.gather(*tasks, desc=desc, unit=unit)

    ok_results = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            logger.exception("Task failed: %s", res)
            if progress_bar:
                progress_bar((idx + 1) / len(items), desc=desc)
            continue
        ok_results.append(res)
        if progress_bar:
            progress_bar((idx + 1) / len(items), desc=desc)

    if progress_bar:
        progress_bar(1.0, desc=desc)
    return ok_results
