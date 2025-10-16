import asyncio
from typing import Awaitable, Callable, List, Optional, TypeVar

import gradio as gr
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.utils.log import logger

T = TypeVar("T")
R = TypeVar("R")


# async def run_concurrent(
#     coro_fn: Callable[[T], Awaitable[R]],
#     items: List[T],
#     *,
#     desc: str = "processing",
#     unit: str = "item",
#     progress_bar: Optional[gr.Progress] = None,
# ) -> List[R]:
#     tasks = [asyncio.create_task(coro_fn(it)) for it in items]
#
#     results = []
#     async for future in tqdm_async(
#         tasks, desc=desc, unit=unit
#     ):
#         try:
#             result = await future
#             results.append(result)
#         except Exception as e: # pylint: disable=broad-except
#             logger.exception("Task failed: %s", e)
#
#         if progress_bar is not None:
#             progress_bar((len(results)) / len(items), desc=desc)
#
#     if progress_bar is not None:
#         progress_bar(1.0, desc=desc)
#     return results

#     results = await tqdm_async.gather(*tasks, desc=desc, unit=unit)
#
#     ok_results = []
#     for idx, res in enumerate(results):
#         if isinstance(res, Exception):
#             logger.exception("Task failed: %s", res)
#             if progress_bar:
#                 progress_bar((idx + 1) / len(items), desc=desc)
#             continue
#         ok_results.append(res)
#         if progress_bar:
#             progress_bar((idx + 1) / len(items), desc=desc)
#
#     if progress_bar:
#         progress_bar(1.0, desc=desc)
#     return ok_results

# async def run_concurrent(
#         coro_fn: Callable[[T], Awaitable[R]],
#         items: List[T],
#         *,
#         desc: str = "processing",
#         unit: str = "item",
#         progress_bar: Optional[gr.Progress] = None,
# ) -> List[R]:
#     tasks = [asyncio.create_task(coro_fn(it)) for it in items]
#
#     results = []
#     # 使用同步方式更新进度条，避免异步冲突
#     for i, task in enumerate(asyncio.as_completed(tasks)):
#         try:
#             result = await task
#             results.append(result)
#             # 同步更新进度条
#             if progress_bar is not None:
#                 # 在同步上下文中更新进度
#                 progress_bar((i + 1) / len(items), desc=desc)
#         except Exception as e:
#             logger.exception("Task failed: %s", e)
#             results.append(e)
#
#     return results


async def run_concurrent(
    coro_fn: Callable[[T], Awaitable[R]],
    items: List[T],
    *,
    desc: str = "processing",
    unit: str = "item",
    progress_bar: Optional[gr.Progress] = None,
) -> List[R]:
    tasks = [asyncio.create_task(coro_fn(it)) for it in items]

    completed_count = 0
    results = []

    pbar = tqdm_async(total=len(items), desc=desc, unit=unit)

    if progress_bar is not None:
        progress_bar(0.0, desc=f"{desc} (0/{len(items)})")

    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            results.append(result)
        except Exception as e:  # pylint: disable=broad-except
            logger.exception("Task failed: %s", e)
            # even if failed, record it to keep results consistent with tasks
            results.append(e)

        completed_count += 1
        pbar.update(1)

        if progress_bar is not None:
            progress = completed_count / len(items)
            progress_bar(progress, desc=f"{desc} ({completed_count}/{len(items)})")

    pbar.close()

    if progress_bar is not None:
        progress_bar(1.0, desc=f"{desc} (completed)")

    # filter out exceptions
    results = [res for res in results if not isinstance(res, Exception)]

    return results
