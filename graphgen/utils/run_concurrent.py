import asyncio
from typing import Awaitable, Callable, List, TypeVar

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.utils.log import logger
from .loop import create_event_loop

T = TypeVar("T")
R = TypeVar("R")


def run_concurrent(
    coro_fn: Callable[[T], Awaitable[R]],
    items: List[T],
    *,
    desc: str = "processing",
    unit: str = "item",
) -> List[R]:
    async def _run_all():
        tasks = [asyncio.create_task(coro_fn(item)) for item in items]

        results = []
        pbar = tqdm_async(total=len(items), desc=desc, unit=unit)

        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
            except Exception as e:  # pylint: disable=broad-except
                logger.exception("Task failed: %s", e)
                results.append(e)

            pbar.update(1)

        pbar.close()
        return [res for res in results if not isinstance(res, Exception)]

    loop = create_event_loop()
    try:
        return loop.run_until_complete(_run_all())
    finally:
        loop.close()
