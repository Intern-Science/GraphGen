import asyncio
import inspect
from functools import wraps


def async_to_sync_method(func):
    """Convert async method to sync method, handling both coroutines and async generators."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = func(*args, **kwargs)

        # handle async generator (STREAMING operation)
        if inspect.isasyncgen(result):
            async_gen = result

            def sync_generator():
                try:
                    while True:
                        item = loop.run_until_complete(anext(async_gen))
                        yield item
                except StopAsyncIteration:
                    pass
                finally:
                    loop.close()

            return sync_generator()

        # handle coroutine (BARRIER operation)
        if inspect.iscoroutine(result):
            try:
                return loop.run_until_complete(result)
            finally:
                loop.close()

        else:
            loop.close()
            return result

    return wrapper
