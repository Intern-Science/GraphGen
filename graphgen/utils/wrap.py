from functools import wraps
from typing import Any, Callable

from .loop import create_event_loop


def async_to_sync_method(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        loop = create_event_loop()
        return loop.run_until_complete(func(self, *args, **kwargs))

    return wrapper
