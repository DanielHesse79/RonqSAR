from __future__ import annotations

import functools
import time
from typing import Any, Callable, Iterable

from loguru import logger


def timed(func: Callable) -> Callable:
    """Simple timing decorator."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} took {time.time()-start:.3f}s")
        return result

    return wrapper


@functools.lru_cache(maxsize=8)
def cached_list(value: Iterable[Any]) -> tuple[Any, ...]:
    """Convert to tuple for hashing."""
    return tuple(value)
