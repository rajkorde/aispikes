import time
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timed(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            end = time.perf_counter()
            print(f"{fn.__name__} executed in {end - start:.4f} seconds")

    return wrapper


def write_text_to_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text_from_file(file_path: str) -> str:
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {e}")
