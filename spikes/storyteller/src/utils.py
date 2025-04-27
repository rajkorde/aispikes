import os
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def deserialize(file_path: str, model_class: Type[T]) -> T:
    try:
        data = Path(file_path).read_text()
        return model_class.model_validate_json(data)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize {file_path}: {e}") from e


def serialize(obj: BaseModel, file_path: str) -> None:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(obj.model_dump_json())
