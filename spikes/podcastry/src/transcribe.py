from typing import Any

import mlx_whisper

from src.utils import timed


@timed
def mlx_transcribe(audio_file: str) -> Any:
    return mlx_whisper.transcribe(
        audio_file, path_or_hf_repo="mlx-community/whisper-large-v3-mlx"
    )
