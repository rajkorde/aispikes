import os
import time
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline  # type: ignore

P = ParamSpec("P")
R = TypeVar("R")
assert load_dotenv()

audio_file = "data/podcasts/latent_space/dm4r0hi8cbxibnjqkh92vyb4.mp3"


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


if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")


# diarization
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"]
)

pipeline.to(device)


@timed
def diarize(filename: str) -> Any:
    return pipeline(filename)


diarization = diarize()
print(diarization)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

diarization_rrtm = diarization.to_rttm()

# whisper
pipe = pipeline(
    file=audio_file,
    task="automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=device,
    torch_dtype=torch.float16,
    # model_kwargs={"attn_implementation": "flash_attention_2"},
)

# Optional: Optimize model with BetterTransformer
# pipe.model = pipe.model.to_bettertransformer()

# Transcribe the audio file
outputs = pipe(
    audio_file,
    chunk_length_s=30,
    batch_size=2,
    return_timestamps=True,
)

print(outputs["text"])


# whisper
import whisper


@timed
def transcribe(audio_file: str) -> Any:
    model = whisper.load_model("large", device=device)
    return model.transcribe(audio_file, fp16=True)


result = transcribe(audio_file)
print(result["text"])
write_text_to_file("data/podcasts/latent_space/whisper.txt", result["text"])

# mlx-whisper
import mlx_whisper


@timed
def mlx_transcribe(audio_file: str) -> Any:
    return mlx_whisper.transcribe(
        audio_file, path_or_hf_repo="mlx-community/whisper-large-v3-mlx"
    )


result = mlx_transcribe(audio_file)
print(result["text"])
write_text_to_file("data/podcasts/latent_space/mlx_whisper.txt", result["text"])


# Insert into vector database
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector


def split_text(text: str, max_len: int = 800) -> list[str]:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence += "."  # re-add period
        if len(current) + len(sentence) + 1 <= max_len:
            current += " " + sentence if current else sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


text = read_text_from_file("data/podcasts/latent_space/mlx_whisper.txt")

chunks = split_text(text)
print(chunks)

registry = get_registry()
emb_model = (
    get_registry()
    .get("sentence-transformers")
    .create(name="BAAI/bge-small-en-v1.5", device="cpu")
)


class EpisodesChunks(LanceModel):
    date: str
    name: str
    text: str = emb_model.SourceField()
    vector: Vector(emb_model.ndims()) = emb_model.VectorField()


db = lancedb.connect("database/podcasts.db")
table = db.create_table("EpisodeChunks", schema=EpisodesChunks)

entries = []
for chunk in chunks:
    entries.append({"date": "2025-05-05", "name": "The Latent Space", "text": chunk})

table.add(entries)

query = "Summarize the podcast in 4 bullet points"
actuals = table.search(query).limit(10).to_pydantic(EpisodesChunks)
docs = "\n\n".join([x.text for x in actuals])

prompt = f"""
DOCUMENT:
{docs}

QUESTION:
{query}

INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn’t contain the facts to answer the QUESTION return empty string
"""

from openai import OpenAI

client = OpenAI()

response = client.responses.create(model="gpt-4.1-mini", input=prompt)

print(response.output_text)
