# Insert into vector database
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI


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


client = OpenAI()

response = client.responses.create(model="gpt-4.1-mini", input=prompt)

print(response.output_text)
