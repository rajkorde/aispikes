from unicodedata import category

import lancedb
import openai
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import CohereReranker
from wikipediaapi import Wikipedia

assert load_dotenv()

wiki = Wikipedia("RAGBot/0.0", "en")

docs = [
    {"text": x, "category": "person"}
    for x in wiki.page("Hayao_Miyazaki").text.split("\n\n")
]

docs += [
    {"text": x, "category": "film"}
    for x in wiki.page("Spirited_Away").text.split("\n\n")
]


model = get_registry().get("openai").create(name="text-embedding-3-small")


class Document(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()
    category: str


db = lancedb.connect("test.db")
tbl = db.create_table("my_table", schema=Document, exist_ok=True)

tbl.add(docs)

tbl.create_fts_index("text", replace=True)

reranker = CohereReranker()

query = "What is Chihiro's new name given to her by the witch?"

results = (
    (tbl.search(query, query_type="hybrid"))
    .where("category='film'", prefilter=True)
    .limit(10)
    .rerank(reranker=reranker)
    .to_pandas()
)


def generate_answer(
    question: str, contexts: list[str], model: str = "gpt-4o-mini"
) -> str:
    context_text = "\n\n".join(contexts)
    prompt = f"""You are an expert assistant. Use the following context to answer the question accurately.

Context:
{context_text}

Question:
{question}

Answer:"""

    try:
        response = openai.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
