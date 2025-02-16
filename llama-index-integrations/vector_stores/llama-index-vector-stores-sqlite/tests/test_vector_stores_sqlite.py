import os

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.vector_stores.sqlite import SQLiteVectorStore
from llama_index.vector_stores.sqlite.base import DBEmbeddingRow
from openai import OpenAI


def test_class():
    names_of_base_classes = [b.__name__ for b in SQLiteVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_insert():
    sentences = [
        "Capri-Sun is a brand of juice concentrate–based drinks manufactured by the German company Wild and regional licensees.",
        "George V was King of the United Kingdom and the British Dominions, and Emperor of India, from 6 May 1910 until his death in 1936.",
        "Alaqua Cox is a Native American (Menominee) actress.",
        "Shohei Ohtani is a Japanese professional baseball pitcher and designated hitter for the Los Angeles Dodgers of Major League Baseball.",
        "Tamarindo, also commonly known as agua de tamarindo, is a non-alcoholic beverage made of tamarind, sugar, and water.",
    ]

    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    nodes = []
    for s in sentences:
        e = client.embeddings.create(input=s, model="text-embedding-3-small")
        nodes.append(TextNode(text=s, embedding=e.data[0].embedding))

    vector_store = SQLiteVectorStore()
    vector_store.add(nodes)

    query = "fruity liquids"
    query_embedding = (
        client.embeddings.create(input=query, model="text-embedding-3-small")
        .data[0]
        .embedding
    )
    result = vector_store._query_with_score(query_embedding)
    assert result, "_query_with_score result is non-empty"

    query = VectorStoreQuery(
        query_embedding=query_embedding,
    )
    result = vector_store.query(query)
    assert result, "query result is non-empty"
