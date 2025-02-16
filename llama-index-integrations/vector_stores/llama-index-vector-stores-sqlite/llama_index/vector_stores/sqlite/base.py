import json
import sqlite3
import struct
from typing import Any, NamedTuple, Optional, Sequence

import fsspec
import sqlite_vec
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from pydantic.fields import PrivateAttr


class DBEmbeddingRow(NamedTuple):
    node_id: str
    text: str
    metadata: dict[str, Any]
    similarity: float


class SQLiteVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    embed_dim: int
    connection_string: str
    table_name: str = "vec_items"
    db: sqlite3.Connection | None = None
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        connection_string: str = ":memory:",
        embed_dim: int = 1536,
    ):
        super().__init__(embed_dim=embed_dim, connection_string=connection_string)  # pyright: ignore

    def _initialize(self) -> None:
        if not self._is_initialized:
            self.db = sqlite3.connect(self.connection_string)
            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
            self.db.execute(
                f"""CREATE VIRTUAL TABLE {self.table_name} USING vec0(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metadata_ TEXT,
                node_id TEXT,
                embedding float[{self.embed_dim}] distance_metric=cosine,
                +text TEXT
                )"""
            )
            self._is_initialized = True

    @property
    def client(self) -> Any:
        """Get client."""
        if not self._is_initialized:
            return None
        return self.db

    def get_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> list[BaseNode]:
        """Get nodes from vector store."""
        raise NotImplementedError("get_nodes not implemented")

    async def aget_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> list[BaseNode]:
        """Asynchronously get nodes from vector store."""
        return self.get_nodes(node_ids, filters)

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> list[str]:
        """Add nodes to vector store."""
        self._initialize()
        items = [
            (
                node.node_id,
                node.embedding,
                json.dumps(node.metadata, separators=(",", ":")),
                node.get_content(),
            )
            for node in nodes
        ]
        with self.db:
            cursor = self.db.cursor()
            ids = []
            for item in items:
                self.db.execute(
                    "INSERT INTO vec_items(node_id, embedding, metadata_, text) VALUES (?, ?, ?, ?)",
                    [item[0], serialize_f32(item[1]), item[2], item[3]],  # pyright: ignore[reportArgumentType]
                )
                self.db.commit()
                ids.append(cursor.lastrowid)
            return ids

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> list[str]:
        """
        Asynchronously add nodes to vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call add synchronously.
        """
        return self.add(nodes, **kwargs)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id."""
        raise NotImplementedError("get_nodes not implemented")

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call delete synchronously.
        """
        self.delete(ref_doc_id, **delete_kwargs)

    def delete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes from vector store."""
        raise NotImplementedError("delete_nodes not implemented")

    async def adelete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Asynchronously delete nodes from vector store."""
        self.delete_nodes(node_ids, filters)

    def clear(self) -> None:
        """Clear all nodes from configured vector store."""
        raise NotImplementedError("clear not implemented")

    async def aclear(self) -> None:
        """Asynchronously clear all nodes from configured vector store."""
        self.clear()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        self._initialize()
        # TODO: query mode hybrid/sparse/text
        if query.mode == VectorStoreQueryMode.DEFAULT:
            results = self._query_with_score(
                query.query_embedding,
                query.similarity_top_k,
                query.filters,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        return self._db_rows_to_query_result(results)

    def _db_rows_to_query_result(
        self, rows: list[DBEmbeddingRow]
    ) -> VectorStoreQueryResult:
        nodes = []
        similarities = []
        ids = []
        for db_embedding_row in rows:
            try:
                node = metadata_dict_to_node(db_embedding_row.metadata)
                node.set_content(str(db_embedding_row.text))
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                node = TextNode(
                    id_=db_embedding_row.node_id,
                    text=db_embedding_row.text,
                    metadata=db_embedding_row.metadata,
                )
            similarities.append(db_embedding_row.similarity)
            ids.append(db_embedding_row.node_id)
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def _query_with_score(
        self,
        embedding: list[float] | None,
        limit: int = 10,
        metadata_filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> list[DBEmbeddingRow]:
        # TODO: filters, see <https://alexgarcia.xyz/sqlite-vec/features/vec0.html>
        if not embedding:
            raise ValueError("Embedding is required for vector search.")
        if not self.db:
            raise sqlite3.ProgrammingError("DB not initialized.")

        res = self.db.execute(
            f"""
            SELECT
                node_id,
                text,
                metadata_,
                distance
            FROM {self.table_name}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT {limit}
            """,
            [serialize_f32(embedding)],
        )
        return [
            DBEmbeddingRow(
                node_id=item[0],
                text=item[1],
                metadata=json.loads(item[2]),
                similarity=(1 - item[3]) if item[3] is not None else 0,
            )
            for item in res.fetchall()
        ]

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Asynchronously query vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call query synchronously.
        """
        return self.query(query, **kwargs)

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        return None


def serialize_f32(vector: list[float]) -> bytes:
    """Serializes a list of floats into a compact "raw bytes" format."""
    return struct.pack(f"{len(vector)}f", *vector)
