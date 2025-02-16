import json
import sqlite3
import struct
from typing import Any, Optional, Sequence

import fsspec
import sqlite_vec
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pydantic.fields import PrivateAttr


class SQLiteVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    embed_dim: int
    connection_string: str
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        connection_string: str = ":memory:",
        embed_dim: int = 1536,
    ):
        super().__init__(embed_dim=embed_dim, connection_string=connection_string)  # pyright: ignore[reportCallIssue]

    def _initialize(self) -> None:
        if not self._is_initialized:
            self.db = sqlite3.connect(":memory:")
            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
            self.db.execute(f"""CREATE VIRTUAL TABLE vec_items USING vec0(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metadata_ TEXT,
                node_id TEXT,
                embedding float[{self.embed_dim}],
                +text TEXT
                )""")
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
        raise NotImplementedError("get_nodes not implemented")

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
