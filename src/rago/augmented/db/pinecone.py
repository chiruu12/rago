"""Module for pinecone database."""

from __future__ import annotations

from typing import Any, Iterable

import pinecone

from typeguard import typechecked

from rago.augmented.db.base import DBBase


@typechecked
class PineconeDB(DBBase):
    """Pinecone Vector Database."""

    def __init__(
        self, api_key: str, environment: str, index_name: str, dimension: int
    ):
        """Initialize the Pinecone database."""
        # TO DO: Enable users to create indexes with configurable metrics
        # like cosine,L2 and support for various vector store types
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name, dimension=dimension, metric='cosine'
            )
        self.index = pinecone.Index(index_name)

    def embed(self, documents: Any) -> None:
        """Embed the documents into the database."""
        self.index.upsert(vectors=documents)

    def search(
        self, query_vector: list[float], top_k: int = 2
    ) -> tuple[Iterable[float], Iterable[int]]:
        """Search an encoded query in the vector database."""
        results = self.index.query(
            vector=query_vector, top_k=top_k, include_values=False
        )
        distances = [match['score'] for match in results['matches']]
        ids = [int(match['id']) for match in results['matches']]
        return distances, ids
