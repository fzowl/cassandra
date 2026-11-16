#!/usr/bin/env python3
"""
Voyage AI by MongoDB Contextual Embeddings (voyage-context-4) + Apache Cassandra Vector Search

This example demonstrates REAL contextual retrieval using Voyage AI by MongoDB's
voyage-context-4:
1. Embedding whole documents with auto-chunking, so every chunk is embedded with the
   context of its document
2. Comparing retrieval accuracy: with vs without context
3. Storing contextual embeddings in Cassandra
4. Implementing RAG (Retrieval-Augmented Generation) with contextual embeddings

Note: Voyage AI is now part of MongoDB. The API, `voyageai` Python package, and
model names are unchanged.

Prerequisites:
- Python 3.9+
- pip install 'voyageai>=0.4.1' cassandra-driver
- Voyage AI by MongoDB API key (set as VOYAGE_API_KEY environment variable)
- Apache Cassandra 5.0+ with vector search support

Key Features of voyage-context-4:
- Encodes both chunk-level details and global document context
- Improved retrieval accuracy over standard embeddings
- Seamless drop-in replacement for existing RAG pipelines
- Supports up to 120K tokens in total per request; without auto-chunking each
  document must fit in the 32K-token context window
- Available dimensions: 256, 512, 1024 (default), 2048

How this example calls contextualized_embed:
- Documents (input_type="document"): a flat List[str], one whole document per
  string, with enable_auto_chunking=True and chunk_size=32_000. The service
  chunks each document and returns the chunk texts (result.chunk_texts) next to
  the embeddings; the example stores those texts. A pre-chunked List[List[str]]
  is rejected by the API when auto-chunking is on, so the example raises an
  error for it instead of falling back silently.
- Queries (input_type="query"): no auto-chunking; the query strings are passed
  as a flat List[str].
Reference: https://docs.voyageai.com/docs/contextualized-chunk-embeddings

Author: Apache Cassandra Documentation Team
License: Apache 2.0
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

try:
    import voyageai
    from cassandra.cluster import Cluster, Session
    from cassandra.auth import PlainTextAuthProvider
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("Install dependencies: pip install 'voyageai>=0.4.1' cassandra-driver")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for contextual vector search."""

    # Voyage AI by MongoDB settings
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    CONTEXTUAL_MODEL = "voyage-context-4"
    EMBEDDING_DIMENSION = 1024  # Options: 256, 512, 1024, 2048
    CHUNK_SIZE = 32_000  # chunk_size for auto-chunking

    # Cassandra settings
    CASSANDRA_HOSTS = os.getenv("CASSANDRA_HOSTS", "127.0.0.1").split(",")
    CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
    CASSANDRA_KEYSPACE = "contextual_search"
    CASSANDRA_USERNAME = os.getenv("CASSANDRA_USERNAME")
    CASSANDRA_PASSWORD = os.getenv("CASSANDRA_PASSWORD")

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.VOYAGE_API_KEY:
            raise ValueError(
                "VOYAGE_API_KEY environment variable is required.\n"
                "Get your API key from: https://dash.voyageai.com/api-keys\n"
                "Set it with: export VOYAGE_API_KEY='your-api-key-here'"
            )


# ============================================================================
# Sample Data: Technical Documentation with Context
# ============================================================================

# Whole technical documents. The service splits each one into chunks (auto-chunking).
# With chunk_size=32_000 a document is split only when it is longer than that,
# so these short samples come back as one chunk each; real documents get several.

SAMPLE_DOCUMENTS = [
    {
        "doc_id": "apache-cassandra-architecture",
        "title": "Apache Cassandra Architecture Overview",
        "text": (
            "Apache Cassandra is a distributed NoSQL database designed for handling large amounts of data across multiple nodes. It provides high availability with no single point of failure.\n\n"
            "The ring architecture in Cassandra distributes data across nodes using consistent hashing. Each node is responsible for a range of tokens on the ring.\n\n"
            "Cassandra uses a peer-to-peer architecture where all nodes are equal. There are no master-slave relationships, eliminating single points of failure.\n\n"
            "Replication in Cassandra is configurable per keyspace. The replication factor determines how many copies of data are stored across the cluster for fault tolerance."
        ),
    },
    {
        "doc_id": "vector-search-guide",
        "title": "Vector Search Implementation Guide",
        "text": (
            "Vector search enables semantic similarity queries by representing data as high-dimensional vectors. These vectors capture semantic meaning rather than just keyword matches.\n\n"
            "Storage Attached Indexes (SAI) in Cassandra 5.0+ provide native vector search capabilities. SAI indexes support approximate nearest neighbor (ANN) search with configurable similarity functions.\n\n"
            "Similarity functions available in Cassandra include COSINE, DOT_PRODUCT, and EUCLIDEAN. COSINE similarity is recommended for normalized embeddings from most modern embedding models.\n\n"
            "The ANN search query syntax uses 'ORDER BY vector_column ANN OF [query_vector]'. This performs fast approximate nearest neighbor search without scanning all rows."
        ),
    },
    {
        "doc_id": "embedding-best-practices",
        "title": "Embedding Generation Best Practices",
        "text": (
            "When generating embeddings for documents, use input_type='document' to optimize for storage. For search queries, use input_type='query' to optimize for retrieval performance.\n\n"
            "Chunk size significantly impacts retrieval quality. Chunks should be large enough to contain meaningful context but small enough to match specific queries. Typical sizes range from 200-500 tokens.\n\n"
            "Contextual embeddings improve retrieval by encoding both local chunk details and global document context. This helps disambiguate chunks that might be unclear when isolated from their document.\n\n"
            "Batch processing embeddings reduces API latency and cost. Process multiple chunks in a single API call when possible, respecting the model's batch size limits."
        ),
    },
]


# ============================================================================
# Voyage AI by MongoDB Contextual Embedder
# ============================================================================

class VoyageContextualEmbedder:
    """
    Handles contextual embedding generation using Voyage AI by MongoDB's voyage-context-4.

    This model embeds chunks while encoding context from other chunks in the same document,
    improving retrieval accuracy compared to isolated chunk embeddings.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-context-4",
        dimension: int = 1024
    ):
        """
        Initialize Voyage AI by MongoDB contextual client.

        Args:
            api_key: Voyage AI by MongoDB API key
            model: Model name (voyage-context-4)
            dimension: Output dimension (256, 512, 1024, 2048)
        """
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dimension = dimension
        print(f"✓ Voyage AI by MongoDB contextual client initialized")
        print(f"  Model: {model}")
        print(f"  Dimension: {dimension}")
        print(f"  Feature: Contextual chunk embeddings")

    def embed_documents_with_context(
        self,
        documents: List[str]
    ) -> List[Tuple[List[str], List[List[float]]]]:
        """
        Chunk whole documents and embed every chunk with its document's context.

        The call always uses enable_auto_chunking=True and chunk_size=32_000,
        which the API accepts only with input_type="document" and one string
        per document.

        Args:
            documents: Whole document texts, one string per document

        Returns:
            One (chunk_texts, embeddings) pair per document, in input order.
            chunk_texts are the chunks chosen by the service; embeddings[i]
            belongs to chunk_texts[i].

        Raises:
            ValueError: if documents is not a flat list of strings. A
                pre-chunked List[List[str]] is rejected by the API when
                auto-chunking is enabled, so it is an error here, not a
                silent fallback.
        """
        if not documents:
            raise ValueError("documents must not be empty")
        for i, document in enumerate(documents):
            if not isinstance(document, str):
                raise ValueError(
                    f"documents[{i}] is {type(document).__name__}, expected str: "
                    "auto-chunking expects one string per document; "
                    "pre-chunked List[List[str]] input is not supported"
                )

        result = self.client.contextualized_embed(
            inputs=documents,  # List[str] - one whole document per string
            model=self.model,
            input_type="document",
            output_dimension=self.dimension,
            enable_auto_chunking=True,
            chunk_size=Config.CHUNK_SIZE
        )

        return [(r.chunk_texts, r.embeddings) for r in result.results]

    def embed_chunks_without_context(self, chunk_texts: List[str]) -> List[List[float]]:
        """
        Embed chunk texts WITHOUT context (using standard embed API).

        This is the baseline: the same chunk texts that the contextual call
        returned, each embedded independently.

        Args:
            chunk_texts: Chunk texts of one document, as returned by
                embed_documents_with_context

        Returns:
            List of standard embeddings, one per chunk
        """
        # Use standard embed API - each chunk is independent
        result = self.client.embed(
            texts=chunk_texts,
            model="voyage-4",  # Use voyage-4 for fair comparison
            input_type="document",
            output_dimension=self.dimension
        )

        return result.embeddings

    def embed_query_without_context(self, query: str) -> List[float]:
        """
        Embed a search query with the standard embed API, for the baseline table.

        The baseline chunks are embedded with voyage-4, so the baseline query
        must come from the same model; vectors of different models are not
        comparable.
        """
        result = self.client.embed(
            texts=[query],
            model="voyage-4",
            input_type="query",
            output_dimension=self.dimension
        )

        return result.embeddings[0]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        # Queries are never auto-chunked: a flat List[str] with
        # input_type="query" and no enable_auto_chunking.
        result = self.client.contextualized_embed(
            inputs=[query],  # List[str] - one query string
            model=self.model,
            input_type="query",
            output_dimension=self.dimension
        )

        return result.results[0].embeddings[0]


# ============================================================================
# Cassandra Vector Store
# ============================================================================

class ContextualVectorStore:
    """Handles contextual vector storage and search in Cassandra."""

    def __init__(
        self,
        hosts: List[str],
        port: int = 9042,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """Initialize Cassandra connection."""
        auth_provider = None
        if username and password:
            auth_provider = PlainTextAuthProvider(username=username, password=password)

        self.cluster = Cluster(
            contact_points=hosts,
            port=port,
            auth_provider=auth_provider
        )
        self.session: Optional[Session] = None
        print(f"✓ Cassandra cluster initialized (hosts: {', '.join(hosts)})")

    def connect(self):
        """Establish connection to Cassandra cluster."""
        try:
            self.session = self.cluster.connect()
            print("✓ Connected to Cassandra cluster")
        except Exception as e:
            print(f"Error connecting to Cassandra: {e}")
            raise

    def close(self):
        """Close Cassandra connection."""
        if self.cluster:
            self.cluster.shutdown()
            print("✓ Cassandra connection closed")

    def setup_schema(
        self,
        keyspace: str,
        dimension: int,
        replication_factor: int = 1
    ):
        """
        Create schema for contextual document storage.

        Creates two tables for comparison:
        - document_chunks_contextual: Uses contextual embeddings
        - document_chunks_standard: Uses standard embeddings (baseline)

        Args:
            keyspace: Keyspace name
            dimension: Dimension of embeddings
            replication_factor: Replication factor
        """
        # Create keyspace
        query = f"""
        CREATE KEYSPACE IF NOT EXISTS {keyspace}
        WITH REPLICATION = {{
            'class': 'SimpleStrategy',
            'replication_factor': {replication_factor}
        }}
        """
        self.session.execute(query)
        print(f"✓ Keyspace '{keyspace}' created")

        self.session.set_keyspace(keyspace)

        # Table for contextual embeddings
        query = f"""
        CREATE TABLE IF NOT EXISTS document_chunks_contextual (
            chunk_id UUID PRIMARY KEY,
            doc_id TEXT,
            doc_title TEXT,
            chunk_text TEXT,
            chunk_index INT,
            embedding VECTOR<FLOAT, {dimension}>,
            created_at TIMESTAMP
        )
        """
        self.session.execute(query)
        print(f"✓ Table 'document_chunks_contextual' created")

        # Table for standard embeddings (baseline comparison)
        query = f"""
        CREATE TABLE IF NOT EXISTS document_chunks_standard (
            chunk_id UUID PRIMARY KEY,
            doc_id TEXT,
            doc_title TEXT,
            chunk_text TEXT,
            chunk_index INT,
            embedding VECTOR<FLOAT, {dimension}>,
            created_at TIMESTAMP
        )
        """
        self.session.execute(query)
        print(f"✓ Table 'document_chunks_standard' created")

        # Create SAI indexes for both tables
        for table_name in ["document_chunks_contextual", "document_chunks_standard"]:
            query = f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {table_name}_vector_idx
            ON {table_name}(embedding)
            USING 'StorageAttachedIndex'
            WITH OPTIONS = {{'similarity_function': 'COSINE'}}
            """
            self.session.execute(query)
            print(f"✓ SAI vector index created on {table_name}")

    def insert_chunk(
        self,
        keyspace: str,
        table_name: str,
        doc_id: str,
        doc_title: str,
        chunk_text: str,
        chunk_index: int,
        embedding: List[float]
    ):
        """Insert a document chunk with its embedding."""
        self.session.set_keyspace(keyspace)

        query = f"""
        INSERT INTO {table_name} (
            chunk_id, doc_id, doc_title, chunk_text, chunk_index,
            embedding, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        self.session.execute(
            self.session.prepare(query),
            (
                uuid.uuid4(),
                doc_id,
                doc_title,
                chunk_text,
                chunk_index,
                embedding,
                datetime.utcnow()
            )
        )

    def search_similar_chunks(
        self,
        keyspace: str,
        table_name: str,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks.

        Args:
            keyspace: Keyspace name
            table_name: Table to search (contextual or standard)
            query_vector: Query embedding
            limit: Maximum results

        Returns:
            List of matching chunks with similarity scores
        """
        self.session.set_keyspace(keyspace)

        query = f"""
        SELECT
            chunk_id, doc_id, doc_title, chunk_text, chunk_index,
            similarity_cosine(embedding, ?) AS similarity
        FROM {table_name}
        ORDER BY embedding ANN OF ?
        LIMIT ?
        """

        rows = self.session.execute(
            self.session.prepare(query), (query_vector, query_vector, limit)
        )

        results = []
        for row in rows:
            results.append({
                "chunk_id": str(row.chunk_id),
                "doc_id": row.doc_id,
                "doc_title": row.doc_title,
                "chunk_text": row.chunk_text,
                "chunk_index": row.chunk_index,
                "similarity": float(row.similarity) if row.similarity else None
            })

        return results


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application demonstrating contextual embeddings."""

    print("\n" + "="*80)
    print("Voyage AI by MongoDB Contextual Embeddings (voyage-context-4) + Cassandra")
    print("="*80 + "\n")

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Initialize components
    print("1. Initializing Voyage AI by MongoDB contextual embedder...")
    print("-" * 80)

    embedder = VoyageContextualEmbedder(
        api_key=Config.VOYAGE_API_KEY,
        model=Config.CONTEXTUAL_MODEL,
        dimension=Config.EMBEDDING_DIMENSION
    )

    vector_store = ContextualVectorStore(
        hosts=Config.CASSANDRA_HOSTS,
        port=Config.CASSANDRA_PORT,
        username=Config.CASSANDRA_USERNAME,
        password=Config.CASSANDRA_PASSWORD
    )

    try:
        vector_store.connect()

        # Setup schema
        print("\n2. Setting up Cassandra schema...")
        print("-" * 80)

        vector_store.setup_schema(
            keyspace=Config.CASSANDRA_KEYSPACE,
            dimension=Config.EMBEDDING_DIMENSION,
            replication_factor=1
        )

        # Process documents with both contextual and standard embeddings
        print("\n3. Generating embeddings (contextual vs standard)...")
        print("-" * 80)

        documents = [doc["text"] for doc in SAMPLE_DOCUMENTS]
        chunked = embedder.embed_documents_with_context(documents)

        for doc, (chunk_texts, contextual_embeddings) in zip(SAMPLE_DOCUMENTS, chunked):
            doc_id = doc["doc_id"]
            doc_title = doc["title"]

            print(f"\nProcessing: {doc_title}")
            print(f"  Chunks (from auto-chunking): {len(chunk_texts)}")
            print(f"  ✓ Generated {len(contextual_embeddings)} contextual embeddings")

            # Generate STANDARD embeddings of the same chunk texts (for comparison)
            standard_embeddings = embedder.embed_chunks_without_context(chunk_texts)
            print(f"  ✓ Generated {len(standard_embeddings)} standard embeddings")

            # Store contextual embeddings
            for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, contextual_embeddings)):
                vector_store.insert_chunk(
                    keyspace=Config.CASSANDRA_KEYSPACE,
                    table_name="document_chunks_contextual",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    chunk_text=chunk_text,
                    chunk_index=i,
                    embedding=embedding
                )

            # Store standard embeddings
            for i, (chunk_text, embedding) in enumerate(zip(chunk_texts, standard_embeddings)):
                vector_store.insert_chunk(
                    keyspace=Config.CASSANDRA_KEYSPACE,
                    table_name="document_chunks_standard",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    chunk_text=chunk_text,
                    chunk_index=i,
                    embedding=embedding
                )

        print(f"\n✓ All documents processed and stored")

        # Perform comparison searches
        print("\n4. Comparing retrieval: Contextual vs Standard embeddings...")
        print("-" * 80)

        test_queries = [
            "How does Cassandra distribute data across nodes?",
            "What similarity functions are available for vector search?",
            "What is the recommended chunk size for embeddings?"
        ]

        for query_text in test_queries:
            print(f"\nQuery: \"{query_text}\"")
            print("=" * 70)

            # Generate query embeddings (one per model: contextual and standard)
            query_vector = embedder.embed_query(query_text)
            standard_query_vector = embedder.embed_query_without_context(query_text)

            # Search with CONTEXTUAL embeddings
            print("\n[CONTEXTUAL EMBEDDINGS]")
            print("-" * 40)
            contextual_results = vector_store.search_similar_chunks(
                keyspace=Config.CASSANDRA_KEYSPACE,
                table_name="document_chunks_contextual",
                query_vector=query_vector,
                limit=3
            )

            for i, result in enumerate(contextual_results, 1):
                print(f"{i}. {result['doc_title']} (chunk {result['chunk_index']})")
                print(f"   Similarity: {result['similarity']:.4f}")
                print(f"   Text: {result['chunk_text'][:100]}...")
                print()

            # Search with STANDARD embeddings
            print("[STANDARD EMBEDDINGS - Baseline]")
            print("-" * 40)
            standard_results = vector_store.search_similar_chunks(
                keyspace=Config.CASSANDRA_KEYSPACE,
                table_name="document_chunks_standard",
                query_vector=standard_query_vector,
                limit=3
            )

            for i, result in enumerate(standard_results, 1):
                print(f"{i}. {result['doc_title']} (chunk {result['chunk_index']})")
                print(f"   Similarity: {result['similarity']:.4f}")
                print(f"   Text: {result['chunk_text'][:100]}...")
                print()

        print("\n" + "="*80)
        print("SUCCESS: Contextual embeddings demonstration complete!")
        print("="*80)

        print("\nKey Features Demonstrated:")
        print("✓ Real Voyage AI by MongoDB voyage-context-4 integration")
        print("✓ Contextual chunk embeddings with global document context")
        print("✓ Side-by-side comparison with standard embeddings")
        print("✓ Improved retrieval accuracy for ambiguous chunks")
        print("✓ Drop-in replacement for existing RAG pipelines")

        print("\nWhen to Use Contextual Embeddings:")
        print("- Long documents split into chunks (technical docs, books)")
        print("- Chunks that need surrounding context for disambiguation")
        print("- Improved precision for RAG applications")
        print("- Knowledge bases with interconnected information")

        print("\nBest Practices:")
        print("- Pass whole documents as a flat list of strings (one string per document)")
        print("- Use enable_auto_chunking=True with chunk_size=32_000 and store the returned chunk_texts")
        print("- Auto-chunking works only with input_type='document'; embed queries with input_type='query'")
        print("- Do not pass pre-chunked List[List[str]] together with auto-chunking")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        vector_store.close()


if __name__ == "__main__":
    sys.exit(main())
