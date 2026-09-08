#!/usr/bin/env python3
"""
VoyageAI by MongoDB Contextual Embeddings (voyage-context-4) + Apache Cassandra Vector Search

This example demonstrates REAL contextual retrieval using VoyageAI by MongoDB's voyage-context-4:
1. Embedding full documents with server-side auto chunking (enable_auto_chunking)
2. Comparing retrieval accuracy: with vs without context
3. Storing contextual embeddings in Cassandra
4. Implementing RAG (Retrieval-Augmented Generation) with contextual embeddings

Prerequisites:
- Python 3.8+
- pip install voyageai cassandra-driver
- VoyageAI by MongoDB API key (set as VOYAGE_API_KEY environment variable)
- Apache Cassandra 5.0+ with vector search support

Key Features of voyage-context-4:
- Encodes both chunk-level details and global document context
- Improved retrieval accuracy over standard embeddings
- Seamless drop-in replacement for existing RAG pipelines
- Server-side auto chunking: pass full documents as a flat list[str]
- Supports documents up to 120K tokens total
- Available dimensions: 256, 512, 1024 (default), 2048

Author: Apache Cassandra Documentation Team
License: Apache 2.0
"""

import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

try:
    import voyageai
    from cassandra.cluster import Cluster, Session
    from cassandra.auth import PlainTextAuthProvider
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("Install dependencies: pip install voyageai cassandra-driver")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for contextual vector search."""

    # VoyageAI by MongoDB settings
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    CONTEXTUAL_MODEL = "voyage-context-4"
    EMBEDDING_DIMENSION = 1024  # Options: 256, 512, 1024, 2048
    AUTO_CHUNK_SIZE = 32000  # Target chunk size in tokens for auto chunking (max 32000)

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

# Simulating a long technical document split into chunks
# Each document has multiple chunks that need context

SAMPLE_DOCUMENTS = [
    {
        "doc_id": "apache-cassandra-architecture",
        "title": "Apache Cassandra Architecture Overview",
        "chunks": [
            "Apache Cassandra is a distributed NoSQL database designed for handling large amounts of data across multiple nodes. "
            "It provides high availability with no single point of failure.",

            "The ring architecture in Cassandra distributes data across nodes using consistent hashing. "
            "Each node is responsible for a range of tokens on the ring.",

            "Cassandra uses a peer-to-peer architecture where all nodes are equal. "
            "There are no master-slave relationships, eliminating single points of failure.",

            "Replication in Cassandra is configurable per keyspace. The replication factor determines "
            "how many copies of data are stored across the cluster for fault tolerance."
        ]
    },
    {
        "doc_id": "vector-search-guide",
        "title": "Vector Search Implementation Guide",
        "chunks": [
            "Vector search enables semantic similarity queries by representing data as high-dimensional vectors. "
            "These vectors capture semantic meaning rather than just keyword matches.",

            "Storage Attached Indexes (SAI) in Cassandra 5.0+ provide native vector search capabilities. "
            "SAI indexes support approximate nearest neighbor (ANN) search with configurable similarity functions.",

            "Similarity functions available in Cassandra include COSINE, DOT_PRODUCT, and EUCLIDEAN. "
            "COSINE similarity is recommended for normalized embeddings from most modern embedding models.",

            "The ANN search query syntax uses 'ORDER BY vector_column ANN OF [query_vector]'. "
            "This performs fast approximate nearest neighbor search without scanning all rows."
        ]
    },
    {
        "doc_id": "embedding-best-practices",
        "title": "Embedding Generation Best Practices",
        "chunks": [
            "When generating embeddings for documents, use input_type='document' to optimize for storage. "
            "For search queries, use input_type='query' to optimize for retrieval performance.",

            "Chunk size significantly impacts retrieval quality. Chunks should be large enough to contain "
            "meaningful context but small enough to match specific queries. Typical sizes range from 200-500 tokens.",

            "Contextual embeddings improve retrieval by encoding both local chunk details and global document context. "
            "This helps disambiguate chunks that might be unclear when isolated from their document.",

            "Batch processing embeddings reduces API latency and cost. Process multiple chunks in a single API call "
            "when possible, respecting the model's batch size limits."
        ]
    }
]


# ============================================================================
# VoyageAI by MongoDB Contextual Embedder
# ============================================================================

class VoyageContextualEmbedder:
    """
    Handles contextual embedding generation using VoyageAI by MongoDB's voyage-context-4.

    This model embeds chunks while encoding context from other chunks in the same document,
    improving retrieval accuracy compared to isolated chunk embeddings.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-context-4",
        dimension: int = 1024,
        chunk_size: int = 32000
    ):
        """
        Initialize VoyageAI by MongoDB contextual client.

        Args:
            api_key: VoyageAI by MongoDB API key
            model: Model name (voyage-context-4)
            dimension: Output dimension (256, 512, 1024, 2048)
            chunk_size: Target chunk size in tokens for auto chunking (max 32000)
        """
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dimension = dimension
        self.chunk_size = chunk_size
        print(f"✓ VoyageAI by MongoDB contextual client initialized")
        print(f"  Model: {model}")
        print(f"  Dimension: {dimension}")
        print(f"  Feature: Contextual chunk embeddings")

    def embed_document_with_context(
        self,
        document: str,
        input_type: str = "document"
    ) -> List[tuple]:
        """
        Embed a full document with context using voyage-context-4 auto chunking.

        The full document is passed as a flat list[str] and Voyage splits it
        server-side (enable_auto_chunking), encoding global document context into
        each chunk's embedding. The backend-generated chunk texts are returned
        alongside their embeddings.

        Args:
            document: Full document text (a single string)
            input_type: "document" or "query"

        Returns:
            List of (chunk_text, embedding) tuples, one per auto-generated chunk
        """
        # Pass the full document as a flat list[str]; Voyage auto-chunks it
        result = self.client.contextualized_embed(
            inputs=[document],              # flat list[str] of full-document strings
            model=self.model,
            input_type=input_type,
            output_dimension=self.dimension,
            enable_auto_chunking=True,      # split each document server-side
            chunk_size=self.chunk_size      # target chunk size in tokens (max 32000)
        )

        # Auto chunking returns the generated chunk texts + their embeddings
        doc_result = result.results[0]
        return list(zip(doc_result.chunk_texts, doc_result.embeddings))

    def embed_document_chunks_without_context(
        self,
        chunks: List[str],
        input_type: str = "document"
    ) -> List[List[float]]:
        """
        Embed document chunks WITHOUT context (using standard embed API).

        This is the baseline approach where each chunk is embedded independently
        without knowledge of surrounding chunks.

        Args:
            chunks: List of text chunks from a single document
            input_type: "document" or "query"

        Returns:
            List of standard embeddings, one per chunk
        """
        # Use standard embed API - each chunk is independent
        result = self.client.embed(
            texts=chunks,
            model="voyage-4",  # Use voyage-4 for fair comparison
            input_type=input_type,
            output_dimension=self.dimension
        )

        return result.embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        # Queries are embedded as a flat list[str] (auto chunking not needed)
        result = self.client.contextualized_embed(
            inputs=[query],  # flat list[str] with a single query
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
            query,
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

        rows = self.session.execute(query, (query_vector, query_vector, limit))

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
    print("VoyageAI by MongoDB Contextual Embeddings (voyage-context-4) + Cassandra")
    print("="*80 + "\n")

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Initialize components
    print("1. Initializing VoyageAI by MongoDB contextual embedder...")
    print("-" * 80)

    embedder = VoyageContextualEmbedder(
        api_key=Config.VOYAGE_API_KEY,
        model=Config.CONTEXTUAL_MODEL,
        dimension=Config.EMBEDDING_DIMENSION,
        chunk_size=Config.AUTO_CHUNK_SIZE
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

        for doc in SAMPLE_DOCUMENTS:
            doc_id = doc["doc_id"]
            doc_title = doc["title"]
            # Pass the full document; voyage-context-4 auto-chunks it server-side
            full_text = "\n\n".join(doc["chunks"])

            print(f"\nProcessing: {doc_title}")

            # Generate CONTEXTUAL embeddings via auto chunking
            contextual_pairs = embedder.embed_document_with_context(full_text)
            chunk_texts = [text for text, _ in contextual_pairs]
            contextual_embeddings = [emb for _, emb in contextual_pairs]
            print(f"  ✓ Auto-chunked into {len(chunk_texts)} contextual embeddings")

            # Generate STANDARD embeddings on the same chunks (for comparison)
            standard_embeddings = embedder.embed_document_chunks_without_context(chunk_texts)
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

            # Generate query embedding
            query_vector = embedder.embed_query(query_text)

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
                query_vector=query_vector,
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
        print("✓ Real VoyageAI by MongoDB voyage-context-4 integration")
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
        print("- Pass full documents as a flat list[str] and let Voyage auto-chunk")
        print("- Use enable_auto_chunking=True with chunk_size up to 32000 tokens")
        print("- Store the returned chunk_texts alongside their embeddings")
        print("- Use input_type='document' for documents, 'query' for searches")

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
