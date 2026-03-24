"""
pipeline/embedder.py

Builds a FAISS vector index for Retrieval-Augmented Generation (RAG).

Why FAISS instead of a cloud vector DB?
- Runs fully air-gapped (critical for data-classified environments like aviation)
- Sub-millisecond search on millions of vectors on a single machine
- No data leaves the premises — essential for ITAR/EASA/FAA controlled documents

Architecture:
  Document chunks → SentenceTransformer (MiniLM-L6-v2) → 384-dim embeddings → FAISS IVFFlat index
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# We use lazy imports so the file is importable even without these packages
def _get_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError("Run: pip install faiss-cpu")


def _get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")


# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------

class AeroDocEmbedder:
    """
    Wraps SentenceTransformers for consistent embedding generation.

    Model choice: all-MiniLM-L6-v2
    - 22M parameters, fast inference
    - 384-dimensional embeddings (good balance of quality vs memory)
    - Strong performance on semantic similarity tasks
    - Runs entirely locally — no API calls
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        SentenceTransformer = _get_sentence_transformer()
        logger.info(f"Loading embedding model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of texts.
        Returns: np.ndarray of shape (N, embedding_dim), dtype float32
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, embedding_dim)."""
        return self.embed([query], show_progress=False)


# ---------------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """
    Wraps a FAISS index with chunk metadata for document retrieval.

    Index type: IndexFlatIP (exact inner product search)
    - With L2-normalized embeddings, inner product == cosine similarity
    - Exact search is fine for <100k documents
    - For millions of docs, switch to IndexIVFFlat (approximate, ~10x faster)

    Storage: index saved as .faiss file, metadata as .pkl
    """

    def __init__(self, embedding_dim: int = 384):
        faiss = _get_faiss()
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Dict] = []  # parallel array to the FAISS index

    def add(self, embeddings: np.ndarray, chunks: List[Dict]):
        """Add embeddings and their associated chunk metadata."""
        assert embeddings.shape[0] == len(chunks), "Mismatch between embeddings and chunks"
        assert embeddings.shape[1] == self.embedding_dim, \
            f"Expected {self.embedding_dim}-dim embeddings, got {embeddings.shape[1]}"

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"Index now contains {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most similar chunks.

        Returns list of dicts with chunk metadata + similarity score.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty!")
            return []

        # query_embedding shape: (1, embedding_dim)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            chunk = self.chunks[idx].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        return results

    def save(self, output_dir: str):
        faiss = _get_faiss()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{output_dir}/vectors.faiss")
        with open(f"{output_dir}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Vector store saved → {output_dir} ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, index_dir: str) -> "FAISSVectorStore":
        faiss = _get_faiss()
        store = cls.__new__(cls)
        store.index = faiss.read_index(f"{index_dir}/vectors.faiss")
        store.embedding_dim = store.index.d
        with open(f"{index_dir}/chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        logger.info(f"Loaded vector store: {store.index.ntotal} vectors, dim={store.embedding_dim}")
        return store


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

class IndexBuilder:
    """
    Builds a FAISS index from a processed JSONL chunks file.

    Usage:
        builder = IndexBuilder()
        builder.build("data/processed/chunks.jsonl", "data/vector_store")
    """

    def __init__(self):
        self.embedder = AeroDocEmbedder()
        self.store = FAISSVectorStore(embedding_dim=self.embedder.embedding_dim)

    def build(self, chunks_path: str, output_dir: str, batch_size: int = 128):
        chunks = []
        texts = []

        logger.info(f"Loading chunks from {chunks_path}...")
        with open(chunks_path) as f:
            for line in f:
                row = json.loads(line.strip())
                chunks.append(row)
                texts.append(row["text"])

        logger.info(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
        embeddings = self.embedder.embed(texts, batch_size=batch_size)

        self.store.add(embeddings, chunks)
        self.store.save(output_dir)

        logger.info("Index build complete.")
        return self.store


# ---------------------------------------------------------------------------
# RAG retriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    Retrieves relevant document chunks for a given query.

    This is the retrieval half of RAG. In a full RAG pipeline, the retrieved
    chunks would be passed as context to a generative model (e.g. local LLaMA).
    Here we demonstrate the retrieval layer independently.
    """

    def __init__(self, index_dir: str):
        self.embedder = AeroDocEmbedder()
        self.store = FAISSVectorStore.load(index_dir)

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: Natural language query
            top_k: Maximum number of results to return
            min_score: Minimum cosine similarity threshold (0-1)

        Returns:
            List of relevant chunk dicts with similarity scores
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.store.search(query_embedding, top_k=top_k)

        # Post-process: filter by minimum score
        results = [r for r in results if r["similarity_score"] >= min_score]

        logger.debug(f"Query: '{query[:50]}...' → {len(results)} results above threshold")
        return results

    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved chunks into a context string for a generative model."""
        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {i} | Label: {result.get('label', 'unknown')} | "
                f"Score: {result['similarity_score']:.3f}]\n{result['text']}"
            )
        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Quick test with mock data
    print("Testing embedder with sample text...")
    try:
        embedder = AeroDocEmbedder()
        texts = [
            "Aircraft underwent A-check maintenance. MEL item cleared.",
            "Engine flame-out at FL350. Emergency descent checklist executed.",
            "P/N 114A1234-001 elevator actuator. Lead time 8 weeks.",
        ]
        embeddings = embedder.embed(texts, show_progress=False)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"All L2 norms ≈ 1.0: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)}")
    except ImportError as e:
        print(f"Skipping test: {e}")
