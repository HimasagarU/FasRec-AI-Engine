"""
FAISS HNSW index building and nearest-neighbor search.
Uses faiss-gpu if available, falls back to faiss-cpu.
"""
import numpy as np
import faiss
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _use_gpu_index(index: faiss.Index) -> faiss.Index:
    """Try to move FAISS index to GPU if faiss-gpu is available."""
    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        print("[FAISS] Using GPU index.")
        return gpu_index
    except Exception:
        print("[FAISS] GPU not available for FAISS, using CPU.")
        return index


def build_hnsw_index(
    embeddings: np.ndarray,
    M: int = 32,
    ef_construction: int = 200,
) -> faiss.Index:
    """
    Build a FAISS HNSW index.

    Args:
        embeddings: np.ndarray of shape (N, D), should be L2-normalized
        M: HNSW connections per node
        ef_construction: construction-time search depth

    Returns:
        faiss.IndexHNSWFlat
    """
    dim = embeddings.shape[1]
    # Use Inner Product (cosine sim for normalized vectors)
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = 128

    print(f"[FAISS] Building HNSW index: {embeddings.shape[0]} vectors, dim={dim}, M={M}...")
    index.add(embeddings)
    print(f"[FAISS] Index built. Total vectors: {index.ntotal}")
    return index


def search_index(
    index: faiss.Index,
    query: np.ndarray,
    top_k: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for nearest neighbors.

    Args:
        index: FAISS index
        query: query vector(s), shape (1, D) or (B, D)
        top_k: number of neighbors to retrieve

    Returns:
        (scores, indices) each of shape (B, top_k)
    """
    if query.ndim == 1:
        query = query.reshape(1, -1)
    scores, indices = index.search(query, top_k)
    return scores, indices


def batch_search_index(
    index: faiss.Index,
    queries: np.ndarray,
    top_k: int = 200,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch search the FAISS index for nearest neighbors.

    Args:
        index: FAISS index
        queries: query vectors, shape (N, D)
        top_k: number of neighbors to retrieve
        batch_size: batch size for search

    Returns:
        (scores, indices) each of shape (N, top_k)
    """
    all_scores = []
    all_indices = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i: i + batch_size]
        scores, indices = index.search(batch, top_k)
        all_scores.append(scores)
        all_indices.append(indices)

    return np.vstack(all_scores), np.vstack(all_indices)


def save_index(index: faiss.Index, name: str, output_dir: Path | None = None):
    """Save FAISS index to disk."""
    out = output_dir or ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.faiss"
    faiss.write_index(index, str(path))
    print(f"[FAISS] Saved index to {path}")


def load_index(name: str, input_dir: Path | None = None) -> faiss.Index:
    """Load FAISS index from disk."""
    d = input_dir or ARTIFACTS_DIR
    path = d / f"{name}.faiss"
    index = faiss.read_index(str(path))
    print(f"[FAISS] Loaded index from {path}, vectors: {index.ntotal}")
    return index
