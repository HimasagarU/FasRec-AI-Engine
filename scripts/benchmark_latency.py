import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.faiss_index import load_index
from src.embeddings import load_embeddings

def benchmark():
    print("Loading embeddings and index...")
    text_emb, image_emb, _ = load_embeddings()
    text_index = load_index("text_index")
    # For a realistic single query request, we'll search top 200 candidates
    
    n_queries = 1000
    np.random.seed(42)
    # Pick random queries from the dataset
    indices = np.random.choice(len(text_emb), size=n_queries, replace=False)
    queries = text_emb[indices]
    
    latencies = []
    
    # Warmup
    for i in range(10):
        q = np.expand_dims(queries[i], axis=0)
        text_index.search(q, 200)
        
    print(f"Benchmarking {n_queries} single text queries...")
    for i in range(n_queries):
        q = np.expand_dims(queries[i], axis=0)
        start = time.perf_counter()
        text_index.search(q, 200)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
        
    latencies = np.array(latencies)
    print("--- Benchmark Results ---")
    print(f"Mean Latency: {np.mean(latencies):.2f} ms")
    print(f"P50 Latency:  {np.percentile(latencies, 50):.2f} ms")
    print(f"P95 Latency:  {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 Latency:  {np.percentile(latencies, 99):.2f} ms")

if __name__ == "__main__":
    benchmark()
