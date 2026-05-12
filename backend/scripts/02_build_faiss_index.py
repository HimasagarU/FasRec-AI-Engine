"""
Script 02: Build FAISS HNSW indexes for text and image embeddings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import load_embeddings
from src.faiss_index import build_hnsw_index, save_index


def main():
    print("=== Building FAISS Indexes ===\n")

    text_emb, image_emb, product_ids = load_embeddings()

    print(f"Text embeddings: {text_emb.shape}")
    print(f"Image embeddings: {image_emb.shape}")

    # Build text index
    text_index = build_hnsw_index(text_emb, M=32, ef_construction=200)
    save_index(text_index, "text_index")

    # Build image index
    image_index = build_hnsw_index(image_emb, M=32, ef_construction=200)
    save_index(image_index, "image_index")

    print("\n=== Done! FAISS indexes built and saved. ===")


if __name__ == "__main__":
    main()
