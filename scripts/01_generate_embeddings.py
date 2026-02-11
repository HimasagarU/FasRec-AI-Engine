"""
Script 01: Generate text and image embeddings.
Uses GPU for both SBERT and CLIP when available.
Supports resume: skips already-generated embeddings.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_products
from src.embeddings import (
    generate_text_embeddings,
    generate_image_embeddings,
    save_embeddings,
    DEVICE,
    ARTIFACTS_DIR,
)
import numpy as np


def main():
    print(f"=== Embedding Generation (device: {DEVICE}) ===\n")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load products
    df = load_products()

    text_emb_path = ARTIFACTS_DIR / "text_embeddings.npy"
    image_emb_path = ARTIFACTS_DIR / "image_embeddings.npy"
    ids_path = ARTIFACTS_DIR / "product_ids.npy"

    # Generate or load text embeddings
    if text_emb_path.exists():
        print(f"[Resume] Text embeddings already exist, loading from {text_emb_path}")
        text_emb = np.load(text_emb_path)
    else:
        texts = df["text_field"].tolist()
        text_emb = generate_text_embeddings(texts)
        np.save(text_emb_path, text_emb)
        print(f"[Saved] text_embeddings.npy ({text_emb.shape})")

    # Generate or load image embeddings
    if image_emb_path.exists():
        print(f"[Resume] Image embeddings already exist, loading from {image_emb_path}")
        image_emb = np.load(image_emb_path)
    else:
        image_paths = df["image_path"].tolist()
        image_emb = generate_image_embeddings(image_paths)
        np.save(image_emb_path, image_emb)
        print(f"[Saved] image_embeddings.npy ({image_emb.shape})")

    # Ensure alignment - truncate to smaller set
    n = min(len(text_emb), len(image_emb), len(df))
    text_emb = text_emb[:n]
    image_emb = image_emb[:n]
    product_ids = df["id"].values[:n]

    # Save final aligned versions
    save_embeddings(text_emb, image_emb, product_ids)

    print(f"\n=== Done! {n} products embedded. ===")
    print(f"Text shape: {text_emb.shape}, Image shape: {image_emb.shape}")


if __name__ == "__main__":
    main()
