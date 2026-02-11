"""
Embedding generation using SBERT (text) and CLIP (images).
Uses GPU when available.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def generate_text_embeddings(
    texts: list[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 128,
) -> np.ndarray:
    """
    Generate L2-normalized text embeddings using SBERT.

    Args:
        texts: list of text strings
        model_name: HuggingFace model name
        batch_size: encoding batch size

    Returns:
        np.ndarray of shape (N, 768), L2-normalized
    """
    print(f"[Embeddings] Loading text model: {model_name} on {DEVICE}...")
    model = SentenceTransformer(model_name, device=DEVICE)

    print(f"[Embeddings] Generating text embeddings for {len(texts)} products...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"[Embeddings] Text embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def generate_image_embeddings(
    image_paths: list[str],
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Generate L2-normalized image embeddings using CLIP.

    Args:
        image_paths: list of paths to images
        model_name: HuggingFace CLIP model name
        batch_size: processing batch size

    Returns:
        np.ndarray of shape (N, 512), L2-normalized
    """
    print(f"[Embeddings] Loading CLIP model: {model_name} on {DEVICE}...")
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    failed_indices = []

    print(f"[Embeddings] Generating image embeddings for {len(image_paths)} products...")
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Image embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        valid_in_batch = []

        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_in_batch.append(i + j)
            except Exception:
                failed_indices.append(i + j)

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            # Use the full model to get image features
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            image_embeds = model.visual_projection(vision_outputs.pooler_output)
            # L2 normalize
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(image_embeds.cpu().numpy())

    if failed_indices:
        print(f"[Embeddings] Warning: {len(failed_indices)} images failed to load.")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"[Embeddings] Image embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(
    text_emb: np.ndarray,
    image_emb: np.ndarray,
    product_ids: np.ndarray,
    output_dir: Path | None = None,
):
    """Save embeddings and IDs to disk."""
    out = output_dir or ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "text_embeddings.npy", text_emb)
    np.save(out / "image_embeddings.npy", image_emb)
    np.save(out / "product_ids.npy", product_ids)
    print(f"[Embeddings] Saved to {out}")


def load_embeddings(input_dir: Path | None = None):
    """Load saved embeddings and IDs."""
    d = input_dir or ARTIFACTS_DIR
    text_emb = np.load(d / "text_embeddings.npy")
    image_emb = np.load(d / "image_embeddings.npy")
    product_ids = np.load(d / "product_ids.npy")
    return text_emb, image_emb, product_ids
