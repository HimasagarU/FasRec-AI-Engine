"""
Recommendation engine: weighted fusion of text + image similarities,
and top-K precomputation.
"""
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.faiss_index import batch_search_index, load_index
from src.embeddings import load_embeddings

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def compute_fusion_scores(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    text_index,
    image_index,
    alpha: float = 0.5,
    candidate_k: int = 200,
    top_k: int = 10,
    batch_size: int = 512,
) -> dict[int, list[dict]]:
    """
    For each item, retrieve top candidates from text and image indexes,
    compute weighted fusion, and return top-K recommendations.

    Args:
        text_embeddings: (N, D_text) L2-normalized
        image_embeddings: (N, D_image) L2-normalized
        text_index: FAISS index for text
        image_index: FAISS index for image
        alpha: weight for text similarity (1-alpha for image)
        candidate_k: number of candidates per modality
        top_k: final number of recommendations per item
        batch_size: batch size for FAISS search

    Returns:
        dict mapping item index -> list of {index, score}
    """
    n = len(text_embeddings)
    print(f"[Recommender] Computing fusion for {n} items (alpha={alpha})...")

    # Batch search both indexes
    print("[Recommender] Searching text index...")
    text_scores, text_indices = batch_search_index(
        text_index, text_embeddings, candidate_k, batch_size
    )
    print("[Recommender] Searching image index...")
    image_scores, image_indices = batch_search_index(
        image_index, image_embeddings, candidate_k, batch_size
    )

    recommendations = {}

    for i in tqdm(range(n), desc="Fusing recommendations"):
        # Merge candidates from both modalities
        candidates = {}

        for j in range(candidate_k):
            idx = int(text_indices[i, j])
            if idx == i or idx < 0:
                continue
            score_t = float(text_scores[i, j])
            candidates.setdefault(idx, {"text_sim": 0.0, "image_sim": 0.0})
            candidates[idx]["text_sim"] = score_t

        for j in range(candidate_k):
            idx = int(image_indices[i, j])
            if idx == i or idx < 0:
                continue
            score_i = float(image_scores[i, j])
            candidates.setdefault(idx, {"text_sim": 0.0, "image_sim": 0.0})
            candidates[idx]["image_sim"] = score_i

        # Compute fused score
        scored = []
        for idx, sims in candidates.items():
            fused = alpha * sims["text_sim"] + (1 - alpha) * sims["image_sim"]
            scored.append({"index": idx, "score": round(fused, 6)})

        # Sort and take top-K
        scored.sort(key=lambda x: x["score"], reverse=True)
        recommendations[i] = scored[:top_k]

    return recommendations


def precompute_and_save(
    alpha: float = 0.5,
    candidate_k: int = 200,
    top_k: int = 10,
    output_dir: Path | None = None,
):
    """
    Full precomputation pipeline: load embeddings + indexes,
    compute fusion, save results.
    """
    out = output_dir or ARTIFACTS_DIR

    # Load embeddings
    text_emb, image_emb, product_ids = load_embeddings(out)

    # Load indexes
    text_index = load_index("text_index", out)
    image_index = load_index("image_index", out)

    # Compute fusion
    recs = compute_fusion_scores(
        text_emb, image_emb, text_index, image_index,
        alpha=alpha, candidate_k=candidate_k, top_k=top_k,
    )

    # Convert index-based recs to product-ID-based recs
    id_recs = {}
    for item_idx, rec_list in recs.items():
        item_id = int(product_ids[item_idx])
        id_recs[item_id] = [
            {"id": int(product_ids[r["index"]]), "score": r["score"]}
            for r in rec_list
        ]

    # Save
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "precomputed_recs.json"
    with open(output_path, "w") as f:
        json.dump(id_recs, f)
    print(f"[Recommender] Saved {len(id_recs)} precomputed recommendations to {output_path}")
    return id_recs


def load_recommendations(input_dir: Path | None = None) -> dict:
    """Load precomputed recommendations."""
    d = input_dir or ARTIFACTS_DIR
    with open(d / "precomputed_recs.json") as f:
        return json.load(f)
