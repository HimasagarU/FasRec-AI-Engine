"""
FastAPI application for the Fashion Recommendation Engine.
Supports real-time semantic text search, visual image search,
cross-category outfit completion, and out-of-stock alternatives.
"""
import io
import json
import os
import random
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# Load .env
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"
IMAGES_DIR = DATA_DIR / "images"

# R2 public URL (set via env var for CDN image serving)
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "").rstrip("/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Fashion Recommendation Engine", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────────────────

products_df: pd.DataFrame | None = None
recommendations: dict | None = None
product_ids: np.ndarray | None = None
text_embeddings: np.ndarray | None = None
image_embeddings: np.ndarray | None = None
text_index = None
image_index = None
sbert_model = None
clip_model = None
clip_processor = None

# Map from product_id -> FAISS row index for fast lookup
id_to_idx: dict[int, int] = {}


@app.on_event("startup")
def startup():
    global products_df, recommendations, product_ids
    global text_embeddings, image_embeddings, text_index, image_index
    global sbert_model, clip_model, clip_processor, id_to_idx

    # ── 1. Load product metadata ─────────────────────────────────────
    csv_path = DATA_DIR / "styles.csv"
    if csv_path.exists():
        products_df = pd.read_csv(csv_path, on_bad_lines="skip")
    else:
        alt_csv = ARTIFACTS_DIR / "styles.csv"
        if alt_csv.exists():
            products_df = pd.read_csv(alt_csv, on_bad_lines="skip")
        else:
            raise FileNotFoundError("styles.csv not found in data/ or artifacts/")

    products_df["id"] = products_df["id"].astype(int)
    text_cols = [
        "productDisplayName", "articleType", "baseColour",
        "gender", "usage", "masterCategory", "subCategory", "season",
    ]
    for col in text_cols:
        if col in products_df.columns:
            products_df[col] = products_df[col].fillna("")

    # Mock in_stock status (80% chance in stock)
    random.seed(42)
    products_df["in_stock"] = [random.random() < 0.80 for _ in range(len(products_df))]

    # ── 2. Load precomputed recommendations (fallback) ───────────────
    recs_path = ARTIFACTS_DIR / "precomputed_recs.json"
    if recs_path.exists():
        with open(recs_path) as f:
            recommendations = json.load(f)
        print(f"[API] Loaded {len(recommendations)} precomputed recommendations.")
    else:
        recommendations = {}
        print("[API] WARNING: No precomputed_recs.json found.")

    # ── 3. Load FAISS indexes & embeddings ───────────────────────────
    try:
        text_index = faiss.read_index(str(ARTIFACTS_DIR / "text_index.faiss"))
        image_index = faiss.read_index(str(ARTIFACTS_DIR / "image_index.faiss"))
        text_embeddings = np.load(ARTIFACTS_DIR / "text_embeddings.npy")
        image_embeddings = np.load(ARTIFACTS_DIR / "image_embeddings.npy")
        product_ids = np.load(ARTIFACTS_DIR / "product_ids.npy")

        # Build reverse lookup: product_id -> row index in FAISS
        for i, pid in enumerate(product_ids):
            id_to_idx[int(pid)] = i

        print(f"[API] FAISS indexes loaded. Text: {text_index.ntotal}, Image: {image_index.ntotal}")
    except Exception as e:
        print(f"[API] WARNING: Could not load FAISS indexes: {e}")

    # ── 4. Load ML models for real-time search (opt-in) ─────────────
    load_models = os.getenv("LOAD_MODELS", "false").lower() == "true"
    if load_models:
        try:
            sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)
            print("[API] SBERT model loaded.")
        except Exception as e:
            print(f"[API] WARNING: Could not load SBERT model: {e}")

        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model.eval()
            print("[API] CLIP model loaded.")
        except Exception as e:
            print(f"[API] WARNING: Could not load CLIP model: {e}")
    else:
        print("[API] ML models not loaded (set LOAD_MODELS=true to enable text/visual search).")

    if R2_PUBLIC_URL:
        print(f"[API] Serving images from R2: {R2_PUBLIC_URL}")
    else:
        print("[API] Serving images locally from /images/")


# ── Helpers ──────────────────────────────────────────────────────────────


def _image_url(product_id: int) -> str:
    """Generate image URL — uses R2 CDN if configured, otherwise local."""
    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL}/images/{product_id}.jpg"
    return f"/images/{product_id}.jpg"


def _product_to_dict(row) -> dict:
    return {
        "id": int(row["id"]),
        "title": str(row.get("productDisplayName", "")),
        "gender": str(row.get("gender", "")),
        "masterCategory": str(row.get("masterCategory", "")),
        "subCategory": str(row.get("subCategory", "")),
        "articleType": str(row.get("articleType", "")),
        "baseColour": str(row.get("baseColour", "")),
        "season": str(row.get("season", "")),
        "year": str(row.get("year", "")),
        "usage": str(row.get("usage", "")),
        "in_stock": bool(row.get("in_stock", True)),
        "image_url": _image_url(int(row["id"])),
    }


def _faiss_search_to_products(
    scores: np.ndarray,
    indices: np.ndarray,
    exclude_id: int | None = None,
    category_filter: str | None = None,
    in_stock_only: bool = False,
    top_k: int = 10,
) -> list[dict]:
    """Convert raw FAISS search results into product dicts with filters."""
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(product_ids):
            continue
        pid = int(product_ids[idx])
        if pid == exclude_id:
            continue
        row = products_df[products_df["id"] == pid]
        if row.empty:
            continue
        r = row.iloc[0]

        if category_filter and str(r.get("subCategory", "")).lower() != category_filter.lower():
            continue
        if in_stock_only and not bool(r.get("in_stock", True)):
            continue

        pdict = _product_to_dict(r)
        pdict["score"] = round(float(score), 4)
        results.append(pdict)

        if len(results) >= top_k:
            break
    return results


# ── Endpoints ────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"message": "Fashion Recommendation Engine API", "version": "2.0.0"}


@app.get("/products")
def list_products(
    page: int = Query(1, ge=1),
    per_page: int = Query(40, ge=1, le=200),
    category: str | None = None,
    gender: str | None = None,
    search: str | None = None,
):
    """List products with pagination and optional filters."""
    df = products_df.copy()

    if category:
        df = df[df["masterCategory"].str.lower() == category.lower()]
    if gender:
        df = df[df["gender"].str.lower() == gender.lower()]
    if search:
        search_cols = ["productDisplayName", "articleType", "baseColour", "subCategory", "usage", "gender"]
        mask = df[search_cols[0]].str.contains(search, case=False, na=False)
        for col in search_cols[1:]:
            mask = mask | df[col].str.contains(search, case=False, na=False)
        df = df[mask]

    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end]

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "products": [_product_to_dict(row) for _, row in page_df.iterrows()],
    }


@app.get("/recommend/{item_id}")
def get_recommendations(item_id: int):
    """Get precomputed top-10 recommendations for an item."""
    source = products_df[products_df["id"] == item_id]
    if source.empty:
        raise HTTPException(404, f"Product {item_id} not found")

    source_dict = _product_to_dict(source.iloc[0])

    recs = recommendations.get(str(item_id), [])
    rec_products = []
    for r in recs:
        rec_row = products_df[products_df["id"] == r["id"]]
        if not rec_row.empty:
            rec_dict = _product_to_dict(rec_row.iloc[0])
            rec_dict["score"] = r["score"]
            rec_products.append(rec_dict)

    return {
        "item": source_dict,
        "recommendations": rec_products,
    }


@app.get("/similar/{item_id}")
def get_similar(
    item_id: int,
    top_k: int = Query(10, ge=1, le=50),
    category_filter: str | None = None,
    in_stock_only: bool = False,
):
    """
    Get similar items using dynamic FAISS search.
    Supports filtering by subCategory and in_stock status.
    Used for both "Complete the Outfit" and "Out-of-Stock Saver".
    """
    source = products_df[products_df["id"] == item_id]
    if source.empty:
        raise HTTPException(404, f"Product {item_id} not found")

    source_dict = _product_to_dict(source.iloc[0])

    # Use dynamic FAISS search if indexes are loaded
    if text_index is not None and image_index is not None and item_id in id_to_idx:
        idx = id_to_idx[item_id]
        text_query = text_embeddings[idx].reshape(1, -1)
        image_query = image_embeddings[idx].reshape(1, -1)

        # Search more candidates to allow for filtering
        search_k = top_k * 20
        t_scores, t_indices = text_index.search(text_query, search_k)
        i_scores, i_indices = image_index.search(image_query, search_k)

        # Fuse scores (alpha=0.5)
        candidates = {}
        for j in range(search_k):
            t_idx = int(t_indices[0, j])
            if t_idx >= 0 and t_idx < len(product_ids):
                candidates.setdefault(t_idx, {"t": 0.0, "i": 0.0})
                candidates[t_idx]["t"] = float(t_scores[0, j])

            i_idx = int(i_indices[0, j])
            if i_idx >= 0 and i_idx < len(product_ids):
                candidates.setdefault(i_idx, {"t": 0.0, "i": 0.0})
                candidates[i_idx]["i"] = float(i_scores[0, j])

        # Sort by fused score
        scored = []
        for c_idx, sims in candidates.items():
            pid = int(product_ids[c_idx])
            if pid == item_id:
                continue
            fused = 0.5 * sims["t"] + 0.5 * sims["i"]
            scored.append((c_idx, pid, fused))
        scored.sort(key=lambda x: x[2], reverse=True)

        # Apply filters
        results = []
        for c_idx, pid, fused_score in scored:
            row = products_df[products_df["id"] == pid]
            if row.empty:
                continue
            r = row.iloc[0]
            if category_filter and str(r.get("subCategory", "")).lower() != category_filter.lower():
                continue
            if in_stock_only and not bool(r.get("in_stock", True)):
                continue
            pdict = _product_to_dict(r)
            pdict["score"] = round(fused_score, 4)
            results.append(pdict)
            if len(results) >= top_k:
                break

        return {"item": source_dict, "similar_items": results}

    # Fallback to precomputed (fetch extra candidates to allow filtering)
    recs = recommendations.get(str(item_id), [])[:top_k * 10]
    results = []
    for r in recs:
        rec_row = products_df[products_df["id"] == r["id"]]
        if not rec_row.empty:
            row = rec_row.iloc[0]
            if category_filter and str(row.get("subCategory", "")).lower() != category_filter.lower():
                continue
            if in_stock_only and not bool(row.get("in_stock", True)):
                continue
            rec_dict = _product_to_dict(row)
            rec_dict["score"] = r["score"]
            results.append(rec_dict)
            if len(results) >= top_k:
                break

    return {"item": source_dict, "similar_items": results}


@app.get("/text_search")
def text_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(20, ge=1, le=100),
):
    """
    Semantic text search using SBERT embeddings.
    Understands meaning, not just keywords.
    """
    if sbert_model is None or text_index is None:
        raise HTTPException(503, "Text search is not available. Models not loaded.")

    # Generate query embedding
    query_emb = sbert_model.encode(
        [q], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    # Search FAISS
    scores, indices = text_index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(product_ids):
            continue
        pid = int(product_ids[idx])
        row = products_df[products_df["id"] == pid]
        if row.empty:
            continue
        pdict = _product_to_dict(row.iloc[0])
        pdict["score"] = round(float(score), 4)
        results.append(pdict)

    return {"query": q, "total": len(results), "products": results}


@app.post("/visual_search")
async def visual_search(
    file: UploadFile = File(...),
    top_k: int = Query(20, ge=1, le=100),
):
    """
    Visual search: upload an image and find similar items using CLIP.
    """
    if clip_model is None or clip_processor is None or image_index is None:
        raise HTTPException(503, "Visual search is not available. Models not loaded.")

    # Read and process the uploaded image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    # Generate CLIP embedding
    with torch.no_grad():
        inputs = clip_processor(images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        image_emb = clip_model.visual_projection(vision_outputs.pooler_output)
        image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
        query_emb = image_emb.cpu().numpy().astype(np.float32)

    # Search FAISS
    scores, indices = image_index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(product_ids):
            continue
        pid = int(product_ids[idx])
        row = products_df[products_df["id"] == pid]
        if row.empty:
            continue
        pdict = _product_to_dict(row.iloc[0])
        pdict["score"] = round(float(score), 4)
        results.append(pdict)

    return {"total": len(results), "products": results}


@app.get("/categories")
def get_categories():
    """Get unique category values for filtering."""
    return {
        "masterCategories": sorted(products_df["masterCategory"].dropna().unique().tolist()),
        "subCategories": sorted(products_df["subCategory"].dropna().unique().tolist()),
        "genders": sorted(products_df["gender"].dropna().unique().tolist()),
    }


# ── Static files ─────────────────────────────────────────────────────────

# Serve product images locally (fallback when R2 not configured)
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Serve frontend
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
