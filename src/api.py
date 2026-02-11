"""
FastAPI application for the Fashion Recommendation Engine.
Supports both local image serving and Cloudflare R2 CDN URLs.
"""
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
from dotenv import load_dotenv

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

app = FastAPI(title="Fashion Recommendation Engine", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load data ──────────────────────────────────────────────────

products_df: pd.DataFrame | None = None
recommendations: dict | None = None


@app.on_event("startup")
def startup():
    global products_df, recommendations

    # Load product metadata
    csv_path = DATA_DIR / "styles.csv"
    if csv_path.exists():
        products_df = pd.read_csv(csv_path, on_bad_lines="skip")
    else:
        # Fallback: check if bundled in artifacts
        alt_csv = ARTIFACTS_DIR / "styles.csv"
        if alt_csv.exists():
            products_df = pd.read_csv(alt_csv, on_bad_lines="skip")
        else:
            raise FileNotFoundError("styles.csv not found in data/ or artifacts/")

    products_df["id"] = products_df["id"].astype(int)
    text_cols = ["productDisplayName", "articleType", "baseColour",
                 "gender", "usage", "masterCategory", "subCategory", "season"]
    for col in text_cols:
        if col in products_df.columns:
            products_df[col] = products_df[col].fillna("")

    # Load precomputed recommendations
    recs_path = ARTIFACTS_DIR / "precomputed_recs.json"
    if recs_path.exists():
        with open(recs_path) as f:
            recommendations = json.load(f)
        print(f"[API] Loaded {len(recommendations)} precomputed recommendations.")
    else:
        recommendations = {}
        print("[API] WARNING: No precomputed_recs.json found.")

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
        "image_url": _image_url(int(row["id"])),
    }


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Fashion Recommendation Engine API", "version": "1.0.0"}


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
        df = df[df["productDisplayName"].str.contains(search, case=False, na=False)]

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
def get_similar(item_id: int, top_k: int = Query(10, ge=1, le=50)):
    """Get similar items with similarity scores."""
    source = products_df[products_df["id"] == item_id]
    if source.empty:
        raise HTTPException(404, f"Product {item_id} not found")

    recs = recommendations.get(str(item_id), [])[:top_k]
    results = []
    for r in recs:
        rec_row = products_df[products_df["id"] == r["id"]]
        if not rec_row.empty:
            rec_dict = _product_to_dict(rec_row.iloc[0])
            rec_dict["similarity_score"] = r["score"]
            results.append(rec_dict)

    return {
        "item_id": item_id,
        "similar_items": results,
    }


@app.get("/categories")
def get_categories():
    """Get unique category values for filtering."""
    return {
        "masterCategories": sorted(products_df["masterCategory"].dropna().unique().tolist()),
        "genders": sorted(products_df["gender"].dropna().unique().tolist()),
    }


# ── Static files ─────────────────────────────────────────────────────────

# Serve product images locally (fallback when R2 not configured)
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Serve frontend
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
