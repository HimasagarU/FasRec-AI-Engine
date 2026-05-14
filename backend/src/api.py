"""
FastAPI application for the Fashion Recommendation Engine.
Supports both local image serving and Cloudflare R2 CDN URLs.
"""
import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import timedelta

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

# Import internal modules
from .database import engine, get_db, Base
from . import models
from .auth import get_password_hash, verify_password, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
from .llm import generate_outfit_narration

# Load .env
load_dotenv()

# Create tables
try:
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables created/verified successfully.")
except Exception as e:
    print(f"Warning: Could not connect to database to create tables: {e}")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
IMAGES_DIR = DATA_DIR / "images"

# R2 public URL (set via env var for CDN image serving)
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "").rstrip("/")

app = FastAPI(title="Fashion Recommendation Engine", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Schemas ─────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class FavoriteCreate(BaseModel):
    product_id: int

class SavedOutfitCreate(BaseModel):
    query_product_id: int
    recommendation_text: str
    occasion_text: str
    recommended_product_ids: List[int]

# ── Startup: load data ──────────────────────────────────────────────────

products_df: pd.DataFrame | None = None
recommendations: dict | None = None

@app.on_event("startup")
def startup():
    global products_df, recommendations

    # Load product metadata — prefer the rich JSON-parsed cache
    parsed_csv = ARTIFACTS_DIR / "parsed_products.csv"
    styles_csv = DATA_DIR / "styles.csv"

    if parsed_csv.exists():
        products_df = pd.read_csv(parsed_csv)
        print(f"[API] Loaded {len(products_df)} products from parsed_products.csv")
    elif styles_csv.exists():
        products_df = pd.read_csv(styles_csv, on_bad_lines="skip")
        print(f"[API] Loaded {len(products_df)} products from styles.csv (fallback)")
    else:
        print("Warning: No product data found in artifacts/ or data/")
        products_df = pd.DataFrame()

    if not products_df.empty:
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

# ── Helpers ──────────────────────────────────────────────────────────────

def _image_url(product_id: int) -> str:
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

def get_product_by_id(item_id: int) -> dict:
    if products_df is None or products_df.empty:
        return None
    source = products_df[products_df["id"] == item_id]
    if source.empty:
        return None
    return _product_to_dict(source.iloc[0])


def get_available_article_types(gender: str = "") -> list[str]:
    """Get unique articleType values from the catalog, optionally filtered by gender."""
    if products_df is None or products_df.empty:
        return []
    df = products_df
    if gender:
        df = df[df["gender"].str.lower() == gender.lower()]
    return sorted(df["articleType"].dropna().unique().tolist())


def search_products_by_criteria(article_type: str = "", color: str = "",
                                 gender: str = "", limit: int = 3) -> list[dict]:
    """
    Smart multi-strategy search of the 44k product catalog.
    Strategy order:
      1. Exact articleType + color + gender
      2. Exact articleType + gender (relax color)
      3. Word-by-word articleType match + color + gender
      4. Word-by-word articleType match + gender (relax color)
      5. Fuzzy: check if any word from query appears in any articleType
    """
    if products_df is None or products_df.empty:
        return []

    base_df = products_df.copy()
    if gender:
        base_df = base_df[base_df["gender"].str.lower() == gender.lower()]

    if not article_type:
        return []

    at_lower = article_type.strip().lower()
    # Extract meaningful search words (skip very short/generic words)
    stop_words = {"a", "an", "the", "in", "of", "for", "and", "or", "with", "casual", "formal", "classic", "modern", "minimalist", "leather", "cotton", "slim", "fit", "regular"}
    words = [w for w in at_lower.split() if len(w) > 2 and w not in stop_words]

    def apply_color_filter(df, clr):
        if not clr:
            return df
        clr_lower = clr.strip().lower()
        clr_words = [w for w in clr_lower.split() if len(w) > 2]
        for cw in clr_words:
            filtered = df[df["baseColour"].str.lower().str.contains(cw, na=False)]
            if not filtered.empty:
                return filtered
        return df  # if color filter gives empty, return unfiltered

    # Strategy 1: exact articleType contains full query + color
    matched = base_df[base_df["articleType"].str.lower().str.contains(at_lower, na=False)]
    if not matched.empty:
        result = apply_color_filter(matched, color)
        if not result.empty:
            return [_product_to_dict(row) for _, row in result.head(limit).iterrows()]
        return [_product_to_dict(row) for _, row in matched.head(limit).iterrows()]

    # Strategy 2: try each word individually against articleType
    for word in words:
        matched = base_df[base_df["articleType"].str.lower().str.contains(word, na=False)]
        if not matched.empty:
            result = apply_color_filter(matched, color)
            if not result.empty:
                return [_product_to_dict(row) for _, row in result.head(limit).iterrows()]
            return [_product_to_dict(row) for _, row in matched.head(limit).iterrows()]

    # Strategy 3: try matching against productDisplayName (broader search)
    for word in words:
        matched = base_df[base_df["productDisplayName"].str.lower().str.contains(word, na=False)]
        if not matched.empty:
            result = apply_color_filter(matched, color)
            if not result.empty:
                return [_product_to_dict(row) for _, row in result.head(limit).iterrows()]
            return [_product_to_dict(row) for _, row in matched.head(limit).iterrows()]

    # Strategy 4: try subCategory
    for word in words:
        matched = base_df[base_df["subCategory"].str.lower().str.contains(word, na=False)]
        if not matched.empty:
            result = apply_color_filter(matched, color)
            if not result.empty:
                return [_product_to_dict(row) for _, row in result.head(limit).iterrows()]
            return [_product_to_dict(row) for _, row in matched.head(limit).iterrows()]

    return []


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
@app.head("/")
def root():
    return {"message": "Fashion Recommendation Engine API", "version": "2.0.0"}

@app.get("/keep-alive")
@app.head("/keep-alive")
async def keep_alive():
    return {"status": "alive"}

# ── Auth Endpoints ───────────────────────────────────────────────────────

@app.post("/auth/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@app.post("/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user.email,
        "user_id": user.id
    }

@app.get("/auth/me")
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return {"email": current_user.email, "id": current_user.id}

# ── Products & Recommendations ───────────────────────────────────────────

@app.get("/products")
@app.head("/products")
def list_products(
    page: int = Query(1, ge=1),
    per_page: int = Query(40, ge=1, le=200),
    category: str | None = None,
    gender: str | None = None,
    search: str | None = None,
):
    if products_df is None or products_df.empty:
        return {"total": 0, "page": page, "per_page": per_page, "products": []}

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
    source_dict = get_product_by_id(item_id)
    if not source_dict:
        raise HTTPException(404, f"Product {item_id} not found")

    recs = recommendations.get(str(item_id), []) if recommendations else []
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

@app.get("/recommend/{item_id}/narration")
def get_recommendation_narration(item_id: int):
    """
    Generates an LLM outfit narration AND searches the 44k product catalog
    to find real products matching the LLM suggestions.
    """
    source_dict = get_product_by_id(item_id)
    if not source_dict:
        raise HTTPException(404, f"Product {item_id} not found")

    recs = recommendations.get(str(item_id), []) if recommendations else []
    rec_products = []
    for r in recs[:5]:
        rec_row = products_df[products_df["id"] == r["id"]]
        if not rec_row.empty:
            rec_dict = _product_to_dict(rec_row.iloc[0])
            rec_products.append(rec_dict)

    if not rec_products:
        return {
            "recommendation": "No recommendations available.",
            "occasion": "Any",
            "outfit_pieces": []
        }
    
    # Get the actual article types available in our catalog for this gender
    gender = source_dict.get("gender", "")
    available_types = get_available_article_types(gender)

    # Call Groq API — enhanced prompt returns structured outfit pieces
    narration = generate_outfit_narration(source_dict, rec_products, available_types)

    # Now search the real catalog for each suggested piece
    gender = source_dict.get("gender", "")
    outfit_pieces_with_products = []

    for piece in narration.get("outfit_pieces", []):
        article_type = piece.get("type", "")
        color = piece.get("color", "")
        matching = search_products_by_criteria(
            article_type=article_type,
            color=color,
            gender=gender,
            limit=3
        )
        outfit_pieces_with_products.append({
            "type": article_type,
            "color": color,
            "why": piece.get("why", ""),
            "products": matching
        })

    return {
        "recommendation": narration.get("recommendation", ""),
        "occasion": narration.get("occasion", ""),
        "outfit_pieces": outfit_pieces_with_products
    }

@app.get("/similar/{item_id}")
def get_similar(item_id: int, top_k: int = Query(10, ge=1, le=50)):
    source_dict = get_product_by_id(item_id)
    if not source_dict:
        raise HTTPException(404, f"Product {item_id} not found")

    recs = recommendations.get(str(item_id), [])[:top_k] if recommendations else []
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
@app.head("/categories")
def get_categories():
    if products_df is None or products_df.empty:
        return {"masterCategories": [], "genders": []}
        
    return {
        "masterCategories": sorted(products_df["masterCategory"].dropna().unique().tolist()),
        "genders": sorted(products_df["gender"].dropna().unique().tolist()),
    }

# ── Favorites Endpoints ──────────────────────────────────────────────────

@app.get("/favorites")
def get_favorites(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    favs = db.query(models.FavoriteProduct).filter(models.FavoriteProduct.user_id == current_user.id).all()
    
    fav_details = []
    for f in favs:
        p = get_product_by_id(f.product_id)
        if p:
            p['favorite_id'] = f.id
            fav_details.append(p)
            
    return fav_details

@app.get("/favorites/ids")
def get_favorite_ids(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Returns just the product IDs that the user has favorited — used for UI heart state."""
    favs = db.query(models.FavoriteProduct).filter(models.FavoriteProduct.user_id == current_user.id).all()
    return {f.product_id: f.id for f in favs}

@app.post("/favorites")
def add_favorite(fav: FavoriteCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    existing = db.query(models.FavoriteProduct).filter(
        models.FavoriteProduct.user_id == current_user.id,
        models.FavoriteProduct.product_id == fav.product_id
    ).first()
    
    if existing:
        return {"message": "Already favorited", "id": existing.id}
        
    new_fav = models.FavoriteProduct(user_id=current_user.id, product_id=fav.product_id)
    db.add(new_fav)
    db.commit()
    db.refresh(new_fav)
    return {"message": "Favorited successfully", "id": new_fav.id}

@app.delete("/favorites/{product_id}")
def remove_favorite(product_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    fav = db.query(models.FavoriteProduct).filter(
        models.FavoriteProduct.product_id == product_id,
        models.FavoriteProduct.user_id == current_user.id
    ).first()
    
    if not fav:
        raise HTTPException(status_code=404, detail="Favorite not found")
        
    db.delete(fav)
    db.commit()
    return {"message": "Removed successfully"}

# ── Saved Outfits Endpoints ──────────────────────────────────────────────

@app.get("/outfits")
def get_saved_outfits(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    outfits = db.query(models.SavedOutfit).filter(models.SavedOutfit.user_id == current_user.id).all()
    results = []
    for o in outfits:
        query_p = get_product_by_id(o.query_product_id)
        rec_ps = [get_product_by_id(pid) for pid in (o.recommended_product_ids or [])]
        rec_ps = [p for p in rec_ps if p is not None]
        
        results.append({
            "id": o.id,
            "query_product": query_p,
            "recommendation_text": o.recommendation_text,
            "occasion_text": o.occasion_text,
            "recommended_products": rec_ps
        })
    return results

@app.post("/outfits")
def save_outfit(outfit: SavedOutfitCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_outfit = models.SavedOutfit(
        user_id=current_user.id,
        query_product_id=outfit.query_product_id,
        recommendation_text=outfit.recommendation_text,
        occasion_text=outfit.occasion_text,
        recommended_product_ids=outfit.recommended_product_ids
    )
    db.add(new_outfit)
    db.commit()
    db.refresh(new_outfit)
    return {"message": "Outfit saved successfully", "id": new_outfit.id}

@app.delete("/outfits/{outfit_id}")
def remove_saved_outfit(outfit_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    outfit = db.query(models.SavedOutfit).filter(
        models.SavedOutfit.id == outfit_id,
        models.SavedOutfit.user_id == current_user.id
    ).first()
    
    if not outfit:
        raise HTTPException(status_code=404, detail="Outfit not found")
        
    db.delete(outfit)
    db.commit()
    return {"message": "Removed successfully"}


# ── Static files ─────────────────────────────────────────────────────────

if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
