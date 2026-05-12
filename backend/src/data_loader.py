"""
Data loading and preprocessing for the Fashion Recommendation Engine.
Reads from rich JSON files instead of basic CSV to build high-quality semantic texts.
"""
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STYLES_DIR = DATA_DIR / "styles"
IMAGES_DIR = DATA_DIR / "images"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _clean_html(text: str) -> str:
    """Very basic HTML tag stripper for product descriptions."""
    if not text:
        return ""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', str(text)).replace('&nbsp;', ' ').strip()


def parse_product_json(filepath: Path) -> dict | None:
    """Parse a rich product JSON file and extract required fields."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # The structure is usually {"data": { ... }}
        if "data" in data:
            item = data["data"]
        else:
            item = data
            
        product_id = item.get("id")
        if not product_id:
            return None

        # Extract core identity & categories
        display_name = item.get("productDisplayName", "")
        brand = item.get("brandName", "")
        gender = item.get("gender", "")
        base_color = item.get("baseColour", "")
        season = item.get("season", "")
        usage = item.get("usage", "")
        
        # Some JSONs have nested objects for categories
        master_category = item.get("masterCategory", {}).get("typeName", "") if isinstance(item.get("masterCategory"), dict) else item.get("masterCategory", "")
        sub_category = item.get("subCategory", {}).get("typeName", "") if isinstance(item.get("subCategory"), dict) else item.get("subCategory", "")
        article_type = item.get("articleType", {}).get("typeName", "") if isinstance(item.get("articleType"), dict) else item.get("articleType", "")

        # Extract fine-grained attributes
        attrs = item.get("articleAttributes", {})
        fit = attrs.get("Fit", "")
        occasion = attrs.get("Occasion", "")
        
        # Extract text content
        descriptors = item.get("productDescriptors", {})
        description = _clean_html(descriptors.get("description", {}).get("value", ""))
        style_note = _clean_html(descriptors.get("style_note", {}).get("value", ""))

        # Build canonical text string
        text_parts = [
            f"{display_name}.",
            f"{brand}." if brand else "",
            f"Gender: {gender}." if gender else "",
            f"Color: {base_color}." if base_color else "",
            f"Season: {season}." if season else "",
            f"Usage: {usage}." if usage else "",
            f"Fit: {fit}." if fit else "",
            f"Occasion: {occasion}." if occasion else "",
            f"Description: {description}" if description else "",
            f"Style note: {style_note}" if style_note else ""
        ]
        text_field = " ".join([p for p in text_parts if p]).strip()

        # Build record matching the old dataframe schema + rich text
        return {
            "id": int(product_id),
            "productDisplayName": display_name,
            "brandName": brand,
            "gender": gender,
            "baseColour": base_color,
            "season": season,
            "usage": usage,
            "masterCategory": master_category,
            "subCategory": sub_category,
            "articleType": article_type,
            "text_field": text_field,
            "image_path": str(IMAGES_DIR / f"{product_id}.jpg")
        }
    except Exception as e:
        print(f"Error parsing {filepath.name}: {e}")
        return None


def load_products(max_products: int | None = None) -> pd.DataFrame:
    """
    Load all product JSONs, filter to products with valid images,
    and create a dataframe with rich semantic text fields.
    Uses caching to avoid parsing 44k JSONs every time.
    """
    cache_path = ARTIFACTS_DIR / "parsed_products.csv"
    if cache_path.exists():
        print(f"[DataLoader] Found cached parsed products at {cache_path}. Loading...")
        df = pd.read_csv(cache_path)
        # Ensure ID is int, as CSV might load it as float if there are NaNs
        df["id"] = df["id"].astype(int)
        if max_products: 
            df = df.head(max_products)
        return df

    if not STYLES_DIR.exists():
        print(f"[DataLoader] Directory not found: {STYLES_DIR}. Falling back to old CSV if exists.")
        # Fallback to styles.csv logic if JSON folder isn't populated
        csv_path = DATA_DIR / "styles.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, on_bad_lines="skip")
            df = df.dropna(subset=["id", "productDisplayName"])
            df["id"] = df["id"].astype(int)
            text_cols = ["productDisplayName", "articleType", "baseColour", "gender", "usage", "masterCategory", "subCategory", "season"]
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna("")
            df["text_field"] = df["productDisplayName"] + " " + df["articleType"] + " " + df["baseColour"] + " " + df["gender"]
            df["image_path"] = df["id"].apply(lambda x: str(IMAGES_DIR / f"{x}.jpg"))
            mask = df["id"].apply(lambda x: (IMAGES_DIR / f"{x}.jpg").exists())
            df = df[mask].reset_index(drop=True)
            if max_products: df = df.head(max_products)
            return df
        return pd.DataFrame()

    print(f"[DataLoader] Parsing JSON files from {STYLES_DIR}...")
    json_files = list(STYLES_DIR.glob("*.json"))
    
    if max_products is not None:
        # Sort or take a subset early to save time if limiting
        json_files = json_files[:max_products*2] # buffer for missing images

    records = []
    for filepath in tqdm(json_files, desc="Parsing JSONs"):
        product_id = filepath.stem
        # Filter to products that have a corresponding image
        if not (IMAGES_DIR / f"{product_id}.jpg").exists():
            continue
            
        record = parse_product_json(filepath)
        if record:
            records.append(record)
            
        if max_products is not None and len(records) >= max_products:
            break

    df = pd.DataFrame(records)
    print(f"[DataLoader] Loaded {len(df)} products with valid JSON and images.")
    
    # Save cache for future runs
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[DataLoader] Cached parsed products to {cache_path}")
    
    return df


def get_product_by_id(df: pd.DataFrame, product_id: int) -> dict | None:
    """Get a single product's metadata by ID."""
    row = df[df["id"] == product_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


if __name__ == "__main__":
    df = load_products()
    if not df.empty:
        print(f"\nSample product:\n{df.iloc[0].to_dict()}")
        print(f"\nCategory distribution:\n{df['masterCategory'].value_counts().head(10)}")
