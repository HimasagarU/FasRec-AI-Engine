"""
Data loading and preprocessing for the Fashion Recommendation Engine.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STYLES_CSV = DATA_DIR / "styles.csv"
IMAGES_DIR = DATA_DIR / "images"


def load_products(max_products: int | None = None) -> pd.DataFrame:
    """
    Load styles.csv, clean data, filter to products with valid images,
    and create a composite text field for embedding.

    Returns:
        DataFrame with columns: id, gender, masterCategory, subCategory,
        articleType, baseColour, season, year, usage, productDisplayName,
        text_field, image_path
    """
    print("[DataLoader] Loading styles.csv...")
    df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")

    # Drop rows missing critical fields
    critical_cols = ["id", "productDisplayName"]
    df = df.dropna(subset=critical_cols)
    df["id"] = df["id"].astype(int)

    # Fill NaN in text columns with empty string
    text_cols = ["productDisplayName", "articleType", "baseColour",
                 "gender", "usage", "masterCategory", "subCategory", "season"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Filter to products that have a corresponding image
    def has_image(pid):
        return (IMAGES_DIR / f"{pid}.jpg").exists()

    print("[DataLoader] Filtering products with available images...")
    df["image_path"] = df["id"].apply(lambda x: str(IMAGES_DIR / f"{x}.jpg"))
    mask = df["id"].apply(has_image)
    df = df[mask].reset_index(drop=True)

    # Create composite text field for embedding
    df["text_field"] = (
        df["productDisplayName"].astype(str) + " " +
        df["articleType"].astype(str) + " " +
        df["baseColour"].astype(str) + " " +
        df["gender"].astype(str) + " " +
        df["usage"].astype(str) + " " +
        df["masterCategory"].astype(str) + " " +
        df["subCategory"].astype(str)
    )

    if max_products is not None:
        df = df.head(max_products)

    print(f"[DataLoader] Loaded {len(df)} products with images.")
    return df


def get_product_by_id(df: pd.DataFrame, product_id: int) -> dict | None:
    """Get a single product's metadata by ID."""
    row = df[df["id"] == product_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


if __name__ == "__main__":
    df = load_products()
    print(f"\nSample product:\n{df.iloc[0].to_dict()}")
    print(f"\nCategory distribution:\n{df['masterCategory'].value_counts().head(10)}")
