"""
Script 04: Evaluate recommendation quality.
Metrics: intra-category precision, diversity.
"""
import sys
import json
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_products

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def evaluate():
    print("=== Evaluation ===\n")

    # Load products
    df = load_products()
    id_to_category = dict(zip(df["id"], df["masterCategory"]))
    id_to_article = dict(zip(df["id"], df["articleType"]))
    id_to_subcategory = dict(zip(df["id"], df["subCategory"]))

    # Load recommendations
    recs_path = ARTIFACTS_DIR / "precomputed_recs.json"
    with open(recs_path) as f:
        recs = json.load(f)

    print(f"Evaluating {len(recs)} items...\n")

    master_precisions = []
    article_precisions = []
    sub_precisions = []
    diversities = []

    for item_id_str, rec_list in recs.items():
        item_id = int(item_id_str)
        if item_id not in id_to_category:
            continue

        source_cat = id_to_category[item_id]
        source_art = id_to_article.get(item_id, "")
        source_sub = id_to_subcategory.get(item_id, "")

        rec_ids = [r["id"] for r in rec_list]

        # Intra-category precision (masterCategory)
        same_cat = sum(1 for rid in rec_ids if id_to_category.get(rid) == source_cat)
        master_precisions.append(same_cat / len(rec_ids) if rec_ids else 0)

        # Intra-articleType precision
        same_art = sum(1 for rid in rec_ids if id_to_article.get(rid) == source_art)
        article_precisions.append(same_art / len(rec_ids) if rec_ids else 0)

        # Intra-subCategory precision
        same_sub = sum(1 for rid in rec_ids if id_to_subcategory.get(rid) == source_sub)
        sub_precisions.append(same_sub / len(rec_ids) if rec_ids else 0)

        # Diversity: number of unique masterCategories in recommendations
        unique_cats = len(set(id_to_category.get(rid, "") for rid in rec_ids))
        diversities.append(unique_cats)

    # Summary
    import numpy as np

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nIntra-masterCategory Precision:")
    print(f"  Mean:   {np.mean(master_precisions):.4f}")
    print(f"  Median: {np.median(master_precisions):.4f}")
    print(f"  Std:    {np.std(master_precisions):.4f}")

    print(f"\nIntra-articleType Precision:")
    print(f"  Mean:   {np.mean(article_precisions):.4f}")
    print(f"  Median: {np.median(article_precisions):.4f}")
    print(f"  Std:    {np.std(article_precisions):.4f}")

    print(f"\nIntra-subCategory Precision:")
    print(f"  Mean:   {np.mean(sub_precisions):.4f}")
    print(f"  Median: {np.median(sub_precisions):.4f}")
    print(f"  Std:    {np.std(sub_precisions):.4f}")

    print(f"\nDiversity (unique masterCategories in top-10):")
    print(f"  Mean:   {np.mean(diversities):.4f}")
    print(f"  Median: {np.median(diversities):.4f}")

    print(f"\nTotal items evaluated: {len(master_precisions)}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
