"""
Script 03: Precompute top-10 recommendations for all products.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender import precompute_and_save


def main():
    print("=== Precomputing Recommendations ===\n")
    # Using Reciprocal Rank Fusion (RRF) for significantly better hybrid results
    recs = precompute_and_save(fusion_strategy="rrf", candidate_k=200, top_k=10)
    print(f"\n=== Done! {len(recs)} items with precomputed recommendations. ===")

    # Show sample
    sample_ids = list(recs.keys())[:3]
    for sid in sample_ids:
        print(f"\nItem {sid} -> {recs[sid]}")


if __name__ == "__main__":
    main()
