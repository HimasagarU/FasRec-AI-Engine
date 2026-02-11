"""
Script 03: Precompute top-10 recommendations for all products.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender import precompute_and_save


def main():
    print("=== Precomputing Recommendations ===\n")
    recs = precompute_and_save(alpha=0.5, candidate_k=200, top_k=10)
    print(f"\n=== Done! {len(recs)} items with precomputed recommendations. ===")

    # Show sample
    sample_ids = list(recs.keys())[:3]
    for sid in sample_ids:
        print(f"\nItem {sid} -> {recs[sid]}")


if __name__ == "__main__":
    main()
