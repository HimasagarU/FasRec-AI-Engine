"""
Upload product images to Cloudflare R2 (S3-compatible).
Uses concurrent uploads for speed.
"""
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from tqdm import tqdm

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=50,
        ),
        region_name="auto",
    )


def upload_single(s3, filepath: Path, bucket: str) -> str:
    """Upload a single image and return its key."""
    key = f"images/{filepath.name}"
    s3.upload_file(
        str(filepath),
        bucket,
        key,
        ExtraArgs={"ContentType": "image/jpeg"},
    )
    return key


def main():
    print("=== Uploading Images to Cloudflare R2 ===\n")

    if not all([R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        print("ERROR: Missing R2 credentials in .env file")
        sys.exit(1)

    # Get all images
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(image_files)} images to upload")

    # Check what's already uploaded
    s3 = get_s3_client()

    print("Checking existing uploads...")
    existing_keys = set()
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix="images/"):
            for obj in page.get("Contents", []):
                existing_keys.add(obj["Key"])
    except Exception as e:
        print(f"Note: Could not list existing objects: {e}")

    # Filter to only new images
    to_upload = [f for f in image_files if f"images/{f.name}" not in existing_keys]
    print(f"Already uploaded: {len(existing_keys)}, remaining: {len(to_upload)}")

    if not to_upload:
        print("All images already uploaded!")
        return

    # Upload with thread pool
    uploaded = 0
    failed = 0

    with tqdm(total=len(to_upload), desc="Uploading") as pbar:
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}
            for f in to_upload:
                # Each thread gets its own S3 client
                client = get_s3_client()
                fut = executor.submit(upload_single, client, f, R2_BUCKET_NAME)
                futures[fut] = f.name

            for fut in as_completed(futures):
                try:
                    fut.result()
                    uploaded += 1
                except Exception as e:
                    failed += 1
                    print(f"\nFailed: {futures[fut]}: {e}")
                pbar.update(1)

    print(f"\n=== Done! Uploaded: {uploaded}, Failed: {failed} ===")
    print(f"\nNext steps:")
    print(f"1. Enable public access on your R2 bucket in Cloudflare dashboard")
    print(f"2. Copy the public URL and set R2_PUBLIC_URL in .env")
    print(f"3. Images will be at: <R2_PUBLIC_URL>/images/<id>.jpg")


if __name__ == "__main__":
    main()
