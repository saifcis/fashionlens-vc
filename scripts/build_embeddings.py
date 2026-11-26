
import sqlite3
import time
import numpy as np

from pathlib import Path
from PIL import Image

from app.embeddings.clip_provider import CLIPProvider

BATCH_SIZE = 16

OUTPUT_DIR = Path("data/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Retrieve products

with sqlite3.connect("data/catalog.db") as connection:

    products = connection.execute("""
        SELECT id, image_path
        FROM products
        ORDER BY id
    """).fetchall()

print("Total products:", len(products))

# Load model once

provider = CLIPProvider()

all_embeddings = []
product_ids = []

start = time.perf_counter()

# Process products in batches

for offset in range(0, len(products), BATCH_SIZE):

    batch = products[offset:offset + BATCH_SIZE]

    images = []

    for product_id, image_path in batch:

        with Image.open(image_path) as source:
            images.append(source.convert("RGB"))

        product_ids.append(product_id)

    embeddings = provider.encode_images(images)

    all_embeddings.append(embeddings)

    print(
        f"Processed "
        f"{len(product_ids)}/{len(products)}",
        flush=True
    )

# Combine all batches

matrix = np.concatenate(all_embeddings, axis=0)

ids = np.array(product_ids, dtype=np.int64)

assert matrix.shape == (len(products), 512)
assert len(ids) == len(matrix)
assert np.isfinite(matrix).all()
assert np.allclose(
    np.linalg.norm(matrix, axis=1),
    1.0,
    atol=1e-5
)

# Save vectors and product IDs together

np.savez(
    OUTPUT_DIR / "clip_catalog.npz",
    embeddings=matrix,
    product_ids=ids,
    model_name=provider.MODEL_NAME
)

elapsed = time.perf_counter() - start

print("\nIndexing completed")
print("Embedding shape:", matrix.shape)
print("Data type:", matrix.dtype)
print("Total time:", round(elapsed, 2), "seconds")
print("Products per second:", round(len(products) / elapsed, 2))
print("Saved:", OUTPUT_DIR / "clip_catalog.npz")

