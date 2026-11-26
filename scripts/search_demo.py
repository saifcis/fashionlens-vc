
import argparse
import sqlite3
import time
import numpy as np

from app.embeddings.clip_provider import CLIPProvider

parser = argparse.ArgumentParser()

query = parser.add_mutually_exclusive_group(required=True)

query.add_argument("--text")
query.add_argument("--image")

parser.add_argument("--top", type=int, default=5)

args = parser.parse_args()

if args.top < 1:
    parser.error("--top must be at least 1")

# Load the stored embeddings

with np.load("data/embeddings/clip_catalog.npz") as data:

    catalog_embeddings = data["embeddings"]
    product_ids = data["product_ids"]
    model_name = data["model_name"].item()

# Load the same model used for indexing

provider = CLIPProvider()

if model_name != provider.MODEL_NAME:
    raise ValueError("Embedding model mismatch")

# Generate query embedding

start = time.perf_counter()

if args.text:
    query_embedding = provider.encode_text(args.text)
else:
    query_embedding = provider.encode_image(args.image)

inference_time = time.perf_counter() - start

# Compare query with every catalog vector

start = time.perf_counter()

scores = catalog_embeddings @ query_embedding

top_indices = np.argsort(-scores)[:args.top]

search_time = time.perf_counter() - start

# Retrieve product information

with sqlite3.connect("data/catalog.db") as connection:

    print("\nSearch results:\n")

    for rank, index in enumerate(top_indices, start=1):

        product_id = int(product_ids[index])

        product = connection.execute("""
            SELECT name, colour, article_type, image_path
            FROM products
            WHERE id = ?
        """, (product_id,)).fetchone()

        if product is None:
            continue

        print(f"{rank}. {product[0]}")
        print(f"   Similarity: {scores[index]:.4f}")
        print(f"   Colour: {product[1]}")
        print(f"   Type: {product[2]}")
        print(f"   Image: {product[3]}")
        print()

print("Inference time:", round(inference_time, 4), "seconds")
print("Vector search time:", round(search_time, 6), "seconds")

