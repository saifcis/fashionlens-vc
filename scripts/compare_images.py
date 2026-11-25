
import sqlite3
import torch

from PIL import Image
from transformers import AutoProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"

# 1. Retrieve three products

connection = sqlite3.connect("data/catalog.db")

product_a = connection.execute("""
    SELECT id, name, article_type, image_path
    FROM products
    WHERE article_type = 'Handbags'
    ORDER BY id
    LIMIT 1
""").fetchone()

product_b = connection.execute("""
    SELECT id, name, article_type, image_path
    FROM products
    WHERE article_type = 'Handbags'
      AND id != ?
    ORDER BY id
    LIMIT 1
""", (product_a[0],)).fetchone()

product_c = connection.execute("""
    SELECT id, name, article_type, image_path
    FROM products
    WHERE article_type = 'Casual Shoes'
    ORDER BY id
    LIMIT 1
""").fetchone()

connection.close()

products = [product_a, product_b, product_c]

if any(product is None for product in products):
    raise RuntimeError("Required products not found")

# 2. Load the model once

processor = AutoProcessor.from_pretrained(
    MODEL_NAME, local_files_only=True
)

model = CLIPModel.from_pretrained(
    MODEL_NAME, local_files_only=True
)

model.eval()

# 3. Create a reusable embedding function

def encode_image(image_path):

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        features = model.get_image_features(**inputs)

        embedding = (
            features if isinstance(features, torch.Tensor)
            else features.pooler_output
        )

    return embedding

# 4. Generate three embeddings

embedding_a = encode_image(product_a[3])
embedding_b = encode_image(product_b[3])
embedding_c = encode_image(product_c[3])

# 5. Calculate cosine similarity

def similarity(embedding_1, embedding_2):

    score = torch.nn.functional.cosine_similarity(
        embedding_1,
        embedding_2,
        dim=-1
    )

    return score.item()

# 6. Display results

for label, product in zip("ABC", products):
    print(f"Product {label}: {product[1]}")
    print(f"Article type: {product[2]}")
    print(f"Image: {product[3]}")
    print()

print("Cosine similarity results:")

print(
    "A vs A:",
    round(similarity(embedding_a, embedding_a), 4)
)

print(
    "A vs B:",
    round(similarity(embedding_a, embedding_b), 4)
)

print(
    "A vs C:",
    round(similarity(embedding_a, embedding_c), 4)
)

print(
    "B vs C:",
    round(similarity(embedding_b, embedding_c), 4)
)

