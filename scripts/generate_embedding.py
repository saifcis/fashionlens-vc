
import time
import sqlite3
import torch

from PIL import Image
from transformers import AutoProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"

# 1. Retrieve a product from SQLite
connection = sqlite3.connect("data/catalog.db")

product = connection.execute("""
    SELECT id, name, image_path
    FROM products
    ORDER BY id
    LIMIT 1
""").fetchone()

connection.close()

product_id, product_name, image_path = product

print("Product ID:", product_id)
print("Product name:", product_name)
print("Image path:", image_path)

# 2. Load the processor and model
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, local_files_only=True
)

model = CLIPModel.from_pretrained(
    MODEL_NAME, local_files_only=True
)

model.eval()

# 3. Open the image
with Image.open(image_path) as source:
    image = source.convert("RGB")

print("Original image size:", image.size)

# 4. Prepare the image for CLIP
inputs = processor(
    images=image,
    return_tensors="pt"
)

print("Input tensor shape:", inputs["pixel_values"].shape)

# 5. Generate the image embedding
start_time = time.perf_counter()

with torch.inference_mode():
    features = model.get_image_features(**inputs)

    embedding = (
        features if isinstance(features, torch.Tensor)
        else features.pooler_output
    )

elapsed = time.perf_counter() - start_time

# 6. Display the result
print("\nEmbedding shape:", embedding.shape)
print("Embedding dimensions:", embedding.shape[-1])
print("First 10 values:", embedding[0][:10].tolist())
print("Inference time:", f"{elapsed:.4f} seconds")

