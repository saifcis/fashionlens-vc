
import time
import torch

from transformers import AutoProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cpu"

start_time = time.perf_counter()

print("Loading processor...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

elapsed = time.perf_counter() - start_time

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("\nModel loaded successfully")
print("Model:", MODEL_NAME)
print("Device:", DEVICE)
print("Embedding dimensions:", model.config.projection_dim)
print("Parameters:", f"{parameter_count:,}")
print("Loading time:", f"{elapsed:.2f} seconds")

