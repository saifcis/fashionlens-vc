
from datasets import load_dataset

# Download and load the dataset
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train"
)

# Display dataset information
print("\nTotal products:", len(dataset))

print("\nAvailable fields:")
print(dataset.column_names)

# Read the first product
product = dataset[0]

print("\nFirst product:")

for key, value in product.items():
    if key != "image":
        print(f"{key}: {value}")

# Save the first product image
product["image"].save("data/sample_product.jpg")

print("\nImage saved to data/sample_product.jpg")

print("\nDataset cache files:")
print(dataset.cache_files)
