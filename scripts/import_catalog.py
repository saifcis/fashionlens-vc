
import sqlite3
from pathlib import Path
from datasets import load_dataset

PRODUCT_LIMIT = 1000

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
DATABASE_PATH = DATA_DIR / "catalog.db"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Load the Hugging Face dataset
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train"
)

# Randomly select 1,000 products
dataset = dataset.shuffle(seed=42).select(
    range(PRODUCT_LIMIT)
)

# Connect to our database
connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()

# Create the products table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        gender TEXT,
        category TEXT,
        subcategory TEXT,
        article_type TEXT,
        colour TEXT,
        season TEXT,
        usage TEXT,
        image_path TEXT NOT NULL
    )
""")

# Import products
for product in dataset:

    product_id = product["id"]
    image_path = IMAGE_DIR / f"{product_id}.jpg"

    # Save the product photograph
    product["image"].convert("RGB").save(image_path)

    # Store product metadata
    cursor.execute("""
        INSERT OR REPLACE INTO products (
            id, name, gender, category,
            subcategory, article_type, colour,
            season, usage, image_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        product_id,
        product["productDisplayName"],
        product["gender"],
        product["masterCategory"],
        product["subCategory"],
        product["articleType"],
        product["baseColour"],
        product["season"],
        product["usage"],
        str(image_path)
    ))

connection.commit()

# Verify the imported records
cursor.execute("SELECT COUNT(*) FROM products")
count = cursor.fetchone()[0]

connection.close()

print(f"Total products imported: {count}")
print(f"Database: {DATABASE_PATH}")
print(f"Images: {IMAGE_DIR}")

