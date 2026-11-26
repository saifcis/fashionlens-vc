
import sqlite3
import time

start = time.perf_counter()

with sqlite3.connect("data/catalog.db") as connection:

    connection.execute("""
        DROP TABLE IF EXISTS products_fts
    """)

    connection.execute("""
        CREATE VIRTUAL TABLE products_fts
        USING fts5(
            product_id UNINDEXED,
            name,
            colour,
            category,
            article_type,
            usage
        )
    """)

    connection.execute("""
        INSERT INTO products_fts (
            product_id,
            name,
            colour,
            category,
            article_type,
            usage
        )
        SELECT
            id,
            name,
            colour,
            category,
            article_type,
            usage
        FROM products
    """)

    count = connection.execute("""
        SELECT COUNT(*) FROM products_fts
    """).fetchone()[0]

print("Text index created")
print("Indexed products:", count)
print(
    "Indexing time:",
    round(time.perf_counter() - start, 4),
    "seconds"
)

