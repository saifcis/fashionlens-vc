
import re
import sqlite3


class KeywordSearch:

    def __init__(self, database="data/catalog.db"):
        self.database = database

    def search(self, query, limit=10):

        tokens = re.findall(r"\w+", query.lower())

        if not tokens:
            return []

        # Match any search term
        expression = " OR ".join(
            f'"{token}"' for token in tokens
        )

        with sqlite3.connect(self.database) as connection:

            connection.row_factory = sqlite3.Row

            results = connection.execute("""
                SELECT
                    p.id,
                    p.name,
                    p.colour,
                    p.article_type,
                    p.image_path,
                    -bm25(products_fts) AS score
                FROM products_fts
                JOIN products p
                    ON p.id = products_fts.product_id
                WHERE products_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
            """, (expression, limit)).fetchall()

        return [dict(row) for row in results]

