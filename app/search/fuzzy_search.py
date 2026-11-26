
import sqlite3

from rapidfuzz import fuzz, process


class FuzzySearch:

    def __init__(self, database="data/catalog.db"):

        with sqlite3.connect(database) as connection:

            connection.row_factory = sqlite3.Row

            self.products = [
                dict(row)
                for row in connection.execute("""
                    SELECT
                        id,
                        name,
                        colour,
                        article_type,
                        image_path
                    FROM products
                """).fetchall()
            ]

        self.choices = {
            index: (
                f"{product['name']} "
                f"{product['colour'] or ''} "
                f"{product['article_type'] or ''}"
            ).lower()

            for index, product in enumerate(self.products)
        }

    def search(self, query, limit=10):

        matches = process.extract(
            query.lower(),
            self.choices,
            scorer=fuzz.WRatio,
            limit=limit,
            score_cutoff=40
        )

        results = []

        for matched_text, score, index in matches:

            product = self.products[index].copy()
            product["score"] = score

            results.append(product)

        return results

