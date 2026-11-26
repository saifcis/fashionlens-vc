import sqlite3

FILTER_FIELDS = {
    "gender": "Gender",
    "category": "Category",
    "article_type": "Article type",
    "colour": "Colour",
}

NAVIGATION = {
    "Discover": (),
    "Women": (("gender", "=", "Women"),),
    "Men": (("gender", "=", "Men"),),
    "Clothing": (("category", "=", "Apparel"),),
    "Bags": (("subcategory", "=", "Bags"),),
    "Shoes": (("category", "=", "Footwear"),),
    "Accessories": (
        ("category", "=", "Accessories"),
        ("subcategory", "!=", "Bags"),
    ),
}


class Catalog:
    def __init__(self, database):
        self.database = database

    def filter_options(self):
        with sqlite3.connect(self.database) as connection:
            return {
                field: [
                    value for (value,) in connection.execute(
                        f"SELECT DISTINCT {field} FROM products "
                        f"WHERE {field} IS NOT NULL AND {field} != '' "
                        f"ORDER BY {field} COLLATE NOCASE"
                    )
                ]
                for field in FILTER_FIELDS
            }

    def matching_products(self, collection="Discover", filters=None, limit=-1):
        conditions = [
            *NAVIGATION[collection],
            *((field, "=", value) for field, value in (filters or {}).items()
              if field in FILTER_FIELDS and value is not None),
        ]
        where = " AND ".join(f"{field} {op} ?" for field, op, _ in conditions)
        params = [value for _, _, value in conditions]
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            return [
                dict(row) for row in connection.execute(
                    "SELECT id, name, article_type, colour, image_path FROM products "
                    f"{'WHERE ' + where if where else ''} ORDER BY id LIMIT ?",
                    (*params, limit),
                )
            ]

    def statistics(self):
        with sqlite3.connect(self.database) as connection:
            return connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT article_type) FROM products"
            ).fetchone()
