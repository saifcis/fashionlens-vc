
import sqlite3
import numpy as np

from app.embeddings.clip_provider import CLIPProvider


class VectorSearch:

    def __init__(
        self,
        database="data/catalog.db",
        index_path="data/embeddings/clip_catalog.npz"
    ):

        self.database = database

        with np.load(index_path, allow_pickle=False) as data:
            self.embeddings = data["embeddings"]
            self.product_ids = data["product_ids"]
            self.model_name = data["model_name"].item()

        self.provider = CLIPProvider()

        if self.model_name != self.provider.MODEL_NAME:
            raise ValueError("Embedding model mismatch")

        if self.embeddings.shape != (
            len(self.product_ids), 512
        ):
            raise ValueError("Invalid embedding shape")

    def search_vector(self, query_embedding, limit=10, *, eligible_ids=None):

        scores = self.embeddings @ query_embedding

        indices = np.argsort(-scores)
        if eligible_ids is not None:
            indices = indices[np.isin(self.product_ids[indices], list(eligible_ids))]
        indices = indices[:limit]

        results = []

        with sqlite3.connect(self.database) as connection:

            connection.row_factory = sqlite3.Row

            for index in indices:

                product_id = int(self.product_ids[index])

                row = connection.execute("""
                    SELECT id, name, colour,
                           article_type, image_path
                    FROM products
                    WHERE id = ?
                """, (product_id,)).fetchone()

                if row is not None:

                    product = dict(row)
                    product["score"] = float(scores[index])

                    results.append(product)

        return results

    def search_text(self, text, limit=10, *, eligible_ids=None):

        embedding = self.provider.encode_text(text)

        return self.search_vector(embedding, limit, eligible_ids=eligible_ids)

    def search_image(self, image_path, limit=10, *, eligible_ids=None):

        embedding = self.provider.encode_image(image_path)

        return self.search_vector(embedding, limit, eligible_ids=eligible_ids)

    def search_multimodal(
        self,
        text,
        image_path,
        image_weight=0.5,
        limit=10,
        *,
        eligible_ids=None,
    ):

        if not 0 <= image_weight <= 1:
            raise ValueError("Invalid image weight")

        image_embedding = self.provider.encode_image(
            image_path
        )

        text_embedding = self.provider.encode_text(text)

        combined = (
            image_weight * image_embedding
            + (1 - image_weight) * text_embedding
        )

        norm = np.linalg.norm(combined)

        if norm < 1e-8:
            raise ValueError("Invalid combined embedding")

        combined = combined / norm

        return self.search_vector(combined, limit, eligible_ids=eligible_ids)

