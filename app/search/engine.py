
from app.search.keyword_search import KeywordSearch
from app.search.fuzzy_search import FuzzySearch
from app.search.vector_search import VectorSearch
from app.search.hybrid_search import HybridSearch


class SearchEngine:

    def __init__(self):

        self.keyword = KeywordSearch()
        self.fuzzy = FuzzySearch()
        self.vector = VectorSearch()

        self.hybrid = HybridSearch(
            self.keyword,
            self.vector
        )

    def search(
        self,
        mode,
        text=None,
        image_path=None,
        image_weight=0.5,
        limit=10
    ):

        if limit < 1:
            raise ValueError("Invalid result limit")

        if mode in (
            "keyword", "fuzzy", "semantic",
            "hybrid", "multimodal"
        ) and not (text and text.strip()):
            raise ValueError("Text query required")

        if mode in ("visual", "multimodal") and not image_path:
            raise ValueError("Image required")

        if mode == "keyword":

            return self.keyword.search(text, limit)

        if mode == "fuzzy":

            return self.fuzzy.search(text, limit)

        if mode == "semantic":

            return self.vector.search_text(text, limit)

        if mode == "visual":

            return self.vector.search_image(
                image_path, limit
            )

        if mode == "hybrid":

            return self.hybrid.search(text, limit)

        if mode == "multimodal":

            return self.vector.search_multimodal(
                text,
                image_path,
                image_weight,
                limit
            )

        raise ValueError(f"Unsupported mode: {mode}")

