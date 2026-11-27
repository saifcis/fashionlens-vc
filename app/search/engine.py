from time import perf_counter

from app.search.catalog import Catalog
from app.search.relevance import FashionRelevance, RelevanceBlocked, RelevanceMode
from app.search.keyword_search import KeywordSearch
from app.search.fuzzy_search import FuzzySearch
from app.search.vector_search import VectorSearch
from app.search.hybrid_search import HybridSearch


class SearchEngine:

    def __init__(self, relevance_mode=None):

        relevance_mode = RelevanceMode.configured() if relevance_mode is None else RelevanceMode(relevance_mode)
        self.keyword = KeywordSearch()
        self.fuzzy = FuzzySearch()
        self.vector = VectorSearch()
        self.relevance = FashionRelevance(self.vector.provider, relevance_mode)
        self.catalog = Catalog(self.keyword.database)

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
        limit=10,
        collection="Discover",
        filters=None,
        exclude_product_id=None,
        *,
        image_query=None,
        override_relevance=False,
    ):

        if limit < 1:
            raise ValueError("Invalid result limit")

        if mode in (
            "keyword", "fuzzy", "semantic",
            "hybrid", "multimodal"
        ) and not (text and text.strip()):
            raise ValueError("Text query required")

        if mode in ("visual", "multimodal") and not (image_path or image_query is not None):
            raise ValueError("Image required")

        eligible_ids = None
        if (
            collection != "Discover"
            or any(value is not None for value in (filters or {}).values())
            or exclude_product_id is not None
        ):
            eligible_ids = {
                product["id"]
                for product in self.catalog.matching_products(collection, filters)
                if product["id"] != exclude_product_id
            }

        if mode == "keyword":

            return self.keyword.search(text, limit, eligible_ids=eligible_ids)

        if mode == "fuzzy":

            return self.fuzzy.search(text, limit, eligible_ids=eligible_ids)

        if mode == "semantic":

            return self.vector.search_text(text, limit, eligible_ids=eligible_ids)

        if mode == "hybrid":
            return self.hybrid.search(text, limit, eligible_ids=eligible_ids)

        if mode in ("visual", "multimodal"):
            if mode == "multimodal" and not 0 <= image_weight <= 1:
                raise ValueError("Invalid image weight")
            query = image_query
            if query is None:
                query = self.relevance.prepare(image_path)
            return self._search_prepared_image(
                mode, query, text, image_weight, limit, eligible_ids, override_relevance
            )

        raise ValueError(f"Unsupported mode: {mode}")

    def _search_prepared_image(self, mode, query, text, image_weight, limit, eligible_ids, override):
        decision = self.relevance.decision(query, mode, override)
        if decision in ("confirm", "reject"):
            status = "awaiting_confirmation" if decision == "confirm" else "blocked"
            self.relevance.record_evaluation(query, mode, decision, search_status=status)
            raise RelevanceBlocked(decision, query.relevance)

        start = perf_counter()
        products = None
        try:
            if mode == "visual":
                products = self.vector.search_vector(query.embedding, limit, eligible_ids=eligible_ids)
            else:
                products = self.vector.search_multimodal_vector(
                    text, query.embedding, image_weight, limit, eligible_ids=eligible_ids
                )
            return products
        finally:
            execution = {
                "search_status": "failed" if products is None else "completed",
                "search_latency_ms": (perf_counter() - start) * 1000,
                "result_count": None if products is None else len(products),
            }
            self.relevance.record_evaluation(query, mode, decision, **execution)
            if decision == "override":
                self.relevance.record_override(query, mode, **execution)
