
class HybridSearch:

    def __init__(
        self,
        keyword_search,
        vector_search
    ):

        self.keyword_search = keyword_search
        self.vector_search = vector_search

    def search(self, query, limit=10, *, eligible_ids=None):

        candidate_limit = max(limit * 5, 20)

        keyword_results = self.keyword_search.search(
            query,
            limit=candidate_limit,
            eligible_ids=eligible_ids,
        )

        semantic_results = self.vector_search.search_text(
            query,
            limit=candidate_limit,
            eligible_ids=eligible_ids,
        )

        return self.fuse(
            [keyword_results, semantic_results],
            limit
        )

    @staticmethod
    def fuse(result_lists, limit, k=60):

        scores = {}
        products = {}

        for results in result_lists:

            for rank, product in enumerate(
                results, start=1
            ):

                product_id = product["id"]

                scores[product_id] = (
                    scores.get(product_id, 0)
                    + 1 / (k + rank)
                )

                products[product_id] = product.copy()

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True
        )[:limit]

        results = []

        for product_id in ranked_ids:

            product = products[product_id]
            product["score"] = scores[product_id]

            results.append(product)

        return results

