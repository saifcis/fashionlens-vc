"""CLIP fashion-domain scoring."""

import numpy as np


class ClipDomainClassifier:
    def __init__(self, provider, config):
        self.groups = {
            label: list(categories)
            for label, categories in config["prompts"].items()
        }
        self.categories = [category for group in self.groups.values() for category in group]
        embeddings = {}
        centroids = []
        for categories in config["prompts"].values():
            for prompts in categories.values():
                for prompt in prompts:
                    if prompt not in embeddings:
                        embeddings[prompt] = provider.encode_text(prompt)
                centroid = np.mean([embeddings[prompt] for prompt in prompts], axis=0)
                centroids.append(centroid / np.linalg.norm(centroid))
        self.centroids = np.stack(centroids)
        self.fashion_count = len(self.groups["fashion"])
        if list(self.groups) != ["fashion", "non-fashion"]:
            raise ValueError("Prompt groups must be ordered fashion, then non-fashion.")

    def score(self, embedding):
        similarities = self.centroids @ embedding
        fashion = int(np.argmax(similarities[:self.fashion_count]))
        non_fashion = self.fashion_count + int(np.argmax(similarities[self.fashion_count:]))
        return {
            "category_scores": {
                category: float(score) for category, score in zip(self.categories, similarities)
            },
            "fashion_category": self.categories[fashion],
            "non_fashion_category": self.categories[non_fashion],
            "fashion_cosine": float(similarities[fashion]),
            "non_fashion_cosine": float(similarities[non_fashion]),
            "margin": float(similarities[fashion] - similarities[non_fashion]),
        }
