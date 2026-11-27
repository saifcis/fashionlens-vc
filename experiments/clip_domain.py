"""Evaluation-only CLIP domain scoring."""

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
            "fashion_category": self.categories[fashion],
            "non_fashion_category": self.categories[non_fashion],
            "fashion_cosine": float(similarities[fashion]),
            "non_fashion_cosine": float(similarities[non_fashion]),
            "margin": float(similarities[fashion] - similarities[non_fashion]),
        }


def decisions(margin, top_cosine, rules):
    retrieval = top_cosine >= rules["retrieval_threshold"]
    zero_shot = margin >= rules["zero_shot_margin"]
    # Retrieval can rescue a borderline query but cannot override stronger negative evidence.
    combined = zero_shot or (margin >= rules["rescue_margin"] and retrieval)
    return {"retrieval": retrieval, "zero_shot": zero_shot, "combined": combined}
