import time
from pathlib import Path

from app.search.engine import SearchEngine

print("Initializing search engine...")

engine = SearchEngine()

query_image = next(
    Path("data/images").glob("*.jpg")
)

tests = [
    ("keyword", {"text": "black handbag"}),
    ("fuzzy", {"text": "blak handbg"}),
    ("semantic", {"text": "black leather handbag"}),
    ("hybrid", {"text": "black leather handbag"}),
    ("visual", {"image_path": query_image}),
    (
        "multimodal",
        {
            "text": "red fashion product",
            "image_path": query_image
        }
    )
]

for mode, params in tests:

    print("\n" + "=" * 50)
    print("SEARCH MODE:", mode.upper())

    start = time.perf_counter()

    results = engine.search(
        mode=mode,
        limit=5,
        **params
    )

    elapsed = time.perf_counter() - start

    for rank, product in enumerate(results, start=1):

        print(
            f"{rank}. {product['name']} "
            f"| Score: {product['score']:.4f}"
        )

    print(f"Results: {len(results)}")
    print(f"Latency: {elapsed * 1000:.2f} ms")

