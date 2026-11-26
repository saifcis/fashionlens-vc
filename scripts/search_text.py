
import argparse
import time

from app.search.keyword_search import KeywordSearch
from app.search.fuzzy_search import FuzzySearch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["keyword", "fuzzy"],
    required=True
)

parser.add_argument("--query", required=True)
parser.add_argument("--top", type=int, default=5)

args = parser.parse_args()

if args.top < 1:
    parser.error("--top must be at least 1")

if args.mode == "keyword":
    engine = KeywordSearch()
else:
    engine = FuzzySearch()

start = time.perf_counter()

results = engine.search(
    args.query,
    limit=args.top
)

elapsed = time.perf_counter() - start

print("\nSearch mode:", args.mode)
print("Query:", args.query)
print("Results:", len(results))
print()

for rank, product in enumerate(results, start=1):

    print(f"{rank}. {product['name']}")
    print(f"   Score: {product['score']:.4f}")
    print(f"   Colour: {product['colour']}")
    print(f"   Type: {product['article_type']}")
    print(f"   Image: {product['image_path']}")
    print()

print(f"Search time: {elapsed * 1000:.2f} ms")

