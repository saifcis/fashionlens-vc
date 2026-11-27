"""Compare frozen CLIP domain decision rules offline on a labeled manifest."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

from app.search.engine import SearchEngine
from experiments.clip_domain import ClipDomainClassifier, decisions
from scripts.evaluate_visual_search import load_manifest, markdown_cell, percentage, write_csv

CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiments/clip_domain_prompts.json"
METHODS = ("retrieval", "zero_shot", "combined")
OUTCOMES = {(True, True): "TP", (True, False): "FN",
            (False, True): "FP", (False, False): "TN"}


def measure(engine, classifier, image_path):
    start = perf_counter()
    embedding = engine.vector.provider.encode_image(image_path)
    encoded = perf_counter()
    products = engine.vector.search_vector(embedding, limit=10)
    retrieved = perf_counter()
    scores = classifier.score(embedding)
    classified = perf_counter()
    if len(products) != 10 or not all(math.isfinite(p["score"]) for p in products):
        raise ValueError(f"Expected 10 finite retrieval scores for {image_path}")
    scores.update(
        embedding_ms=(encoded - start) * 1000,
        retrieval_ms=(retrieved - encoded) * 1000,
        classification_ms=(classified - retrieved) * 1000,
    )
    return scores, products


def evaluate(engine, classifier, rules, rows, input_dir):
    results, rankings = [], []
    for row in rows:
        scores, products = measure(engine, classifier, input_dir / row["filename"])
        best = products[0]
        classification = decisions(scores["margin"], best["score"], rules)
        expected = row["expected_classification"] == "fashion"
        results.append({
            "filename": row["filename"],
            "expected_classification": row["expected_classification"],
            "category": row["category"],
            "ambiguous": row.get("ambiguous", "no"),
            "catalog_image": row["catalog_image"],
            "difficulty_note": row.get("difficulty_note", ""),
            "top_product_id": best["id"],
            "top_product": best["name"],
            "top_cosine": best["score"],
            **scores,
            **{method: OUTCOMES[expected, classification[method]] for method in METHODS},
        })
        rankings.extend({
            "filename": row["filename"], "rank": rank, "product_id": product["id"],
            "product_name": product["name"], "cosine_similarity": product["score"],
        } for rank, product in enumerate(products, 1))
    return results, rankings


def counts(rows, method):
    return tuple(sum(row[method] == outcome for row in rows) for outcome in ("TP", "TN", "FP", "FN"))


def metric_table(rows):
    yield "| Method | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | FPR | FNR |"
    yield "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    for method in METHODS:
        tp, tn, fp, fn = counts(rows, method)
        metrics = [(tp + tn, len(rows)), (tp, tp + fp), (tp, tp + fn),
                   (2 * tp, 2 * tp + fp + fn), (fp, fp + tn), (fn, fn + tp)]
        yield "| " + " | ".join(map(str, (method, tp, tn, fp, fn,
                                           *(percentage(a, b) for a, b in metrics)))) + " |"


def latency(rows, *fields):
    times = sorted(sum(row[field] for field in fields) for row in rows)
    return f"{statistics.median(times):.3f} | {times[math.ceil(0.95 * len(times)) - 1]:.3f}"


def report(path, rows, hashes, setup_ms, rules):
    clear = [row for row in rows if row["ambiguous"] == "no" and row["catalog_image"] == "no"]
    lines = [
        "# CLIP fashion-domain evaluation", "",
        f"Evaluated **{len(rows)}** images and retained **{10 * len(rows)}** ranked products.",
        f"Primary metrics use **{len(clear)}** unambiguous external images.", "",
        f"Configuration SHA-256 `{hashes[0]}` and manifest SHA-256 `{hashes[1]}`.", "",
        "A category score is the image cosine against the normalized mean of its text prompts. "
        "The margin is the strongest fashion cosine minus the strongest non-fashion cosine. "
        "These scores are uncalibrated.", "",
        f"Retrieval accepts top cosine ≥ {rules['retrieval_threshold']}. Zero-shot accepts margin "
        f"≥ {rules['zero_shot_margin']}. Combined also accepts retrieval-positive queries with margin "
        f"≥ {rules['rescue_margin']}. All comparisons are inclusive.", "",
        "## Confusion matrices and metrics", "",
        *metric_table(clear), "",
        "## Category performance", "",
        "| Category | Clear N | Retrieval correct | Zero-shot correct | Combined correct |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category in sorted({row["category"] for row in clear}):
        group = [row for row in clear if row["category"] == category]
        rates = [percentage(sum(row[method] in {"TP", "TN"} for row in group), len(group))
                 for method in METHODS]
        lines.append(f"| {category} | {len(group)} | " + " | ".join(rates) + " |")
    lines.extend([
        "", "## Warmed local CPU latency", "",
        "| Path | Median ms | P95 ms (nearest rank) |", "| --- | ---: | ---: |",
        f"| Retrieval | {latency(rows, 'embedding_ms', 'retrieval_ms')} |",
        f"| Zero-shot alone | {latency(rows, 'embedding_ms', 'classification_ms')} |",
        f"| Combined | {latency(rows, 'embedding_ms', 'retrieval_ms', 'classification_ms')} |",
        f"| Additional classification scoring | {latency(rows, 'classification_ms')} |",
        "",
        f"Text prompt encoding and caching took {setup_ms:.1f} ms once. Model load and a synthetic "
        "warm-up image are excluded. Query timing includes image decoding and inference. Substep times "
        "are composed from the same run and exclude report generation and boolean decisions.",
    ])
    for heading, group in [
        ("Unambiguous images misclassified by any method",
         [row for row in clear if any(row[m] in {"FP", "FN"} for m in METHODS)]),
        ("Ambiguous cases and catalog controls",
         [row for row in rows if row["ambiguous"] == "yes" or row["catalog_image"] == "yes"]),
    ]:
        lines.extend([
            "", f"## {heading}", "",
            "| Image | Expected | Retrieval | Zero-shot | Combined | Best fashion / other prompt | Top product | Top cosine | Margin | Image note |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ])
        for row in group:
            cells = [row["filename"], row["expected_classification"], *(row[m] for m in METHODS),
                     f"{row['fashion_category']} / {row['non_fashion_category']}", row["top_product"],
                     f"{row['top_cosine']:.4f}", f"{row['margin']:.4f}", row["difficulty_note"]]
            lines.append("| " + " | ".join(map(markdown_cell, cells)) + " |")
    lines.extend(["", "The labels concern query domain only. Establishing individual product "
                  "recommendation relevance requires separate product-level judgments."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    rows = load_manifest(args.manifest)
    engine = SearchEngine()
    if engine.vector.provider.MODEL_NAME != config["model"]:
        raise ValueError("Model mismatch")
    start = perf_counter()
    classifier = ClipDomainClassifier(engine.vector.provider, config)
    setup_ms = (perf_counter() - start) * 1000
    from PIL import Image
    warm_embedding = engine.vector.provider.encode_images([Image.new("RGB", (224, 224))])[0]
    classifier.score(warm_embedding)
    engine.vector.search_vector(warm_embedding, limit=10)

    results, rankings = evaluate(engine, classifier, config["rules"], rows, args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "clip_domain_results.csv", tuple(results[0]), results)
    write_csv(args.output_dir / "clip_domain_rankings.csv", tuple(rankings[0]), rankings)
    hashes = (hashlib.sha256(config_bytes).hexdigest(),
              hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    report(args.output_dir / "clip_domain_report.md", results, hashes, setup_ms, config["rules"])
    print(f"Evaluated {len(results)} images in {args.output_dir}")


if __name__ == "__main__":
    main()
