"""Evaluate the existing visual search engine against a labeled image manifest."""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from PIL import Image

from app.search.engine import SearchEngine


EVALUATION_DIR = Path("data/evaluation")
TOP_K = 10
RESULT_FIELDS = (
    "filename", "expected_classification", "category", "catalog_image",
    "top_ranked_product", "highest_similarity", "lowest_top_10_similarity",
    "mean_top_10_similarity", "first_second_gap", "search_latency_ms",
)
FIXED_THRESHOLD_FIELDS = (
    "ambiguous", "difficulty_note", "evaluation_threshold",
    "predicted_classification", "classification_outcome",
)
RANK_FIELDS = (
    "filename", "expected_classification", "category", "catalog_image",
    "rank", "product_id", "product_name", "cosine_similarity",
)


def load_manifest(path):
    with path.open(encoding="utf-8", newline="") as manifest:
        rows = list(csv.DictReader(manifest))

    if not rows:
        raise ValueError("The visual manifest is empty.")

    seen = set()
    for row in rows:
        filename = row["filename"]
        if not filename or filename in {".", ".."} or Path(filename).name != filename or filename in seen:
            raise ValueError(f"Invalid or duplicate manifest filename: {filename!r}")
        if row["expected_classification"] not in {"fashion", "non-fashion"}:
            raise ValueError(f"Invalid classification for {filename}.")
        if row["catalog_image"] not in {"yes", "no"} or (
            row["catalog_image"] == "yes" and row["expected_classification"] != "fashion"
        ):
            raise ValueError(f"Invalid catalog flag for {filename}.")
        if row.get("ambiguous", "no") not in {"yes", "no"}:
            raise ValueError(f"Invalid ambiguity flag for {filename}.")
        seen.add(filename)
    return rows


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(engine, rows, image_dir):
    results, rankings, failures = [], [], []
    warmed_up = False
    for row in rows:
        image_path = image_dir / row["filename"]
        try:
            with Image.open(image_path) as image:
                image.load()
            if not warmed_up:
                engine.search("visual", image_path=image_path, limit=TOP_K)
                warmed_up = True

            start = perf_counter()
            products = engine.search("visual", image_path=image_path, limit=TOP_K)
            latency_ms = (perf_counter() - start) * 1000
            scores = [product["score"] for product in products]
            if len(scores) != TOP_K or not all(math.isfinite(score) for score in scores):
                raise ValueError(f"Expected {TOP_K} finite cosine scores, received {scores!r}.")

            labels = {key: row[key] for key in RESULT_FIELDS[:4]}
            results.append({
                **labels,
                "top_ranked_product": products[0]["name"],
                "highest_similarity": scores[0],
                "lowest_top_10_similarity": min(scores),
                "mean_top_10_similarity": statistics.mean(scores),
                "first_second_gap": scores[0] - scores[1],
                "search_latency_ms": latency_ms,
            })
            rankings.extend({
                **labels,
                "rank": rank,
                "product_id": product["id"],
                "product_name": product["name"],
                "cosine_similarity": product["score"],
            } for rank, product in enumerate(products, start=1))
        except (OSError, ValueError, RuntimeError) as error:
            failures.append((row["filename"], str(error)))
            print(f"FAILED {row['filename']}: {error}", file=sys.stderr)
            continue

        print(f"{row['filename']}: {scores[0]:.4f}, {latency_ms:.1f} ms", flush=True)
    return results, rankings, failures


def distribution_row(label, rows):
    if not rows:
        return f"| {label} | 0 | — | — | — | — | — |"
    scores = [row["highest_similarity"] for row in rows]
    median_latency = statistics.median(row["search_latency_ms"] for row in rows)
    return (
        f"| {label} | {len(rows)} | {min(scores):.4f} | {max(scores):.4f} | "
        f"{statistics.median(scores):.4f} | {statistics.mean(scores):.4f} | "
        f"{median_latency:.1f} |"
    )


def threshold_analysis(fashion, non_fashion):
    if not fashion or not non_fashion:
        return ["Both external groups are required to analyze score overlap."]

    scores = sorted({row["highest_similarity"] for row in fashion + non_fashion})
    thresholds = [scores[0], *((a + b) / 2 for a, b in zip(scores, scores[1:])),
                  math.nextafter(scores[-1], math.inf)]

    def errors(threshold):
        false_positives = [r for r in non_fashion if r["highest_similarity"] >= threshold]
        false_negatives = [r for r in fashion if r["highest_similarity"] < threshold]
        return false_positives, false_negatives

    def balanced_error(threshold):
        false_positives, false_negatives = errors(threshold)
        return (len(false_positives) / len(non_fashion) + len(false_negatives) / len(fashion)) / 2

    best = min(thresholds, key=balanced_error)
    fp, fn = errors(best)
    weakest_positive = min(fashion, key=lambda r: r["highest_similarity"])
    strongest_negative = max(non_fashion, key=lambda r: r["highest_similarity"])
    positive_scores = [row["highest_similarity"] for row in fashion]
    negative_scores = [row["highest_similarity"] for row in non_fashion]
    overlap_low = max(min(positive_scores), min(negative_scores))
    overlap_high = min(max(positive_scores), max(negative_scores))
    if overlap_low <= overlap_high:
        positive_count = sum(overlap_low <= score <= overlap_high for score in positive_scores)
        negative_count = sum(overlap_low <= score <= overlap_high for score in negative_scores)
        overlap = (
            f"The top-1 score ranges overlap from **{overlap_low:.4f} to {overlap_high:.4f}**. "
            f"The overlap interval contains {positive_count} external fashion and "
            f"{negative_count} non-fashion scores."
        )
    else:
        gap = max(min(positive_scores), min(negative_scores)) - min(max(positive_scores), max(negative_scores))
        overlap = f"The observed top-1 score ranges have no overlap and are separated by **{gap:.4f} cosine points**."
    retain_all = sum(score >= min(positive_scores) for score in negative_scores)
    reject_all = sum(score <= max(negative_scores) for score in positive_scores)
    lines = [
        overlap,
        "",
        f"- Highest-scoring non-fashion query: `{strongest_negative['filename']}` "
        f"({strongest_negative['highest_similarity']:.4f}).",
        f"- Lowest-scoring external fashion query: `{weakest_positive['filename']}` "
        f"({weakest_positive['highest_similarity']:.4f}).",
        f"- A cutoff retaining every external fashion image would accept {retain_all}/{len(non_fashion)} "
        f"non-fashion images. A cutoff rejecting every non-fashion image would reject "
        f"{reject_all}/{len(fashion)} external fashion images.",
        f"- The observed cutoff that minimizes balanced error is approximately **{best:.6f}** with "
        f"{len(fp)}/{len(non_fashion)} false positives and "
        f"{len(fn)}/{len(fashion)} false negatives. This cutoff was selected on the evaluation sample.",
    ]
    if fp:
        lines.append("- False positives at that cutoff: " + ", ".join(f"`{r['filename']}`" for r in fp) + ".")
    if fn:
        lines.append("- False negatives at that cutoff: " + ", ".join(f"`{r['filename']}`" for r in fn) + ".")
    lines.append("")
    if fp or fn:
        lines.append(
            "**Threshold assessment.** Top-1 cosine alone cannot separate both classes on this sample. "
            "A separate fashion-domain relevance classifier is worth evaluating on a larger, held-out set "
            "if the product needs to reject unrelated photos."
        )
    else:
        lines.append(
            "**Threshold assessment.** A top-1 cosine cutoff separates this sample. A larger, held-out "
            "test is needed to determine whether a cutoff is sufficient or a separate domain classifier is warranted."
        )
    lines.extend([
        "",
        "**Risks to test next.** Fashion-adjacent negatives such as packaging or brand marks resembling "
        "garment graphics could score higher than the negatives here. Low-light photos, unusual crops, "
        "and products poorly represented in the catalog could score below the external fashion examples.",
    ])
    return lines


def classify(results, manifest, threshold):
    labels = {row["filename"]: row for row in manifest}
    for row in results:
        label = labels[row["filename"]]
        expected_fashion = row["expected_classification"] == "fashion"
        accepted = row["highest_similarity"] >= threshold
        outcome = { (True, True): "TP", (True, False): "FN",
                    (False, True): "FP", (False, False): "TN" }[expected_fashion, accepted]
        row.update(
            ambiguous=label.get("ambiguous", "no"),
            difficulty_note=label.get("difficulty_note", ""),
            evaluation_threshold=threshold,
            predicted_classification="fashion" if accepted else "non-fashion",
            classification_outcome=outcome,
        )


def classification_counts(rows):
    return tuple(sum(row["classification_outcome"] == key for row in rows)
                 for key in ("TP", "TN", "FP", "FN"))


def percentage(numerator, denominator):
    return f"{numerator / denominator:.1%}" if denominator else "n/a"


def markdown_cell(value):
    return str(value).replace("|", r"\|").replace("\n", " ")


def classification_table(rows):
    return [
        "| Filename | Expected | Outcome | Top-ranked product | Top-1 cosine | Difficulty note |",
        "| --- | --- | --- | --- | ---: | --- |",
        *(
            "| " + " | ".join(map(markdown_cell, (
                row["filename"], row["expected_classification"], row["classification_outcome"],
                row["top_ranked_product"], f'{row["highest_similarity"]:.4f}',
                row["difficulty_note"] or "No image-specific note in the manifest.",
            ))) + " |"
            for row in rows
        ),
    ]


def fixed_threshold_analysis(results, threshold):
    external = [row for row in results if row["catalog_image"] == "no"]
    clear = [row for row in external if row["ambiguous"] == "no"]
    ambiguous = [row for row in external if row["ambiguous"] == "yes"]
    tp, tn, fp, fn = classification_counts(clear)
    all_tp, all_tn, all_fp, all_fn = classification_counts(external)
    errors = [row for row in clear if row["classification_outcome"] in {"FP", "FN"}]
    clear_positive = [row["highest_similarity"] for row in clear if row["expected_classification"] == "fashion"]
    clear_negative = [row["highest_similarity"] for row in clear if row["expected_classification"] == "non-fashion"]
    lines = [
        f"The preselected cutoff was **{threshold:.15g}**. A top-1 cosine score at or above it is classified "
        "as fashion. The cutoff was held fixed for all queries.",
        "",
        "### Confusion matrix",
        "",
        f"Primary metrics cover **{len(clear)} unambiguous external images**. "
        f"The **{len(ambiguous)} ambiguous images** are reported separately, and catalog controls are excluded.",
        "",
        "| Expected / predicted | Fashion | Non-fashion |",
        "| --- | ---: | ---: |",
        f"| Fashion | {tp} (TP) | {fn} (FN) |",
        f"| Non-fashion | {fp} (FP) | {tn} (TN) |",
        "",
        "| Metric | Unambiguous images | All external images using provisional labels |",
        "| --- | ---: | ---: |",
    ]
    for label, numerator, denominator, all_numerator, all_denominator in (
        ("Accuracy", tp + tn, len(clear), all_tp + all_tn, len(external)),
        ("Precision", tp, tp + fp, all_tp, all_tp + all_fp),
        ("Recall", tp, tp + fn, all_tp, all_tp + all_fn),
        ("F1 score", 2 * tp, 2 * tp + fp + fn, 2 * all_tp, 2 * all_tp + all_fp + all_fn),
        ("False-positive rate", fp, fp + tn, all_fp, all_fp + all_tn),
        ("False-negative rate", fn, fn + tp, all_fn, all_fn + all_tp),
    ):
        lines.append(f"| {label} | {percentage(numerator, denominator)} | "
                     f"{percentage(all_numerator, all_denominator)} |")
    if ambiguous:
        lines.extend(["", f"With provisional labels included, the counts are {all_tp} TP, {all_tn} TN, "
                          f"{all_fp} FP, and {all_fn} FN."])
    if clear_positive and clear_negative:
        overlap = min(max(clear_positive), max(clear_negative)) - max(min(clear_positive), min(clear_negative))
        lines.extend(["", f"The unambiguous group score ranges overlap by {overlap:.4f} cosine points."
                           if overlap > 0 else "The unambiguous group score ranges do not overlap."])

    lines.extend(["", "### Classification errors", ""])
    lines.extend(classification_table(errors) if errors else ["No unambiguous examples were misclassified."])
    lines.extend(["", "### Ambiguous queries", ""])
    lines.extend(classification_table(ambiguous) if ambiguous else ["The manifest flagged no ambiguous queries."])
    lines.extend(["", "### Recommendation", ""])
    if fp or fn:
        lines.append(
            "The fixed cutoff makes errors on held-out images. Evaluate zero-shot fashion-domain "
            "classification using the same CLIP model, and compare options on a fresh labeled set. "
            "Report trade-offs for rejecting unrelated images and retaining difficult fashion photos."
        )
    else:
        lines.append(
            "The fixed cutoff separates the unambiguous holdout. Test more fashion-adjacent negatives "
            "and difficult positives before using it in production or deciding whether zero-shot classification is needed."
        )
    return lines


def write_summary(path, results, rankings, manifest_size, failures, threshold=None):
    fashion = [r for r in results if r["expected_classification"] == "fashion" and r["catalog_image"] == "no"]
    catalog = [r for r in results if r["catalog_image"] == "yes"]
    non_fashion = [r for r in results if r["expected_classification"] == "non-fashion"]
    categories = defaultdict(list)
    for row in results:
        categories[(row["expected_classification"], row["catalog_image"], row["category"])].append(row)

    heading = "| Group | Images | Min | Max | Median | Mean | Median latency (ms) |"
    separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [
        "# Visual search relevance evaluation", "",
        f"Successfully evaluated **{len(results)}/{manifest_size}** images and recorded **{len(rankings)}** ranked products.",
        f"Fashion: **{len(fashion) + len(catalog)}** ({len(fashion)} external, {len(catalog)} catalog). "
        f"Non-fashion: **{len(non_fashion)}**. Evaluation errors: **{len(failures)}**.", "",
        "## Method", "",
        "The existing SearchEngine ran CLIP image-to-image retrieval once per measured query with a top-10 limit. "
        "An untimed warm-up search preceded the measurements. Latency covers image preprocessing, inference, "
        "ranking, and catalog lookups. Model initialization and image validation are outside the timed interval.", "",
        "## Top-1 cosine similarity by group", "", heading, separator,
        distribution_row("External fashion", fashion),
        distribution_row("Existing catalog controls", catalog),
        distribution_row("Non-fashion", non_fashion), "",
        "## Top-1 cosine similarity by category", "", heading, separator,
    ]
    lines.extend(
        distribution_row(category.replace("-", " ").title(), rows)
        for (_, _, category), rows in sorted(categories.items())
    )
    if threshold is None:
        analysis = threshold_analysis(fashion, non_fashion)
        sample_note = "The external samples are a small, curated set with three images per category."
        threshold_note = "The threshold sweep is exploratory."
    else:
        analysis = fixed_threshold_analysis(results, threshold)
        sample_note = "The holdout is a curated sample with equal category quotas. Its accuracy and precision depend on that mix."
        threshold_note = "Labels describe the main subject or the visible fashion item. Ambiguous cases carry provisional labels."
        if results:
            median_latency = statistics.median(row["search_latency_ms"] for row in results)
            lines.extend(["", f"Median search latency across all queries was **{median_latency:.1f} ms**."])
    lines.extend(["", "## Score overlap and threshold assessment", "", *analysis])
    lines.extend([
        "", "## Limitations", "",
        f"- {sample_note} Framing, background, lighting, and product coverage can affect results.",
        "- Each query has a fashion-domain label. The study has no per-product relevance judgments, so these results do not establish ranking quality or retrieval precision.",
        "- Latencies are single-run local observations after warm-up. The engine always returns nearest neighbors for these queries, including non-fashion inputs.",
        f"- {threshold_note} The application still uses its existing search behavior.",
    ])
    if failures:
        lines.extend(["", "## Evaluation errors", ""])
        lines.extend(f"- `{filename}`: {error}" for filename, error in failures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=EVALUATION_DIR / "visual")
    parser.add_argument("--manifest", type=Path, default=EVALUATION_DIR / "visual_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--threshold", type=float, help="Fixed evaluation-only top-1 cosine cutoff.")
    args = parser.parse_args(argv)
    if args.threshold is not None:
        if not -1 <= args.threshold <= 1:
            parser.error("--threshold must be a finite cosine score between -1 and 1.")
        if args.output_dir.resolve() == EVALUATION_DIR.resolve():
            parser.error("Use a separate --output-dir for fixed-threshold runs to preserve calibration reports.")

    rows = load_manifest(args.manifest)
    engine = SearchEngine()
    results, rankings, failures = evaluate(engine, rows, args.input_dir)
    fields = RESULT_FIELDS
    if args.threshold is not None:
        classify(results, rows, args.threshold)
        fields += FIXED_THRESHOLD_FIELDS

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "visual_results.csv", fields, results)
    write_csv(args.output_dir / "visual_rankings.csv", RANK_FIELDS, rankings)
    write_summary(
        args.output_dir / "visual_summary.md", results, rankings, len(rows), failures, args.threshold,
    )
    print(f"Evaluated {len(results)}/{len(rows)} images. Reports saved in {args.output_dir}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
