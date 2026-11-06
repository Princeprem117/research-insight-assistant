
import csv
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path("evaluation_results/retrieval_comparison.csv")
OUTPUT_FILE = Path("evaluation_results/retrieval_metrics.csv")


def calculate_metrics():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {INPUT_FILE}"
        )

    groups = defaultdict(list)

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            relevance = row["relevant"].strip().lower()

            if relevance not in ("yes", "no"):
                raise ValueError(
                    "Please label every retrieved chunk as yes or no. "
                    f"Missing/invalid label for: {row['question']}"
                )

            key = (
                row["question"],
                row["search_type"],
                int(row["k"]),
            )

            groups[key].append({
                "rank": int(row["rank"]),
                "relevant": relevance == "yes",
            })

    results = []

    for (question, search_type, k), chunks in groups.items():
        chunks.sort(key=lambda item: item["rank"])

        relevant_count = sum(
            chunk["relevant"] for chunk in chunks
        )

        # Precision@K: relevant results divided by requested K.
        precision = relevant_count / k if k else 0

        # Reciprocal rank: inverse rank of the first relevant chunk.
        first_relevant = next(
            (
                chunk["rank"]
                for chunk in chunks
                if chunk["relevant"]
            ),
            None,
        )

        reciprocal_rank = (
            1 / first_relevant
            if first_relevant is not None
            else 0
        )

        results.append({
            "question": question,
            "search_type": search_type,
            "k": k,
            "relevant_chunks": relevant_count,
            "precision_at_k": round(precision, 4),
            "reciprocal_rank": round(reciprocal_rank, 4),
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question",
        "search_type",
        "k",
        "relevant_chunks",
        "precision_at_k",
        "reciprocal_rank",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    # Aggregate results across all test questions.
    summary = defaultdict(list)

    for result in results:
        key = (result["search_type"], result["k"])
        summary[key].append(result)

    print("\nRETRIEVAL EVALUATION RESULTS")
    print("-" * 55)

    for (search_type, k), values in sorted(summary.items()):
        avg_precision = sum(
            item["precision_at_k"] for item in values
        ) / len(values)

        avg_mrr = sum(
            item["reciprocal_rank"] for item in values
        ) / len(values)

        print(f"\nStrategy: {search_type.upper()}, K={k}")
        print(f"Average Precision@{k}: {avg_precision:.4f}")
        print(f"Mean Reciprocal Rank: {avg_mrr:.4f}")

    print(f"\nDetailed metrics saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    calculate_metrics()
