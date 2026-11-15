import math


def retrieval_metrics(ranked, relevant, k=10):
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Query has no judged relevant authorized item")
    if len(ranked) != len(set(ranked)):
        raise ValueError("Retrieval metrics require deduplicated IDs")
    top = ranked[:k]
    hits = [int(i in relevant) for i in top]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        "recall10": sum(hits) / len(relevant),
        "ndcg10": sum(h / math.log2(i + 2) for i, h in enumerate(hits)) / ideal,
    }
