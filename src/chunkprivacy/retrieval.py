"""Capability boundary: the attack sees only endpoint responses and public inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class AttackInput:
    family_id: str
    candidates: tuple[str, ...]
    target_id: str
    public_target_text: str


def endpoint(ids, texts, vectors, encode_query: Callable, tier="S", topk=10, query_budget=None):
    if (
        len(ids) != len(set(ids))
        or vectors.ndim != 2
        or vectors.shape[0] != len(ids)
        or len(texts) != len(ids)
    ):
        raise ValueError("Invalid authorized search index")
    if tier not in ("S", "R", "O"):
        raise ValueError("Invalid endpoint tier")
    # Inputs contain only authorized documents, so filtering happens BEFORE top-k.
    calls = 0

    def search(query, target_id=None):
        nonlocal calls
        if query_budget is not None and calls >= query_budget:
            raise RuntimeError("Endpoint query budget exhausted")
        calls += 1
        scores = np.clip(vectors @ encode_query(query), -1.0, 1.0)
        if tier == "O":
            if target_id not in ids:
                raise PermissionError("Direct lookup must still authorize the document")
            return [{"id": target_id, "score": float(scores[ids.index(target_id)])}]
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:topk]
        rows = [{"id": ids[i], "text": texts[i], "rank": rank + 1} for rank, i in enumerate(order)]
        if tier == "S":
            for row, i in zip(rows, order, strict=True):
                row["score"] = float(scores[i])
        return rows

    return search
