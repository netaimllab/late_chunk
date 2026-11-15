from __future__ import annotations

import random

from researchkit.io import stable_seed


def infer(public, search, tier, wording, seed=20261010):
    if wording not in ("bare", "context"):
        raise ValueError("Unknown frozen query wording")
    transcript, values = [], []
    for candidate in public.candidates:
        query = (
            candidate
            if wording == "bare"
            else f"Information about {candidate}. {public.public_target_text}"
        )
        response = search(query, public.target_id) if tier == "O" else search(query)
        target = next((r for r in response if r["id"] == public.target_id), None)
        value = (
            (target["score"] if target else -2.0)
            if tier in ("S", "O")
            else (1 / target["rank"] if target else 0.0)
        )
        transcript.append({"query": query, "response": response})
        values.append(value)
    # Coupled across worlds, methods and endpoint tiers, with no truth in the key.
    order = list(range(4))
    random.Random(stable_seed(seed, public.family_id, "candidate-tie-v1")).shuffle(order)
    best = max(values)
    prediction = next(i for i in order if values[i] == best)
    return {
        "prediction": prediction,
        "values": values,
        "tied_best": values.count(best) > 1,
        "target_absent_queries": sum(
            not any(r["id"] == public.target_id for r in t["response"]) for t in transcript
        ),
        "transcript": transcript,
    }
