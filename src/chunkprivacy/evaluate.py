from __future__ import annotations

import numpy as np
import pandas as pd


def template_mean(frame, column="correct"):
    # Equal worlds/family, equal families/template, then macro templates.
    return float(
        frame.groupby(["type", "template_id", "family_id"])[column]
        .mean()
        .groupby(level=[0, 1])
        .mean()
        .mean()
    )


def hierarchical_interval(frame, value="correct", repetitions=2000, seed=20261010):
    frame = frame.copy()
    # Legacy fixtures without shared designs use their template as the design.
    # Production observations always carry design_id and instance_id explicitly.
    if "design_id" not in frame:
        frame["design_id"] = frame.template_id
        frame["instance_id"] = frame.family_id
    family = (
        frame.groupby(["design_id", "instance_id", "type", "template_id", "family_id"])[value]
        .mean()
        .reset_index()
    )
    designs = []
    for _, group in family.groupby("design_id", sort=True):
        # Keep attribute types together when a structural design/public instance
        # is reused. Forty-eight type/template IDs are NOT 48 independent designs.
        designs.append(group.groupby("instance_id", sort=True)[value].mean().to_numpy())
    if not designs:
        raise ValueError("No observations")
    rng, values = np.random.default_rng(seed), []
    for _ in range(repetitions):
        sampled = []
        for index in rng.integers(0, len(designs), len(designs)):
            instances = designs[index]
            sampled.append(
                float(np.mean(instances[rng.integers(0, len(instances), len(instances))]))
            )
        values.append(float(np.mean(sampled)))
    return {
        "estimate": template_mean(frame, value),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "replicates": repetitions,
        "templates": int(family.template_id.nunique()),
        "families": len(family),
        "structural_designs": len(designs),
        "unit": "structural design then public instance; paired attribute types and worlds stay together",
        "limited_independent_designs": len(designs) < 10,
    }


def summarize(rows, repetitions=2000, seed=20261010):
    frame = pd.DataFrame(rows)
    results, paired = {}, {}
    for (model, method, tier), group in frame.groupby(["model", "method", "tier"]):
        name = f"{model}/{method}/{tier}"
        results[name] = {
            "accuracy": hierarchical_interval(group, repetitions=repetitions, seed=seed),
            "per_type": {kind: template_mean(g) for kind, g in group.groupby("type")},
            "per_template": group.groupby("template_id").correct.mean().to_dict(),
            "tie_rate": float(group.tied_best.mean()),
            "target_absence_query_rate": float(group.target_absent_queries.mean() / 4),
            "diagnostic_only": tier == "O",
            "worlds": len(group),
        }
        if "same_token_length" in group:
            equal = group[group.same_token_length]
            results[name]["same_token_length_subset"] = (
                hierarchical_interval(equal, repetitions=repetitions, seed=seed)
                if len(equal)
                else {"families": 0, "estimate": None, "ci95": None}
            )
    for (model, tier), subset in frame.groupby(["model", "tier"]):
        keys = [
            c
            for c in ["design_id", "instance_id", "type", "template_id", "family_id", "world_id"]
            if c in subset
        ]
        pivot = subset.pivot(index=keys, columns="method", values="correct")
        for method in ["E", "P", "A"]:
            if method in pivot and "L" in pivot:
                difference = (pivot[method] - pivot.L).rename("difference").reset_index()
                paired[f"{model}/{method}-L/{tier}"] = hierarchical_interval(
                    difference, "difference", repetitions, seed
                )
    return {
        "results": results,
        "paired_accuracy_differences": paired,
        "caution": "Synthetic event documents and known candidates limit generalization; O is diagnostic.",
    }
