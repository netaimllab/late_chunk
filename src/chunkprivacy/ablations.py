from __future__ import annotations

import copy
import random

import numpy as np

from researchkit.io import stable_seed
from .attacks import infer
from .data import PLAIN, distractors
from .encoders import document
from .retrieval import AttackInput, endpoint


def subset(families, size=40, seed=20261010):
    # Fixed round-robin stratification across attribute types, without attack outcomes.
    pools = {}
    for f in families:
        pools.setdefault(f["type"], []).append(f)
    for key, values in pools.items():
        values.sort(key=lambda f: f["family_id"])
        random.Random(stable_seed(seed, "ablation", key)).shuffle(values)
    selected = []
    while len(selected) < min(size, len(families)):
        for kind in sorted(pools):
            if pools[kind] and len(selected) < size:
                selected.append(pools[kind].pop())
    return selected


def transformed(family, position="before", gap=0, no_anaphora=False):
    result = copy.deepcopy(family)
    if no_anaphora:
        result["public_target_text"] = PLAIN[family["type"]]
    for world in result["worlds"]:
        hidden = next(c for c in world["chunks"] if c["id"] == "h0")
        public = [c for c in world["chunks"] if c["id"] != "h0"]
        if no_anaphora:
            next(c for c in public if c["id"] == "p2")["text"] = result["public_target_text"]
        target = next(i for i, c in enumerate(public) if c["id"] == "p2")
        # 'routine' is checked to be one token for the selected tokenizer. All
        # filler remains forbidden; authorized views stay identical across worlds.
        filler = (
            [{"id": "gap", "text": " ".join(["routine"] * gap), "readers": ["owner"]}]
            if gap
            else []
        )
        insert = [hidden] + filler if position == "before" else filler + [hidden]
        point = target if position == "before" else target + 1
        world["chunks"] = public[:point] + insert + public[point:]
    return result


def evaluate(corpus, encoder, model, wording, distractor_count=200, seed=20261010):
    selected = subset(corpus, seed=seed)
    conditions = [
        (pos, gap, False, False, distractor_count)
        for pos in ["before", "after"]
        for gap in [0, 128, 512]
    ]
    conditions += [
        ("before", 0, True, False, distractor_count),
        ("before", 0, False, True, distractor_count),
        ("before", 0, False, False, 1000),
    ]
    rows = []
    for position, gap, no_anaphora, constant, count in conditions:
        if (
            gap
            and len(
                encoder.tokenizer(" ".join(["routine"] * gap), add_special_tokens=False)[
                    "input_ids"
                ]
            )
            != gap
        ):
            raise ValueError("Filler token count differs from specified gap")
        for original in selected:
            family = transformed(original, position, gap, no_anaphora)
            background = distractors(family, count)
            dv = np.stack(
                [encoder.encode([c], method="standalone-distractor")[0] for c in background]
            )
            public = AttackInput(
                family["family_id"], tuple(family["candidates"]), "p2", family["public_target_text"]
            )
            for method in ["L0"] if constant else ["E", "L", "P", "A"]:
                coupled = []
                for world in family["worlds"]:
                    visible = [c for c in world["chunks"] if "reader" in c["readers"]]
                    vectors = document(encoder, world["chunks"], method)
                    coupled.append(np.stack([vectors[c["id"]] for c in visible]))
                    allrows = visible + background
                    matrix = np.concatenate([coupled[-1], dv])
                    for tier in ["S", "R", "O"]:
                        search = endpoint(
                            [c["id"] for c in allrows],
                            [c["text"] for c in allrows],
                            matrix,
                            encoder.query,
                            tier,
                            query_budget=4,
                        )
                        response = infer(public, search, tier, wording, seed)
                        rows.append(
                            {
                                "model": model,
                                "method": method,
                                "tier": tier,
                                "family_id": family["family_id"],
                                "template_id": family["template_id"],
                                "design_id": family["design_id"],
                                "instance_id": family["instance_id"],
                                "type": family["type"],
                                "world_id": world["world_id"],
                                "position": position,
                                "filler_tokens": gap,
                                "no_anaphora": no_anaphora,
                                "distractor_count": count,
                                "constant_hidden_context": constant,
                                "correct": int(response["prediction"] == world["truth"]),
                                "truth": world["truth"],
                                **response,
                            }
                        )
                if method in ["E", "P", "A", "L0"] and not all(
                    np.allclose(v, coupled[0], rtol=1e-5, atol=1e-6) for v in coupled
                ):
                    raise ValueError("Ablation hidden-input invariance failed")
    return {
        "selected_development_families": [f["family_id"] for f in selected],
        "rows": rows,
        "gap_definition": "exact filler token count; excludes secret-sentence and target-internal tokens",
        "exploratory": True,
        "selection_uses_test_outcomes": False,
    }
