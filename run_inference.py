"""Rerun a selected experiment using the pinned CPU encoders."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["security", "utility", "sensitivity", "scifact", "cost"], required=True
    )
    parser.add_argument("--model", choices=["small", "base"], required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; existing output is never overwritten",
    )
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache")
    parser.add_argument(
        "--family",
        help="One family for a security or utility run; omit for all held-out families",
    )
    args = parser.parse_args()
    if args.family and args.stage not in ["security", "utility"]:
        parser.error("--family applies only to security or utility")
    config = json.loads((ROOT / "config.json").read_text())
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[key] = str(config["retrieval_threads"])
    import numpy as np
    from chunkprivacy.encoders import Encoder, document, joined
    from chunkprivacy.attacks import infer
    from chunkprivacy.data import distractors, PUBLIC
    from chunkprivacy.retrieval import AttackInput, endpoint
    from chunkprivacy.utility import controlled_foils, utility_ids, acquire, external_evaluate
    from researchkit.metrics import retrieval_metrics
    from researchkit.io import read_jsonl, write_json, write_jsonl
    from itertools import groupby
    from verify import check

    check()
    corpus = list(read_jsonl(ROOT / "data/corpus-corrected.jsonl"))
    if args.family and args.family not in {f["family_id"] for f in corpus if f["split"] == "test"}:
        parser.error("--family must name a supplied held-out family")
    args.output.mkdir(parents=True, exist_ok=False)
    # Model weights are pinned. Numerical comparisons concern CPU float32.
    encoder = Encoder(
        args.model,
        args.cache,
        max_tokens=config["max_tokens"],
        threads=config["threads"],
        cache=args.stage != "cost",
    )
    started = time.perf_counter()
    if args.stage == "security" or args.stage == "utility":
        rows = []
        selected = [
            f
            for f in corpus
            if f["split"] == "test" and (not args.family or f["family_id"] == args.family)
        ]
        for family in selected:
            public = AttackInput(
                family["family_id"],
                tuple(family["candidates"]),
                family["target_id"],
                family["public_target_text"],
            )
            background = distractors(family, config["distractors"])
            dv = np.stack(
                [encoder.encode([c], method="standalone-distractor")[0] for c in background]
            )
            token_counts = [
                len(encoder.tokenizer(joined(w["chunks"])[0], add_special_tokens=True)["input_ids"])
                for w in family["worlds"]
            ]
            for method in config["methods"]:
                invariants = []
                foil_chunks, foil_vectors = [], []
                if args.stage == "utility":
                    for foil in controlled_foils(family):
                        vectors = document(encoder, foil, method)
                        foil_chunks.extend(foil)
                        foil_vectors.extend(vectors[c["id"]] for c in foil)
                for world in family["worlds"]:
                    visible = [c for c in world["chunks"] if family["principal"] in c["readers"]]
                    vectors = document(encoder, world["chunks"], method, family["principal"])
                    matrix = np.stack([vectors[c["id"]] for c in visible])
                    invariants.append(matrix)
                    identifiers = dict(
                        model=args.model,
                        method=method,
                        family_id=family["family_id"],
                        template_id=family["template_id"],
                        design_id=family["design_id"],
                        instance_id=family["instance_id"],
                        type=family["type"],
                        world_id=world["world_id"],
                    )
                    if args.stage == "security":
                        indexed = visible + background
                        matrix = np.concatenate([matrix, dv])
                        for wording in ["context", "bare"]:
                            for tier in config["tiers"]:
                                search = endpoint(
                                    [c["id"] for c in indexed],
                                    [c["text"] for c in indexed],
                                    matrix,
                                    encoder.query,
                                    tier,
                                    config["top_k"],
                                    query_budget=4,
                                )
                                response = infer(public, search, tier, wording, config["seed"])
                                rows.append(
                                    dict(
                                        **identifiers,
                                        tier=tier,
                                        wording=wording,
                                        truth=world["truth"],
                                        same_token_length=len(set(token_counts)) == 1,
                                        correct=int(response["prediction"] == world["truth"]),
                                        **response,
                                    )
                                )
                    else:
                        indexed = visible + background + foil_chunks
                        mapping = utility_ids(family["family_id"], [c["id"] for c in indexed])
                        matrix = np.concatenate([matrix, dv, foil_vectors])
                        search = endpoint(
                            [mapping[c["id"]] for c in indexed],
                            [c["text"] for c in indexed],
                            matrix,
                            encoder.query,
                            "R",
                            config["top_k"],
                        )
                        for qid, q in enumerate(family["utility"]):
                            response = search(q["query"])
                            rows.append(
                                dict(
                                    **identifiers,
                                    query_id=qid,
                                    response=response,
                                    **retrieval_metrics(
                                        [r["id"] for r in response],
                                        [mapping[k] for k in q["relevant"]],
                                    ),
                                )
                            )
                if method in ["E", "P", "A"]:
                    assert all(
                        np.allclose(v, invariants[0], rtol=1e-5, atol=1e-6) for v in invariants
                    )
            print(f"Completed {family['family_id']}", flush=True)
        write_jsonl(args.output / f"{args.stage}.jsonl", rows)
    elif args.stage == "sensitivity":
        from chunkprivacy.ablations import evaluate

        result = evaluate(
            [f for f in corpus if f["split"] == "dev"],
            encoder,
            args.model,
            "context",
            config["distractors"],
            config["seed"],
        )
        write_json(args.output / "sensitivity.json", result)
    elif args.stage == "scifact":
        acquire(args.output)
        write_json(args.output / "scifact.json", external_evaluate(args.output, encoder))
    else:
        fixture = [dict(c) for c in corpus[0]["worlds"][0]["chunks"]]
        fixture[1]["readers"] = ["owner", "reader", "team_a"]
        fixture[3]["readers"] = ["owner", "reader", "team_b"]
        results = []
        for method in config["methods"]:
            if method == "E":
                groups = [[c] for c in fixture]
            elif method == "L":
                groups = [fixture]
            elif method == "P":
                groups = [list(g) for _, g in groupby(fixture, lambda c: tuple(c["readers"]))]
            else:
                groups = [[c for c in fixture if principal in c["readers"]] for principal in PUBLIC]
            for group in groups:
                encoder.encode(group, method=method)
            times = []
            for _ in range(5):
                start = time.perf_counter()
                for group in groups:
                    encoder.encode(group, method=method)
                times.append(time.perf_counter() - start)
            results.append(
                dict(
                    model=args.model,
                    method=method,
                    index_construction_seconds=times,
                    encoder_invocations_per_build=len(groups),
                    stored_vectors=sum(map(len, groups)),
                )
            )
        write_json(args.output / "indexing-cost.json", {"fixture": fixture, "results": results})
    write_json(
        args.output / "execution.json",
        {
            "stage": args.stage,
            "model": args.model,
            "family": args.family,
            "elapsed_seconds": time.perf_counter() - started,
            "encoder": encoder.profile(),
        },
    )


if __name__ == "__main__":
    main()
