"""Regenerate the synthetic corpus and apply the recorded wording correction."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from chunkprivacy.data import generate, validate
from researchkit.io import read_jsonl, write_jsonl


def corpora():
    config = json.loads((ROOT / "config.json").read_text())
    original = generate(
        config["seed"], config["families_per_template"], config["templates_per_type"]
    )
    # Materialize independent JSON records before applying per-world edits.
    corrected = json.loads(json.dumps(original))
    ledger = json.loads((ROOT / "data/correction.json").read_text())
    changes = {(r["family_id"], r["world_id"]): r for r in ledger["changes"]}
    applied = 0
    for family in corrected:
        for world in family["worlds"]:
            key = (family["family_id"], world["world_id"])
            if key in changes:
                chunk = next(c for c in world["chunks"] if c["id"] == "p1")
                if chunk["text"] != changes[key]["old"]:
                    raise ValueError("Correction does not match the generated original sentence")
                chunk["text"] = changes[key]["new"]
                applied += 1
    assert applied == 60 and len({k[0] for k in changes}) == 15
    for name, values in [("original", original), ("corrected", corrected)]:
        validate(values, expected_main=True)
        assert values == list(read_jsonl(ROOT / f"data/corpus-{name}.jsonl")), name
    return original, corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    original, corrected = corpora()
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "corpus-original.jsonl", original)
    write_jsonl(args.output / "corpus-corrected.jsonl", corrected)
    print("Verified 240 families, 960 worlds, and the 15-family/60-world correction.")


if __name__ == "__main__":
    main()
