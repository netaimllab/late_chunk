"""Recompute the manuscript results from complete saved observations."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from chunkprivacy.attacks import infer
from chunkprivacy.ablations import subset, transformed
from chunkprivacy.retrieval import AttackInput
from chunkprivacy.evaluate import hierarchical_interval, summarize, template_mean
from chunkprivacy.utility import utility_ids, external_bootstrap
from researchkit.metrics import retrieval_metrics
from generate_corpus import corpora
from verify import check


def load(name):
    with gzip.open(ROOT / "data/observations" / name, "rt") as f:
        return [json.loads(line) for line in f] if ".jsonl." in name else json.load(f)


def compare(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, path + "/" + str(i))
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (path, a, b)
    else:
        assert a == b, (path, a, b)


def replay(rows, families, seed, wording=None, ablation=False):
    for row in rows:
        family = families[row["family_id"]]
        if ablation:
            family = transformed(family, row["position"], row["filler_tokens"], row["no_anaphora"])
        world = next(w for w in family["worlds"] if w["world_id"] == row["world_id"])
        assert row["truth"] == world["truth"]
        for key in ["design_id", "instance_id", "type", "template_id"]:
            assert row[key] == family[key]
        public = AttackInput(
            family["family_id"],
            tuple(family["candidates"]),
            family["target_id"],
            family["public_target_text"],
        )
        assert len(row["transcript"]) == 4
        pending = iter(row["transcript"])

        def search(query, target_id=None):
            saved = next(pending)
            assert query == saved["query"]
            response = saved["response"]
            if row["tier"] == "O":
                assert target_id == public.target_id
                assert len(response) == 1 and response[0]["id"] == target_id
                assert set(response[0]) == {"id", "score"}
            else:
                assert len(response) <= 10
                assert [r["rank"] for r in response] == list(range(1, len(response) + 1))
                expected = {"id", "text", "rank"} | ({"score"} if row["tier"] == "S" else set())
                assert all(set(r) == expected and r["id"] not in {"h0", "gap"} for r in response)
            return response

        got = infer(public, search, row["tier"], row.get("wording", wording), seed)
        for key in ["prediction", "values", "tied_best", "target_absent_queries", "transcript"]:
            assert got[key] == row[key], (row["family_id"], key)
        assert row["correct"] == int(got["prediction"] == world["truth"])
        assert next(pending, None) is None
    return len(rows)


def table(out, name, rows):
    with (out / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expected_csv(name, rows, keys):
    expected = pd.read_csv(ROOT / "expected" / name).sort_values(keys).reset_index(drop=True)
    actual = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    assert set(expected.columns) <= set(actual.columns), name
    assert len(expected) == len(actual), name
    for col in expected:
        if pd.api.types.is_numeric_dtype(expected[col]):
            np.testing.assert_allclose(
                actual[col], expected[col], rtol=0, atol=1e-12, err_msg=name + ":" + col
            )
        elif col not in {"ci95"}:
            assert actual[col].astype(str).tolist() == expected[col].astype(str).tolist(), (
                name,
                col,
            )


def figure2(out, attacks, families):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fid = sorted(f for f, v in families.items() if v["split"] == "test" and v["type"] == "month")[0]
    family = families[fid]
    selected = [r for r in attacks if r["family_id"] == fid and r["model"] == "small"]
    values = []
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for ax, (method, tier, title) in zip(
        axes,
        [
            ("L", "S", "L: target score"),
            ("L", "R", "L: target rank"),
            ("P", "S", "P: target score"),
        ],
        strict=True,
    ):
        rows = sorted(
            (r for r in selected if r["method"] == method and r["tier"] == tier),
            key=lambda r: r["truth"],
        )
        matrix = np.full((4, 4), np.nan)
        for row in rows:
            assert row["same_token_length"]
            for j, transcript in enumerate(row["transcript"]):
                target = next(
                    (v for v in transcript["response"] if v["id"] == family["target_id"]), None
                )
                value = target["score" if tier == "S" else "rank"] if target else None
                if value is not None:
                    matrix[row["truth"], j] = value
                values.append(
                    {
                        "family": fid,
                        "method": method,
                        "tier": tier,
                        "truth": row["truth"],
                        "candidate": j,
                        "value": value,
                        "chosen": j == row["prediction"],
                    }
                )
            ax.add_patch(
                plt.Rectangle(
                    (row["prediction"] - 0.5, row["truth"] - 0.5),
                    1,
                    1,
                    fill=False,
                    ec="black",
                    lw=1.5,
                )
            )
        ax.imshow(
            matrix,
            cmap="Blues" if tier == "S" else "Greys",
            vmin=0.8 if tier == "S" else 0,
            vmax=1 if tier == "S" else 10,
        )
        for (i, j), x in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                "NA" if np.isnan(x) else (f"{x:.2f}" if tier == "S" else str(int(x))),
                ha="center",
                va="center",
                fontsize=8,
            )
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xticklabels=range(1, 5),
            yticklabels=range(1, 5),
            title=title,
            xlabel="Query candidate",
        )
    axes[0].set_ylabel("Private value")
    fig.text(
        0.5,
        0.02,
        "; ".join(f"{i + 1}: {c}" for i, c in enumerate(family["candidates"])),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "figure-2.png", dpi=200)
    plt.close(fig)
    table(out, "figure-2.csv", values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New result directory")
    args = parser.parse_args()
    started = time.perf_counter()
    count = check()
    _, corpus = corpora()
    families = {f["family_id"]: f for f in corpus}
    config = json.loads((ROOT / "config.json").read_text())
    seed, reps = config["seed"], config["bootstrap_repetitions"]
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    report = {
        "files_verified": count,
        "corpus_families": len(corpus),
        "changed_families": 15,
        "changed_worlds": 60,
        "model_inference_repeated": False,
        "bootstrap_replicates": reps,
        "held_out_designs": 4,
    }
    attacks = load("security.jsonl.gz")
    assert len(attacks) == 5760
    report["security_records_replayed"] = replay(attacks, families, seed)
    lengths = json.loads((ROOT / "data/length-controls.json").read_text())
    for row in attacks:
        assert (
            row["same_token_length"] == lengths[row["model"]][row["family_id"]]["same_token_length"]
        )
    frame = pd.DataFrame(attacks)
    assert set(frame[frame.same_token_length]["type"]) == {"month"}
    assert frame[frame.same_token_length].family_id.nunique() == 20
    for _, g in frame[frame.method.isin(["E", "P", "A"])].groupby(
        ["model", "method", "tier", "family_id"]
    ):
        assert len(g) == 4 and g.correct.sum() == 1 and g.prediction.nunique() == 1
    print("Corpus and held-out decisions verified; computing descriptive intervals.", flush=True)
    sec = summarize(attacks, reps, seed)
    sec.pop("caution")
    compare(sec, json.loads((ROOT / "expected/security-summary.json").read_text()))
    (out / "security-summary.json").write_text(json.dumps(sec, indent=2) + "\n")
    rows = []
    for model in ["small", "base"]:
        for tier in ["S", "R", "O"]:
            r = sec["results"][f"{model}/L/{tier}"]["accuracy"]
            rows.append(
                {
                    "model": model,
                    "endpoint": tier,
                    "accuracy_percent": 100 * r["estimate"],
                    "lower_percent": 100 * r["ci95"][0],
                    "upper_percent": 100 * r["ci95"][1],
                }
            )
    table(out, "table-2.csv", rows)
    type_rows = [
        {"model": m, "attribute": t, "accuracy": template_mean(g)}
        for (m, t), g in frame[(frame.method == "L") & (frame.tier == "S")].groupby(
            ["model", "type"]
        )
    ]
    table(out, "attribute-accuracy.csv", type_rows)
    calibration = load("query-selection.jsonl.gz")
    assert len(calibration) == 480
    report["query_selection_records_replayed"] = replay(calibration, families, seed)
    scores = {w: template_mean(g) for w, g in pd.DataFrame(calibration).groupby("wording")}
    rule = json.loads((ROOT / "data/query-rule.json").read_text())
    compare(scores, rule["development_macro_template_S_accuracy"])
    assert rule["wording"] == ("context" if scores["context"] > scores["bare"] else "bare")
    bare = load("candidate-only.jsonl.gz")
    assert len(bare) == 5760
    report["candidate_only_records_replayed"] = replay(bare, families, seed)
    rows = [
        dict(model=m, method=method, tier=t, accuracy=template_mean(g))
        for (m, method, t), g in pd.DataFrame(bare).groupby(["model", "method", "tier"])
    ]
    expected_csv("bare-query-test.csv", rows, ["model", "method", "tier"])
    table(out, "candidate-only.csv", rows)
    del bare, calibration
    utility = load("controlled-utility.json.gz")
    assert len(utility["rows"]) == 3840
    for row in utility["rows"]:
        family = families[row["family_id"]]
        q = family["utility"][row["query_id"]]
        relevant = list(utility_ids(family["family_id"], q["relevant"]).values())
        metrics = retrieval_metrics([r["id"] for r in row["response"]], relevant)
        compare(metrics, {k: row[k] for k in metrics})
    uf = pd.DataFrame(utility["rows"])
    averages = {
        f"{m}/{method}": {k: template_mean(g, k) for k in ["recall10", "ndcg10"]}
        for (m, method), g in uf.groupby(["model", "method"])
    }
    compare(averages, utility["macro_template"])
    intervals = {
        f"{m}/{method}/{metric}": hierarchical_interval(g, metric, reps, seed)
        for (m, method), g in uf.groupby(["model", "method"])
        for metric in ["recall10", "ndcg10"]
    }
    compare(intervals, utility["intervals"])
    paired = {}
    for model, g in uf.groupby("model"):
        for metric in ["recall10", "ndcg10"]:
            pivot = g.pivot(
                index=[
                    "design_id",
                    "instance_id",
                    "type",
                    "template_id",
                    "family_id",
                    "world_id",
                    "query_id",
                ],
                columns="method",
                values=metric,
            )
            for method in ["E", "P", "A"]:
                d = (pivot[method] - pivot.L).rename("difference").reset_index()
                paired[f"{model}/{method}-L/{metric}"] = hierarchical_interval(
                    d, "difference", reps, seed
                )
    compare(paired, utility["paired_differences"])
    (out / "utility-intervals.json").write_text(
        json.dumps({"intervals": intervals, "paired": paired}, indent=2) + "\n"
    )
    table(
        out,
        "table-3.csv",
        [
            dict(model=m, method=method, **averages[f"{m}/{method}"])
            for m in ["small", "base"]
            for method in ["E", "L", "P", "A"]
        ],
    )
    report["controlled_utility_rankings_verified"] = len(uf)
    del uf, utility
    qrels = {}
    with (ROOT / "data/scifact-qrels.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    external = load("scifact.json.gz")
    sc = []
    for model, value in external["results"].items():
        for row in value["rows"]:
            metrics = retrieval_metrics(row["ranked_document_ids"], qrels[row["query_id"]])
            compare(metrics, {k: row[k] for k in metrics})
        computed = external_bootstrap(value["rows"], qrels, reps, seed)
        compare(computed, value["paired_bootstrap"])
        assert computed["clusters"] == 247 and computed["queries"] == 300
        for metric, v in computed["metrics"].items():
            sc.append(
                dict(
                    model=model,
                    metric=metric,
                    E=v["E"],
                    L=v["L"],
                    difference=v["L_minus_E"],
                    low=v["paired_ci95"][0],
                    high=v["paired_ci95"][1],
                )
            )
    expected_csv("scifact.csv", sc, ["model", "metric"])
    table(out, "table-4.csv", sc)
    report["scifact_rankings_verified"] = sum(len(v["rows"]) for v in external["results"].values())
    cost = load("indexing-cost.json.gz")
    costs = []
    for row in cost["results"]:
        ms = [1000 * t for t in row["index_construction_seconds"]]
        costs.append(
            dict(
                model=row["model"],
                method=row["method"],
                calls=row["encoder_invocations_per_build"],
                vectors=row["stored_vectors"],
                median_ms=statistics.median(ms),
                min_ms=min(ms),
                max_ms=max(ms),
            )
        )
    expected_csv("cost.csv", costs, ["model", "method"])
    table(out, "table-5.csv", costs)
    print(
        "Security and retrieval summaries verified; replaying sensitivity observations.", flush=True
    )
    ablations = load("sensitivity.json.gz")
    abrows = []
    keys = [
        "model",
        "method",
        "tier",
        "position",
        "filler_tokens",
        "no_anaphora",
        "constant_hidden_context",
        "distractor_count",
    ]
    report["sensitivity_records_replayed"] = 0
    for model, value in ablations.items():
        selected = [
            f["family_id"] for f in subset([f for f in corpus if f["split"] == "dev"], seed=seed)
        ]
        assert selected == value["selected_development_families"]
        assert len(value["rows"]) == 15840
        report["sensitivity_records_replayed"] += replay(
            value["rows"], families, seed, rule["wording"], ablation=True
        )
        for group, g in pd.DataFrame(value["rows"]).groupby(keys):
            abrows.append(dict(zip(keys, group, strict=True), accuracy=template_mean(g)))
    expected_csv("ablations-all.csv", abrows, keys)
    assert all(r["accuracy"] == 0.25 for r in abrows if r["method"] in ["E", "P", "A", "L0"])
    table(out, "sensitivity-all.csv", abrows)
    table(
        out,
        "table-6.csv",
        [r for r in abrows if r["method"] in ["L", "L0"] and r["tier"] in ["S", "R"]],
    )
    report["sensitivity_conditions_verified"] = len(abrows)
    figure2(out, attacks, families)
    report["elapsed_seconds"] = time.perf_counter() - started
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
