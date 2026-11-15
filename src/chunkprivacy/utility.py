from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import re
import zipfile

import numpy as np

from researchkit.io import digest, download, read_jsonl, write_json
from researchkit.metrics import retrieval_metrics
from .data import PUBLIC
from .encoders import document, joined

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def controlled_foils(family):
    """Ten public competing events; identical boundaries under every method.

    Deadline chunks deliberately need their preceding event name to distinguish
    equally relevant-looking registrations. These are utility-only negatives.
    """
    world = family["worlds"][0]
    public = {c["id"]: c for c in world["chunks"]}
    source_name = family["utility_context"]["event_name"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    source_day = family["utility_context"]["deadline_day"]
    for i, name in enumerate(
        ["Aster", "Bramble", "Clover", "Dahlia", "Elm", "Fern", "Gorse", "Hazel", "Iris", "Laurel"]
    ):
        day = [d for d in days if d != source_day][i % 4]
        yield [
            {
                "id": f"foil-{i}-intro",
                "text": public["p0"]["text"].replace(source_name, name),
                "readers": PUBLIC,
            },
            {
                "id": f"foil-{i}-deadline",
                "text": public["p1"]["text"].replace(source_day, day),
                "readers": PUBLIC,
            },
        ]


def utility_ids(family_id, original_ids):
    # Target IDs must not win or lose all score ties simply because of a prefix.
    return {key: digest(["utility-public-id-v1", family_id, key]) for key in original_ids}


def acquire(root):
    folder = Path(root) / "data" / "external"
    metadata = download(SCIFACT_URL, folder / "scifact.zip", max_bytes=100_000_000)
    # BEIR publishes this archive checksum. SHA-256 is also recorded by download().
    published_md5 = "5f7d1de60b170fc8027bb7898e2efca1"
    with (folder / "scifact.zip").open("rb") as fp:
        actual_md5 = hashlib.file_digest(fp, "md5").hexdigest()
    if actual_md5 != published_md5:
        raise ValueError(
            "SciFact archive differs from the declared BEIR release; review before changing protocol"
        )
    license_file = folder / "scifact-LICENSE-upstream.md"
    license_metadata = download(
        "https://raw.githubusercontent.com/allenai/scifact/master/LICENSE.md",
        license_file,
        max_bytes=100_000,
    )
    with zipfile.ZipFile(folder / "scifact.zip") as archive:
        if sum(z.file_size for z in archive.infolist()) > 150_000_000:
            raise ValueError("External corpus expansion exceeds bound")
        for entry in archive.infolist():
            target = (folder / entry.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError("Unsafe ZIP member")
            if not entry.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(entry))
    metadata.update(
        {
            "dataset": "BEIR/SciFact",
            "split": "test",
            "official_qrels_preserved": True,
            "published_md5_verified": actual_md5,
            "upstream_license_record": license_metadata,
            "source": "https://github.com/beir-cellar/beir/wiki/Datasets-available",
            "upstream": "https://github.com/allenai/scifact",
            "license_components": {
                "upstream_claims_and_evidence_annotations": "CC BY 4.0",
                "upstream_S2ORC_abstracts": "ODC-By 1.0",
                "upstream_code": "Apache 2.0",
            },
            "BEIR_dataset_card_license": "CC BY-SA 4.0 (distribution metadata; retain upstream component notices)",
            "BEIR_license_source": "https://huggingface.co/datasets/BeIR/scifact/blob/b3b5335604bf5ee3c4447671af975ea25143d4f5/README.md",
        }
    )
    write_json(folder / "scifact-acquisition.json", metadata)
    return metadata


def chunk_document(text, tokenizer, target_tokens=128):
    """Sentence-first deterministic chunks; split oversized sentences at token offsets."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    pieces = []
    for sentence in sentences:
        offsets = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        if len(offsets) <= target_tokens:
            pieces.append(sentence)
        else:
            for i in range(0, len(offsets), target_tokens):
                selected = offsets[i : i + target_tokens]
                pieces.append(sentence[selected[0][0] : selected[-1][1]])
    chunks, current = [], ""
    for piece in pieces:
        trial = (current + " " + piece).strip()
        if current and len(tokenizer(trial, add_special_tokens=False)["input_ids"]) > target_tokens:
            chunks.append(current)
            current = piece
        else:
            current = trial
    if current:
        chunks.append(current)
    if not chunks:
        raise ValueError("Empty external document")
    return [{"id": str(i), "text": t, "readers": PUBLIC} for i, t in enumerate(chunks)]


def windows(chunks, tokenizer, maximum=1024):
    current = []
    for chunk in chunks:
        trial = current + [chunk]
        if len(tokenizer(joined(trial)[0], add_special_tokens=True)["input_ids"]) > maximum:
            if not current:
                raise ValueError("Single chunk exceeds context window")
            yield current
            current = [chunk]
        else:
            current = trial
    if current:
        yield current


def external_evaluate(root, encoder):
    folder = Path(root) / "data" / "external" / "scifact"
    corpus = list(read_jsonl(folder / "corpus.jsonl"))
    queries = {r["_id"]: r["text"] for r in read_jsonl(folder / "queries.jsonl")}
    qrels = {}
    with (folder / "qrels" / "test.tsv").open() as fp:
        for row in csv.DictReader(fp, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    documents = {r["_id"] for r in corpus}
    if any(not relevance <= documents or qid not in queries for qid, relevance in qrels.items()):
        raise ValueError("Missing judged document/query; do not silently drop qrels")
    if len(documents) != len(corpus):
        raise ValueError("Duplicate external document ID")
    matrices = {"E": [], "L": []}
    doc_indices, chunk_counts, window_count = [], [], 0
    for index, row in enumerate(corpus):
        chunks = chunk_document(
            (row.get("title", "") + ". " + row["text"]).strip(), encoder.tokenizer
        )
        chunk_counts.append(len(chunks))
        for window in windows(chunks, encoder.tokenizer, encoder.max_tokens):
            window_count += 1
            for method in matrices:
                vectors = document(encoder, window, method)
                matrices[method].extend(vectors[c["id"]] for c in window)
            doc_indices.extend([index] * len(window))
        if index % 500 == 0:
            print(f"SciFact encoded {index}/{len(corpus)} documents", flush=True)
    matrices = {k: np.stack(v) for k, v in matrices.items()}
    doc_indices = np.asarray(doc_indices)
    ids = [r["_id"] for r in corpus]
    results = []
    for qid in sorted(qrels):
        query = encoder.query(queries[qid])
        for method, matrix in matrices.items():
            scores = np.full(len(corpus), -np.inf)
            np.maximum.at(scores, doc_indices, matrix @ query)
            ranked = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:10]
            results.append(
                {
                    "query_id": qid,
                    "method": method,
                    "ranked_document_ids": [ids[i] for i in ranked],
                    **retrieval_metrics([ids[i] for i in ranked], qrels[qid]),
                }
            )
    # Every external chunk is public: P and A have exactly the same encoder inputs
    # and pooling as L; the results therefore apply to P and A.
    return {
        "rows": results,
        "paired_bootstrap": external_bootstrap(results, qrels),
        "documents": len(corpus),
        "queries": len(qrels),
        "chunks": sum(chunk_counts),
        "windows": window_count,
        "equivalences": {
            "P": "L (all-public input equality)",
            "A": "L (all-public input equality)",
        },
        "ranking": "maximum chunk cosine per original document; deduplicate before official qrels",
    }


def external_bootstrap(rows, qrels, repetitions=2000, seed=20261010):
    import pandas as pd

    qids = sorted(qrels)
    parent = {q: q for q in qids}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    by_document = {}
    for qid in qids:
        for did in qrels[qid]:
            if did in by_document:
                parent[find(qid)] = find(by_document[did])
            else:
                by_document[did] = qid
    clusters = {}
    for i, qid in enumerate(qids):
        clusters.setdefault(find(qid), []).append(i)
    groups = list(clusters.values())
    frame = pd.DataFrame(rows)
    rng, report = np.random.default_rng(seed), {}
    draws = [
        np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        for _ in range(repetitions)
    ]
    for metric in ["recall10", "ndcg10"]:
        pivot = frame.pivot(index="query_id", columns="method", values=metric).reindex(qids)
        if pivot.isna().any().any():
            raise ValueError("Incomplete paired external retrieval results")
        a, b = pivot.E.to_numpy(), pivot.L.to_numpy()
        report[metric] = {
            "E": float(a.mean()),
            "L": float(b.mean()),
            "L_minus_E": float((b - a).mean()),
            "paired_ci95": np.quantile(
                [float((b - a)[d].mean()) for d in draws], [0.025, 0.975]
            ).tolist(),
        }
    return {
        "metrics": report,
        "clusters": len(groups),
        "queries": len(qids),
        "repetitions": repetitions,
        "unit": "connected query clusters sharing a judged relevant source document; query-weighted mean",
    }
