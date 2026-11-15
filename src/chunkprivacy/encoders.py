from __future__ import annotations

from itertools import groupby
from pathlib import Path
import os
import platform
import time

import numpy as np

from researchkit.io import digest, file_hash, save_array

MODELS = {
    "small": {
        "name": "jinaai/jina-embeddings-v2-small-en",
        "revision": "44e7d1d6caec8c883c2d4b207588504d519788d0",
    },
    "base": {
        "name": "jinaai/jina-embeddings-v2-base-en",
        "revision": "322d4d7e2f35e84137961a65af894fda0385eb7a",
    },
}
CODE_REVISION = "f3ec4cf7de7e561007f27c9efc7148b0bd713f81"
POOLING = "content-only-wholly-contained-token-mean-l2-v2"


def joined(chunks):
    texts, spans, cursor = [], [], 0
    for chunk in chunks:
        if texts:
            cursor += 2
        texts.append(chunk["text"])
        spans.append((cursor, cursor + len(chunk["text"])))
        cursor = spans[-1][1]
    return "\n\n".join(texts), spans


def span_indices(offsets, special, attention, spans):
    result = []
    for start, end in spans:
        selected = []
        for i, ((a, b), sp, mask) in enumerate(zip(offsets, special, attention, strict=True)):
            if sp or not mask or a == b:
                continue
            overlaps = a < end and b > start
            if overlaps and not (start <= a < b <= end):
                raise ValueError("Token straddles a declared character span")
            if overlaps:
                selected.append(i)
        if not selected:
            raise ValueError("Empty content span after tokenization")
        result.append(selected)
    return result


class Encoder:
    def __init__(self, model, cache_dir, *, max_tokens=1024, threads=4, cache=True):
        # Keep downloaded pinned remote implementation code inside the portable tree.
        os.environ.setdefault("HF_MODULES_CACHE", str(Path(cache_dir).resolve() / "modules"))
        os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_dir).resolve() / "huggingface"))
        import torch
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import snapshot_download
        from importlib.metadata import version

        self.spec = {
            **MODELS[model],
            "code_revision": CODE_REVISION,
            "pooling": POOLING,
            "device": "cpu",
            "dtype": "float32",
            "max_tokens": max_tokens,
            "threads": threads,
            "encoder_source_sha256": file_hash(__file__),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "os": platform.system(),
            "architecture": platform.machine(),
        }
        self.spec["transformers_version"] = version("transformers")
        self.spec["tokenizers_version"] = version("tokenizers")
        self.cache_dir, self.cache = Path(cache_dir), cache
        self.max_tokens = max_tokens
        self.calls = self.cache_hits = 0
        self.forward_seconds = 0.0
        torch.set_num_threads(threads)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        start = time.perf_counter()
        snapshot = snapshot_download(
            self.spec["name"],
            revision=self.spec["revision"],
            cache_dir=str(self.cache_dir / "huggingface"),
            allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            code_revision=CODE_REVISION,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir / "huggingface"),
            use_fast=True,
        )
        if not self.tokenizer.is_fast:
            raise ValueError("Character-span pooling requires a fast tokenizer")
        self.model = AutoModel.from_pretrained(
            snapshot,
            code_revision=CODE_REVISION,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir / "huggingface"),
            torch_dtype=torch.float32,
        )
        self.model.to("cpu").eval()
        self.load_seconds = time.perf_counter() - start

    def encode(self, chunks, *, method="query"):
        import torch

        if not chunks:
            raise ValueError("Cannot encode an empty input")
        text, spans = joined(chunks)
        key = digest(
            {
                "spec": self.spec,
                "text": text,
                "spans": spans,
                "method": method,
                "acl": [c.get("readers", []) for c in chunks],
            }
        )
        path = self.cache_dir / "vectors" / f"{key}.npy"
        if self.cache and path.exists():
            self.cache_hits += 1
            result = np.load(path, allow_pickle=False)
            if (
                result.shape != (len(chunks), self.model.config.hidden_size)
                or not np.isfinite(result).all()
            ):
                raise ValueError("Invalid embedding cache entry")
            return result
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=False,
        )
        if encoded["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("Input exceeds protocol token limit; silent truncation is forbidden")
        offsets = encoded.pop("offset_mapping")[0].tolist()
        special = encoded.pop("special_tokens_mask")[0].tolist()
        groups = span_indices(offsets, special, encoded["attention_mask"][0].tolist(), spans)
        start = time.perf_counter()
        with torch.inference_mode():
            states = self.model(**encoded).last_hidden_state[0]
            pooled = torch.stack([states[g].mean(0) for g in groups])
            norms = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
            if torch.any(norms <= 0):
                raise ValueError("Zero embedding")
            values = (pooled / norms).cpu().numpy().astype(np.float32)
        self.forward_seconds += time.perf_counter() - start
        self.calls += 1
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite embedding")
        if self.cache:
            save_array(path, values)
        return values

    def query(self, text):
        return self.encode([{"text": text}], method="query")[0]

    def profile(self):
        return {
            "model": self.spec,
            "model_load_seconds": self.load_seconds,
            "uncached_encoder_calls": self.calls,
            "cache_hits": self.cache_hits,
            "forward_seconds": self.forward_seconds,
            "cache_enabled": self.cache,
        }


def document(encoder, chunks, method, principal="reader"):
    allowed = [c for c in chunks if principal in c["readers"]]
    if method == "E":
        groups = [[c] for c in allowed]
    elif method in ("L", "L0"):
        work = chunks
        if method == "L0":
            work = [
                {**c, "text": "Confidential office record: the selected value is undisclosed."}
                if principal not in c["readers"]
                else c
                for c in chunks
            ]
        groups = [work]
    elif method == "P":
        # Equality of complete ACL sets is intentionally more conservative than
        # inclusion; noncontiguous runs are not joined across forbidden text.
        groups = [
            list(group) for _, group in groupby(chunks, lambda c: tuple(sorted(c["readers"])))
        ]
        groups = [g for g in groups if principal in g[0]["readers"]]
    elif method == "A":
        groups = [allowed]
    else:
        raise ValueError(f"Unknown method: {method}")
    vectors = {}
    for group in groups:
        if not group:
            continue
        for chunk, vector in zip(group, encoder.encode(group, method=method), strict=True):
            if principal in chunk["readers"]:
                vectors[chunk["id"]] = vector
    return vectors
