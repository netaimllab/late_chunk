from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
import numpy as np


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(
        path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    )


def read_json(path):
    return json.loads(Path(path).read_text())


def write_jsonl(path, rows):
    # Write the bounded family and observation records atomically.
    atomic_bytes(path, "".join(canonical(row) + "\n" for row in rows).encode())


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def stable_seed(*parts):
    return int(digest([p.item() if isinstance(p, np.generic) else p for p in parts])[:16], 16)


def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as fp:
        temporary = Path(fp.name)
        np.save(fp, array, allow_pickle=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download(url, path, *, expected_hash=None, max_bytes=1_000_000_000):
    path = Path(path)
    if not url.startswith("https://"):
        raise ValueError("Acquisition requires HTTPS")
    if path.exists():
        actual = file_hash(path)
        if expected_hash and actual != expected_hash:
            raise ValueError(f"Existing download hash mismatch: {path.name}")
        if path.stat().st_size > max_bytes:
            raise ValueError("Existing download exceeds configured size limit")
        return {"url": url, "sha256": actual, "bytes": path.stat().st_size}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    h = hashlib.sha256()
    size = 0
    try:
        with urlopen(
            Request(url, headers={"User-Agent": "late-chunking-attribute-inference/1.0"}), timeout=60
        ) as src:
            advertised = src.headers.get("Content-Length")
            if advertised and int(advertised) > max_bytes:
                raise ValueError("Download exceeds configured size limit")
            with temporary.open("wb") as dst:
                while block := src.read(1024 * 1024):
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError("Download exceeds configured size limit")
                    dst.write(block)
                    h.update(block)
            if advertised and size != int(advertised):
                raise ValueError("Incomplete download")
        if expected_hash and h.hexdigest() != expected_hash:
            raise ValueError("Downloaded hash mismatch")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"url": url, "sha256": h.hexdigest(), "bytes": size}
