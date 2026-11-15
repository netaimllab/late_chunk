"""Verify package contents and compressed observation integrity without dependencies."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for relative, expected in manifest.items():
        p = (ROOT / relative).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            raise ValueError(f"Missing or invalid package member: {relative}")
        with p.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {relative}")
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    for name, record in provenance["observations"].items():
        with gzip.open(ROOT / "data/observations" / name, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != record["uncompressed_sha256"]:
                raise ValueError(f"Uncompressed observation mismatch: {name}")
    return len(manifest)


if __name__ == "__main__":
    print(f"Verified {check()} files and all original observation checksums.")
