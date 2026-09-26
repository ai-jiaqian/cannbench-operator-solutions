#!/usr/bin/env python3
"""Verify every published byte against the recorded official-source hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_files(directory: Path) -> None:
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, expected in provenance["files"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"source hash mismatch: {directory / name}")


def main() -> None:
    catalog = json.loads((ROOT / "results/current.json").read_text())
    rows = catalog["operators"]
    if len(rows) != 53 or len({row["operator_key"] for row in rows}) != 53:
        raise ValueError("operator catalog is not complete")
    for row in rows:
        directory = ROOT / "operators" / row["operator_key"]
        provenance = json.loads((directory / "provenance.json").read_text())
        if any(provenance[key] != row[key] for key in row):
            raise ValueError(f"official metadata mismatch: {row['operator_key']}")
        verify_files(directory)
    print("Verified 53 operator source trees.")


if __name__ == "__main__":
    main()
